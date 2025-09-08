#!/usr/bin/env python3
"""
Math RL Training Script with DDP Support

Usage:
    # Single GPU
    python rl_ddp.py --config AdderRLConfig
    
    # Multi-GPU DDP
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 rl_ddp.py --config AdderRLConfig
"""

import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import argparse
import torch
import wandb
import setproctitle
from torch.distributed import init_process_group, destroy_process_group
from utils.utils import set_seed, create_folder_if_not_exist
from configs import get_experiment_config
from rl.math_trainer import MathRLTrainer
setproctitle.setproctitle("MathRL@Training")

def ddp_setup():
    num_cores = os.cpu_count()
    num_threads = max(1, min(4, num_cores // 4))    # Each process uses part of the CPU cores
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MASTER_ADDR"] = "localhost"         # localhost for single node
    os.environ["MASTER_PORT"] = "21662"             # Any free port
    init_process_group(backend="nccl")              # nccl for linux, gloo for windows
    torch.cuda.set_device(int(os.environ.get("RANK", default='0')))

def setup_model_and_args(config_name: str, world_size: int, rank: int):
    """设置模型和参数"""
    # 临时清除命令行参数，避免与base config的parse_args冲突
    import sys
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]  # 只保留脚本名
    
    try:
        # 获取配置
        config = get_experiment_config(config_name)
        
        # 设置world_size用于配置计算
        config.world_size = world_size
        args = config.get_args_ready()
    finally:
        # 恢复原始命令行参数
        sys.argv = original_argv
    
    # 设置分布式相关参数
    args.world_size = world_size
    args.rank = rank
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # 设置输出目录
    if not hasattr(args, 'out_dir'):
        args.out_dir = f'{base_path}/output'
    
    create_folder_if_not_exist(args.out_dir)
    
    # 加载模型
    if args.model == 'NanoGPT':
        from model.NanoGPT import NanoGPT, NanoGPTConfig
        
        # 创建模型配置
        model_config = NanoGPTConfig(
            vocab_size=13,  # 0-9 + math symbols
            n_position=args.n_position,
            n_embed=args.n_embed,
            n_layer=args.n_layer,
            n_head=args.n_head,
            dropout=args.dropout,
            dropout_attn=args.dropout_attn,
            weight_tying=args.weight_tying,
            add_bias=args.add_bias
        )
        
        model = NanoGPT(model_config)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # 如果有预训练模型，加载它
    if hasattr(args, 'init_from') and args.init_from:
        print(f"Loading pretrained model from {args.init_from}")
        checkpoint = torch.load(args.init_from, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
    
    return args, model

def main():
    parser = argparse.ArgumentParser(description='Math RL Training')
    parser.add_argument('--config', type=str, required=True, 
                       choices=['AdderRLConfig', 'MultiplierRLConfig'],
                       help='Configuration class name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--debug', action='store_true', help='Debug mode (fewer steps)')
    
    args_cmd = parser.parse_args()
    
    # init DDP process group
    ddp_setup()
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", default='1'))
    RANK = int(os.environ.get("RANK", default='0'))

    # activate tf32 on matmul and cudnn to boost NVIDIA Ampere GPU performance
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True 
    
    # 设置随机种子
    set_seed(args_cmd.seed + rank)
    
    # 设置模型和参数
    args, model = setup_model_and_args(args_cmd.config, WORLD_SIZE, rank)
    
    # 调试模式
    if args_cmd.debug:
        args.step_num = min(args.step_num, 20)
        args.eval_interval = 5
        args.save_interval = 10
    
    # 创建训练器
    trainer = MathRLTrainer(args, model, device_id=RANK)
    
    # 设置wandb（仅在rank 0）
    if rank == 0 and not args_cmd.no_wandb and args.wandb:
        wandb_id = wandb.util.generate_id()
        trainer.wandb_id = wandb_id
        trainer.wandb_enabled = True
        
        wandb.init(
            project=args.wandb_project,
            name=f"{args.exp_name}_seed{args_cmd.seed}",
            id=wandb_id,
            config={
                'config_name': args_cmd.config,
                'seed': args_cmd.seed,
                'world_size': WORLD_SIZE,
                **{k: v for k, v in vars(args).items() if not k.startswith('_')}
            },
            resume='allow'
        )
        
        # 监控模型
        raw_model = trainer.raw_model
        wandb.watch(raw_model, log='parameters', log_freq=100)
    else:
        trainer.wandb_enabled = False
    
    # 打印配置信息（仅在rank 0）
    if rank == 0:
        print(f"\n=== Math RL Training Configuration ===")
        print(f"Config: {args_cmd.config}")
        print(f"Dataset: {args.dataset}")
        print(f"Model: {args.model}")
        print(f"Steps: {args.step_num}")
        print(f"Batch size per GPU: {args.q_batch_size_per_gpu}")
        print(f"Total batch size: {args.q_batch_size_per_gpu * WORLD_SIZE}")
        print(f"Learning rate: {args.lr_max}")
        print(f"World size: {WORLD_SIZE}")
        print(f"Device: cuda:{RANK}" if torch.cuda.is_available() else "cpu")
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"="*50)
    
    try:
        # 开始训练
        best_accuracy = trainer.train()
        
        if rank == 0:
            print(f"\nTraining completed successfully!")
            print(f"Best accuracy: {best_accuracy:.4f}")
            
            # 记录最终结果
            if trainer.wandb_enabled:
                wandb.log({'final_best_accuracy': best_accuracy})
    
    except KeyboardInterrupt:
        if rank == 0:
            print("\nTraining interrupted by user")
    
    except Exception as e:
        if rank == 0:
            print(f"\nTraining failed with error: {e}")
            import traceback
            traceback.print_exc()
        raise
    
    finally:
        # 清理
        if rank == 0 and trainer.wandb_enabled:
            wandb.finish()
        
        if is_ddp:
            destroy_process_group()

if __name__ == "__main__":
    main()