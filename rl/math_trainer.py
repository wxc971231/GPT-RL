import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import torch
import torch.nn.functional as F
import numpy as np
import wandb
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import OrderedDict
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributions import Categorical

from rl.math_reward import batch_compute_math_reward, parse_math_sequence
from train.scheduler import OptimizerParamScheduler
from utils.utils import set_seed

@dataclass
class MathRLSnapshot:
    model_state: OrderedDict
    optimizer_state: Dict
    scheduler_state: Dict
    total_steps: int
    trained_time: float
    best_accuracy: float
    wandb_id: str

class MathRLTrainer:
    def __init__(self, args, model, device_id: int = 0):
        self.args = args
        self.model = model
        self.device_id = device_id
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        
        # 移动模型到设备
        self.model = self.model.to(self.device)
        
        # 如果是分布式训练，包装模型
        if hasattr(args, 'world_size') and args.world_size > 1:
            self.model = DDP(self.model, device_ids=[device_id])
            self.raw_model = self.model.module
        else:
            self.raw_model = self.model
        
        # 设置优化器
        self.optimizer = self._setup_optimizer()
        self.scheduler = OptimizerParamScheduler(args, self.optimizer)
        
        # 训练状态
        self.step = 0
        self.best_accuracy = 0.0
        self.trained_time = 0.0
        
        # 数学问题相关设置
        self.dataset_type = args.dataset
        self.math_vocab = getattr(args, 'math_vocab', {'=': 10, '+': 11, 'x': 12})
        self.vocab_size = 10  # 数字0-9
        
        # 计算序列长度
        if self.dataset_type == "adder":
            ndigit = getattr(args, 'adder_ndigit', 3)
            use_format = getattr(args, 'adder_use_format', True)
            self.input_len = ndigit * 2 + 2 if use_format else ndigit * 2
            self.answer_len = ndigit + 1
        elif self.dataset_type == "multiplier":
            ndigit = getattr(args, 'multiplier_ndigit', 3)
            use_format = getattr(args, 'multiplier_use_format', True)
            self.input_len = ndigit * 2 + 2 if use_format else ndigit * 2
            self.answer_len = ndigit * 2
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
        
        self.total_len = self.input_len + self.answer_len
        
        # 快照路径
        self.snapshot_path = f'{args.out_dir}/rl_snapshot_{args.seeds[0]}.pt'
        
    def _setup_optimizer(self):
        """设置优化器"""
        # 获取可训练参数
        param_dict = {pn: p for pn, p in self.raw_model.named_parameters() if p.requires_grad}
        
        # 分离权重衰减和非权重衰减参数
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.args.wd_begin},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # 检查是否使用fused AdamW
        use_fused = 'fused' in torch.optim.AdamW.__doc__ and torch.cuda.is_available()
        
        optimizer = torch.optim.AdamW(
            optim_groups, 
            lr=self.args.lr_max,
            betas=(0.9, self.args.adam_beta2),
            fused=use_fused
        )
        
        return optimizer
    
    def generate_problems(self, batch_size: int) -> List[str]:
        """生成数学问题"""
        problems = []
        
        if self.dataset_type == "adder":
            ndigit = getattr(self.args, 'adder_ndigit', 3)
            use_format = getattr(self.args, 'adder_use_format', True)
            
            for _ in range(batch_size):
                # 生成随机数字
                num1 = np.random.randint(10**(ndigit-1), 10**ndigit)
                num2 = np.random.randint(10**(ndigit-1), 10**ndigit)
                
                if use_format:
                    problem = f"{num1}+{num2}="
                else:
                    problem = f"{num1}{num2}"
                
                problems.append(problem)
                
        elif self.dataset_type == "multiplier":
            ndigit = getattr(self.args, 'multiplier_ndigit', 3)
            use_format = getattr(self.args, 'multiplier_use_format', True)
            
            for _ in range(batch_size):
                # 生成随机数字
                num1 = np.random.randint(10**(ndigit-1), 10**ndigit)
                num2 = np.random.randint(10**(ndigit-1), 10**ndigit)
                
                if use_format:
                    problem = f"{num1}x{num2}="
                else:
                    problem = f"{num1}{num2}"
                
                problems.append(problem)
        
        return problems
    
    def problems_to_tokens(self, problems: List[str]) -> torch.Tensor:
        """将问题转换为token序列"""
        batch_size = len(problems)
        sequences = torch.zeros(batch_size, self.input_len, dtype=torch.long, device=self.device)
        
        # 创建反向词汇表
        reverse_vocab = {v: k for k, v in self.math_vocab.items()}
        
        for i, problem in enumerate(problems):
            tokens = []
            for char in problem:
                if char.isdigit():
                    tokens.append(int(char))
                elif char in self.math_vocab:
                    tokens.append(self.math_vocab[char])
                # 忽略其他字符
            
            # 截断或填充到input_len
            if len(tokens) > self.input_len:
                tokens = tokens[:self.input_len]
            else:
                tokens.extend([0] * (self.input_len - len(tokens)))  # 用0填充
            
            sequences[i] = torch.tensor(tokens, dtype=torch.long)
        
        return sequences
    
    def rollout(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """执行rollout采样"""
        self.model.eval()
        
        # 生成问题
        problems = self.generate_problems(batch_size)
        input_ids = self.problems_to_tokens(problems)
        
        # 生成答案
        with torch.no_grad():
            generated_ids = torch.zeros(batch_size, self.total_len, dtype=torch.long, device=self.device)
            generated_ids[:, :self.input_len] = input_ids
            
            log_probs = []
            
            for pos in range(self.input_len, self.total_len):
                # 前向传播
                logits, _ = self.model(generated_ids[:, :pos])
                
                # 获取下一个token的logits
                next_logits = logits[:, -1, :] / self.args.temperature
                
                # 采样
                probs = F.softmax(next_logits, dim=-1)
                dist = Categorical(probs)
                next_token = dist.sample()
                
                # 记录log概率
                log_probs.append(dist.log_prob(next_token))
                
                # 更新序列
                generated_ids[:, pos] = next_token
            
            log_probs = torch.stack(log_probs, dim=1)  # [batch_size, answer_len]
        
        # 计算奖励
        rewards = batch_compute_math_reward(
            generated_ids, self.input_len, self.vocab_size, 
            self.math_vocab, self.dataset_type
        )
        
        return generated_ids, log_probs, rewards
    
    def compute_policy_loss(self, generated_ids: torch.Tensor, old_log_probs: torch.Tensor, 
                           rewards: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """计算策略损失"""
        self.model.train()
        
        batch_size = generated_ids.shape[0]
        
        # 前向传播获取logits
        logits, _ = self.model(generated_ids[:, :-1], targets=generated_ids[:, 1:])
        
        # 提取答案部分的logits
        answer_logits = logits[:, self.input_len-1:self.input_len-1+self.answer_len, :]
        answer_targets = generated_ids[:, self.input_len:self.input_len+self.answer_len]
        
        # 计算当前策略的log概率
        log_probs = F.log_softmax(answer_logits, dim=-1)
        current_log_probs = log_probs.gather(2, answer_targets.unsqueeze(-1)).squeeze(-1)
        
        # 计算重要性采样比率
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # 计算优势（这里简化为直接使用奖励）
        advantages = rewards.unsqueeze(1).expand(-1, self.answer_len)
        
        # PPO损失
        clip_low, clip_high = self.args.clip_adv
        clipped_ratio = torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
        
        policy_loss1 = ratio * advantages
        policy_loss2 = clipped_ratio * advantages
        policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
        
        # 熵损失
        entropy = -(F.softmax(answer_logits, dim=-1) * log_probs).sum(dim=-1).mean()
        entropy_loss = -self.args.entropy_coef * entropy
        
        # 总损失
        total_loss = policy_loss + entropy_loss
        
        # 统计信息
        stats = {
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'entropy': entropy.item(),
            'ratio_mean': ratio.mean().item(),
            'advantages_mean': advantages.mean().item()
        }
        
        return total_loss, stats
    
    def train_step(self) -> Dict:
        """执行一个训练步骤"""
        start_time = time.time()
        
        # Rollout
        generated_ids, log_probs, rewards = self.rollout(self.args.q_group_size)
        
        # 计算准确率
        accuracy = rewards.mean().item()
        
        # 多次策略更新
        total_loss = 0.0
        all_stats = {}
        
        for update_idx in range(self.args.update_num):
            # 计算损失
            loss, stats = self.compute_policy_loss(generated_ids, log_probs, rewards)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 累积统计信息
            for key, value in stats.items():
                if key not in all_stats:
                    all_stats[key] = []
                all_stats[key].append(value)
        
        # 更新调度器
        self.scheduler.step()
        
        # 平均统计信息
        for key in all_stats:
            all_stats[key] = np.mean(all_stats[key])
        
        step_time = time.time() - start_time
        self.trained_time += step_time
        
        # 返回训练统计
        train_stats = {
            'step': self.step,
            'loss': total_loss / self.args.update_num,
            'accuracy': accuracy,
            'reward_mean': rewards.mean().item(),
            'reward_std': rewards.std().item(),
            'lr': self.optimizer.param_groups[0]['lr'],
            'step_time': step_time,
            **all_stats
        }
        
        return train_stats
    
    def evaluate(self, num_samples: int = 1000) -> Dict:
        """评估模型性能"""
        self.model.eval()
        
        total_rewards = []
        
        # 分批评估
        batch_size = min(self.args.q_group_size, num_samples)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for _ in range(num_batches):
                current_batch_size = min(batch_size, num_samples - len(total_rewards))
                if current_batch_size <= 0:
                    break
                
                _, _, rewards = self.rollout(current_batch_size)
                total_rewards.extend(rewards.cpu().numpy().tolist())
        
        total_rewards = total_rewards[:num_samples]
        
        eval_stats = {
            'eval_accuracy': np.mean(total_rewards),
            'eval_std': np.std(total_rewards),
            'eval_samples': len(total_rewards)
        }
        
        return eval_stats
    
    def save_snapshot(self):
        """保存训练快照"""
        snapshot = MathRLSnapshot(
            model_state=self.raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            total_steps=self.step,
            trained_time=self.trained_time,
            best_accuracy=self.best_accuracy,
            wandb_id=getattr(self, 'wandb_id', '')
        )
        
        os.makedirs(os.path.dirname(self.snapshot_path), exist_ok=True)
        torch.save(snapshot, self.snapshot_path)
    
    def load_snapshot(self):
        """加载训练快照"""
        if os.path.exists(self.snapshot_path):
            snapshot = torch.load(self.snapshot_path, map_location=self.device)
            
            self.raw_model.load_state_dict(snapshot.model_state)
            self.optimizer.load_state_dict(snapshot.optimizer_state)
            self.scheduler.load_state_dict(snapshot.scheduler_state)
            self.step = snapshot.total_steps
            self.trained_time = snapshot.trained_time
            self.best_accuracy = snapshot.best_accuracy
            
            return True
        return False
    
    def train(self):
        """主训练循环"""
        print(f"Starting RL training for {self.args.step_num} steps...")
        
        # 尝试加载快照
        if self.load_snapshot():
            print(f"Resumed from step {self.step}")
        
        for step in range(self.step, self.args.step_num):
            self.step = step
            
            # 训练步骤
            train_stats = self.train_step()
            
            # 评估
            if step % self.args.eval_interval == 0:
                eval_stats = self.evaluate()
                train_stats.update(eval_stats)
                
                # 更新最佳准确率
                if eval_stats['eval_accuracy'] > self.best_accuracy:
                    self.best_accuracy = eval_stats['eval_accuracy']
                    train_stats['best_accuracy'] = self.best_accuracy
            
            # 记录日志
            if hasattr(self, 'wandb_enabled') and self.wandb_enabled:
                wandb.log(train_stats)
            
            # 打印进度
            if step % 10 == 0 or step == self.args.step_num - 1:
                print(f"Step {step}: Loss={train_stats['loss']:.4f}, "
                      f"Accuracy={train_stats['accuracy']:.4f}, "
                      f"LR={train_stats['lr']:.2e}")
            
            # 保存快照
            if step % self.args.save_interval == 0 or step == self.args.step_num - 1:
                self.save_snapshot()
        
        print(f"Training completed! Best accuracy: {self.best_accuracy:.4f}")
        return self.best_accuracy