from configs.tinystory import *
from configs.mathAdder import *
from configs.mathMulitplier import *
from configs.base import BaseExperimentConfig

# 实验注册表
EXPERIMENT_REGISTRY = {
    'TinyStory_SFT': lambda model: TinyStoryLlamaConfig() if model == 'llama' else TinyStoryNanoGPTConfig(),
    'Adder_SFT': lambda model: AdderSFTConfig(model),
    'Adder_RL': lambda model: AdderRLConfig(model),
    'Multiplier_SFT': lambda model: MultiplierSFTConfig(model),
    'Multiplier_RL': lambda model: MultiplierRLConfig(model),
}

def get_experiment_config(exp_setting) -> BaseExperimentConfig:
    """获取实验配置"""
    exp_name = exp_setting.pop('name')
    if exp_name not in EXPERIMENT_REGISTRY:
        available = list(EXPERIMENT_REGISTRY.keys())
        raise ValueError(f"Unknown experiment: {exp_name}. Available: {available}")
    
    return EXPERIMENT_REGISTRY[exp_name](**exp_setting).get_args_ready()
