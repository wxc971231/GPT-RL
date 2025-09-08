import torch
import numpy as np
from typing import List, Tuple, Union

def compute_math_reward(problems: List[str], predictions: List[str], dataset_type: str = "adder") -> torch.Tensor:
    """
    计算数学问题的奖励
    
    Args:
        problems: 原始问题列表，如 ["123+456=", "789*012="]
        predictions: 模型预测列表，如 ["579", "9468"]
        dataset_type: 数据集类型，"adder" 或 "multiplier"
    
    Returns:
        奖励张量，正确为1.0，错误为0.0
    """
    rewards = []
    for problem, prediction in zip(problems, predictions):
        try:
            if dataset_type == "adder":
                reward = _compute_adder_reward(problem, prediction)
            elif dataset_type == "multiplier":
                reward = _compute_multiplier_reward(problem, prediction)
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")
            rewards.append(reward)
        except Exception:
            # 如果解析失败，给予0奖励
            rewards.append(0.0)
    
    return torch.tensor(rewards, dtype=torch.float32)

def _compute_adder_reward(problem: str, prediction: str) -> float:
    """
    计算加法问题的奖励
    
    Args:
        problem: 如 "123+456="
        prediction: 如 "579"
    
    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    # 解析问题
    if '=' in problem:
        equation = problem.replace('=', '').strip()
    else:
        equation = problem.strip()
    
    if '+' not in equation:
        return 0.0
    
    parts = equation.split('+')
    if len(parts) != 2:
        return 0.0
    
    try:
        num1 = int(parts[0].strip())
        num2 = int(parts[1].strip())
        correct_answer = num1 + num2
        
        predicted_answer = int(prediction.strip())
        
        return 1.0 if predicted_answer == correct_answer else 0.0
    except ValueError:
        return 0.0

def _compute_multiplier_reward(problem: str, prediction: str) -> float:
    """
    计算乘法问题的奖励
    
    Args:
        problem: 如 "123*456="
        prediction: 如 "56088"
    
    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    # 解析问题
    if '=' in problem:
        equation = problem.replace('=', '').strip()
    else:
        equation = problem.strip()
    
    # 支持 'x' 和 '*' 两种乘法符号
    if 'x' in equation:
        parts = equation.split('x')
    elif '*' in equation:
        parts = equation.split('*')
    else:
        return 0.0
    
    if len(parts) != 2:
        return 0.0
    
    try:
        num1 = int(parts[0].strip())
        num2 = int(parts[1].strip())
        correct_answer = num1 * num2
        
        predicted_answer = int(prediction.strip())
        
        return 1.0 if predicted_answer == correct_answer else 0.0
    except ValueError:
        return 0.0

def parse_math_sequence(sequence: List[int], vocab_size: int = 10, 
                       math_vocab: dict = None, dataset_type: str = "adder") -> Tuple[str, str]:
    """
    将token序列解析为问题和答案
    
    Args:
        sequence: token序列
        vocab_size: 基础词汇表大小（数字0-9）
        math_vocab: 数学符号词汇表，如 {'=': 10, '+': 11, 'x': 12}
        dataset_type: 数据集类型
    
    Returns:
        (problem, answer) 元组
    """
    if math_vocab is None:
        math_vocab = {'=': 10, '+': 11, 'x': 12}
    
    # 创建反向词汇表
    reverse_vocab = {v: k for k, v in math_vocab.items()}
    
    # 转换序列为字符串
    tokens = []
    for token in sequence:
        if token < vocab_size:
            tokens.append(str(token))
        elif token in reverse_vocab:
            tokens.append(reverse_vocab[token])
        else:
            tokens.append('?')  # 未知token
    
    sequence_str = ''.join(tokens)
    
    # 找到等号位置分割问题和答案
    if '=' in sequence_str:
        parts = sequence_str.split('=', 1)
        problem = parts[0] + '='
        answer = parts[1] if len(parts) > 1 else ''
    else:
        # 如果没有等号，尝试根据数据集类型推断分割点
        if dataset_type == "adder" and '+' in sequence_str:
            # 对于加法，假设格式为 "num1+num2answer"
            plus_idx = sequence_str.index('+')
            # 找到第二个数字后的位置
            problem_part = sequence_str[:plus_idx+1]
            remaining = sequence_str[plus_idx+1:]
            
            # 尝试找到第二个数字的结束位置
            i = 0
            while i < len(remaining) and remaining[i].isdigit():
                i += 1
            
            problem = problem_part + remaining[:i] + '='
            answer = remaining[i:]
        elif dataset_type == "multiplier" and ('x' in sequence_str or '*' in sequence_str):
            # 类似处理乘法
            mult_char = 'x' if 'x' in sequence_str else '*'
            mult_idx = sequence_str.index(mult_char)
            problem_part = sequence_str[:mult_idx+1]
            remaining = sequence_str[mult_idx+1:]
            
            i = 0
            while i < len(remaining) and remaining[i].isdigit():
                i += 1
            
            problem = problem_part + remaining[:i] + '='
            answer = remaining[i:]
        else:
            # 无法解析，返回原始字符串
            problem = sequence_str
            answer = ''
    
    return problem, answer

def batch_compute_math_reward(sequences: torch.Tensor, input_len: int, 
                             vocab_size: int = 10, math_vocab: dict = None,
                             dataset_type: str = "adder") -> torch.Tensor:
    """
    批量计算数学序列的奖励
    
    Args:
        sequences: 形状为 [batch_size, seq_len] 的token序列
        input_len: 输入部分的长度
        vocab_size: 基础词汇表大小
        math_vocab: 数学符号词汇表
        dataset_type: 数据集类型
    
    Returns:
        奖励张量，形状为 [batch_size]
    """
    batch_size = sequences.shape[0]
    rewards = []
    
    for i in range(batch_size):
        sequence = sequences[i].cpu().numpy().tolist()
        problem, answer = parse_math_sequence(sequence, vocab_size, math_vocab, dataset_type)
        
        reward = compute_math_reward([problem], [answer], dataset_type)[0].item()
        rewards.append(reward)
    
    return torch.tensor(rewards, dtype=torch.float32, device=sequences.device)