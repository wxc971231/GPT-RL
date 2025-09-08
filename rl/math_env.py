import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from data.adder.prepare import AdditionDataset, AdditionTokenizer
from data.multiplier.prepare import MultiplicationDataset, MultiplicationTokenizer
from configs import get_experiment_config

@dataclass
class MathProblem:
    """数学问题数据结构"""
    input_seq: torch.Tensor     # 输入序列 (问题部分)
    target_seq: torch.Tensor    # 目标序列 (完整序列)
    answer: torch.Tensor        # 答案部分
    raw_problem: str            # 原始问题字符串
    raw_answer: str             # 原始答案字符串
    a: int                      # 第一个操作数
    b: int                      # 第二个操作数
    result: int                 # 正确结果

class MathEnvironment:    
    def __init__(self, dataset_type: str = "adder", ndigit: int = 3, format_vocab: dict = None):
        self.dataset_type = dataset_type
        self.ndigit = ndigit
        self.device = f'cuda:{int(os.environ.get("LOCAL_RANK", default="0"))}'
        self.format_vocab = format_vocab
        self.use_format = format_vocab is not None
        
        # 创建tokenizer
        if dataset_type == "adder":
            self.tokenizer = AdditionTokenizer(ndigit=ndigit, format_vocab=self.format_vocab)
        elif dataset_type == "multiplier":
            self.tokenizer = MultiplicationTokenizer(ndigit=ndigit, format_vocab=self.format_vocab)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
            
    def generate_problems(self, batch_size: int, seed: Optional[int] = None) -> List[MathProblem]:
        """生成一批数学问题"""
        if seed is not None:
            np.random.seed(seed)
            
        # 向量化生成随机操作数
        max_val = 10 ** self.ndigit - 1
        min_val = 0
        
        # 一次性生成所有随机数
        a_values = np.random.randint(min_val, max_val + 1, size=batch_size)
        b_values = np.random.randint(min_val, max_val + 1, size=batch_size)
        
        # 向量化计算结果
        if self.dataset_type == "adder":
            results = a_values + b_values
        elif self.dataset_type == "multiplier":
            results = a_values * b_values
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
            
        # 批量创建问题
        problems = []
        for a, b, result in zip(a_values, b_values, results):
            problem = self._create_problem(int(a), int(b), int(result))
            problems.append(problem)
            
        return problems
    
    def _create_problem(self, a: int, b: int, result: int) -> MathProblem:
        """创建单个问题"""
        if self.dataset_type == "adder":
            if self.use_format:
                # 格式: "a+b=result" (result reversed)
                raw_problem = f"{a}+{b}="
                raw_answer = str(result)[::-1]  # 反转答案
                
                # 转换为token序列
                input_tokens = []
                for char in str(a).zfill(self.ndigit):
                    input_tokens.append(int(char))
                input_tokens.append(self.format_vocab['+'])
                for char in str(b).zfill(self.ndigit):
                    input_tokens.append(int(char))
                input_tokens.append(self.format_vocab['='])
                
                answer_tokens = []
                answer_str = str(result)[::-1].zfill(self.ndigit + 1)
                for char in answer_str:
                    answer_tokens.append(int(char))
            else:
                # 无格式: 直接拼接数字
                raw_problem = f"{a}{b}"
                raw_answer = str(result)[::-1]
                
                input_tokens = []
                for char in str(a).zfill(self.ndigit):
                    input_tokens.append(int(char))
                for char in str(b).zfill(self.ndigit):
                    input_tokens.append(int(char))
                    
                answer_tokens = []
                answer_str = str(result)[::-1].zfill(self.ndigit + 1)
                for char in answer_str:
                    answer_tokens.append(int(char))
                    
        elif self.dataset_type == "multiplier":
            if self.use_format:
                # 格式: "axb=result" (result reversed and padded)
                raw_problem = f"{a}x{b}="
                raw_answer = str(result)[::-1]
                
                input_tokens = []
                for char in str(a).zfill(self.ndigit):
                    input_tokens.append(int(char))
                input_tokens.append(self.format_vocab['x'])
                for char in str(b).zfill(self.ndigit):
                    input_tokens.append(int(char))
                input_tokens.append(self.format_vocab['='])
                
                answer_tokens = []
                answer_str = str(result)[::-1].zfill(self.ndigit * 2)
                for char in answer_str:
                    answer_tokens.append(int(char))
            else:
                raw_problem = f"{a}{b}"
                raw_answer = str(result)[::-1]
                
                input_tokens = []
                for char in str(a).zfill(self.ndigit):
                    input_tokens.append(int(char))
                for char in str(b).zfill(self.ndigit):
                    input_tokens.append(int(char))
                    
                answer_tokens = []
                answer_str = str(result)[::-1].zfill(self.ndigit * 2)
                for char in answer_str:
                    answer_tokens.append(int(char))
        
        # 转换为tensor
        input_seq = torch.tensor(input_tokens, dtype=torch.long)
        answer_seq = torch.tensor(answer_tokens, dtype=torch.long)
        target_seq = torch.cat([input_seq, answer_seq], dim=0)
        
        return MathProblem(
            input_seq=input_seq,
            target_seq=target_seq,
            answer=answer_seq,
            raw_problem=raw_problem,
            raw_answer=raw_answer,
            a=a,
            b=b,
            result=result
        )
    
    def evaluate_answers(self, problems: List[MathProblem], generated_answers: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """评估生成的答案
        Args:
            problems: 问题列表
            generated_answers: 生成的答案 (batch_size, answer_len)
            
        Returns:
            rewards: 奖励张量 (batch_size,)
            metrics: 评估指标字典
        """
        batch_size = len(problems)
        rewards = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        correct_count = 0
        
        for i, problem in enumerate(problems):
            generated_answer = generated_answers[i]
            
            # 解码生成的答案
            if self.dataset_type == "adder":
                decoded_answer = self.tokenizer.decode(
                    problem.input_seq.unsqueeze(0),
                    torch.cat([problem.input_seq, generated_answer], dim=0).unsqueeze(0)
                )
            elif self.dataset_type == "multiplier":
                decoded_answer = self.tokenizer.decode(
                    problem.input_seq.unsqueeze(0),
                    torch.cat([problem.input_seq, generated_answer], dim=0).unsqueeze(0)
                )
            
            # 计算奖励
            if decoded_answer[0].item() == 1:  # 正确
                rewards[i] = 1.0
                correct_count += 1
            else:  # 错误
                rewards[i] = 0.0
                
        # 计算指标
        accuracy = correct_count / batch_size
        metrics = {
            'accuracy': accuracy,
            'avg_reward': rewards.mean().item(),
            'correct_count': correct_count,
            'total_count': batch_size
        }
        
        return rewards, metrics

if __name__ == "__main__":
    # 测试加法环境    
    print("Testing Adder Environment...")
    experiment_name = "Adder_RL"
    args = get_experiment_config(experiment_name)
    adder_env = MathEnvironment("adder", ndigit=2, format_vocab=args.adder_format_vocab)
    problems = adder_env.generate_problems(5, seed=42)
    
    for i, problem in enumerate(problems):
        print(f"Problem {i+1}:")
        print(f"  Raw: {problem.raw_problem}{problem.raw_answer}")
        print(f"  Input: {problem.input_seq}")
        print(f"  Answer: {problem.answer}")
        print(f"  Target: {problem.target_seq}")
        print(f"  a={problem.a}, b={problem.b}, result={problem.result}")
        print()
    
    # 测试乘法环境
    print("Testing Multiplication Environment...")
    experiment_name = "Multiplier_RL"
    args = get_experiment_config(experiment_name)
    mult_env = MathEnvironment("multiplier", ndigit=2, format_vocab=args.multiplier_format_vocab)
    problems = mult_env.generate_problems(3, seed=42)
    
    for i, problem in enumerate(problems):
        print(f"Problem {i+1}:")
        print(f"  Raw: {problem.raw_problem}{problem.raw_answer}")
        print(f"  Input: {problem.input_seq}")
        print(f"  Answer: {problem.answer}")
        print(f"  Target: {problem.target_seq}")
        print(f"  a={problem.a}, b={problem.b}, result={problem.result}")
        print()