# 数学任务强化学习训练系统

本系统为加法和乘法任务实现了完整的强化学习训练框架，基于REINFORCE算法，支持分布式训练和多种配置选项。

## 系统架构

### 核心组件

1. **环境系统** (`math_env.py`)
   - `MathEnvironment`: 数学问题生成和评估
   - `MathProblem`: 问题数据结构
   - 支持加法和乘法任务

2. **奖励系统** (`math_reward.py`)
   - 基于规则的奖励函数
   - 精确匹配评估
   - 支持自定义奖励策略

3. **RL训练器** (`math_trainer.py`)
   - `MathRLTrainer`: 核心训练逻辑
   - REINFORCE算法实现
   - 快照保存/恢复机制

4. **分布式训练脚本** (`rl_ddp.py`)
   - 支持DDP分布式训练
   - 动态配置加载
   - Wandb集成

## 实现思路与技术细节

### 1. 架构设计理念

本系统采用模块化设计，将强化学习的各个组件解耦：
- **环境抽象**: 将数学问题生成与模型训练分离
- **奖励解耦**: 独立的奖励计算模块，便于扩展不同的评估策略
- **配置驱动**: 通过配置类动态控制训练参数和模型架构
- **分布式友好**: 原生支持多GPU训练，提高训练效率

### 2. 核心算法实现

#### REINFORCE算法
```python
# 策略梯度计算
policy_loss = -torch.mean(log_probs * advantages)
```

**关键技术点：**
- **优势函数计算**: 使用奖励减去基线来减少方差
- **基线估计**: 采用移动平均作为基线，提高训练稳定性
- **梯度裁剪**: 防止梯度爆炸，确保训练稳定

#### 采样策略
- **温度采样**: 通过temperature参数控制探索与利用的平衡
- **批量rollout**: 并行生成多个轨迹，提高样本效率
- **序列生成**: 使用autoregressive方式生成完整答案

### 3. 数学环境设计

#### 问题生成策略
```python
class MathEnvironment:
    def generate_problem(self, ndigit, operation):
        # 动态生成指定位数的数学问题
        # 支持加法和乘法运算
```

**设计特点：**
- **动态难度**: 根据ndigit参数生成不同难度的问题
- **格式统一**: 标准化的输入输出格式，便于模型学习
- **批量处理**: 支持批量问题生成，提高训练效率

### 4. 奖励函数设计

#### 精确匹配奖励
```python
def calculate_reward(prediction, target):
    return 1.0 if prediction.strip() == target.strip() else 0.0
```

**设计考虑：**
- **二元奖励**: 简单明确的0/1奖励，避免奖励稀疏性问题
- **字符串匹配**: 基于精确字符串匹配，确保答案正确性
- **可扩展性**: 预留接口支持更复杂的奖励函数

### 5. 分布式训练实现

#### DDP集成
```python
# 模型包装
model = DDP(model, device_ids=[local_rank])

# 梯度同步
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**技术要点：**
- **进程组管理**: 自动处理多进程通信
- **梯度同步**: 确保各GPU上的梯度一致性
- **负载均衡**: 动态分配训练任务到不同GPU

### 6. 配置系统设计

#### 动态配置加载
```python
def get_config_class(config_name):
    # 根据配置名动态导入对应的配置类
    module = importlib.import_module(f"configs.{module_name}")
    return getattr(module, class_name)
```

**设计优势：**
- **热插拔**: 无需修改代码即可切换不同配置
- **参数验证**: 自动验证配置参数的合法性
- **继承机制**: 支持配置继承，减少重复代码

### 7. 内存优化策略

#### 梯度累积
```python
for i in range(gradient_accumulation_steps):
    loss = loss / gradient_accumulation_steps
    loss.backward()
optimizer.step()
```

#### 动态批处理
- **自适应批大小**: 根据GPU内存动态调整批大小
- **序列长度优化**: 根据问题复杂度调整序列长度
- **内存回收**: 及时释放不需要的中间变量

### 8. 训练稳定性保障

#### 数值稳定性
- **梯度裁剪**: 防止梯度爆炸
- **学习率调度**: 动态调整学习率
- **权重初始化**: 合理的权重初始化策略

#### 训练监控
- **实时指标**: 训练过程中的关键指标监控
- **异常检测**: 自动检测训练异常并处理
- **检查点机制**: 定期保存训练状态，支持断点续训

## 快速开始

### 1. 环境准备

```bash
# 确保CUDA环境可用
nvidia-smi

# 安装依赖（如果需要）
pip install torch torchvision torchaudio wandb tqdm numpy
```

### 2. 单GPU训练

```bash
# 加法任务训练
python rl/rl_ddp.py --config_name AdderRLConfig --debug --no_wandb

# 乘法任务训练
python rl/rl_ddp.py --config_name MultiplierRLConfig --debug --no_wandb

# 启用Wandb监控
python rl/rl_ddp.py --config_name AdderRLConfig
```

### 3. 多GPU分布式训练

```bash
# 使用4个GPU训练加法任务
torchrun --nproc_per_node=4 rl/rl_ddp.py --config_name AdderRLConfig

# 使用2个GPU训练乘法任务
torchrun --nproc_per_node=2 rl/rl_ddp.py --config_name MultiplierRLConfig

# 指定GPU设备
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 rl/rl_ddp.py --config_name AdderRLConfig
```

### 4. 配置选项

```bash
# 调试模式（减少输出，快速验证）
python rl/rl_ddp.py --config_name AdderRLConfig --debug

# 禁用Wandb
python rl/rl_ddp.py --config_name AdderRLConfig --no_wandb

# 组合使用
python rl/rl_ddp.py --config_name MultiplierRLConfig --debug --no_wandb
```

## 配置参数详解

### 配置类选择
- `AdderRLConfig`: 加法任务配置
- `MultiplierRLConfig`: 乘法任务配置

### 模型配置 (在配置类中定义)
- `n_layer`: Transformer层数 (加法: 4, 乘法: 6)
- `n_head`: 注意力头数 (加法: 8, 乘法: 8)
- `n_embed`: 嵌入维度 (加法: 256, 乘法: 256)
- `n_position`: 最大序列长度 (加法: 64, 乘法: 128)
- `vocab_size`: 词汇表大小 (根据任务自动计算)

### RL训练配置
- `rl_train_steps`: 训练步数 (默认: 200)
- `rl_rollout_size`: 每步rollout数量 (默认: 8)
- `rl_batch_size`: 批次大小 (默认: 64)
- `rl_ga_steps`: 梯度累积步数 (默认: 2)
- `rl_temperature`: 采样温度 (默认: 1.0)

### 优化器配置
- `learning_rate`: 学习率 (默认: 1e-4)
- `weight_decay`: 权重衰减 (默认: 0.01)
- `adam_beta1`: Adam beta1参数 (默认: 0.9)
- `adam_beta2`: Adam beta2参数 (默认: 0.99)
- `grad_clip`: 梯度裁剪阈值 (默认: 1.0)

### 数据配置
- `ndigit`: 数字位数 (加法: 3, 乘法: 2)
- `use_format`: 是否使用格式化符号 (默认: True)
- `max_seq_len`: 最大序列长度 (自动计算)

## 算法说明

### REINFORCE算法
本系统实现了经典的REINFORCE策略梯度算法：

#### 核心原理
```python
# 策略梯度公式
∇θ J(θ) = E[∇θ log π(a|s) * A(s,a)]

# 实际实现
policy_loss = -torch.mean(log_probs * advantages)
```

#### 关键特性
- **策略梯度**: 直接优化策略参数，无需价值函数
- **蒙特卡洛采样**: 使用完整轨迹计算回报
- **基线减方差**: 使用移动平均基线减少梯度方差
- **无偏估计**: 保证梯度估计的无偏性

### 基线机制
为了减少梯度方差，系统采用移动平均基线：

```python
# 基线更新
baseline = 0.9 * baseline + 0.1 * current_reward

# 优势函数计算
advantages = rewards - baseline
```

**优势：**
- 减少方差，提高训练稳定性
- 保持梯度估计的无偏性
- 自适应调整，无需手动调参

## 输出文件结构

```
out/
├── rl_snapshot_*.pt     # 训练快照文件
└── wandb/               # Wandb日志目录 (如果启用)
    └── run-*/
        ├── files/
        └── logs/
```

**快照文件说明：**
- 自动保存训练状态，支持断点续训
- 包含模型参数、优化器状态、调度器状态
- 文件名格式：`rl_snapshot_{step}.pt`

## 监控和日志

### Wandb集成
- 实时记录训练指标和损失曲线
- 自动记录配置参数和系统信息
- 支持多实验对比和超参数分析
- 可通过 `--no_wandb` 参数禁用

### 关键指标
- `train/policy_loss`: REINFORCE策略损失
- `train/accuracy`: 当前批次准确率
- `train/avg_reward`: 平均奖励值
- `train/baseline`: 当前基线值
- `train/lr`: 当前学习率
- `train/step`: 训练步数

### 控制台输出
```
Step 50/200 | Loss: 0.234 | Acc: 0.750 | Reward: 0.750 | LR: 1.00e-04
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少批次大小和梯度累积
   # 修改配置类中的 rl_batch_size 和 rl_ga_steps
   ```

2. **模型加载失败 (size mismatch)**
   ```bash
   # 删除不兼容的快照文件
   rm out/rl_snapshot_*.pt
   ```

3. **训练不收敛**
   - 检查奖励函数是否正确
   - 调整学习率和训练步数
   - 确保数据生成逻辑正确

4. **分布式训练问题**
   ```bash
   # 检查GPU可见性
   nvidia-smi
   
   # 设置特定GPU
   CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 rl/rl_ddp.py --config_name AdderRLConfig
   ```

5. **Wandb连接问题**
   ```bash
   # 使用离线模式或禁用wandb
   python rl/rl_ddp.py --config_name AdderRLConfig --no_wandb
   ```

### 调试模式

```bash
# 快速验证训练流程
python rl/rl_ddp.py --config_name AdderRLConfig --debug --no_wandb

# 检查配置加载
python -c "from rl.rl_ddp import get_config_class; config = get_config_class('AdderRLConfig')(); print(config.get_args_ready())"

# 测试数学环境
python -c "from rl.math_env import MathEnvironment; env = MathEnvironment('adder', 3); print(env.generate_problem())"
```

## 扩展指南

### 添加新的数学任务
1. **扩展MathEnvironment类**
   ```python
   # 在 math_env.py 中添加新的操作类型
   def generate_problem(self, operation='subtraction'):
       # 实现减法问题生成逻辑
   ```

2. **创建新的配置类**
   ```python
   # 在 configs/ 目录下创建新的配置文件
   class SubtractionRLConfig(BaseRLConfig):
       def get_args_ready(self):
           # 定义减法任务的特定参数
   ```

3. **更新奖励函数**
   ```python
   # 在 math_reward.py 中添加任务特定的奖励逻辑
   def calculate_reward(self, prediction, target, task_type):
       # 实现任务特定的奖励计算
   ```

### 添加新的RL算法
1. **扩展MathRLTrainer类**
   ```python
   # 在 math_trainer.py 中添加新的损失函数
   def compute_ppo_loss(self, log_probs, old_log_probs, advantages):
       # 实现PPO损失计算
   ```

2. **更新配置参数**
   ```python
   # 在配置类中添加算法特定参数
   self.rl_algorithm = 'ppo'  # 或 'reinforce'
   self.ppo_clip_ratio = 0.2
   ```

### 自定义模型架构
1. **实现新的模型类**
   ```python
   # 在 model/ 目录下创建新的模型文件
   class CustomTransformer(nn.Module):
       def generate(self, idx, max_new_tokens, **kwargs):
           # 实现生成接口
   ```

2. **更新配置类**
   ```python
   # 在配置类中指定模型类型
   self.model_type = 'custom_transformer'
   ```

## 性能基准

### 当前实现性能

**3位数加法任务 (AdderRLConfig):**
- 模型架构: NanoGPT (4层, 8头, 256维)
- 训练步数: 200步
- 预期准确率: ~90-95%
- 训练时间: ~5-10分钟 (单GPU)

**2位数乘法任务 (MultiplierRLConfig):**
- 模型架构: NanoGPT (6层, 8头, 256维)
- 训练步数: 200步
- 预期准确率: ~80-90%
- 训练时间: ~8-15分钟 (单GPU)

### 系统要求
- **最低配置**: 1x GPU (4GB+ VRAM)
- **推荐配置**: 1x GPU (8GB+ VRAM)
- **分布式训练**: 2-8x GPU

## 技术栈

- **深度学习框架**: PyTorch 2.0+
- **分布式训练**: PyTorch DDP
- **实验管理**: Weights & Biases
- **数值计算**: NumPy
- **配置管理**: Python dataclass