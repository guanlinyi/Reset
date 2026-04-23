# Reset

# 项目说明文档
readme_content = '''# Universal AI Project Framework

通用人工智能项目代码框架，适用于构建任何与AI相关的项目，包括深度学习、强化学习、Agent系统等。

## 核心设计理念

### 1. Agentic工作流集成
框架内置目标驱动的嵌套循环引擎，实现你描述的9步工作流：

```
1. 确立全局大目标（粗糙容忍度）
2. 目标分解（粗糙容忍度）
3. 获取外部信息/搜索优秀样本
4. 外源学习模式（模仿/推测/提问/质疑）
5. 内源学习模式（自我博弈/AlphaGo模式）
6. 实践验证
7. 分析差异
8. 修正/更新理解（贝叶斯更新）
9. 循环（停机判断）
```

### 2. 多范式统一支持
- **监督学习**：分类、回归、预测（如InfoGCN++骨架识别）
- **自监督学习**：对比学习、掩码预测
- **强化学习**：PPO、SAC等算法
- **自我博弈**：AlphaGo模式的内源学习

### 3. 模块化设计
- **配置中心**：OmegaConf/YAML集中管理所有超参数
- **注册表模式**：模型、数据集、优化器动态注册
- **策略模式**：训练策略可插拔切换
- **工厂模式**：根据配置自动实例化组件

## 项目结构

```
universal_ai_project/
├── config/                    # 配置中心
│   ├── base_config.py         # 配置管理器
│   └── experiment/            # 实验配置
│       ├── default.yaml
│       └── infogcn_like.yaml  # InfoGCN++配置示例
├── src/
│   ├── core/                  # Agentic工作流引擎
│   │   ├── agentic_loop.py    # 主循环引擎
│   │   ├── knowledge_base.py  # 知识库
│   │   └── belief_updater.py  # 贝叶斯更新
│   ├── data/                  # 数据处理层
│   │   ├── base_dataset.py    # 数据集基类
│   │   ├── transforms/        # 数据变换
│   │   └── loaders/           # 数据加载器
│   ├── models/                # 模型层
│   │   ├── base_model.py      # 模型基类与工厂
│   │   ├── layers/            # 可复用组件
│   │   └── architectures/     # 完整架构
│   ├── training/              # 训练引擎
│   │   ├── trainer.py         # 通用训练器
│   │   └── strategies/        # 训练策略
│   ├── evaluation/            # 评估与验证
│   │   ├── evaluator.py       # 统一评估器
│   │   └── analyzers/         # 差异分析器
│   ├── agents/                # Agent系统
│   │   ├── base_agent.py      # Agent基类
│   │   ├── goal_manager.py    # 目标管理
│   │   └── memory/            # 记忆系统
│   └── utils/                 # 工具库
│       ├── registry.py        # 注册表模式
│       └── seed.py            # 随机种子
└── scripts/                   # 入口脚本
    ├── train.py               # 训练入口
    ├── eval.py                # 评估入口
    └── agent_loop.py          # Agent主循环
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 标准训练模式（如InfoGCN++）

```bash
python scripts/train.py --config config/experiment/infogcn_like.yaml
```

### 3. Agentic训练模式（目标驱动循环）

```bash
python scripts/train.py --config config/experiment/infogcn_like.yaml --agentic
```

### 4. 自我博弈模式

```bash
python scripts/train.py --config config/experiment/agent_rl.yaml --self_play
```

## 使用示例

### 定义新模型

```python
from src.models.model_base import BaseModel, ModelRegistry

@ModelRegistry.register("MyModel")
class MyModel(BaseModel):
    def build(self):
        # 定义架构
        pass
    
    def forward(self, x):
        # 定义前向传播
        pass
```

### 定义新数据集

```python
from src.data.data_base import BaseDataset

class MyDataset(BaseDataset):
    def _load_data(self):
        # 加载数据
        pass
    
    def _get_item(self, index):
        # 返回单个样本
        pass
```

### 使用Agentic循环

```python
from src.core.agentic_loop import AgenticLoop, Goal

loop = AgenticLoop(config={})
global_goal = Goal(
    id="root",
    description="Build state-of-the-art skeleton recognition system",
    tolerance="rough"
)
completed = loop.run(global_goal)
```

## 配置说明

配置文件使用YAML格式，支持分层覆盖：

```yaml
name: "my_experiment"

data:
  dataset_type: "skeleton"
  batch_size: 32

model:
  model_name: "InfoGCNPlus"
  use_neural_ode: true

training:
  strategy: "supervised"
  num_epochs: 70
  learning_rate: 0.1

agentic:
  enabled: true
  global_goal: "Build online skeleton action recognition"
  tolerance_mode: "rough"
```

## 核心特性

### 粗糙容忍度（Rough Tolerance）
- 先完成Demo，再循环改进
- 避免过早优化
- 快速验证核心假设

### 外源学习（Exogenous Learning）
- 自动搜索GitHub/ArXiv优秀样本
- 分析代码：模仿、推测、提问、质疑
- 理解层级随循环提升

### 内源学习（Endogenous Learning）
- AlphaGo式自我博弈
- 探索-利用平衡
- 自主发现策略

### 贝叶斯更新
- 根据实验反馈调整置信度
- 有选择地更新理解
- 取其精华，去其糟粕

## 扩展指南

### 添加新的训练策略

在 `src/training/strategies/` 中继承 `TrainingStrategy`：

```python
class MyStrategy(TrainingStrategy):
    def setup(self, model, config):
        pass
    
    def train_step(self, batch, model, state):
        pass
```

### 添加新的评估指标

在 `src/evaluation/metrics/` 中继承 `Metric`：

```python
class MyMetric(Metric):
    def compute(self, predictions, targets):
        pass
    
    def reset(self):
        pass
```

## 参考项目

- **InfoGCN++**: [stnoah1/infogcn2](https://github.com/stnoah1/infogcn2) - 在线骨架动作识别
- **AlphaGo**: 自我博弈与蒙特卡洛树搜索
- **PPO/SAC**: 强化学习算法

## 许可证

MIT License
'''

requirements_content = '''
torch>=1.9.0
numpy>=1.19.0
tqdm>=4.60.0
tensorboardX>=2.4
wandb>=0.12.0
einops>=0.4.0
omegaconf>=2.1.0
pyyaml>=5.4.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
pandas>=1.2.0

# 可选依赖
# torchdiffeq>=0.2.0  # 神经ODE（InfoGCN++需要）
# gym>=0.21.0         # 强化学习环境
# transformers>=4.20.0 # NLP模型
'''

with open('/mnt/agents/output/README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

with open('/mnt/agents/output/requirements.txt', 'w', encoding='utf-8') as f:
    f.write(requirements_content)

print("✅ README.md 已生成")
print("✅ requirements.txt 已生成")
