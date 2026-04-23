
# 生成完整的框架总结文档和架构图说明
summary_doc = """
# 通用AI项目框架 - 完整架构说明

## 一、框架核心设计哲学

### 1. 分层架构（Layered Architecture）
```
┌─────────────────────────────────────────────────────────────┐
│                    Agentic Loop Layer                        │
│  (目标管理 → 搜索/学习 → 实践验证 → 差异分析 → 贝叶斯更新)      │
├─────────────────────────────────────────────────────────────┤
│                    Task Layer                                │
│  (Classification | Prediction | RL | Agent Task)             │
├─────────────────────────────────────────────────────────────┤
│                    Training Layer                            │
│  (Supervised | Self-Supervised | RL | Self-Play Strategies)    │
├─────────────────────────────────────────────────────────────┤
│                    Model Layer                                 │
│  (BaseModel → Architectures | Layers | Neural ODE)           │
├─────────────────────────────────────────────────────────────┤
│                    Data Layer                                  │
│  (Datasets | Transforms | Loaders | Preprocess Pipeline)     │
├─────────────────────────────────────────────────────────────┤
│                    Evaluation Layer                            │
│  (Metrics | Validators | Discrepancy Analyzers)            │
└─────────────────────────────────────────────────────────────┘
```

### 2. 设计模式应用
- **工厂模式**: ModelFactory, FeederFactory, DataLoaderFactory
- **策略模式**: TrainingStrategy (Supervised/RL/SelfPlay)
- **注册表模式**: ModelRegistry, Registry (通用)
- **观察者模式**: TrainingCallback 系统
- **管道模式**: PreprocessPipeline, Compose (数据变换)

### 3. Agentic工作流状态机
```
[Global Goal]
    │
    ▼ (Decompose)
[Sub-Goal 1] ←──────→ [Sub-Goal 2] ←──────→ [Sub-Goal N]
    │                      │                      │
    ▼                      ▼                      ▼
[Search Samples]      [Search Samples]      [Search Samples]
    │                      │                      │
    ▼                      ▼                      ▼
[Exogenous] or [Endogenous] Learning
    │                      │                      │
    ▼                      ▼                      ▼
[Practice Verify]      [Practice Verify]      [Practice Verify]
    │                      │                      │
    ▼                      ▼                      ▼
[Analyze Discrepancy]  [Analyze Discrepancy]  [Analyze Discrepancy]
    │                      │                      │
    ▼                      ▼                      ▼
[Bayesian Update]      [Bayesian Update]      [Bayesian Update]
    │                      │                      │
    └──────────────────────┼──────────────────────┘
                           ▼
              [Stopping Criteria Check]
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        [Continue Refine]         [Mark Tech Debt]
              │                         │
              ▼                         ▼
        [Next Iteration]          [Next Sub-Goal]
```

## 二、各模块详细说明

### 2.1 Agentic Loop Engine (src/core/)

**agentic_loop.py** - 核心引擎
- `AgenticLoop`: 主循环控制器
- `Goal`: 目标节点（支持嵌套）
- `Belief`: 信念单元（贝叶斯更新）
- `StoppingCriteria`: 停机判断（成本-收益分析）

**关键特性**:
- 粗糙容忍度：先完成Demo再循环改进
- 理解层级：循环次数越多，理解层级越高
- 选择性更新：取其精华，去其糟粕

### 2.2 数据层 (src/data/)

**base_dataset.py**
- `BaseDataset`: 抽象基类，统一接口
- `SkeletonDataset`: 骨架数据专用（NTU RGB+D兼容）
- `FeederFactory`: 兼容InfoGCN的feeder模式
- `DataLoaderFactory`: 统一DataLoader创建

**数据变换组件**:
- `SkeletonNormalize`: 归一化
- `SkeletonRandomRotate/Scale`: 数据增强
- `SkeletonMultiModalRepresentation`: 多模态表示（joint/bone/motion）
- `Compose`: 变换管道组合

### 2.3 模型层 (src/models/)

**model_base.py**
- `BaseModel`: 抽象基类（参数量统计、保存/加载）
- `ModelRegistry`: 装饰器注册机制
- `ModelFactory`: 自动实例化

**可复用组件 (layers/)**:
- `SAGraphConv`: 自注意力图卷积（InfoGCN核心）
- `MultiScaleTemporalConv`: 多尺度时间卷积
- `NeuralODEFunction`: 神经ODE函数（InfoGCN++）

**完整架构 (architectures/)**:
- `InfoGCN`: 基础骨架识别
- `InfoGCNPlus`: 在线识别+未来预测
- `RLAgent`: 强化学习网络

### 2.4 训练层 (src/training/)

**trainer.py**
- `UniversalTrainer`: 通用训练器
- `TrainingState`: 状态管理（断点续训）
- `TrainingCallback`: 回调接口

**训练策略 (strategies/)**:
- `SupervisedStrategy`: 监督学习（支持多任务）
- `SelfPlayStrategy`: AlphaGo式自我博弈
- `RLStrategy`: PPO/SAC强化学习

**回调 (callbacks/)**:
- `CheckpointCallback`: 检查点保存
- `LoggingCallback`: 日志记录
- `EarlyStoppingCallback`: 早停

### 2.5 评估层 (src/evaluation/)

**evaluator.py**
- `Evaluator`: 统一评估器
- `Metric`: 指标基类

**指标库 (metrics/)**:
- `AccuracyMetric`: Top-K准确率
- `SkeletonRecognitionMetric`: 混淆矩阵+每类准确率
- `OnlineRecognitionMetric`: 不同观察比例下的性能
- `RLMetric`: 回合奖励/成功率

**分析器 (analyzers/)**:
- `DiscrepancyAnalyzer`: 实验反馈与理解差异分析
- `ErrorAnalyzer`: 错误模式分析

### 2.6 Agent系统 (src/agents/)

**base_agent.py**
- `BaseAgent`: 抽象基类
- `GoalManager`: 目标分解与管理
- `ShortTermMemory`: 短期记忆（FIFO）
- `LongTermMemory`: 长期记忆（持久化+索引）
- `ToolRegistry`: 工具注册与调用

**工具 (tools/)**:
- `CodeExecutionTool`: 代码执行
- `SearchTool`: 外部搜索

### 2.7 配置系统 (config/)

**base_config.py**
- `ExperimentConfig`: 顶层配置（数据/模型/训练/Agentic）
- `ConfigManager`: YAML加载/保存/合并

## 三、使用场景示例

### 场景1: 构建InfoGCN++骨架识别系统

```bash
# 1. 准备数据
# 2. 配置实验
# 3. 启动训练
python scripts/train.py --config config/experiment/infogcn_like.yaml

# 4. 评估
python scripts/eval.py --config config/experiment/infogcn_like.yaml --checkpoint checkpoints/infogcn/best.pt
```

### 场景2: Agentic目标驱动开发

```bash
# 启用Agentic模式，自动搜索、学习、验证
python scripts/train.py --config config/experiment/infogcn_like.yaml --agentic
```

工作流程:
1. 框架自动搜索GitHub/ArXiv上的InfoGCN相关项目
2. 分析优秀样本的代码结构和设计决策
3. 生成初步实现（粗糙容忍度）
4. 运行训练验证
5. 分析实验结果与预期的差异
6. 贝叶斯更新理解
7. 判断是否继续优化或标记技术债

### 场景3: 强化学习Agent训练

```bash
# 自我博弈模式
python scripts/train.py --config config/experiment/agent_rl.yaml --self_play
```

### 场景4: 从零开始新项目

```python
# 1. 定义配置
config = {
    "model_name": "MyCustomModel",
    "dataset_type": "my_data",
    "strategy": "supervised"
}

# 2. 注册组件
@ModelRegistry.register("MyCustomModel")
class MyCustomModel(BaseModel):
    def build(self): ...
    def forward(self, x): ...

# 3. 启动训练
trainer = UniversalTrainer(model, strategy, config)
trainer.fit(train_loader, val_loader)
```

## 四、关键设计决策

### 4.1 为什么使用注册表模式？
- **解耦**: 模型定义与使用分离
- **扩展性**: 新模型只需添加装饰器，无需修改工厂
- **配置驱动**: 通过字符串名称即可实例化任意模型

### 4.2 为什么分离TrainingStrategy？
- **多范式支持**: 同一模型可用不同策略训练
- **实验便利**: 快速切换监督/自监督/RL
- **代码复用**: 通用训练器逻辑与具体策略分离

### 4.3 为什么内置Agentic循环？
- **目标驱动**: 所有开发活动围绕明确目标
- **快速迭代**: 粗糙容忍度避免过早优化
- **持续学习**: 外源+内源学习循环改进
- **理性决策**: 成本-收益分析指导资源分配

### 4.4 如何处理技术债？
当满足以下条件时标记技术债:
- Demo已跑通但优化成本过高
- 当前目标不阻碍后续进展
- 成本-收益性价比低于阈值

技术债记录到长期记忆，后续循环中可重新评估。

## 五、扩展接口

### 5.1 添加自定义模型
```python
from src.models.model_base import BaseModel, ModelRegistry

@ModelRegistry.register("CustomModel")
class CustomModel(BaseModel):
    def build(self):
        # 定义层
        pass
    
    def forward(self, x):
        # 定义前向传播
        pass
```

### 5.2 添加自定义数据集
```python
from src.data.data_base import BaseDataset

class CustomDataset(BaseDataset):
    def _load_data(self):
        # 加载数据到self.samples
        pass
    
    def _get_item(self, index):
        # 返回{"inputs": ..., "labels": ...}
        pass
```

### 5.3 添加自定义训练策略
```python
from src.training.trainer import TrainingStrategy

class CustomStrategy(TrainingStrategy):
    def setup(self, model, config): ...
    def train_step(self, batch, model, state): ...
    def validate(self, dataloader, model): ...
```

## 六、与参考项目的对应关系

### InfoGCN++ (stnoah1/infogcn2)
| 参考项目组件 | 框架对应位置 | 说明 |
|------------|-----------|------|
| feeder.py | src/data/data_base.py | FeederFactory兼容 |
| model.py | src/models/architectures/infogcn.py | InfoGCN/InfoGCNPlus |
| main.py | scripts/train.py | 统一入口 |
| graph/ntu_rgb_d.py | src/models/layers/graph_conv.py | 图结构定义 |
| SA-GC | src/models/layers/graph_conv.py | SAGraphConv |
| MS-TC | src/models/layers/temporal.py | MultiScaleTemporalConv |
| Neural ODE | src/models/layers/neural_ode.py | NeuralODEFunction |

### Agent项目
| Agent需求 | 框架对应位置 | 说明 |
|----------|-----------|------|
| 目标分解 | src/agents/goal_manager.py | GoalManager |
| 外源学习 | src/search/ | 搜索与分析模块 |
| 内源学习 | src/training/strategies/self_play.py | SelfPlayStrategy |
| 贝叶斯更新 | src/core/belief_updater.py | BeliefUpdater |
| 记忆系统 | src/agents/memory/ | Short/LongTermMemory |
| 工具调用 | src/agents/tools/ | ToolRegistry |

## 七、文件清单

### 核心框架文件（已生成）
1. `agentic_loop.py` - Agentic循环引擎
2. `trainer.py` - 通用训练器
3. `model_base.py` - 模型基类与工厂
4. `data_base.py` - 数据层基类
5. `evaluator.py` - 评估与差异分析
6. `config.py` - 配置系统
7. `train.py` - 主训练入口
8. `agent_system.py` - Agent核心组件
9. `utils.py` - 工具库
10. `infogcn_example.py` - InfoGCN++实现示例

### 配置文件
11. `infogcn_like.yaml` - InfoGCN++实验配置
12. `requirements.txt` - 依赖列表
13. `README.md` - 项目说明

## 八、后续扩展建议

### 短期（Demo级）
- [ ] 实现具体数据加载器（NTU RGB+D, NW-UCLA）
- [ ] 添加更多数据增强变换
- [ ] 实现分布式训练支持
- [ ] 添加TensorBoard/WandB日志

### 中期（优化级）
- [ ] 集成更多RL算法（SAC, TD3, DQN）
- [ ] 添加模型压缩工具（剪枝、量化）
- [ ] 实现AutoML超参搜索
- [ ] 添加多模态融合支持

### 长期（系统级）
- [ ] 接入LLM进行代码生成/分析
- [ ] 实现自动论文阅读与理解
- [ ] 构建知识图谱存储学习成果
- [ ] 支持跨项目经验迁移
"""

with open('/mnt/agents/output/FRAMEWORK_GUIDE.md', 'w', encoding='utf-8') as f:
    f.write(summary_doc)

print("=" * 60)
print("🎉 通用AI项目框架构建完成！")
print("=" * 60)
print("\n📦 已生成文件清单:")
print("-" * 60)

files = [
    ("agentic_loop.py", "Agentic循环引擎（9步工作流）"),
    ("trainer.py", "通用训练器（多范式支持）"),
    ("model_base.py", "模型基类与工厂（注册表模式）"),
    ("data_base.py", "数据层基类（骨架/CV/通用）"),
    ("evaluator.py", "评估器与差异分析器"),
    ("config.py", "配置系统（OmegaConf/YAML）"),
    ("train.py", "主训练入口脚本"),
    ("agent_system.py", "Agent系统（目标/记忆/工具）"),
    ("utils.py", "工具库（种子/日志/注册表）"),
    ("infogcn_example.py", "InfoGCN++完整实现示例"),
    ("infogcn_like.yaml", "InfoGCN++实验配置模板"),
    ("requirements.txt", "项目依赖列表"),
    ("README.md", "项目说明文档"),
    ("FRAMEWORK_GUIDE.md", "框架架构详细指南")
]

for i, (fname, desc) in enumerate(files, 1):
    print(f"  {i:2d}. {fname:25s} - {desc}")

print("-" * 60)
print(f"\n总计: {len(files)} 个核心文件")
print("\n所有文件路径: sandbox:///mnt/agents/output/")
print("\n💡 使用建议:")
print("   1. 复制框架到项目目录")
print("   2. 根据具体任务修改 config/experiment/ 下的YAML配置")
print("   3. 使用 scripts/train.py --config <config> [--agentic] 启动")
print("   4. 参考 infogcn_example.py 实现具体模型架构")
