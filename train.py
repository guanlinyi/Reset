
# 核心代码文件6: 配置系统与主入口脚本
config_code = '''"""
Configuration System - 配置中心

使用OmegaConf/YAML管理所有超参数
支持：
- 分层配置（基础配置 + 实验配置 + 覆盖）
- 命令行参数自动映射
- 配置验证
- 实验跟踪集成
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """数据配置"""
    dataset_type: str = "skeleton"           # skeleton, image, text, rl_env
    data_path: str = "./data"
    batch_size: int = 32
    num_workers: int = 4
    num_classes: int = 60
    num_nodes: int = 25
    num_persons: int = 2
    window_size: int = 52                   # 时序窗口大小
    transform_config: Dict[str, Any] = field(default_factory=lambda: {
        "normalize": True,
        "random_rotate": True,
        "random_scale": True,
        "multimodal": False
    })


@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "InfoGCN"
    in_channels: int = 3                    # 输入通道数（骨架坐标xyz）
    hidden_dim: int = 64
    num_layers: int = 9                     # 编码器层数
    num_heads: int = 3                      # SA-GC多头数
    dropout: float = 0.1
    use_neural_ode: bool = False            # 是否使用神经ODE（InfoGCN++）
    graph_type: str = "ntu"                 # 图结构类型


@dataclass
class TrainingConfig:
    """训练配置"""
    strategy: str = "supervised"            # supervised, self_play, reinforcement
    num_epochs: int = 70
    learning_rate: float = 0.1
    weight_decay: float = 0.0003
    scheduler: str = "step"                 # step, cosine, plateau
    step_size: List[int] = field(default_factory=lambda: [50, 60])
    lr_decay: float = 0.1
    grad_clip: float = 1.0
    device: str = "cuda"
    mixed_precision: bool = True            # 混合精度训练
    seed: int = 1


@dataclass
class AgenticConfig:
    """Agentic工作流配置"""
    enabled: bool = True
    global_goal: str = "Build skeleton action recognition system"
    tolerance_mode: str = "rough"           # rough（粗糙容忍）/ strict（严格）
    max_iterations: int = 10
    cost_threshold: float = 5.0             # 成本-收益比阈值
    search_sources: List[str] = field(default_factory=lambda: ["github", "arxiv", "paperswithcode"])


@dataclass
class ExperimentConfig:
    """实验配置 - 顶层配置"""
    name: str = "default_experiment"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)
    
    # 附加配置
    eval_frequency: int = 1                 # 每几个epoch评估一次
    save_frequency: int = 5                 # 每几个epoch保存检查点
    wandb_project: Optional[str] = None


class ConfigManager:
    """配置管理器"""
    
    @staticmethod
    def from_yaml(path: str) -> ExperimentConfig:
        """从YAML文件加载配置"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # 递归构建配置对象
        data_cfg = DataConfig(**config_dict.get("data", {}))
        model_cfg = ModelConfig(**config_dict.get("model", {}))
        train_cfg = TrainingConfig(**config_dict.get("training", {}))
        agentic_cfg = AgenticConfig(**config_dict.get("agentic", {}))
        
        return ExperimentConfig(
            name=config_dict.get("name", "default"),
            output_dir=config_dict.get("output_dir", "./outputs"),
            data=data_cfg,
            model=model_cfg,
            training=train_cfg,
            agentic=agentic_cfg
        )
    
    @staticmethod
    def to_yaml(config: ExperimentConfig, path: str):
        """保存配置到YAML"""
        config_dict = {
            "name": config.name,
            "data": config.data.__dict__,
            "model": config.model.__dict__,
            "training": config.training.__dict__,
            "agentic": config.agentic.__dict__
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @staticmethod
    def merge(base: ExperimentConfig, overrides: Dict[str, Any]) -> ExperimentConfig:
        """合并配置覆盖"""
        # 简化的合并逻辑
        for key, value in overrides.items():
            if hasattr(base, key):
                setattr(base, key, value)
        return base
'''

# 主入口脚本: train.py
main_train_code = '''#!/usr/bin/env python3
"""
训练入口脚本 - 通用AI项目训练启动器

支持模式：
1. 标准训练（监督学习）
2. Agentic训练（目标驱动循环）
3. 自我博弈训练（AlphaGo模式）

用法：
    python scripts/train.py --config config/experiment/default.yaml
    python scripts/train.py --config config/experiment/infogcn_like.yaml --agentic
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.agentic_loop import AgenticLoop, Goal
from src.data.data_base import FeederFactory, DataLoaderFactory
from src.models.model_base import ModelFactory
from src.training.trainer import UniversalTrainer, SupervisedStrategy, SelfPlayStrategy
from src.evaluation.evaluator import Evaluator, AccuracyMetric, SkeletonRecognitionMetric
from src.utils.seed import set_seed
from src.utils.logging import setup_logging
from config.base_config import ConfigManager, ExperimentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Universal AI Project Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--agentic", action="store_true", help="Enable Agentic learning loop")
    parser.add_argument("--self_play", action="store_true", help="Enable self-play mode")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 加载配置
    config = ConfigManager.from_yaml(args.config)
    
    # 设置日志和随机种子
    setup_logging(config.log_dir)
    set_seed(config.training.seed)
    
    logger.info("=" * 60)
    logger.info(f"Experiment: {config.name}")
    logger.info(f"Mode: {'Agentic' if args.agentic else 'Standard'}")
    logger.info("=" * 60)
    
    # Agentic模式：启动目标驱动循环
    if args.agentic:
        agentic_loop = AgenticLoop(config=config.__dict__)
        global_goal = Goal(
            id="root",
            description=config.agentic.global_goal,
            tolerance=config.agentic.tolerance_mode
        )
        completed_goal = agentic_loop.run(global_goal)
        logger.info("Agentic loop completed!")
        return
    
    # 标准训练模式
    # 1. 创建数据Feeder
    train_dataset = FeederFactory.create(
        dataset_type=config.data.dataset_type,
        data_path=config.data.data_path,
        split="train",
        config=config.data.transform_config,
        num_classes=config.data.num_classes,
        num_nodes=config.data.num_nodes,
        num_persons=config.data.num_persons
    )
    
    val_dataset = FeederFactory.create(
        dataset_type=config.data.dataset_type,
        data_path=config.data.data_path,
        split="val",
        config=config.data.transform_config,
        num_classes=config.data.num_classes,
        num_nodes=config.data.num_nodes,
        num_persons=config.data.num_persons
    )
    
    train_loader = DataLoaderFactory.create(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers
    )
    
    val_loader = DataLoaderFactory.create(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )
    
    # 2. 创建模型
    model = ModelFactory.create(config.model.__dict__)
    model.summary()
    
    # 3. 选择训练策略
    if args.self_play:
        strategy = SelfPlayStrategy()
    else:
        strategy = SupervisedStrategy()
    
    # 4. 创建评估器
    evaluator = Evaluator()
    if config.data.dataset_type == "skeleton":
        evaluator.add_metric(SkeletonRecognitionMetric(config.data.num_classes))
    else:
        evaluator.add_metric(AccuracyMetric(topk=(1, 5)))
    
    # 5. 创建训练器并启动训练
    trainer = UniversalTrainer(
        model=model,
        strategy=strategy,
        config=config.training.__dict__,
        callbacks=[
            # 可添加回调
        ]
    )
    
    # 断点续训
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    final_state = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.num_epochs
    )
    
    # 保存最终模型
    final_checkpoint = Path(config.checkpoint_dir) / "final_model.pt"
    trainer.save_checkpoint(str(final_checkpoint))
    
    logger.info(f"Training completed! Best metric: {final_state.best_metric:.4f}")
    logger.info(f"Final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
'''

# 保存配置文件示例
infogcn_config = '''
name: "infogcn_skeleton_recognition"
output_dir: "./outputs/infogcn"
checkpoint_dir: "./checkpoints/infogcn"
log_dir: "./logs/infogcn"

data:
  dataset_type: "skeleton"
  data_path: "./data/ntu_rgbd"
  batch_size: 32
  num_workers: 4
  num_classes: 60
  num_nodes: 25
  num_persons: 2
  window_size: 52
  transform_config:
    normalize: true
    random_rotate: true
    random_scale: true
    multimodal: true  # 启用多模态表示

model:
  model_name: "InfoGCN"
  in_channels: 3
  hidden_dim: 64
  num_layers: 9
  num_heads: 3
  dropout: 0.1
  use_neural_ode: false  # InfoGCN++设为true
  graph_type: "ntu"

training:
  strategy: "supervised"
  num_epochs: 70
  learning_rate: 0.1
  weight_decay: 0.0003
  scheduler: "step"
  step_size: [50, 60]
  lr_decay: 0.1
  grad_clip: 1.0
  device: "cuda"
  mixed_precision: true
  seed: 1

agentic:
  enabled: true
  global_goal: "Build online skeleton action recognition system with InfoGCN++"
  tolerance_mode: "rough"
  max_iterations: 10
  cost_threshold: 5.0
  search_sources:
    - "github"
    - "arxiv"
    - "paperswithcode"

eval_frequency: 1
save_frequency: 5
'''

# 保存所有文件
with open('/mnt/agents/output/config.py', 'w', encoding='utf-8') as f:
    f.write(config_code)

with open('/mnt/agents/output/train.py', 'w', encoding='utf-8') as f:
    f.write(main_train_code)

with open('/mnt/agents/output/infogcn_like.yaml', 'w', encoding='utf-8') as f:
    f.write(infogcn_config)

print("✅ config.py 已生成")
print("✅ train.py 已生成")
print("✅ infogcn_like.yaml 已生成")
