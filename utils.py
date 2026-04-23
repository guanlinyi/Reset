
# 核心代码文件8: 实用工具库
utils_code = '''"""
Utilities - 通用工具库

包含：
- 随机种子管理
- 日志设置
- 检查点工具
- 可视化
- 分布式训练工具
- 注册表模式实现
"""

import logging
import os
import random
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.distributed as dist


# ============ 随机种子 ============

def set_seed(seed: int = 42):
    """
    设置全局随机种子，确保实验可复现
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set to {seed}")


# ============ 日志设置 ============

def setup_logging(
    log_dir: str = "./logs",
    level: int = logging.INFO,
    format_str: Optional[str] = None
):
    """
    设置日志系统
    
    Args:
        log_dir: 日志保存目录
        level: 日志级别
        format_str: 自定义格式字符串
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    if format_str is None:
        format_str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    formatter = logging.Formatter(format_str)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 文件处理器
    log_file = Path(log_dir) / f"train_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # 根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logging.info(f"Logging setup complete. Log file: {log_file}")


# ============ 注册表模式 ============

class Registry:
    """
    通用注册表 - 支持模块的动态注册和查找
    
    用于：
    - 模型注册
    - 数据集注册
    - 优化器注册
    - 损失函数注册
    """
    
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Any] = {}
    
    def register(self, name: str, obj: Any = None):
        """
        注册对象
        
        可用作装饰器：
            @registry.register("my_model")
            class MyModel:
                pass
        """
        if obj is not None:
            self._registry[name] = obj
            return obj
        
        def decorator(cls_or_fn):
            self._registry[name] = cls_or_fn
            return cls_or_fn
        return decorator
    
    def get(self, name: str) -> Any:
        """获取已注册的对象"""
        if name not in self._registry:
            raise KeyError(f"'{name}' not found in {self.name} registry. "
                          f"Available: {list(self._registry.keys())}")
        return self._registry[name]
    
    def list(self) -> Dict[str, str]:
        """列出所有注册项"""
        return {k: str(v) for k, v in self._registry.items()}
    
    def has(self, name: str) -> bool:
        """检查是否已注册"""
        return name in self._registry


# 创建全局注册表实例
MODEL_REGISTRY = Registry("model")
DATASET_REGISTRY = Registry("dataset")
OPTIMIZER_REGISTRY = Registry("optimizer")
LOSS_REGISTRY = Registry("loss")


# ============ 检查点工具 ============

class CheckpointManager:
    """检查点管理器"""
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_keep: int = 5,
        metric_name: str = "accuracy",
        mode: str = "max"
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.metric_name = metric_name
        self.mode = mode
        self.checkpoints: list = []  # [(metric_value, path)]
        
    def save(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        metric_value: float
    ) -> str:
        """
        保存检查点
        
        Args:
            state_dict: 模型状态字典
            epoch: 当前epoch
            metric_value: 评估指标值
            
        Returns:
            保存路径
        """
        filename = f"checkpoint_epoch{epoch}_{self.metric_name}{metric_value:.4f}.pt"
        filepath = self.checkpoint_dir / filename
        
        torch.save(state_dict, filepath)
        
        # 维护检查点列表
        self.checkpoints.append((metric_value, filepath))
        self.checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
        
        # 删除旧检查点
        if len(self.checkpoints) > self.max_keep:
            _, old_path = self.checkpoints.pop()
            if old_path.exists() and old_path != filepath:
                old_path.unlink()
        
        logging.info(f"Checkpoint saved: {filepath}")
        return str(filepath)
    
    def load_best(self) -> Optional[Dict[str, Any]]:
        """加载最佳检查点"""
        if not self.checkpoints:
            return None
        
        best_path = self.checkpoints[0][1]
        return torch.load(best_path)


# ============ 可视化工具 ============

class Visualizer:
    """训练过程可视化"""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history: Dict[str, list] = {}
        
    def log_metric(self, name: str, value: float, step: int):
        """记录指标"""
        if name not in self.metrics_history:
            self.metrics_history[name] = []
        self.metrics_history[name].append((step, value))
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """绘制指标曲线"""
        try:
            import matplotlib.pyplot as plt
            
            num_metrics = len(self.metrics_history)
            fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 3 * num_metrics))
            
            if num_metrics == 1:
                axes = [axes]
            
            for ax, (name, values) in zip(axes, self.metrics_history.items()):
                steps, vals = zip(*values)
                ax.plot(steps, vals, label=name)
                ax.set_xlabel("Step")
                ax.set_ylabel(name)
                ax.legend()
                ax.grid(True)
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = self.log_dir / "metrics.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
            
            logging.info(f"Metrics plot saved: {save_path}")
        except ImportError:
            logging.warning("matplotlib not installed, skipping visualization")


# ============ 分布式训练 ============

def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """初始化分布式训练"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    logging.info(f"Distributed training initialized: rank {rank}/{world_size}")


def cleanup_distributed():
    """清理分布式训练"""
    if dist.is_initialized():
        dist.destroy_process_group()
        logging.info("Distributed training cleaned up")


class AverageMeter:
    """计算并存储平均值和当前值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
'''

with open('/mnt/agents/output/utils.py', 'w', encoding='utf-8') as f:
    f.write(utils_code)

print("✅ utils.py 已生成")
print(f"文件大小: {len(utils_code)} 字符")
