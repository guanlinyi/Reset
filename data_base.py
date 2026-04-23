
# 核心代码文件4: 数据层基类与流水线 - 支持多种数据类型
data_base_code = '''"""
Data Layer - 数据处理层基类与流水线

支持：
- 骨架数据（NTU RGB+D, NW-UCLA等）
- 图像/视频数据
- 文本数据
- 强化学习环境数据

设计模式：
- 管道模式（Pipeline）：可组合的数据处理步骤
- 工厂模式：根据数据集类型自动创建Feeder
- 适配器模式：统一不同数据源的接口
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class DataTransform(ABC):
    """数据变换抽象基类"""
    
    @abstractmethod
    def __call__(self, data: Any) -> Any:
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Compose:
    """组合多个变换"""
    
    def __init__(self, transforms: List[DataTransform]):
        self.transforms = transforms
    
    def __call__(self, data: Any) -> Any:
        for t in self.transforms:
            data = t(data)
        return data


# ============ 骨架数据变换 ============

class SkeletonNormalize(DataTransform):
    """骨架数据归一化"""
    
    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.mean = mean
        self.std = std
    
    def __call__(self, skeleton: np.ndarray) -> np.ndarray:
        # skeleton shape: [frames, nodes, coords]
        if self.mean is not None and self.std is not None:
            return (skeleton - self.mean) / (self.std + 1e-8)
        # 否则做实例归一化
        mean = np.mean(skeleton, axis=(0, 1), keepdims=True)
        std = np.std(skeleton, axis=(0, 1), keepdims=True)
        return (skeleton - mean) / (std + 1e-8)


class SkeletonRandomRotate(DataTransform):
    """骨架随机旋转（数据增强）"""
    
    def __init__(self, angle_range: Tuple[float, float] = (-0.3, 0.3)):
        self.angle_range = angle_range
    
    def __call__(self, skeleton: np.ndarray) -> np.ndarray:
        # skeleton: [frames, nodes, 3]
        angle = np.random.uniform(*self.angle_range)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # 绕Y轴旋转
        rotation_matrix = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
        
        return np.dot(skeleton, rotation_matrix.T)


class SkeletonRandomScale(DataTransform):
    """骨架随机缩放"""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2)):
        self.scale_range = scale_range
    
    def __call__(self, skeleton: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(*self.scale_range)
        return skeleton * scale


class SkeletonToTensor(DataTransform):
    """转换为Tensor"""
    
    def __call__(self, skeleton: np.ndarray) -> torch.Tensor:
        # [frames, nodes, coords] -> [coords, frames, nodes]
        tensor = torch.from_numpy(skeleton).float()
        return tensor.permute(2, 0, 1)


class SkeletonMultiModalRepresentation(DataTransform):
    """
    骨架多模态表示（InfoGCN特性）
    
    生成关节相对位置等互补空间信息
    """
    
    def __init__(self, mode: str = "joint"):
        self.mode = mode  # "joint", "bone", "motion"
    
    def __call__(self, skeleton: np.ndarray) -> np.ndarray:
        if self.mode == "joint":
            return skeleton
        elif self.mode == "bone":
            # 计算骨骼向量（关节相对位置）
            # 这里简化实现，实际应根据骨架拓扑结构计算
            return self._compute_bone_vectors(skeleton)
        elif self.mode == "motion":
            # 计算运动向量（帧间差分）
            return self._compute_motion_vectors(skeleton)
        return skeleton
    
    def _compute_bone_vectors(self, skeleton: np.ndarray) -> np.ndarray:
        # skeleton: [frames, nodes, coords]
        # 简化：相邻节点差分
        return np.diff(skeleton, axis=1, prepend=skeleton[:, :1, :])
    
    def _compute_motion_vectors(self, skeleton: np.ndarray) -> np.ndarray:
        # 帧间差分
        return np.diff(skeleton, axis=0, prepend=skeleton[:1, :, :])


# ============ 数据集基类 ============

class BaseDataset(Dataset, ABC):
    """
    抽象数据集基类
    
    统一接口支持：
    - 骨架识别（NTU RGB+D, NW-UCLA）
    - 图像分类
    - 强化学习环境
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform: Optional[Compose] = None,
        config: Optional[Dict] = None
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform or Compose([])
        self.config = config or {}
        self.samples = []
        
        self._load_data()
    
    @abstractmethod
    def _load_data(self):
        """加载数据到self.samples"""
        pass
    
    @abstractmethod
    def _get_item(self, index: int) -> Dict[str, Any]:
        """获取单个样本"""
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self._get_item(index)
        
        # 应用变换
        if "input" in item:
            item["input"] = self.transform(item["input"])
        
        return item


class SkeletonDataset(BaseDataset):
    """
    骨架数据集 - 兼容InfoGCN/InfoGCN++风格
    
    支持NTU RGB+D, NW-UCLA等标准骨架数据集
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform: Optional[Compose] = None,
        config: Optional[Dict] = None,
        num_classes: int = 60,
        num_nodes: int = 25,
        num_persons: int = 2
    ):
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.num_persons = num_persons
        super().__init__(data_path, split, transform, config)
    
    def _load_data(self):
        """加载骨架数据"""
        # 这里实现具体的数据加载逻辑
        # 可以从.npy, .pkl, 或原始txt文件加载
        data_file = self.data_path / f"{self.split}_data.npy"
        label_file = self.data_path / f"{self.split}_label.pkl"
        
        if data_file.exists():
            self.data = np.load(data_file, allow_pickle=True)
            # 假设label在data中或单独加载
            self.samples = list(range(len(self.data)))
        else:
            logger.warning(f"Data file not found: {data_file}")
            self.data = []
            self.samples = []
    
    def _get_item(self, index: int) -> Dict[str, Any]:
        """获取单个骨架样本"""
        sample = self.data[index]
        
        # 假设sample格式: (skeleton_data, label)
        if isinstance(sample, tuple):
            skeleton, label = sample
        else:
            skeleton = sample
            label = 0  # 默认标签
        
        return {
            "inputs": skeleton,  # [frames, nodes, coords]
            "labels": label,
            "index": index
        }


class FeederFactory:
    """
    Feeder工厂 - 兼容InfoGCN的feeder模式
    
    根据配置自动创建对应的数据Feeder
    """
    
    _feeders = {
        "skeleton": SkeletonDataset,
        "image": None,  # 可扩展
        "text": None,
    }
    
    @classmethod
    def create(
        cls,
        dataset_type: str,
        data_path: str,
        split: str = "train",
        config: Optional[Dict] = None,
        **kwargs
    ) -> BaseDataset:
        """创建数据Feeder"""
        if dataset_type not in cls._feeders:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        dataset_class = cls._feeders[dataset_type]
        if dataset_class is None:
            raise NotImplementedError(f"Dataset type '{dataset_type}' not implemented yet")
        
        # 构建变换管道
        transforms = cls._build_transforms(dataset_type, config)
        
        dataset = dataset_class(
            data_path=data_path,
            split=split,
            transform=transforms,
            config=config,
            **kwargs
        )
        
        logger.info(f"Feeder created: {dataset_type} | Split: {split} | Samples: {len(dataset)}")
        return dataset
    
    @classmethod
    def _build_transforms(cls, dataset_type: str, config: Optional[Dict]) -> Compose:
        """构建变换管道"""
        transforms = []
        
        if dataset_type == "skeleton":
            # 骨架数据变换管道
            transforms.extend([
                SkeletonNormalize(),
                SkeletonRandomRotate(),
                SkeletonRandomScale(),
                SkeletonToTensor()
            ])
            
            # 多模态表示（可选）
            if config and config.get("multimodal", False):
                transforms.append(SkeletonMultiModalRepresentation(mode="bone"))
        
        return Compose(transforms)


class DataLoaderFactory:
    """数据加载器工厂"""
    
    @staticmethod
    def create(
        dataset: BaseDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        **kwargs
    ) -> DataLoader:
        """创建DataLoader"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs
        )


class PreprocessPipeline:
    """
    预处理流水线
    
    可组合的预处理步骤，支持：
    - 数据清洗
    - 特征提取
    - 数据增强
    - 格式转换
    """
    
    def __init__(self, steps: Optional[List[Callable]] = None):
        self.steps = steps or []
    
    def add_step(self, step: Callable):
        """添加处理步骤"""
        self.steps.append(step)
    
    def process(self, data: Any) -> Any:
        """执行流水线"""
        for step in self.steps:
            data = step(data)
        return data
    
    def process_batch(self, data_list: List[Any]) -> List[Any]:
        """批量处理"""
        return [self.process(d) for d in data_list]
'''

with open('/mnt/agents/output/data_base.py', 'w', encoding='utf-8') as f:
    f.write(data_base_code)

print("✅ data_base.py 已生成")
print(f"文件大小: {len(data_base_code)} 字符")
