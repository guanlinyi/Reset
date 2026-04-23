
# 核心代码文件3: 模型基类与工厂 - 支持多种架构
model_base_code = '''"""
Base Model & Model Factory - 模型层基类与工厂

支持：
- 骨架识别模型（InfoGCN/InfoGCN++风格）
- Transformer架构
- 强化学习Agent网络
- 神经ODE（Neural ODE）

设计模式：
- 工厂模式：根据配置自动实例化模型
- 注册表模式：动态注册新架构
- 组件复用：layers目录下的可复用模块
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type
import importlib

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelRegistry:
    """模型注册表 - 支持动态注册和查找"""
    
    _registry: Dict[str, Type[nn.Module]] = {}
    
    @classmethod
    def register(cls, name: str):
        """装饰器：注册模型类"""
        def decorator(model_class: Type[nn.Module]):
            cls._registry[name] = model_class
            logger.info(f"Model registered: {name} -> {model_class.__name__}")
            return model_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
        """获取已注册的模型类"""
        if name not in cls._registry:
            raise ValueError(f"Model '{name}' not found. Available: {list(cls._registry.keys())}")
        return cls._registry[name]
    
    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """列出所有已注册模型"""
        return {name: cls_.__name__ for name, cls_ in cls._registry.items()}


class BaseModel(nn.Module, ABC):
    """
    抽象模型基类
    
    所有具体模型应继承此类，实现forward方法。
    支持多任务输出（如InfoGCN++的分类+预测）
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model_name = self.__class__.__name__
        self._built = False
        
    @abstractmethod
    def build(self):
        """构建模型架构 - 子类应在此方法中定义层"""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Any:
        """
        前向传播
        
        支持返回多值tuple（用于多任务学习）
        """
        pass
    
    def get_num_parameters(self) -> Dict[str, int]:
        """获取模型参数量统计"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": total - trainable
        }
    
    def summary(self):
        """打印模型摘要"""
        params = self.get_num_parameters()
        logger.info(f"Model: {self.model_name}")
        logger.info(f"  Total parameters: {params['total']:,}")
        logger.info(f"  Trainable parameters: {params['trainable']:,}")
        
    def save(self, path: str):
        """保存模型"""
        torch.save({
            "state_dict": self.state_dict(),
            "config": self.config,
            "model_name": self.model_name
        }, path)
        logger.info(f"Model saved: {path}")
    
    @classmethod
    def load(cls, path: str, **kwargs) -> "BaseModel":
        """加载模型"""
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(checkpoint["config"])
        model.build()
        model.load_state_dict(checkpoint["state_dict"])
        logger.info(f"Model loaded: {path}")
        return model


# 可复用组件示例

class SAGraphConv(nn.Module):
    """
    自注意力图卷积（SA-GC）模块
    
    来自InfoGCN/InfoGCN++的核心组件
    推断上下文依赖的内在拓扑
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_nodes: int,
        num_heads: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        
        # 共享拓扑（可学习）
        self.shared_topology = nn.Parameter(
            torch.randn(num_heads, num_nodes, num_nodes)
        )
        
        # 查询、键、值投影
        self.query_proj = nn.Linear(in_channels, out_channels)
        self.key_proj = nn.Linear(in_channels, out_channels)
        self.value_proj = nn.Linear(in_channels, out_channels)
        
        # 输出投影
        self.out_proj = nn.Linear(out_channels, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, nodes, channels]
            adjacency: 可选的预定义邻接矩阵
        
        Returns:
            [batch, nodes, out_channels]
        """
        batch_size, num_nodes, _ = x.shape
        
        # 计算查询、键、值
        Q = self.query_proj(x)  # [B, N, C]
        K = self.key_proj(x)
        V = self.value_proj(x)
        
        # 多头分割
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 自注意力图
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_graph = torch.softmax(attention_scores, dim=-1)
        
        # 结合共享拓扑和自注意力图
        topology = torch.sigmoid(self.shared_topology).unsqueeze(0)  # [1, H, N, N]
        intrinsic_topology = topology * attention_graph  # 广播逐元素乘积
        
        # 图卷积：聚合邻居信息
        out = torch.matmul(intrinsic_topology, V)  # [B, H, N, D]
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)
        
        # 输出投影
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return out


class MultiScaleTemporalConv(nn.Module):
    """
    多尺度时间卷积（MS-TC）
    
    多个不同大小的卷积核并行提取时间特征
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list = [3, 5, 7],
        dropout: float = 0.1
    ):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels,
                out_channels // len(kernel_sizes),
                kernel_size=k,
                padding=k // 2
            )
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, time]
        Returns:
            [batch, out_channels, time]
        """
        outputs = [conv(x) for conv in self.convs]
        out = torch.cat(outputs, dim=1)
        return self.dropout(out)


class NeuralODEFunction(nn.Module):
    """
    神经ODE函数（Neural ODE Function）
    
    用于InfoGCN++的未来运动预测
    建模隐藏状态的连续演化
    """
    
    def __init__(self, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
        self.net = nn.Sequential(*layers)
        
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 时间（ODE求解器传入）
            x: 状态 [batch, hidden_dim]
        Returns:
            dx/dt: 状态导数
        """
        return self.net(x)


class ModelFactory:
    """
    模型工厂
    
    根据配置字符串自动实例化对应模型
    支持从注册表动态查找
    """
    
    @staticmethod
    def create(config: Dict[str, Any]) -> BaseModel:
        """
        创建模型实例
        
        Args:
            config: 必须包含 "model_name" 字段
        
        Returns:
            模型实例
        """
        model_name = config.get("model_name", "base")
        
        # 从注册表获取模型类
        try:
            model_class = ModelRegistry.get(model_name)
        except ValueError:
            # 动态导入
            module_path = config.get("model_module", f"src.models.architectures.{model_name}")
            try:
                module = importlib.import_module(module_path)
                model_class = getattr(module, model_name)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Cannot load model '{model_name}': {e}")
        
        # 实例化
        model = model_class(config)
        model.build()
        
        logger.info(f"Model created: {model_name}")
        return model
'''

with open('/mnt/agents/output/model_base.py', 'w', encoding='utf-8') as f:
    f.write(model_base_code)

print("✅ model_base.py 已生成")
print(f"文件大小: {len(model_base_code)} 字符")
