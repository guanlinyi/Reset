
# 核心代码文件9: 完整示例 - InfoGCN++风格模型实现
infogcn_example = '''"""
InfoGCN++ Model Implementation Example

基于通用框架实现的具体模型示例
展示如何使用框架构建骨架动作识别系统
"""

import logging
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.model_base import BaseModel, ModelRegistry, SAGraphConv, MultiScaleTemporalConv, NeuralODEFunction

logger = logging.getLogger(__name__)


@ModelRegistry.register("InfoGCN")
class InfoGCN(BaseModel):
    """
    InfoGCN模型 - 基于信息瓶颈的骨架动作识别
    
    架构：
    - Embedding Layer
    - Encoder (SA-GC + MS-TC) x L
    - Classifier
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_classes = config.get("num_classes", 60)
        self.num_nodes = config.get("num_nodes", 25)
        self.in_channels = config.get("in_channels", 3)
        self.hidden_dim = config.get("hidden_dim", 64)
        self.num_layers = config.get("num_layers", 9)
        self.num_heads = config.get("num_heads", 3)
        self.dropout = config.get("dropout", 0.1)
        
    def build(self):
        """构建模型架构"""
        # 嵌入层
        self.embedding = nn.Linear(self.in_channels, self.hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_nodes, self.hidden_dim))
        
        # 编码器层
        self.encoder_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            layer = nn.ModuleDict({
                "sa_gc": SAGraphConv(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.num_nodes,
                    self.num_heads,
                    self.dropout
                ),
                "ms_tc": MultiScaleTemporalConv(
                    self.hidden_dim,
                    self.hidden_dim
                ),
                "norm1": nn.LayerNorm(self.hidden_dim),
                "norm2": nn.LayerNorm(self.hidden_dim),
                "dropout": nn.Dropout(self.dropout)
            })
            self.encoder_layers.append(layer)
        
        # 信息瓶颈参数（重参数化技巧）
        self.mu_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.logvar_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
        
        self._built = True
        logger.info("InfoGCN model built successfully")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: [batch, coords, frames, nodes] -> 输入骨架数据
        
        Returns:
            predictions: [batch, num_classes]
            mu: 均值（信息瓶颈）
            logvar: 对数方差（信息瓶颈）
        """
        # 调整维度: [batch, coords, frames, nodes] -> [batch, frames, nodes, coords]
        x = x.permute(0, 2, 3, 1)
        batch_size, num_frames, num_nodes, coords = x.shape
        
        # 嵌入层
        h = self.embedding(x)  # [B, T, N, C]
        h = h + self.pos_embedding.unsqueeze(1)  # 添加位置嵌入
        
        # 编码器
        for layer in self.encoder_layers:
            # 空间建模 (SA-GC)
            # 重塑为 [B*T, N, C] 进行图卷积
            h_reshaped = h.reshape(-1, num_nodes, self.hidden_dim)
            h_spatial = layer["sa_gc"](h_reshaped)
            h_spatial = h_spatial.reshape(batch_size, num_frames, num_nodes, self.hidden_dim)
            h = layer["norm1"](h + layer["dropout"](h_spatial))
            
            # 时间建模 (MS-TC)
            # 重塑为 [B*N, C, T] 进行时间卷积
            h_temp = h.permute(0, 2, 3, 1)  # [B, N, C, T]
            h_temp = h_temp.reshape(-1, self.hidden_dim, num_frames)
            h_temp = layer["ms_tc"](h_temp)
            h_temp = h_temp.reshape(batch_size, num_nodes, self.hidden_dim, num_frames)
            h_temp = h_temp.permute(0, 3, 1, 2)  # [B, T, N, C]
            h = layer["norm2"](h + layer["dropout"](h_temp))
        
        # 全局平均池化 -> [B, C]
        h_pooled = h.mean(dim=(1, 2))  # 在时间和节点维度上平均
        
        # 信息瓶颈：重参数化
        mu = self.mu_proj(h_pooled)
        logvar = self.logvar_proj(h_pooled)
        
        # 重参数化技巧
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        
        # 分类
        predictions = self.classifier(z.unsqueeze(-1).unsqueeze(-1))
        
        return predictions, mu, logvar


@ModelRegistry.register("InfoGCNPlus")
class InfoGCNPlus(InfoGCN):
    """
    InfoGCN++模型 - 在线骨架动作识别
    
    扩展InfoGCN，增加：
    - 未来运动预测（神经ODE）
    - 在线识别能力（因果掩码）
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.use_neural_ode = config.get("use_neural_ode", True)
        self.n_step = config.get("n_step", 3)  # 预测步数
        
    def build(self):
        """构建InfoGCN++架构"""
        # 先构建基础InfoGCN
        super().build()
        
        # 添加未来运动预测器（神经ODE）
        if self.use_neural_ode:
            self.ode_func = NeuralODEFunction(self.hidden_dim)
            self.future_predictor = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            logger.info("Neural ODE future predictor added")
        
        # 未来运动预测解码器
        self.motion_decoder = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self._built = True
        logger.info("InfoGCN++ model built successfully")
    
    def extrapolate(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        使用神经ODE外推未来状态
        
        Args:
            z: 当前状态 [batch, hidden_dim]
            t: 时间步 [n_step]
        
        Returns:
            z_future: 未来状态 [batch, n_step, hidden_dim]
        """
        try:
            from torchdiffeq import odeint
            
            # 求解ODE
            z_future = odeint(self.ode_func, z, t, method='rk4')
            # z_future: [n_step, batch, hidden_dim]
            z_future = z_future.permute(1, 0, 2)  # [batch, n_step, hidden_dim]
            
            return z_future
        except ImportError:
            logger.warning("torchdiffeq not installed, using simple extrapolation")
            # 简化：线性外推
            z_future = z.unsqueeze(1).repeat(1, len(t), 1)
            return z_future
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        InfoGCN++前向传播
        
        Args:
            x: 部分观察的骨架序列 [batch, coords, observe_frames, nodes]
        
        Returns:
            y_hat: 动作类别预测
            x_hat: 未来运动预测
            z_0: 当前表示
            z_hat: 未来表示
            kl_div: KL散度（信息瓶颈）
        """
        batch_size = x.size(0)
        
        # 使用父类编码器获取当前表示
        predictions, mu, logvar = super().forward(x)
        
        z_0 = mu  # 当前表示
        
        # 未来运动预测
        if self.use_neural_ode and self.training:
            # 创建时间步
            t = torch.linspace(0, 1, self.n_step + 1, device=x.device)
            
            # 外推到未来
            z_future = self.extrapolate(z_0, t[1:])  # [batch, n_step, hidden_dim]
            
            # 解码未来运动
            x_hat = self.motion_decoder(z_future)  # [batch, n_step, hidden_dim]
            
            # 计算KL散度（信息瓶颈正则化）
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_div = kl_div / batch_size
        else:
            x_hat = None
            z_future = None
            kl_div = torch.tensor(0.0, device=x.device)
        
        return predictions, x_hat, z_0, z_future, kl_div


def create_infogcn_loss(lambda_1: float = 1.0, lambda_2: float = 0.1, lambda_3: float = 0.001):
    """
    创建InfoGCN损失函数
    
    包含：
    1. 分类损失（交叉熵）
    2. 未来预测损失（MSE）
    3. KL散度（信息瓶颈）
    
    Args:
        lambda_1: 分类损失权重
        lambda_2: 预测损失权重
        lambda_3: KL散度权重
    """
    def loss_fn(outputs, targets):
        predictions, x_hat, z_0, z_future, kl_div = outputs
        labels, future_targets = targets
        
        # 分类损失
        cls_loss = F.cross_entropy(predictions, labels)
        
        # 未来预测损失
        pred_loss = 0
        if x_hat is not None and future_targets is not None:
            pred_loss = F.mse_loss(x_hat, future_targets)
        
        # 总损失
        total_loss = lambda_1 * cls_loss + lambda_2 * pred_loss + lambda_3 * kl_div
        
        return total_loss, {
            "cls_loss": cls_loss.item(),
            "pred_loss": pred_loss.item() if isinstance(pred_loss, torch.Tensor) else 0,
            "kl_div": kl_div.item()
        }
    
    return loss_fn
'''

with open('/mnt/agents/output/infogcn_example.py', 'w', encoding='utf-8') as f:
    f.write(infogcn_example)

print("✅ infogcn_example.py 已生成")
print(f"文件大小: {len(infogcn_example)} 字符")
