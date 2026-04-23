
# 核心代码文件5: 评估器与差异分析器
evaluator_code = '''"""
Evaluation & Discrepancy Analysis - 评估与差异分析

包含：
- 统一评估器（支持分类/预测/RL任务）
- 多指标库
- 差异分析器（Agent工作流第7步：分析实验反馈与理解的差异）
- 在线验证器（InfoGCN++风格）

设计理念：
- 指标即插即用
- 差异分析支持贝叶斯更新输入
- 在线验证支持实时评估
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ============ 评估指标 ============

class Metric(ABC):
    """评估指标基类"""
    
    @abstractmethod
    def compute(self, predictions: Any, targets: Any) -> float:
        pass
    
    @abstractmethod
    def reset(self):
        pass


class AccuracyMetric(Metric):
    """准确率指标"""
    
    def __init__(self, topk: tuple = (1,)):
        self.topk = topk
        self.correct = {k: 0 for k in topk}
        self.total = 0
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Args:
            predictions: [batch, num_classes]
            targets: [batch]
        """
        maxk = max(self.topk)
        batch_size = targets.size(0)
        
        _, pred = predictions.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        results = {}
        for k in self.topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results[f"top{k}_acc"] = (correct_k / batch_size).item()
            self.correct[k] += correct_k.item()
        
        self.total += batch_size
        return results
    
    def reset(self):
        self.correct = {k: 0 for k in self.topk}
        self.total = 0
    
    def get_overall(self) -> Dict[str, float]:
        return {f"top{k}_acc": self.correct[k] / self.total if self.total > 0 else 0 
                for k in self.topk}


class SkeletonRecognitionMetric(Metric):
    """骨架识别专用指标"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """计算骨架识别指标"""
        pred_labels = predictions.argmax(dim=1).cpu().numpy()
        target_labels = targets.cpu().numpy()
        
        # 更新混淆矩阵
        for p, t in zip(pred_labels, target_labels):
            self.confusion_matrix[t, p] += 1
        
        # 计算各类指标
        accuracy = np.trace(self.confusion_matrix) / np.sum(self.confusion_matrix)
        
        # 每类准确率
        per_class_acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-8)
        
        return {
            "accuracy": accuracy,
            "mean_class_accuracy": per_class_acc.mean(),
            "worst_class_accuracy": per_class_acc.min()
        }
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))


class OnlineRecognitionMetric(Metric):
    """
    在线识别指标（InfoGCN++风格）
    
    评估不同观察比例下的识别性能
    """
    
    def __init__(self, observation_ratios: List[float] = [0.1, 0.3, 0.5, 0.7, 1.0]):
        self.ratios = observation_ratios
        self.results = {r: {"correct": 0, "total": 0} for r in observation_ratios}
    
    def compute(self, predictions: Dict[float, torch.Tensor], targets: torch.Tensor) -> Dict[str, float]:
        """
        Args:
            predictions: {observation_ratio: prediction_tensor}
            targets: [batch]
        """
        results = {}
        for ratio, pred in predictions.items():
            pred_labels = pred.argmax(dim=1)
            correct = (pred_labels == targets).sum().item()
            total = targets.size(0)
            
            self.results[ratio]["correct"] += correct
            self.results[ratio]["total"] += total
            
            results[f"acc_ratio_{ratio}"] = correct / total if total > 0 else 0
        
        return results
    
    def reset(self):
        self.results = {r: {"correct": 0, "total": 0} for r in self.ratios}
    
    def get_overall(self) -> Dict[str, float]:
        return {
            f"acc_ratio_{r}": v["correct"] / v["total"] if v["total"] > 0 else 0
            for r, v in self.results.items()
        }


class RLMetric(Metric):
    """强化学习指标"""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_flags = []
    
    def compute(self, episode_data: Dict) -> Dict[str, float]:
        """计算RL指标"""
        reward = episode_data.get("reward", 0)
        length = episode_data.get("length", 0)
        success = episode_data.get("success", False)
        
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.success_flags.append(1 if success else 0)
        
        return {
            "episode_reward": reward,
            "episode_length": length,
            "success": 1 if success else 0
        }
    
    def reset(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_flags = []
    
    def get_summary(self) -> Dict[str, float]:
        if not self.episode_rewards:
            return {}
        
        return {
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "mean_length": np.mean(self.episode_lengths),
            "success_rate": np.mean(self.success_flags),
            "max_reward": np.max(self.episode_rewards)
        }


# ============ 评估器 ============

class Evaluator:
    """
    统一评估器
    
    支持多种任务类型的评估
    """
    
    def __init__(self, metrics: Optional[List[Metric]] = None):
        self.metrics = metrics or []
        self.results_history = []
    
    def add_metric(self, metric: Metric):
        """添加评估指标"""
        self.metrics.append(metric)
    
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        task_type: str = "classification"
    ) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            model: 待评估模型
            dataloader: 验证数据
            task_type: 任务类型（classification/prediction/rl）
        """
        model.eval()
        
        # 重置指标
        for metric in self.metrics:
            metric.reset()
        
        all_results = defaultdict(list)
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["inputs"]
                labels = batch["labels"]
                
                # 前向传播
                outputs = model(inputs)
                
                if isinstance(outputs, tuple):
                    predictions = outputs[0]
                else:
                    predictions = outputs
                
                # 计算各指标
                for metric in self.metrics:
                    result = metric.compute(predictions, labels)
                    if isinstance(result, dict):
                        for k, v in result.items():
                            all_results[k].append(v)
                    else:
                        all_results[metric.__class__.__name__].append(result)
        
        # 汇总结果
        summary = {}
        for k, v in all_results.items():
            summary[k] = np.mean(v) if v else 0
        
        self.results_history.append(summary)
        
        model.train()
        return summary
    
    def evaluate_online(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        observation_ratios: List[float]
    ) -> Dict[str, float]:
        """
        在线评估（InfoGCN++风格）
        
        在不同观察比例下评估模型性能
        """
        model.eval()
        
        online_metric = OnlineRecognitionMetric(observation_ratios)
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["inputs"]  # [batch, coords, frames, nodes]
                labels = batch["labels"]
                
                batch_size, coords, total_frames, nodes = inputs.shape
                
                # 在不同观察比例下进行预测
                predictions = {}
                for ratio in observation_ratios:
                    observe_frames = max(1, int(total_frames * ratio))
                    observed = inputs[:, :, :observe_frames, :]
                    
                    # 填充或截断到固定长度（模型需要）
                    # 这里简化处理
                    pred = model(observed)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    predictions[ratio] = pred
                
                online_metric.compute(predictions, labels)
        
        model.train()
        return online_metric.get_overall()


# ============ 差异分析器 ============

@dataclass
class Understanding:
    """理解/假设单元"""
    content: str                    # 理解内容
    source: str                     # 来源：外源/内源
    confidence: float               # 置信度
    predictions: Dict[str, Any]     # 基于理解做出的预测


@dataclass
class ExperimentFeedback:
    """实验反馈"""
    metrics: Dict[str, float]       # 实际指标
    observations: List[str]         # 观察记录
    anomalies: List[str]            # 异常点


class DiscrepancyAnalyzer:
    """
    差异分析器
    
    Agent工作流第7步：分析实验反馈与最初理解的差异
    输出用于第8步贝叶斯更新的信号
    """
    
    def __init__(self, tolerance: float = 0.05):
        self.tolerance = tolerance  # 可接受的差异容忍度
    
    def analyze(
        self,
        understanding: Understanding,
        feedback: ExperimentFeedback
    ) -> Dict[str, Any]:
        """
        分析理解与实际反馈的差异
        
        Returns:
            discrepancy_report: 差异分析报告
        """
        discrepancies = []
        
        # 比较预测值与实际值
        for metric_name, predicted_value in understanding.predictions.items():
            if metric_name in feedback.metrics:
                actual_value = feedback.metrics[metric_name]
                gap = abs(predicted_value - actual_value)
                relative_gap = gap / (abs(predicted_value) + 1e-8)
                
                discrepancy = {
                    "metric": metric_name,
                    "predicted": predicted_value,
                    "actual": actual_value,
                    "absolute_gap": gap,
                    "relative_gap": relative_gap,
                    "significant": relative_gap > self.tolerance
                }
                discrepancies.append(discrepancy)
        
        # 分析异常
        for anomaly in feedback.anomalies:
            discrepancies.append({
                "type": "anomaly",
                "description": anomaly,
                "significant": True
            })
        
        # 生成报告
        significant_discrepancies = [d for d in discrepancies if d.get("significant", False)]
        
        report = {
            "total_discrepancies": len(discrepancies),
            "significant_discrepancies": len(significant_discrepancies),
            "details": discrepancies,
            "understanding_valid": len(significant_discrepancies) == 0,
            "recommended_action": self._recommend_action(significant_discrepancies)
        }
        
        logger.info(f"Discrepancy analysis: {report['significant_discrepancies']} significant out of {report['total_discrepancies']}")
        
        return report
    
    def _recommend_action(self, significant: List[Dict]) -> str:
        """基于差异推荐行动"""
        if not significant:
            return "continue"  # 理解正确，继续
        
        # 判断是否需要外源学习还是内源调整
        if any(d.get("relative_gap", 0) > 0.3 for d in significant):
            return "search_external"  # 差异太大，需要寻找新的优秀样本
        else:
            return "refine_internal"  # 差异较小，内部微调


class ErrorAnalyzer:
    """错误分析器 - 深入分析模型错误模式"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.error_cases = defaultdict(list)
    
    def analyze_errors(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        inputs: torch.Tensor
    ) -> Dict[str, Any]:
        """分析错误案例"""
        pred_labels = predictions.argmax(dim=1)
        errors = pred_labels != targets
        
        error_indices = errors.nonzero(as_tuple=True)[0]
        
        for idx in error_indices:
            true_class = targets[idx].item()
            pred_class = pred_labels[idx].item()
            
            self.error_cases[true_class].append({
                "predicted_as": pred_class,
                "input_sample": inputs[idx].cpu().numpy()
            })
        
        # 分析错误模式
        class_error_rates = {}
        for cls in range(self.num_classes):
            total = (targets == cls).sum().item()
            errors = len(self.error_cases[cls])
            class_error_rates[cls] = errors / total if total > 0 else 0
        
        # 找出最容易混淆的类别对
        confusion_pairs = defaultdict(int)
        for true_cls, cases in self.error_cases.items():
            for case in cases:
                confusion_pairs[(true_cls, case["predicted_as"])] += 1
        
        top_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_errors": len(error_indices),
            "error_rate": len(error_indices) / len(targets),
            "class_error_rates": class_error_rates,
            "top_confusions": top_confusions,
            "hardest_classes": sorted(class_error_rates.items(), key=lambda x: x[1], reverse=True)[:5]
        }
'''

with open('/mnt/agents/output/evaluator.py', 'w', encoding='utf-8') as f:
    f.write(evaluator_code)

print("✅ evaluator.py 已生成")
print(f"文件大小: {len(evaluator_code)} 字符")
