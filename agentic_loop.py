# 核心代码文件1: Agentic循环引擎 - 实现你描述的9步工作流
agentic_loop_code = '''"""
Agentic Loop Engine - 目标驱动的嵌套循环引擎
实现你描述的9步Agentic工作流：
1. 确立全局大目标（粗糙容忍度）
2. 目标分解
3. 获取外部信息/搜索优秀样本
4. 外源学习模式（模仿/推测/提问/质疑）
5. 内源学习模式（自我博弈/AlphaGo模式）
6. 实践验证
7. 分析差异
8. 修正/更新理解（贝叶斯更新）
9. 循环（停机判断）
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum, auto
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """学习模式枚举"""
    EXOGENOUS = auto()   # 外源学习：存在优秀样本时
    ENDOGENOUS = auto()  # 内源学习：无优秀样本时，自我博弈


class GoalStatus(Enum):
    """目标状态"""
    PENDING = auto()
    IN_PROGRESS = auto()
    DEMO_COMPLETED = auto()  # 粗糙容忍度：Demo完成即可
    REFINING = auto()
    COMPLETED = auto()
    BLOCKED = auto()
    TECH_DEBT = auto()       # 标记为技术债，转向下一目标


@dataclass
class Goal:
    """目标节点 - 支持嵌套分解"""
    id: str
    description: str
    parent_id: Optional[str] = None
    status: GoalStatus = GoalStatus.PENDING
    tolerance: str = "rough"  # 粗糙容忍度：先完成demo再循环改进
    sub_goals: List['Goal'] = field(default_factory=list)
    learned_knowledge: Dict[str, Any] = field(default_factory=dict)
    experiments: List[Dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    
    def is_demo_ready(self) -> bool:
        """判断当前目标是否达到Demo完成标准（粗糙容忍度）"""
        return self.status == GoalStatus.DEMO_COMPLETED
    
    def add_sub_goal(self, sub_goal: 'Goal'):
        """添加子目标，实现目标分解"""
        sub_goal.parent_id = self.id
        self.sub_goals.append(sub_goal)


@dataclass
class Belief:
    """信念/理解单元 - 用于贝叶斯更新"""
    hypothesis: str           # 假设/理解内容
    confidence: float         # 置信度 [0, 1]
    evidence: List[Dict] = field(default_factory=list)  # 支持证据
    source: str = "internal"  # 来源：external（外源）/ internal（内源）
    level: int = 1            # 理解层级，循环次数越多层级越高
    
    def update_confidence(self, new_evidence: Dict, likelihood: float):
        """贝叶斯更新：根据新证据调整置信度"""
        # 简化的贝叶斯更新：P(H|E) ∝ P(E|H) * P(H)
        prior = self.confidence
        posterior = (likelihood * prior) / ((likelihood * prior) + (1 - likelihood) * (1 - prior) + 1e-8)
        self.confidence = posterior
        self.evidence.append(new_evidence)
        logger.info(f"Belief updated: confidence {prior:.3f} -> {posterior:.3f}")


class StoppingCriteria:
    """停机判断条件"""
    
    @staticmethod
    def should_continue(goal: Goal, cost_so_far: float, expected_benefit: float) -> bool:
        """
        判断是否继续当前目标还是转向下一目标
        
        判断逻辑：
        1. 是否跑通当前demo？
        2. 不做X是否阻碍继续？
        3. 成本-收益性价比？
        
        Returns:
            True -> 继续分解/优化
            False -> 标记技术债，转向下一目标
        """
        # 条件1: Demo是否跑通
        if not goal.is_demo_ready():
            logger.info(f"Goal {goal.id}: Demo not ready, continue...")
            return True
        
        # 条件2: 是否阻碍继续
        if goal.status == GoalStatus.BLOCKED:
            logger.info(f"Goal {goal.id}: Blocked, marking as tech debt...")
            goal.status = GoalStatus.TECH_DEBT
            return False
        
        # 条件3: 成本-收益性价比
        cost_benefit_ratio = cost_so_far / (expected_benefit + 1e-8)
        if cost_benefit_ratio > 5.0:  # 阈值可配置
            logger.info(f"Goal {goal.id}: Cost-benefit ratio {cost_benefit_ratio:.2f} too high, pivot...")
            goal.status = GoalStatus.TECH_DEBT
            return False
        
        return True


class AgenticLoop:
    """
    Agentic主循环引擎
    
    核心设计理念：
    - 粗糙容忍度：先完成Demo，再循环改进
    - 嵌套循环：大目标分解为小目标，每个小目标内部再循环
    - 外源优先：有优秀样本时优先外源学习
    - 贝叶斯更新：根据实验反馈持续修正理解
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.knowledge_base = {}  # 知识库
        self.beliefs: List[Belief] = []  # 信念集合
        self.current_goal: Optional[Goal] = None
        self.iteration_count = 0
        self.stopping_criteria = StoppingCriteria()
        
    def run(self, global_goal: Goal) -> Goal:
        """
        启动Agentic主循环
        
        Args:
            global_goal: 全局大目标
            
        Returns:
            完成后的目标树
        """
        logger.info(f"=" * 60)
        logger.info(f"Starting Agentic Loop for goal: {global_goal.description}")
        logger.info(f"=" * 60)
        
        self.current_goal = global_goal
        
        # 第1步: 确立全局大目标（已在参数中）
        # 第2步: 目标分解
        self._decompose_goal(global_goal)
        
        # 对每个子目标执行循环
        for sub_goal in global_goal.sub_goals:
            self._process_sub_goal(sub_goal)
        
        logger.info("Agentic Loop completed!")
        return global_goal
    
    def _decompose_goal(self, goal: Goal):
        """
        第2步: 目标分解
        将大目标分解为可执行的小目标，保持粗糙容忍度
        """
        logger.info(f"[Step 2] Decomposing goal: {goal.description}")
        
        # 这里可以接入LLM或规则引擎进行智能分解
        # 示例分解逻辑：
        if "skeleton recognition" in goal.description.lower():
            sub_goals = [
                Goal(id="g1_1", description="Build data pipeline (rough: loadable)"),
                Goal(id="g1_2", description="Implement base GCN model (rough: forward pass works)"),
                Goal(id="g1_3", description="Add SA-GC attention module (rough: attention scores computed)"),
                Goal(id="g1_4", description="Train single epoch (rough: loss decreases)"),
                Goal(id="g1_5", description="Evaluate on test set (rough: accuracy > random)"),
            ]
        elif "agent" in goal.description.lower():
            sub_goals = [
                Goal(id="g2_1", description="Setup environment wrapper (rough: gym API works)"),
                Goal(id="g2_2", description="Implement policy network (rough: outputs action distribution)"),
                Goal(id="g2_3", description="Train for 100 episodes (rough: reward increases)"),
            ]
        else:
            # 通用分解模板
            sub_goals = [
                Goal(id="gx_1", description="Data preprocessing pipeline (rough: data flows)"),
                Goal(id="gx_2", description="Model architecture (rough: forward pass works)"),
                Goal(id="gx_3", description="Training loop (rough: loss converges)"),
                Goal(id="gx_4", description="Evaluation (rough: metrics computed)"),
            ]
        
        for sg in sub_goals:
            goal.add_sub_goal(sg)
            logger.info(f"  Created sub-goal: {sg.id} - {sg.description}")
    
    def _process_sub_goal(self, goal: Goal):
        """处理单个子目标的核心循环"""
        logger.info(f"\\n[Processing] {goal.id}: {goal.description}")
        goal.status = GoalStatus.IN_PROGRESS
        
        # 第3步: 获取外部信息，判断是否存在优秀样本
        samples = self._search_samples(goal)
        
        if samples:
            # 第4步: 外源学习模式
            self._exogenous_learning(goal, samples)
        else:
            # 第5步: 内源学习模式
            self._endogenous_learning(goal)
        
        # 第6步: 实践验证
        experiment_result = self._practice_verify(goal)
        
        # 第7步: 分析差异
        discrepancy = self._analyze_discrepancy(goal, experiment_result)
        
        # 第8步: 修正/更新理解（贝叶斯更新）
        self._update_beliefs(goal, experiment_result, discrepancy)
        
        # 第9步: 循环判断
        should_continue = self.stopping_criteria.should_continue(
            goal, 
            cost_so_far=experiment_result.get("cost", 0),
            expected_benefit=experiment_result.get("benefit", 1.0)
        )
        
        if should_continue:
            goal.status = GoalStatus.REFINING
            # 递归循环：在当前目标上继续优化
            self._process_sub_goal(goal)
        else:
            if goal.status != GoalStatus.TECH_DEBT:
                goal.status = GoalStatus.DEMO_COMPLETED
    
    def _search_samples(self, goal: Goal) -> List[Dict]:
        """
        第3步: 基于过去经验和目标驱动的搜索/调研
        搜索方向直接决定是否能获取优秀样本
        """
        logger.info(f"[Step 3] Searching for excellent samples...")
        
        # 这里可以接入GitHub API、ArXiv API、PaperWithCode等
        # 返回优秀样本列表（代码仓库、论文、教程等）
        
        # 模拟搜索结果
        samples = []
        if "skeleton" in goal.description.lower():
            samples = [
                {
                    "type": "github",
                    "url": "https://github.com/stnoah1/infogcn2",
                    "description": "InfoGCN++: Online skeleton action recognition",
                    "relevance": 0.95
                },
                {
                    "type": "paper",
                    "url": "https://arxiv.org/abs/2310.10547",
                    "description": "InfoGCN++ paper",
                    "relevance": 0.90
                }
            ]
        
        logger.info(f"  Found {len(samples)} relevant samples")
        return samples
    
    def _exogenous_learning(self, goal: Goal, samples: List[Dict]):
        """
        第4步: 外源学习模式
        分析优秀样本，输出基于过去经验的理解
        包括：模仿、推测、提问、质疑
        理解有层级，循环次数越多层级越高
        取其精华，去其糟粕
        """
        logger.info(f"[Step 4] Exogenous learning mode activated")
        
        for sample in samples:
            logger.info(f"  Analyzing: {sample['description']}")
            
            # 模仿：提取可复用的代码结构/架构模式
            imitation = self._imitate(sample)
            
            # 推测：基于样本推测其设计决策背后的原因
            speculation = self._speculate(sample)
            
            # 提问：对样本中的不确定点提出问题
            questions = self._question(sample)
            
            # 质疑：识别样本中的潜在问题或改进空间
            challenges = self._challenge(sample)
            
            # 整合理解到知识库
            understanding = {
                "source": sample,
                "imitation": imitation,
                "speculation": speculation,
                "questions": questions,
                "challenges": challenges,
                "level": self.iteration_count + 1
            }
            
            goal.learned_knowledge[sample['url']] = understanding
            
            # 创建/更新信念
            belief = Belief(
                hypothesis=f"Understanding from {sample['description']}",
                confidence=sample['relevance'],
                source="external",
                level=self.iteration_count + 1
            )
            self.beliefs.append(belief)
    
    def _endogenous_learning(self, goal: Goal):
        """
        第5步: 内源学习模式
        基于AlphaGo模式的"自我博弈"
        当不存在优秀样本时，通过自我对抗/探索来学习
        """
        logger.info(f"[Step 5] Endogenous learning mode: Self-play / Exploration")
        
        # 自我博弈逻辑：
        # - 对于RL：Agent与环境（或自身副本）对抗
        # - 对于监督学习：数据增强 + 对抗训练
        # - 对于Agent：探索-利用平衡
        
        self_play_result = {
            "episodes": 100,
            "exploration_rate": 0.3,
            "best_performance": 0.6
        }
        
        belief = Belief(
            hypothesis="Self-play derived strategy",
            confidence=0.5,  # 初始置信度较低
            source="internal",
            level=1
        )
        self.beliefs.append(belief)
        
        goal.learned_knowledge["self_play"] = self_play_result
    
    def _practice_verify(self, goal: Goal) -> Dict:
        """
        第6步: 实践验证
        运行实验，获取实际反馈
        """
        logger.info(f"[Step 6] Practice verification")
        
        # 这里调用实际的训练/评估流程
        # 返回实验结果
        experiment = {
            "status": "completed",
            "metrics": {"accuracy": 0.75, "loss": 0.5},
            "cost": 2.5,  # GPU hours
            "benefit": 1.0,
            "timestamp": time.time()
        }
        
        goal.experiments.append(experiment)
        return experiment
    
    def _analyze_discrepancy(self, goal: Goal, experiment: Dict) -> Dict:
        """
        第7步: 分析差异
        分析实验反馈与最初理解的差异
        """
        logger.info(f"[Step 7] Analyzing discrepancy")
        
        # 比较实验结果与预期（基于外源/内源学习的理解）
        expected_accuracy = 0.85  # 基于外源学习的预期
        actual_accuracy = experiment["metrics"]["accuracy"]
        
        discrepancy = {
            "expected": expected_accuracy,
            "actual": actual_accuracy,
            "gap": expected_accuracy - actual_accuracy,
            "analysis": "Performance gap indicates understanding needs refinement"
        }
        
        logger.info(f"  Expected: {expected_accuracy:.3f}, Actual: {actual_accuracy:.3f}, Gap: {discrepancy['gap']:.3f}")
        
        return discrepancy
    
    def _update_beliefs(self, goal: Goal, experiment: Dict, discrepancy: Dict):
        """
        第8步: 修正/更新理解
        贝叶斯更新：根据反馈调整假设概率
        更新第4或5步的理解，更新第3步的搜索方向直觉
        """
        logger.info(f"[Step 8] Updating beliefs via Bayesian update")
        
        for belief in self.beliefs:
            if belief.source == "external" and belief.hypothesis.startswith("Understanding"):
                # 根据实验结果调整置信度
                likelihood = 0.8 if discrepancy["gap"] < 0.1 else 0.4
                belief.update_confidence(
                    new_evidence=experiment,
                    likelihood=likelihood
                )
        
        # 更新搜索方向直觉（基于差异分析）
        if discrepancy["gap"] > 0.2:
            logger.info("  Large gap detected: need to search for more advanced samples")
            # 调整搜索策略：提高筛选标准，寻找更优秀的样本
    
    def _imitate(self, sample: Dict) -> Dict:
        """模仿：提取可复用的代码模式"""
        return {"pattern": "extracted_architecture", "code_snippets": []}
    
    def _speculate(self, sample: Dict) -> Dict:
        """推测：分析设计决策"""
        return {"design_rationale": "inferred", "trade_offs": []}
    
    def _question(self, sample: Dict) -> List[str]:
        """提问：识别不确定点"""
        return ["Why use this specific attention mechanism?", "How does this scale to larger graphs?"]
    
    def _challenge(self, sample: Dict) -> List[str]:
        """质疑：识别潜在问题"""
        return ["Memory efficiency concerns", "Generalization to unseen datasets"]
    
    def save_state(self, path: str):
        """保存Agentic循环状态"""
        state = {
            "knowledge_base": self.knowledge_base,
            "beliefs": [
                {
                    "hypothesis": b.hypothesis,
                    "confidence": b.confidence,
                    "source": b.source,
                    "level": b.level
                }
                for b in self.beliefs
            ],
            "iteration_count": self.iteration_count
        }
        Path(path).write_text(json.dumps(state, indent=2))
'''

# 保存到文件
with open('/mnt/agents/output/agentic_loop.py', 'w', encoding='utf-8') as f:
    f.write(agentic_loop_code)

print("✅ agentic_loop.py 已生成")
print(f"文件大小: {len(agentic_loop_code)} 字符")
