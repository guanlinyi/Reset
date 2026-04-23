
# 核心代码文件7: Agent系统（目标管理、记忆、工具调用）
agent_system_code = '''"""
Agent System - 自主Agent核心组件

包含：
- GoalManager: 目标管理（分解、粗糙容忍度）
- Memory: 短期/长期记忆系统
- ToolRegistry: 工具注册与调用
- BaseAgent: Agent抽象基类

设计理念：
- 目标驱动：所有行为围绕目标展开
- 记忆增强：经验积累与检索
- 工具使用：扩展Agent能力边界
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import json
import heapq

logger = logging.getLogger(__name__)


# ============ 目标管理 ============

@dataclass
class GoalNode:
    """目标节点 - 支持嵌套和优先级"""
    id: str
    description: str
    priority: int = 5                    # 1-10, 数字越小优先级越高
    status: str = "pending"              # pending, active, completed, failed, tech_debt
    parent_id: Optional[str] = None
    sub_goals: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_leaf(self) -> bool:
        return len(self.sub_goals) == 0
    
    def is_ready(self) -> bool:
        return self.status == "pending"


class GoalManager:
    """
    目标管理器
    
    实现目标分解和粗糙容忍度管理
    支持优先级队列和依赖关系
    """
    
    def __init__(self):
        self.goals: Dict[str, GoalNode] = {}
        self.root_goal_id: Optional[str] = None
        
    def set_global_goal(self, description: str) -> str:
        """设置全局大目标"""
        goal_id = f"goal_root_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        goal = GoalNode(
            id=goal_id,
            description=description,
            priority=1
        )
        self.goals[goal_id] = goal
        self.root_goal_id = goal_id
        logger.info(f"Global goal set: {description}")
        return goal_id
    
    def decompose(self, goal_id: str, sub_descriptions: List[str]) -> List[str]:
        """
        目标分解
        
        将大目标分解为可执行的小目标
        保持粗糙容忍度：每个子目标先完成demo即可
        """
        if goal_id not in self.goals:
            raise ValueError(f"Goal {goal_id} not found")
        
        parent = self.goals[goal_id]
        sub_goal_ids = []
        
        for i, desc in enumerate(sub_descriptions):
            sub_id = f"{goal_id}_sub_{i}"
            sub_goal = GoalNode(
                id=sub_id,
                description=desc,
                priority=parent.priority + 1,
                parent_id=goal_id,
                metadata={"tolerance": "rough", "demo_ready": False}
            )
            self.goals[sub_id] = sub_goal
            parent.sub_goals.append(sub_id)
            sub_goal_ids.append(sub_id)
            logger.info(f"  Sub-goal created: {sub_id} - {desc}")
        
        return sub_goal_ids
    
    def get_next_goal(self) -> Optional[GoalNode]:
        """获取下一个待执行的目标（按优先级）"""
        ready_goals = [
            g for g in self.goals.values()
            if g.status == "pending" and g.is_leaf()
        ]
        
        if not ready_goals:
            return None
        
        # 按优先级排序
        ready_goals.sort(key=lambda g: g.priority)
        return ready_goals[0]
    
    def mark_completed(self, goal_id: str, demo_only: bool = True):
        """
        标记目标完成
        
        Args:
            demo_only: 是否仅完成Demo（粗糙容忍度）
        """
        if goal_id not in self.goals:
            return
        
        goal = self.goals[goal_id]
        goal.status = "completed"
        goal.completed_at = datetime.now().isoformat()
        goal.metadata["demo_ready"] = True
        
        if demo_only:
            logger.info(f"Goal {goal_id} marked as DEMO COMPLETED (rough tolerance)")
        else:
            logger.info(f"Goal {goal_id} marked as FULLY COMPLETED")
        
        # 检查父目标是否可以标记完成
        if goal.parent_id:
            self._check_parent_completion(goal.parent_id)
    
    def mark_tech_debt(self, goal_id: str):
        """标记为技术债，转向下一目标"""
        if goal_id in self.goals:
            self.goals[goal_id].status = "tech_debt"
            logger.info(f"Goal {goal_id} marked as TECH DEBT")
    
    def _check_parent_completion(self, parent_id: str):
        """检查父目标是否所有子目标都完成"""
        parent = self.goals[parent_id]
        all_completed = all(
            self.goals[sg_id].status in ["completed", "tech_debt"]
            for sg_id in parent.sub_goals
        )
        
        if all_completed:
            parent.status = "completed"
            logger.info(f"Parent goal {parent_id} auto-completed")
    
    def get_progress(self) -> Dict[str, Any]:
        """获取整体进度"""
        total = len(self.goals)
        completed = sum(1 for g in self.goals.values() if g.status == "completed")
        tech_debt = sum(1 for g in self.goals.values() if g.status == "tech_debt")
        pending = sum(1 for g in self.goals.values() if g.status == "pending")
        
        return {
            "total_goals": total,
            "completed": completed,
            "tech_debt": tech_debt,
            "pending": pending,
            "completion_rate": completed / total if total > 0 else 0
        }


# ============ 记忆系统 ============

@dataclass
class MemoryEntry:
    """记忆条目"""
    content: Any
    entry_type: str                      # experience, knowledge, observation
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    importance: float = 1.0              # 重要性评分
    tags: List[str] = field(default_factory=list)
    source: str = "internal"             # internal, external


class ShortTermMemory:
    """短期记忆 - 最近的经验和上下文"""
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.buffer: List[MemoryEntry] = []
        
    def add(self, entry: MemoryEntry):
        """添加记忆"""
        self.buffer.append(entry)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)  # FIFO
    
    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        """获取最近的n条记忆"""
        return self.buffer[-n:]
    
    def get_by_type(self, entry_type: str) -> List[MemoryEntry]:
        """按类型获取记忆"""
        return [e for e in self.buffer if e.entry_type == entry_type]
    
    def clear(self):
        """清空短期记忆"""
        self.buffer = []


class LongTermMemory:
    """长期记忆 - 持久化的知识和经验"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.knowledge_base: Dict[str, MemoryEntry] = {}
        self.experience_index: Dict[str, List[str]] = {}  # 按标签索引
        
    def store(self, key: str, entry: MemoryEntry):
        """存储长期记忆"""
        self.knowledge_base[key] = entry
        
        # 更新索引
        for tag in entry.tags:
            if tag not in self.experience_index:
                self.experience_index[tag] = []
            self.experience_index[tag].append(key)
    
    def retrieve(self, key: str) -> Optional[MemoryEntry]:
        """检索记忆"""
        return self.knowledge_base.get(key)
    
    def search_by_tags(self, tags: List[str]) -> List[MemoryEntry]:
        """按标签搜索"""
        results = set()
        for tag in tags:
            if tag in self.experience_index:
                results.update(self.experience_index[tag])
        return [self.knowledge_base[k] for k in results]
    
    def get_learnings(self) -> List[MemoryEntry]:
        """获取所有学习成果"""
        return [
            e for e in self.knowledge_base.values()
            if e.entry_type == "knowledge"
        ]
    
    def save(self):
        """持久化到磁盘"""
        if self.storage_path:
            data = {
                k: {
                    "content": v.content,
                    "type": v.entry_type,
                    "timestamp": v.timestamp,
                    "importance": v.importance,
                    "tags": v.tags,
                    "source": v.source
                }
                for k, v in self.knowledge_base.items()
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)


# ============ 工具系统 ============

class Tool:
    """工具抽象类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        pass
    
    def __repr__(self):
        return f"Tool({self.name}: {self.description})"


class CodeExecutionTool(Tool):
    """代码执行工具"""
    
    def __init__(self):
        super().__init__(
            name="code_executor",
            description="Execute Python code and return results"
        )
    
    def execute(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """执行代码"""
        try:
            # 这里应使用安全的执行环境（如subprocess或sandbox）
            # 简化示例：
            local_vars = {}
            exec(code, {"__builtins__": __builtins__}, local_vars)
            return {
                "success": True,
                "output": local_vars.get("result", None),
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "output": None,
                "error": str(e)
            }


class SearchTool(Tool):
    """搜索工具"""
    
    def __init__(self):
        super().__init__(
            name="search",
            description="Search for code samples, papers, or documentation"
        )
    
    def execute(self, query: str, source: str = "github") -> List[Dict]:
        """执行搜索"""
        # 这里接入实际搜索API
        logger.info(f"Searching '{query}' on {source}...")
        return [
            {
                "title": f"Sample result for {query}",
                "url": f"https://example.com/{source}/result",
                "relevance": 0.9
            }
        ]


class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        
    def register(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool
        logger.info(f"Tool registered: {tool.name}")
    
    def get(self, name: str) -> Tool:
        """获取工具"""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")
        return self.tools[name]
    
    def list_tools(self) -> List[str]:
        """列出所有工具"""
        return list(self.tools.keys())
    
    def execute(self, tool_name: str, **kwargs) -> Any:
        """执行工具"""
        tool = self.get(tool_name)
        return tool.execute(**kwargs)


# ============ Agent基类 ============

class BaseAgent(ABC):
    """
    抽象Agent基类
    
    整合目标管理、记忆、工具调用
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.goal_manager = GoalManager()
        self.short_term_memory = ShortTermMemory(capacity=config.get("stm_capacity", 100))
        self.long_term_memory = LongTermMemory(storage_path=config.get("ltm_path"))
        self.tool_registry = ToolRegistry()
        
        # 注册默认工具
        self._register_default_tools()
    
    def _register_default_tools(self):
        """注册默认工具"""
        self.tool_registry.register(CodeExecutionTool())
        self.tool_registry.register(SearchTool())
    
    def set_goal(self, description: str) -> str:
        """设置全局目标"""
        return self.goal_manager.set_global_goal(description)
    
    def decompose_goal(self, goal_id: str, sub_goals: List[str]) -> List[str]:
        """分解目标"""
        return self.goal_manager.decompose(goal_id, sub_goals)
    
    def remember(self, content: Any, entry_type: str = "experience", tags: List[str] = None):
        """记录到短期记忆"""
        entry = MemoryEntry(
            content=content,
            entry_type=entry_type,
            tags=tags or []
        )
        self.short_term_memory.add(entry)
    
    def learn(self, key: str, content: Any, tags: List[str] = None):
        """学习到长期记忆"""
        entry = MemoryEntry(
            content=content,
            entry_type="knowledge",
            tags=tags or [],
            source="learning"
        )
        self.long_term_memory.store(key, entry)
    
    def use_tool(self, tool_name: str, **kwargs) -> Any:
        """使用工具"""
        result = self.tool_registry.execute(tool_name, **kwargs)
        
        # 记录工具使用
        self.remember(
            content={"tool": tool_name, "args": kwargs, "result": result},
            entry_type="tool_usage",
            tags=["tool", tool_name]
        )
        
        return result
    
    @abstractmethod
    def run(self):
        """Agent主循环 - 子类实现"""
        pass


class ResearchAgent(BaseAgent):
    """
    研究型Agent
    
    实现外源学习模式：搜索、分析、理解
    """
    
    def run(self):
        """执行研究任务"""
        # 获取当前目标
        goal = self.goal_manager.get_next_goal()
        if not goal:
            logger.info("No pending goals")
            return
        
        logger.info(f"Researching: {goal.description}")
        
        # 搜索优秀样本
        search_results = self.use_tool("search", query=goal.description)
        
        # 分析样本并记录理解
        for result in search_results:
            self.learn(
                key=f"understanding_{result['title']}",
                content={
                    "source": result,
                    "imitation": "extracted_patterns",
                    "speculation": "design_rationale",
                    "questions": ["q1", "q2"],
                    "challenges": ["c1"]
                },
                tags=["external_learning", "understanding"]
            )
        
        # 标记完成
        self.goal_manager.mark_completed(goal.id, demo_only=True)
'''

with open('/mnt/agents/output/agent_system.py', 'w', encoding='utf-8') as f:
    f.write(agent_system_code)

print("✅ agent_system.py 已生成")
print(f"文件大小: {len(agent_system_code)} 字符")
