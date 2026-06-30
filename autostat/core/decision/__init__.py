"""
智能决策支持模块

提供从"发现问题"到"给出方案"的完整决策支持：
- 异常发现 (Anomaly Detection)
- 根因分析 (Root Cause Analysis)
- 行动建议 (Action Recommendation)
- 效果追踪 (Effect Tracking)
"""

from autostat.core.decision.anomaly import AnomalyDetector, AnomalyType, Anomaly
from autostat.core.decision.root_cause import RootCauseAnalyzer, RootCause, RootCauseResult
from autostat.core.decision.recommender import ActionRecommender, ActionSuggestion, ActionPriority, ActionDifficulty
from autostat.core.decision.tracker import ActionTracker, ActionRecord, TrackerResult

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class DecisionResult:
    """决策结果 - 组合异常、根因和建议"""
    anomaly: Dict[str, Any]
    root_causes: List[RootCause]
    suggestions: List[ActionSuggestion]
    summary: str
    timestamp: str = field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())


__all__ = [
    # 异常发现
    "AnomalyDetector",
    "AnomalyType",
    "Anomaly",
    # 根因分析
    "RootCauseAnalyzer",
    "RootCause",
    "RootCauseResult",
    # 行动建议
    "ActionRecommender",
    "ActionSuggestion",
    "ActionPriority",
    "ActionDifficulty",
    # 效果追踪
    "ActionTracker",
    "ActionRecord",
    "TrackerResult",
    # 组合结果
    "DecisionResult",
]