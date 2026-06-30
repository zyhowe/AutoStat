"""
数据质量监控模块

提供五维数据质量评分体系：
- 完整性 (Completeness)
- 准确性 (Accuracy)
- 一致性 (Consistency)
- 及时性 (Timeliness)
- 唯一性 (Uniqueness)
"""

from autostat.core.quality.scorer import QualityScorer, QualityGrade, QUALITY_GRADES
from autostat.core.quality.monitor import QualityMonitor
from autostat.core.quality.alert import QualityAlert

__all__ = [
    "QualityScorer",
    "QualityGrade",
    "QUALITY_GRADES",
    "QualityMonitor",
    "QualityAlert",
]