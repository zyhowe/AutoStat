"""
核心分析模块
"""

from autostat.core.analyzer import AutoStatisticalAnalyzer
from autostat.core.base import BaseAnalyzer
from autostat.core.timeseries import TimeSeriesAnalyzer
from autostat.core.relationship import RelationshipAnalyzer
from autostat.core.recommendation import RecommendationAnalyzer
from autostat.core.report_data import ReportDataBuilder
from autostat.core.audit import AuditRuleDiscoverer, discover_audit_rules
from autostat.core.date_rules import DateRuleDiscoverer, discover_date_rules
from autostat.core.plots import (
    PlotGenerator,
    plot_categorical,
    plot_continuous,
    plot_timeseries,
    plot_correlation,
    plot_categorical_correlation,
    plot_numeric_categorical_eta
)

# ==================== 新增模块 ====================

# 数据质量监控
from autostat.core.scorer import (
    QualityScorer,
    QualityGrade,
    QUALITY_GRADES
)


# ==================================================

__all__ = [
    # 原有导出
    "AutoStatisticalAnalyzer",
    "BaseAnalyzer",
    "TimeSeriesAnalyzer",
    "RelationshipAnalyzer",
    "RecommendationAnalyzer",
    "ReportDataBuilder",
    "AuditRuleDiscoverer",
    "discover_audit_rules",
    "DateRuleDiscoverer",
    "discover_date_rules",
    "PlotGenerator",
    "plot_categorical",
    "plot_continuous",
    "plot_timeseries",
    "plot_correlation",
    "plot_categorical_correlation",
    "plot_numeric_categorical_eta",

    # 数据质量监控
    "QualityScorer",
    "QualityGrade",
    "QUALITY_GRADES",
 
]