# autostat/core/__init__.py
"""核心分析模块"""

from autostat.core.analyzer import AutoStatisticalAnalyzer
from autostat.core.base import BaseAnalyzer
from autostat.core.timeseries import TimeSeriesAnalyzer
from autostat.core.relationship import RelationshipAnalyzer
from autostat.core.recommendation import RecommendationAnalyzer
from autostat.core.report_data import ReportDataBuilder
from autostat.core.audit import AuditRuleDiscoverer, discover_audit_rules  # 新增

__all__ = [
    "AutoStatisticalAnalyzer",
    "BaseAnalyzer",
    "TimeSeriesAnalyzer",
    "RelationshipAnalyzer",
    "RecommendationAnalyzer",
    "ReportDataBuilder",
    "AuditRuleDiscoverer",      # 新增
    "discover_audit_rules"       # 新增
]