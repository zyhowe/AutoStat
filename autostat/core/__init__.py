# autostat/core/__init__.py
"""核心分析模块"""

from autostat.core.analyzer import AutoStatisticalAnalyzer
from autostat.core.base import BaseAnalyzer
from autostat.core.timeseries import TimeSeriesAnalyzer
from autostat.core.relationship import RelationshipAnalyzer
from autostat.core.recommendation import RecommendationAnalyzer
from autostat.core.report_data import ReportDataBuilder
from autostat.core.audit import AuditRuleDiscoverer, discover_audit_rules
from autostat.core.date_rules import DateRuleDiscoverer, discover_date_rules

__all__ = [
    "AutoStatisticalAnalyzer",
    "BaseAnalyzer",
    "TimeSeriesAnalyzer",
    "RelationshipAnalyzer",
    "RecommendationAnalyzer",
    "ReportDataBuilder",
    "AuditRuleDiscoverer",
    "discover_audit_rules",
    "DateRuleDiscoverer",
    "discover_date_rules"
]