# autostat/__init__.py

"""
AutoStat - 智能统计分析工具
"""

__version__ = "0.2.0"
__author__ = "AutoStat Team"

# 从兼容层导入（保持原有路径）
from autostat.analyzer import AutoStatisticalAnalyzer
from autostat.multi_analyzer import MultiTableStatisticalAnalyzer
from autostat.loader import DataLoader
from autostat.checker import ConditionChecker
from autostat.reporter import Reporter

# 新增：核心模块导出
from autostat.core.base import BaseAnalyzer
from autostat.core.timeseries import TimeSeriesAnalyzer
from autostat.core.relationship import RelationshipAnalyzer
from autostat.core.recommendation import RecommendationAnalyzer
from autostat.core.report_data import ReportDataBuilder
from autostat.core.plots import (
    PlotGenerator,
    plot_categorical,
    plot_continuous,
    plot_timeseries,
    plot_correlation,
    plot_categorical_correlation,
    plot_numeric_categorical_eta
)

__all__ = [
    # 原有导出
    "AutoStatisticalAnalyzer",
    "MultiTableStatisticalAnalyzer",
    "DataLoader",
    "ConditionChecker",
    "Reporter",

    # 新增导出
    "BaseAnalyzer",
    "TimeSeriesAnalyzer",
    "RelationshipAnalyzer",
    "RecommendationAnalyzer",
    "ReportDataBuilder",
    "PlotGenerator",
    "plot_categorical",
    "plot_continuous",
    "plot_timeseries",
    "plot_correlation",
    "plot_categorical_correlation",
    "plot_numeric_categorical_eta"
]