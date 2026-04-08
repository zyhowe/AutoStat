"""
AutoStat - 智能统计分析工具
"""

__version__ = "0.1.0"
__author__ = "AutoStat Team"

from autostat.analyzer import AutoStatisticalAnalyzer
from autostat.multi_analyzer import MultiTableStatisticalAnalyzer
from autostat.loader import DataLoader
from autostat.checker import ConditionChecker

__all__ = [
    "AutoStatisticalAnalyzer",
    "MultiTableStatisticalAnalyzer",
    "DataLoader",
    "ConditionChecker"
]