# autostat/__init__.py
"""
AutoStat - 智能统计分析工具
"""

__version__ = "0.1.0"
__author__ = "AutoStat Team"

# 从兼容层导入（保持原有路径）
from autostat.analyzer import AutoStatisticalAnalyzer
from autostat.multi_analyzer import MultiTableStatisticalAnalyzer
from autostat.loader import DataLoader
from autostat.checker import ConditionChecker
from autostat.reporter import Reporter

__all__ = [
    "AutoStatisticalAnalyzer",
    "MultiTableStatisticalAnalyzer",
    "DataLoader",
    "ConditionChecker",
    "Reporter"
]