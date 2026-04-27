"""
AutoText - 智能文本分析工具
"""

__version__ = "0.1.0"
__author__ = "AutoStat Team"

from autotext.analyzer import TextAnalyzer
from autotext.loader import TextLoader
from autotext.checker import TextChecker
from autotext.reporter import TextReporter

__all__ = [
    "TextAnalyzer",
    "TextLoader",
    "TextChecker",
    "TextReporter",
]