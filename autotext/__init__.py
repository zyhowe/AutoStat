"""AutoText - 智能文本分析工具"""

__version__ = "0.2.0"
__author__ = "AutoStat Team"

from autotext.analyzer import TextAnalyzer
from autotext.reporter import TextReporter
from autotext.loader import TextLoader
from autotext.checker import TextChecker

# 新增模块导出
from autotext.core.summarizer import TextRankSummarizer, LLMSummarizer
from autotext.core.relation_mining import RelationMiner
from autotext.core.trend_detector import TrendDetector
from autotext.core.info_extractor import InfoExtractor

__all__ = [
    "TextAnalyzer",
    "TextReporter",
    "TextLoader",
    "TextChecker",
    "TextRankSummarizer",
    "LLMSummarizer",
    "RelationMiner",
    "TrendDetector",
    "InfoExtractor",
]