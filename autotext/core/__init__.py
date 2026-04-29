"""文本分析核心模块"""

from autotext.core.detector import FieldDetector
from autotext.core.preprocessor import TextPreprocessor
from autotext.core.stats import TextStats
from autotext.core.quality import TextQuality
from autotext.core.keyword import KeywordExtractor
from autotext.core.sentiment import SentimentAnalyzer
from autotext.core.entity import EntityRecognizer
from autotext.core.cluster import TextClusterer
from autotext.core.topic import TopicModeler
from autotext.core.trend import TrendAnalyzer

# 新增模块
try:
    from autotext.core.summarizer import TextRankSummarizer, LLMSummarizer
except ImportError:
    TextRankSummarizer = None
    LLMSummarizer = None

try:
    from autotext.core.relation_mining import RelationMiner
except ImportError:
    RelationMiner = None

try:
    from autotext.core.trend_detector import TrendDetector
except ImportError:
    TrendDetector = None

try:
    from autotext.core.info_extractor import InfoExtractor
except ImportError:
    InfoExtractor = None

__all__ = [
    "FieldDetector",
    "TextPreprocessor",
    "TextStats",
    "TextQuality",
    "KeywordExtractor",
    "SentimentAnalyzer",
    "EntityRecognizer",
    "TextClusterer",
    "TopicModeler",
    "TrendAnalyzer",
    "TextRankSummarizer",
    "LLMSummarizer",
    "RelationMiner",
    "TrendDetector",
    "InfoExtractor",
]