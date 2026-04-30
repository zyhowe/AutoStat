"""文本分析核心模块"""

from autotext.core.detector import FieldDetector
from autotext.core.preprocessor import TextPreprocessor
from autotext.core.stats import TextStats
from autotext.core.quality import TextQuality
from autotext.core.keyword import KeywordExtractor
from autotext.core.sentiment import SentimentAnalyzer
from autotext.core.topic_model import TopicModeler
from autotext.core.relation import RelationDiscoverer
from autotext.core.summarizer import TextRankSummarizer, LLMSummarizer
from autotext.core.event_extractor import EventExtractor
from autotext.core.graph_builder import GraphBuilder
from autotext.core.graph_analyzer import GraphAnalyzer
from autotext.core.entity_profile import EntityProfileBuilder
from autotext.core.timeline_builder import TimelineBuilder, build_timeline_from_analyzer

# entity.py 保留但不导出
try:
    from autotext.core.entity import EntityRecognizer
except ImportError:
    EntityRecognizer = None

__all__ = [
    "FieldDetector",
    "TextPreprocessor",
    "TextStats",
    "TextQuality",
    "KeywordExtractor",
    "SentimentAnalyzer",
    "TopicModeler",
    "RelationDiscoverer",
    "EntityRecognizer",
    "TextRankSummarizer",
    "LLMSummarizer",
    "EventExtractor",
    "GraphBuilder",
    "GraphAnalyzer",
    "EntityProfileBuilder",
    "TimelineBuilder",
    "build_timeline_from_analyzer"
]