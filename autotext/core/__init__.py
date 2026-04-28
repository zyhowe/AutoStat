"""文本分析核心模块"""

# 尝试导入各模块，失败时给出提示
try:
    from autotext.core.detector import FieldDetector
except ImportError:
    FieldDetector = None

try:
    from autotext.core.preprocessor import TextPreprocessor
except ImportError:
    TextPreprocessor = None

try:
    from autotext.core.stats import TextStats
except ImportError:
    TextStats = None

try:
    from autotext.core.quality import TextQuality
except ImportError:
    TextQuality = None

try:
    from autotext.core.keyword import KeywordExtractor
except ImportError:
    KeywordExtractor = None

try:
    from autotext.core.sentiment import SentimentAnalyzer
except ImportError:
    SentimentAnalyzer = None

try:
    from autotext.core.entity import EntityRecognizer
except ImportError:
    EntityRecognizer = None

# 聚类模块（带降级）
try:
    from autotext.core.cluster import TextClusterer
except ImportError:
    TextClusterer = None
    print("⚠️ 聚类模块导入失败")

try:
    from autotext.core.topic import TopicModeler
except ImportError:
    TopicModeler = None
    print("⚠️ 主题建模模块导入失败")

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
]