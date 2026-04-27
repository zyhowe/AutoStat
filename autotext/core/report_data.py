"""
报告数据构建模块 - 为 Reporter 提供完整的报告数据
"""

from typing import Dict, Any, List, Optional
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


class ReportDataBuilder:
    """报告数据构建器"""

    def __init__(self, analyzer):
        """
        初始化报告数据构建器

        参数:
        - analyzer: TextAnalyzer 实例
        """
        self.analyzer = analyzer

    def build(self) -> Dict[str, Any]:
        """
        构建报告所需的所有数据

        返回: 完整的报告数据字典
        """
        data = self.analyzer.data
        texts = self.analyzer.texts
        preprocessed = self.analyzer.preprocessed_data

        # 字段检测结果
        detector = FieldDetector(data) if data is not None else None
        field_info = detector.detect_all() if detector else {}

        # 基础统计
        stats = TextStats(texts, preprocessed)
        stats_result = stats.compute_stats()

        # 数据质量
        quality = TextQuality(texts)
        quality_result = quality.check()
        cleaning_suggestions = quality.get_cleaning_suggestions()

        # 关键词
        keyword_extractor = KeywordExtractor()
        tokens_list = [p.get("tokens_cleaned", []) for p in preprocessed]
        keywords = keyword_extractor.extract_frequency(tokens_list, top_n=30)
        tfidf_keywords = []
        if len(texts) > 1:
            try:
                tfidf_keywords = keyword_extractor.extract_tfidf(texts, top_n=30)
            except:
                pass

        # 情感分析
        sentiment_analyzer = SentimentAnalyzer()
        sentiment_results = sentiment_analyzer.analyze_batch(texts)
        sentiment_dist = sentiment_analyzer.get_distribution(sentiment_results)
        sentiment_summary = sentiment_analyzer.get_summary(sentiment_results)

        # 实体识别
        entity_recognizer = EntityRecognizer()
        entity_results = entity_recognizer.recognize_batch(texts)
        entity_stats = entity_recognizer.get_statistics(entity_results)

        # 聚类
        cluster_info = []
        if self.analyzer.clusterer and self.analyzer.clusterer._fitted:
            cluster_info = self.analyzer.clusterer.get_cluster_info(texts)

        # 主题
        topic_info = []
        if self.analyzer.topic_modeler and self.analyzer.topic_modeler._fitted:
            topic_info = self.analyzer.topic_modeler.get_topics()
            topic_distribution = self.analyzer.topic_modeler.get_topic_distribution()

        # 时间趋势
        trend_info = {}
        if self.analyzer.time_col and self.analyzer.dates:
            trend_analyzer = TrendAnalyzer(texts, self.analyzer.dates)
            trend_info = {
                "time_range": trend_analyzer.get_time_range(),
                "trend_line": trend_analyzer.get_trend_line(),
                "seasonal_pattern": trend_analyzer.get_seasonal_pattern(),
                "anomalies": trend_analyzer.detect_anomalies()
            }

        # 大模型增强
        llm_insights = {}
        if self.analyzer.llm_enhancer and self.analyzer.llm_enhancer.is_available():
            # 为每个簇命名
            if cluster_info and not any(c.get("name") for c in cluster_info):
                for cluster in cluster_info[:5]:
                    name = self.analyzer.llm_enhancer.name_cluster(
                        cluster.get("top_words", [])[:10],
                        cluster.get("sample_texts", [])
                    )
                    if name:
                        cluster["name"] = name

            # 生成整体洞察
            overall_insight = self.analyzer.llm_enhancer.generate_insights(
                stats_result, sentiment_dist, topic_info, cluster_info
            )
            if overall_insight:
                llm_insights["overall"] = overall_insight

        return {
            "field_info": field_info,
            "stats": stats_result,
            "quality": quality_result,
            "cleaning_suggestions": cleaning_suggestions,
            "keywords": {
                "frequency": [{"word": w, "count": c} for w, c in keywords],
                "tfidf": [{"word": w, "score": s} for w, s in tfidf_keywords]
            },
            "sentiment": {
                "distribution": sentiment_dist,
                "summary": sentiment_summary,
                "results": sentiment_results[:10]  # 只返回前10条
            },
            "entity": entity_stats,
            "clusters": cluster_info,
            "topics": {
                "list": topic_info,
                "distribution": topic_distribution if self.analyzer.topic_modeler else []
            },
            "trend": trend_info,
            "llm_insights": llm_insights,
            "sample_texts": texts[:20]
        }