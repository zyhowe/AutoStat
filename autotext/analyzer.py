"""
主分析器类 - 整合所有文本分析模块
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union

from autotext.loader import TextLoader
from autotext.checker import TextChecker
from autotext.core.preprocessor import TextPreprocessor
from autotext.core.stats import TextStats
from autotext.core.quality import TextQuality
from autotext.core.keyword import KeywordExtractor
from autotext.core.sentiment import SentimentAnalyzer
from autotext.core.entity import EntityRecognizer
from autotext.core.cluster import TextClusterer
from autotext.core.topic import TopicModeler
from autotext.core.trend import TrendAnalyzer
from autotext.core.llm_enhance import LLMEnhancer
from autotext.reporter import TextReporter


class TextAnalyzer:
    """智能文本分析器"""

    def __init__(self, data: Union[pd.DataFrame, str, List[str]],
                 text_col: Optional[str] = None,
                 title_col: Optional[str] = None,
                 time_col: Optional[str] = None,
                 metric_cols: Optional[Dict[str, str]] = None,
                 source_name: Optional[str] = None,
                 quiet: bool = False):
        """
        初始化文本分析器
        """
        self.source_name = source_name
        self.quiet = quiet
        self.text_col = text_col
        self.title_col = title_col
        self.time_col = time_col
        self.metric_cols = metric_cols

        # 初始化结果容器
        self.texts = []
        self.titles = []
        self.dates = []
        self.metrics = {}

        # 多版本数据存储
        self.raw_texts = []
        self.cleaned_texts = []
        self.content_texts = []
        self.filtered_texts = []
        self.tokens_list = []
        self.tokens_cleaned_list = []

        self.preprocessed_data = []
        self.field_info = {}
        self.quality_report = {}
        self.stats_result = {}
        self.keywords = {}
        self.sentiment_results = []
        self.sentiment_distribution = {}
        self.entity_results = []
        self.entity_stats = {}
        self.clusterer = None
        self.cluster_info = []
        self.topic_modeler = None
        self.topics = []
        self.trend_analyzer = None
        self.trend_info = {}
        self.llm_enhancer = None
        self.cleaning_suggestions = []

        # 模板词
        self.start_templates = set()
        self.end_templates = set()

        # 加载数据
        self._load_data(data, text_col, title_col, time_col, metric_cols)

        # 检查器
        self.checker = TextChecker(self.texts, self.titles, self.dates)

        if not self.quiet:
            print("\n" + "=" * 70)
            print("📝 启动文本分析流程")
            print("=" * 70)

    def _load_data(self, data, text_col, title_col, time_col, metric_cols):
        """加载数据"""
        if isinstance(data, pd.DataFrame):
            loaded = TextLoader.from_dataframe(data, text_col, title_col, time_col, metric_cols)
            self.data = data
            self.texts = loaded["texts"]
            self.titles = loaded.get("titles", [])
            self.dates = loaded.get("dates", [])
            self.metrics = loaded.get("metrics", {})
        elif isinstance(data, str):
            if os.path.isdir(data):
                result = TextLoader.from_folder(data)
                self.texts = []
                for file_name, file_texts in result.items():
                    self.texts.extend(file_texts)
                self.source_name = self.source_name or data
                self.data = None
            else:
                self.texts = TextLoader.from_file(data)
                self.source_name = self.source_name or data
                self.data = None
        elif isinstance(data, list):
            self.texts = data
            self.data = None
        else:
            raise ValueError("data 必须是 DataFrame、文件路径、文件夹路径或文本列表")

        if not self.texts:
            raise ValueError("没有加载到有效的文本数据")

        self.raw_texts = self.texts.copy()

    def _preprocess(self):
        """统一预处理所有文本"""
        if not self.texts:
            if not self.quiet:
                print("  ⚠️ 没有文本数据")
            return

        if not self.quiet:
            print("\n【阶段1】文本预处理...")

        preprocessor = TextPreprocessor()  # 保存实例

        # 第一步：基础清洗，获取清洗后文本用于模板检测
        cleaned_texts = [preprocessor.clean_text(t) for t in self.texts if t]
        effective_texts = [t for t in cleaned_texts if len(t) > 50]

        if not self.quiet:
            print(f"  🔍 有效文本数: {len(effective_texts)}/{len(self.texts)}")
            print("  🔍 检测模板词...")

        # 第二步：检测模板词
        if len(effective_texts) >= 10:
            start_tmpl, end_tmpl = preprocessor.detect_template_words(
                effective_texts,
                min_abs=10,
                min_ratio=0.05,
                start_thresholds={1: 0.3, 2: 0.2, 3: 0.1},
                end_thresholds={-1: 0.3, -2: 0.2, -3: 0.1},
                verbose=not self.quiet
            )
            preprocessor.start_templates = start_tmpl
            preprocessor.end_templates = end_tmpl
            self.start_templates = start_tmpl
            self.end_templates = end_tmpl
        else:
            if not self.quiet:
                print("  ⚠️ 有效文本不足，跳过模板检测")

        # 第三步：统一处理每条文本
        if not self.quiet:
            print("  🔧 执行完整预处理...")

        self.preprocessed_data = []
        self.cleaned_texts = []
        self.content_texts = []
        self.filtered_texts = []
        self.tokens_list = []
        self.tokens_cleaned_list = []

        for text in self.texts:
            result = preprocessor.full_process(
                text,
                remove_templates=True,
                filter_noise=True,
                min_len=20
            )
            self.preprocessed_data.append(result)
            self.cleaned_texts.append(result["cleaned"])
            self.content_texts.append(result["content"])
            self.filtered_texts.append(result["filtered"])
            self.tokens_list.append(result["tokens"])
            self.tokens_cleaned_list.append(result["tokens_cleaned"])

        # 统计语言分布
        lang_dist = {}
        for d in self.preprocessed_data:
            lang = d.get("language", "unknown")
            if lang != "unknown":
                lang_dist[lang] = lang_dist.get(lang, 0) + 1

        if not self.quiet:
            print(f"  ✅ 完成，共 {len(self.texts)} 条文本")
            print(f"  📊 语言分布: 中文 {lang_dist.get('zh', 0)} 条, 英文 {lang_dist.get('en', 0)} 条")
            if self.start_templates or self.end_templates:
                print(f"  🧹 已切除首尾模板句子")

    def _compute_stats(self):
        """计算基础统计（使用 content_texts）"""
        if not self.quiet:
            print("\n【阶段2】基础统计...")

        stats = TextStats(self.content_texts)
        self.stats_result = stats.compute_stats()

        if not self.quiet:
            print(f"  ✅ 总文本数: {self.stats_result['total_count']}")
            print(f"  ✅ 平均长度: {self.stats_result['char_length']['mean']:.1f} 字符")

    def _check_quality(self):
        """检查数据质量（使用 content_texts）"""
        if not self.quiet:
            print("\n【阶段3】数据质量检查...")

        quality = TextQuality(self.content_texts)
        self.quality_report = quality.check()
        self.cleaning_suggestions = quality.get_cleaning_suggestions()

        if not self.quiet:
            summary = quality.get_summary()
            print(f"  ✅ 空文本: {summary['empty_count']} 条")
            print(f"  ✅ 重复文本: {summary['duplicate_count']} 对")

    def _extract_keywords(self):
        """提取关键词（使用 tokens_cleaned_list）"""
        if not self.quiet:
            print("\n【阶段4】关键词提取...")

        extractor = KeywordExtractor()
        self.keywords["frequency"] = extractor.extract_frequency(self.tokens_cleaned_list, top_n=50)

        if len(self.filtered_texts) > 1:
            try:
                self.keywords["tfidf"] = extractor.extract_tfidf(self.filtered_texts, top_n=50)
            except Exception as e:
                if not self.quiet:
                    print(f"  ⚠️ TF-IDF 关键词提取失败: {e}")

        if not self.quiet:
            print(f"  ✅ 提取高频词 {len(self.keywords.get('frequency', []))} 个")

    def _analyze_sentiment(self):
        """情感分析（使用 content_texts）"""
        if not self.quiet:
            print("\n【阶段5】情感分析...")

        analyzer = SentimentAnalyzer()
        self.sentiment_results = analyzer.analyze_batch(self.content_texts)
        self.sentiment_distribution = analyzer.get_distribution(self.sentiment_results)

        if not self.quiet:
            pos_rate = self.sentiment_distribution["positive_rate"]
            neg_rate = self.sentiment_distribution["negative_rate"]
            print(f"  ✅ 积极: {pos_rate:.1%}  消极: {neg_rate:.1%}  中性: {self.sentiment_distribution['neutral_rate']:.1%}")

    def _recognize_entities(self):
        """实体识别（使用 content_texts）"""
        if not self.quiet:
            print("\n【阶段6】实体识别...")

        recognizer = EntityRecognizer()
        self.entity_results = recognizer.recognize_batch(self.content_texts)
        self.entity_stats = recognizer.get_statistics(self.entity_results)

        if not self.quiet:
            person_count = self.entity_stats.get("person", {}).get("unique", 0)
            location_count = self.entity_stats.get("location", {}).get("unique", 0)
            print(f"  ✅ 识别到人名: {person_count} 个, 地名: {location_count} 个")

    def _cluster_texts(self):
        """文本聚类（与主题建模使用相同数据）"""
        clustering_check = self.checker.check_clustering()
        if not clustering_check.get("suitable"):
            if not self.quiet:
                print(f"\n【阶段7】文本聚类... 跳过 ({clustering_check.get('reason')})")
            return

        if not self.quiet:
            print("\n【阶段7】文本聚类...")

        # 与主题建模使用相同数据
        texts_for_modeling = self._get_texts_for_modeling()

        if len(texts_for_modeling) < 20:
            if not self.quiet:
                print(f"  ⚠️ 有效文本不足 ({len(texts_for_modeling)} < 20)，跳过聚类")
            return

        self.clusterer = TextClusterer()
        try:
            self.clusterer.fit(texts_for_modeling)
            self.cluster_info = self.clusterer.get_cluster_info(texts_for_modeling)
            if not self.quiet:
                print(f"  ✅ 聚类完成，共 {len(self.cluster_info)} 个簇")
                # 打印关键词示例
                for cluster in self.cluster_info[:3]:
                    print(f"    簇{cluster['cluster_id']}: {cluster['top_words'][:5]}")
        except Exception as e:
            if not self.quiet:
                print(f"  ⚠️ 聚类失败: {e}")

    def _topic_modeling(self):
        """主题建模（使用建模专用文本）"""
        topic_check = self.checker.check_topic_modeling()
        if not topic_check.get("suitable"):
            if not self.quiet:
                print(f"\n【阶段8】主题建模... 跳过 ({topic_check.get('reason')})")
            return

        if not self.quiet:
            print("\n【阶段8】主题建模...")

        # 获取建模专用文本
        texts_for_modeling = self._get_texts_for_modeling()

        if len(texts_for_modeling) < 50:
            if not self.quiet:
                print(f"  ⚠️ 有效文本不足 ({len(texts_for_modeling)} < 50)，跳过分组")
            return

        self.topic_modeler = TopicModeler()
        try:
            self.topic_modeler.fit(texts_for_modeling)
            self.topics = self.topic_modeler.get_topics()
            if not self.quiet:
                print(f"  ✅ 主题建模完成，共 {len(self.topics)} 个主题")
                # 打印每个主题的关键词示例
                for topic in self.topics[:3]:
                    print(f"    主题{topic['topic_id']}: {topic['keywords'][:5]}")
        except Exception as e:
            if not self.quiet:
                print(f"  ⚠️ 主题建模失败: {e}")

    def _analyze_trend(self):
        """时间趋势分析（使用 dates 和 raw_texts）"""
        if self.dates:
            trend_check = self.checker.check_time_series()
            if trend_check.get("suitable"):
                if not self.quiet:
                    print("\n【阶段9】时间趋势分析...")
                self.trend_analyzer = TrendAnalyzer(self.content_texts, self.dates)
                self.trend_info = {
                    "time_range": self.trend_analyzer.get_time_range(),
                    "trend_line": self.trend_analyzer.get_trend_line(),
                    "seasonal_pattern": self.trend_analyzer.get_seasonal_pattern(),
                    "anomalies": self.trend_analyzer.detect_anomalies()
                }
                if not self.quiet:
                    print(f"  ✅ 时间范围: {self.trend_info['time_range'].get('start')} -> {self.trend_info['time_range'].get('end')}")
                    print(f"  ✅ 趋势方向: {self.trend_info['trend_line'].get('direction', 'stable')}")

    def set_llm_client(self, llm_client):
        """设置大模型客户端"""
        self.llm_enhancer = LLMEnhancer(llm_client)
        if not self.quiet:
            print("\n【阶段10】大模型增强已启用")

    def generate_full_report(self):
        """生成完整分析报告"""
        self._preprocess()
        self._compute_stats()
        self._check_quality()
        self._extract_keywords()
        self._analyze_sentiment()
        self._recognize_entities()
        self._cluster_texts()
        self._topic_modeling()
        self._analyze_trend()

        if not self.quiet:
            print("\n" + "=" * 70)
            print("✅ 文本分析完成")
            print("=" * 70)

    def to_html(self, output_file: str = None, title: str = "文本分析报告") -> str:
        """生成 HTML 报告"""
        reporter = TextReporter(self)
        return reporter.to_html(output_file, title)

    def to_json(self, output_file: str = None) -> str:
        """生成 JSON 报告"""
        reporter = TextReporter(self)
        return reporter.to_json(output_file)

    def to_markdown(self, output_file: str = None) -> str:
        """生成 Markdown 报告"""
        reporter = TextReporter(self)
        return reporter.to_markdown(output_file)

    # ==================== 保存方法 ====================

    def save_raw_texts(self, output_path: str = None):
        """保存原始文本"""
        if output_path is None:
            output_path = "raw_texts.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(self.raw_texts):
                f.write(f"=== 文本 {i+1} ===\n")
                f.write(text[:500] + "..." if len(text) > 500 else text)
                f.write("\n\n")
        print(f"✅ 原始文本已保存到 {output_path}")

    def save_cleaned_texts(self, output_path: str = None):
        """保存清洗后文本"""
        if output_path is None:
            output_path = "cleaned_texts.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(self.cleaned_texts):
                if text:
                    f.write(f"=== 文本 {i+1} ===\n")
                    f.write(text[:500] + "..." if len(text) > 500 else text)
                    f.write("\n\n")
        print(f"✅ 清洗后文本已保存到 {output_path}")

    def save_content_texts(self, output_path: str = None):
        """保存正文文本（切除模板后）"""
        if output_path is None:
            output_path = "content_texts.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(self.content_texts):
                if text:
                    f.write(f"=== 文本 {i+1} ===\n")
                    f.write(text[:500] + "..." if len(text) > 500 else text)
                    f.write("\n\n")
        print(f"✅ 正文文本已保存到 {output_path}")

    def save_filtered_texts(self, output_path: str = None):
        """保存过滤后文本（去除噪音）"""
        if output_path is None:
            output_path = "filtered_texts.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(self.filtered_texts):
                if text:
                    f.write(f"=== 文本 {i+1} ===\n")
                    f.write(text[:500] + "..." if len(text) > 500 else text)
                    f.write("\n\n")
        print(f"✅ 过滤后文本已保存到 {output_path}")

    def save_templates(self, output_path: str = None):
        """保存检测到的模板词"""
        if output_path is None:
            output_path = "templates.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== 开头模板词 ===\n")
            for word in sorted(self.start_templates):
                f.write(word + "\n")
            f.write("\n=== 结尾模板词 ===\n")
            for word in sorted(self.end_templates):
                f.write(word + "\n")
        print(f"✅ 模板词已保存到 {output_path}")

    def _get_texts_for_modeling(self) -> List[str]:
        """
        获取用于建模（聚类、主题建模）的清洗文本

        额外处理：
        1. 过滤包含股票代码的句子
        2. 过滤过短句子
        3. 使用词级别过滤后的 tokens 重建文本
        """
        texts = []

        for i, text in enumerate(self.content_texts):
            if not text:
                continue

            # 使用预处理后的 tokens
            tokens = self.tokens_cleaned_list[i] if i < len(self.tokens_cleaned_list) else []

            # 应用词级别过滤
            if hasattr(self, 'preprocessor') and self.preprocessor:
                tokens = self.preprocessor.filter_tokens(tokens)

            if not tokens:
                continue

            # 重建文本（用空格连接）
            cleaned_text = ' '.join(tokens)

            # 过滤过短文本
            if len(cleaned_text) < 50:
                continue

            # 过滤包含股票代码的文本
            import re
            if re.search(r'[06]\d{5}', cleaned_text):
                continue

            texts.append(cleaned_text)

        return texts