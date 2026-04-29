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
from autotext.reporter import TextReporter

# 新增导入
try:
    from autotext.core.vectorizer import BertVectorizer
except ImportError:
    BertVectorizer = None
    print("⚠️ BertVectorizer 导入失败，请安装 transformers")

try:
    from autotext.core.ner import EntityRecognizer
except ImportError:
    EntityRecognizer = None
    print("⚠️ EntityRecognizer 导入失败")

try:
    from autotext.core.topic_model import TopicModeler
except ImportError:
    TopicModeler = None
    print("⚠️ TopicModeler 导入失败")

try:
    from autotext.core.insight import InsightDiscoverer, format_insights_for_report
except ImportError:
    InsightDiscoverer = None
    format_insights_for_report = lambda x: ""
    print("⚠️ InsightDiscoverer 导入失败")

# 新增模块导入
try:
    from autotext.core.event_extractor import EventExtractor
except ImportError:
    EventExtractor = None
    print("⚠️ EventExtractor 导入失败")

try:
    from autotext.core.graph_builder import GraphBuilder
except ImportError:
    GraphBuilder = None
    print("⚠️ GraphBuilder 导入失败")

try:
    from autotext.core.graph_analyzer import GraphAnalyzer
except ImportError:
    GraphAnalyzer = None
    print("⚠️ GraphAnalyzer 导入失败")

try:
    from autotext.core.entity_profile import EntityProfileBuilder
except ImportError:
    EntityProfileBuilder = None
    print("⚠️ EntityProfileBuilder 导入失败")

try:
    from autotext.core.timeline_builder import build_timeline_from_analyzer
except ImportError:
    build_timeline_from_analyzer = None
    print("⚠️ TimelineBuilder 导入失败")


class TextAnalyzer:
    """智能文本分析器"""

    def __init__(self, data: Union[pd.DataFrame, str, List[str]],
                 text_col: Optional[str] = None,
                 title_col: Optional[str] = None,
                 time_col: Optional[str] = None,
                 metric_cols: Optional[Dict[str, str]] = None,
                 source_name: Optional[str] = None,
                 quiet: bool = False,
                 use_bert: bool = True):
        """
        初始化文本分析器
        """
        self.source_name = source_name
        self.quiet = quiet
        self.text_col = text_col
        self.title_col = title_col
        self.time_col = time_col
        self.metric_cols = metric_cols
        self.use_bert = use_bert

        # 初始化结果容器
        self.texts = []
        self.titles = []
        self.dates = []
        self.metrics = {}
        self.data = None

        # 多版本数据存储
        self.raw_texts = []
        self.cleaned_texts = []
        self.content_texts = []
        self.filtered_texts = []
        self.tokens_list = []
        self.tokens_cleaned_list = []
        self.tokens_filtered_list = []

        self.preprocessed_data = []
        self.field_info = {}
        self.quality_report = {}
        self.stats_result = {}
        self.keywords = {}
        self.sentiment_results = []
        self.sentiment_distribution = {}
        self.trend_info = {}
        self.cleaning_suggestions = []

        # BERT分析结果容器
        self.embeddings = None
        self.entity_stats = {}
        self.entity_results = []
        self.topics = []
        self.event_timeline = {}
        self.insights = []

        # 新增结果容器
        self.events = []              # 事件列表
        self.graph = None             # 关系图谱
        self.graph_insights = {}      # 图分析洞察
        self.entity_profiles = []     # 实体档案列表
        self.timeline = None          # 时间线

        # 模块实例
        self.topic_modeler = None
        self.vectorizer = None
        self.ner_model = None
        self.event_extractor = None
        self.graph_builder = None
        self.graph_analyzer = None
        self.entity_profile_builder = None

        # 模板词
        self.start_templates = set()
        self.end_templates = set()

        # 语言分布
        self.language_distribution = {}

        # 加载数据
        self._load_data(data, text_col, title_col, time_col, metric_cols)

        # 检查器
        self.checker = TextChecker(self.texts, self.titles, self.dates)

        if not self.quiet:
            print("\n" + "=" * 70)
            print("📝 启动文本分析流程")
            print("=" * 70)

    def set_llm_client(self, llm_client):
        """设置大模型客户端"""
        self.llm_client = llm_client
        if not self.quiet:
            print("  ✅ 大模型客户端已设置")

    def _load_data(self, data, text_col, title_col, time_col, metric_cols):
        """加载数据"""
        # 情况1: DataFrame
        if isinstance(data, pd.DataFrame):
            loaded = TextLoader.from_dataframe(data, text_col, title_col, time_col, metric_cols)
            self.data = data
            self.texts = loaded["texts"]
            self.titles = loaded.get("titles", [])
            self.dates = loaded.get("dates", [])
            self.metrics = loaded.get("metrics", {})
            if not self.quiet:
                print(f"  ✅ 从 DataFrame 加载: {len(self.texts)} 条文本")

        # 情况2: 文件路径
        elif isinstance(data, str):
            if os.path.isdir(data):
                result = TextLoader.from_folder(data)
                self.texts = []
                for file_name, file_texts in result.items():
                    self.texts.extend(file_texts)
                self.source_name = self.source_name or data
                self.data = None
                if not self.quiet:
                    print(f"  ✅ 从文件夹加载: {len(self.texts)} 条文本")
            else:
                file_ext = os.path.splitext(data)[1].lower()

                if file_ext in ['.csv', '.xlsx', '.xls', '.json'] and text_col:
                    try:
                        if file_ext == '.csv':
                            df = pd.read_csv(data, encoding='utf-8-sig', engine='python', on_bad_lines='skip')
                        elif file_ext in ['.xlsx', '.xls']:
                            df = pd.read_excel(data, engine='openpyxl')
                        elif file_ext == '.json':
                            df = pd.read_json(data)
                        else:
                            df = None

                        if df is not None:
                            df.columns = [str(col).strip().replace('\n', '_').replace('\r', '_') for col in df.columns]
                            if text_col not in df.columns:
                                raise ValueError(f"文本列 '{text_col}' 不存在于文件中")
                            loaded = TextLoader.from_dataframe(df, text_col, title_col, time_col, metric_cols)
                            self.data = df
                            self.texts = loaded["texts"]
                            self.titles = loaded.get("titles", [])
                            self.dates = loaded.get("dates", [])
                            self.metrics = loaded.get("metrics", {})
                            if not self.quiet:
                                print(f"  ✅ 从 CSV/Excel 加载: {len(self.texts)} 条文本")
                            return
                    except Exception as e:
                        if not self.quiet:
                            print(f"  ⚠️ 结构化加载失败: {e}，尝试作为纯文本加载")

                self.texts = TextLoader.from_file(data)
                self.source_name = self.source_name or data
                self.data = None
                if not self.quiet:
                    print(f"  ✅ 从文件加载: {len(self.texts)} 条文本")

        # 情况3: 文本列表
        elif isinstance(data, list):
            self.texts = data
            self.data = None
            if not self.quiet:
                print(f"  ✅ 从列表加载: {len(self.texts)} 条文本")

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

        preprocessor = TextPreprocessor()

        cleaned_texts = [preprocessor.clean_text(t) for t in self.texts if t]
        effective_texts = [t for t in cleaned_texts if len(t) > 50]

        if not self.quiet:
            print(f"  🔍 有效文本数: {len(effective_texts)}/{len(self.texts)}")
            print("  🔍 检测模板词...")

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
            if not self.quiet and (start_tmpl or end_tmpl):
                print(f"  🧹 检测到开头模板词: {len(start_tmpl)} 个, 结尾模板词: {len(end_tmpl)} 个")
        else:
            if not self.quiet:
                print("  ⚠️ 有效文本不足，跳过模板检测")

        if not self.quiet:
            print("  🔧 执行完整预处理...")

        self.preprocessed_data = []
        self.cleaned_texts = []
        self.content_texts = []
        self.filtered_texts = []
        self.tokens_list = []
        self.tokens_cleaned_list = []
        self.tokens_filtered_list = []
        self.language_distribution = {}

        for text in self.texts:
            result = preprocessor.full_process(text, remove_templates=True, filter_noise=True, min_len=20)
            self.preprocessed_data.append(result)
            self.cleaned_texts.append(result["cleaned"])
            self.content_texts.append(result["content"])
            self.filtered_texts.append(result["filtered"])
            self.tokens_list.append(result["tokens"])
            self.tokens_cleaned_list.append(result["tokens_cleaned"])
            self.tokens_filtered_list.append(result.get("tokens_filtered", result["tokens_cleaned"]))

            lang = result.get("language", "unknown")
            if lang != "unknown":
                self.language_distribution[lang] = self.language_distribution.get(lang, 0) + 1

        if not self.quiet:
            print(f"  ✅ 完成，共 {len(self.texts)} 条文本")
            if self.language_distribution:
                print(f"  📊 语言分布: 中文 {self.language_distribution.get('zh', 0)} 条, "
                      f"英文 {self.language_distribution.get('en', 0)} 条")
            if self.start_templates or self.end_templates:
                print(f"  🧹 已切除首尾模板句子")

    def _compute_stats(self):
        """基础统计"""
        if not self.quiet:
            print("\n【阶段2】基础统计...")

        stats = TextStats(self.content_texts)
        self.stats_result = stats.compute_stats()

        if not self.quiet:
            print(f"  ✅ 总文本数: {self.stats_result['total_count']}")
            print(f"  ✅ 平均长度: {self.stats_result['char_length']['mean']:.1f} 字符")
            print(f"  ✅ 空文本率: {self.stats_result['empty_rate']:.1%}")

    def _check_quality(self):
        """数据质量检查"""
        if not self.quiet:
            print("\n【阶段3】数据质量检查...")

        quality = TextQuality(self.content_texts)
        self.quality_report = quality.check()
        self.cleaning_suggestions = quality.get_cleaning_suggestions()

        if not self.quiet:
            summary = quality.get_summary()
            print(f"  ✅ 空文本: {summary['empty_count']} 条")
            print(f"  ✅ 重复文本: {summary['duplicate_count']} 对")
            if summary.get('short_count', 0) > 0:
                print(f"  ⚠️ 过短文本: {summary['short_count']} 条")
            if summary.get('long_count', 0) > 0:
                print(f"  ⚠️ 过长文本: {summary['long_count']} 条")

    def _extract_keywords(self):
        """关键词提取"""
        if not self.quiet:
            print("\n【阶段4】关键词提取...")

        extractor = KeywordExtractor()
        self.keywords["frequency"] = extractor.extract_frequency(self.tokens_filtered_list, top_n=50)

        if len(self.filtered_texts) > 1:
            try:
                self.keywords["tfidf"] = extractor.extract_tfidf(self.filtered_texts, top_n=50)
            except Exception as e:
                if not self.quiet:
                    print(f"  ⚠️ TF-IDF 关键词提取失败: {e}")

        if not self.quiet:
            print(f"  ✅ 提取高频词 {len(self.keywords.get('frequency', []))} 个")

    def _analyze_sentiment(self):
        """情感分析（辅助功能）"""
        if not self.quiet:
            print("\n【阶段5】情感分析...")

        analyzer = SentimentAnalyzer()
        self.sentiment_results = analyzer.analyze_batch(self.content_texts)
        self.sentiment_distribution = analyzer.get_distribution(self.sentiment_results)

        if not self.quiet:
            pos_rate = self.sentiment_distribution["positive_rate"]
            neg_rate = self.sentiment_distribution["negative_rate"]
            print(f"  ✅ 积极: {pos_rate:.1%}  消极: {neg_rate:.1%}  中性: {self.sentiment_distribution['neutral_rate']:.1%}")

    def _filter_valid_entities(self, entity_stats: Dict) -> Dict:
        """过滤无效实体"""
        stopwords = {'的', '了', '是', '在', '和', '与', '或', '也', '都', '还',
                     '这', '那', '有', '为', '对', '而', '并', '且', '但', '就',
                     '到', '从', '由', '于', '之', '将', '会', '能', '可', '以',
                     '年', '月', '日', '时', '分', '秒', '上', '下', '中', '内',
                     '外', '前', '后', '左', '右', '高', '低', '大', '小', '多',
                     '少', '新', '旧', '好', '坏', '正', '负', '涨', '跌', '不'}

        filtered = {}
        for entity_type, stats in entity_stats.items():
            filtered_top = []
            for name, count in stats.get('top', []):
                if len(name) < 2:
                    continue
                if name in stopwords:
                    continue
                if name.isdigit():
                    continue
                import re
                if re.match(r'^[A-Z0-9]{6,}$', name):
                    continue
                filtered_top.append((name, count))

            filtered[entity_type] = {
                'total': stats.get('total', 0),
                'unique': len(filtered_top),
                'top': filtered_top[:20]
            }

        return filtered

    def _bert_analysis(self):
        """BERT增强分析 - 核心分析流程"""
        if not self.use_bert or len(self.texts) < 5:
            if not self.quiet:
                print("\n【BERT增强】跳过（文本数量不足或未启用）")
            return

        if not self.quiet:
            print("\n【BERT增强】深度语义分析...")

        valid_texts = [t if t and len(t) > 0 else " " for t in self.content_texts]

        # ==================== 1. 向量化 ====================
        if not self.quiet:
            print("  🔄 文本向量化...")

        try:
            if BertVectorizer is None:
                raise ImportError("BertVectorizer 未导入")
            self.vectorizer = BertVectorizer(device="cpu")
            self.embeddings = self.vectorizer.get_embeddings(valid_texts)
            if not self.quiet:
                print(f"  ✅ 向量化完成，维度: {self.embeddings.shape}")
        except Exception as e:
            if not self.quiet:
                print(f"  ❌ 向量化失败: {e}")
            return

        # ==================== 2. 实体识别 ====================
        if not self.quiet:
            print("  🔍 实体识别...")

        try:
            if EntityRecognizer is None:
                raise ImportError("EntityRecognizer 未导入")
            self.ner_model = EntityRecognizer(device="cpu")
            self.entity_results = self.ner_model.recognize(valid_texts)
            entity_stats_raw = self.ner_model.get_entity_stats(self.entity_results)
            self.entity_stats = self._filter_valid_entities(entity_stats_raw)
            if not self.quiet:
                total_entities = sum(len(ents) for ents in self.entity_results)
                print(f"  ✅ 识别到 {total_entities} 个实体")
        except Exception as e:
            if not self.quiet:
                print(f"  ❌ 实体识别失败: {e}")
            self.entity_results = []
            self.entity_stats = {}

        # ==================== 3. 主题建模 ====================
        if not self.quiet:
            print("  📚 主题建模...")

        try:
            if TopicModeler is None:
                raise ImportError("TopicModeler 未导入")
            self.topic_modeler = TopicModeler(n_topics=10)
            self.topic_modeler.fit(self.tokens_filtered_list)
            self.topics = self.topic_modeler.get_topics()
            if not self.quiet:
                print(f"  ✅ 主题建模完成，共 {len(self.topics)} 个主题")
        except Exception as e:
            if not self.quiet:
                print(f"  ❌ 主题建模失败: {e}")
            self.topics = []
            self.topic_modeler = None

        # ==================== 4. 事件抽取 ====================
        if not self.quiet:
            print("  📰 事件抽取...")

        try:
            if EventExtractor is not None:
                self.event_extractor = EventExtractor(use_model=True)
                events_results = self.event_extractor.extract(valid_texts)
                # 扁平化事件列表
                self.events = []
                for idx, events in enumerate(events_results):
                    for event in events:
                        event["text_index"] = idx
                        self.events.append(event)
                if not self.quiet:
                    print(f"  ✅ 抽取到 {len(self.events)} 个事件")
            else:
                self.events = []
        except Exception as e:
            if not self.quiet:
                print(f"  ❌ 事件抽取失败: {e}")
            self.events = []

        # ==================== 5. 关系图谱构建 ====================
        if not self.quiet:
            print("  🕸️ 关系图谱构建...")

        try:
            if GraphBuilder is not None:
                self.graph_builder = GraphBuilder()

                # 添加实体节点
                for entity_type, stats in self.entity_stats.items():
                    for entity_name, count in stats.get("top", [])[:30]:
                        self.graph_builder.add_entity_node(
                            f"{entity_type}:{entity_name}",
                            entity_name,
                            entity_type.upper(),
                            count
                        )

                # 添加事件节点
                for event in self.events[:50]:
                    self.graph_builder.add_event_node(
                        f"event_{event.get('text_index', 0)}",
                        event.get("event_type", "未知"),
                        event.get("trigger", ""),
                        event.get("timestamp", "")
                    )

                # 添加主题节点
                for topic in self.topics[:10]:
                    self.graph_builder.add_topic_node(
                        topic.get("topic_id", 0),
                        f"主题{topic.get('topic_id', 0)}",
                        topic.get("keywords", [])
                    )

                # 添加实体-实体边（基于共现）
                if hasattr(self, 'relation_result') and self.relation_result:
                    for pair in self.relation_result.get('cooccurrence_pairs', [])[:30]:
                        e1 = pair.get("entity1", "").split(":", 1)[-1]
                        e2 = pair.get("entity2", "").split(":", 1)[-1]
                        pmi = pair.get("pmi", 0)
                        self.graph_builder.add_entity_entity_edge(e1, e2, pmi, pmi)

                self.graph = self.graph_builder
        except Exception as e:
            if not self.quiet:
                print(f"  ❌ 图谱构建失败: {e}")
            self.graph = None

        # ==================== 6. 图算法分析 ====================
        if not self.quiet:
            print("  📊 图算法分析...")

        try:
            if GraphAnalyzer is not None and self.graph is not None:
                self.graph_analyzer = GraphAnalyzer(self.graph.get_graph())
                self.graph_insights = self.graph_analyzer.get_summary_insights()
                if not self.quiet:
                    insights = self.graph_insights
                    print(f"  ✅ 图分析完成: {insights['statistics']['node_count']} 节点, "
                          f"{insights['statistics']['edge_count']} 边")
        except Exception as e:
            if not self.quiet:
                print(f"  ❌ 图分析失败: {e}")
            self.graph_insights = {}

        # ==================== 7. 实体档案生成 ====================
        if not self.quiet:
            print("  📁 实体档案生成...")

        try:
            if EntityProfileBuilder is not None:
                self.entity_profile_builder = EntityProfileBuilder()
                self.entity_profiles = self.entity_profile_builder.build_from_analyzer(self)
                if not self.quiet:
                    print(f"  ✅ 生成 {len(self.entity_profiles)} 个实体档案")
        except Exception as e:
            if not self.quiet:
                print(f"  ❌ 实体档案生成失败: {e}")
            self.entity_profiles = []

        # ==================== 8. 时间线构建 ====================
        if not self.quiet:
            print("  📅 时间线构建...")

        try:
            if build_timeline_from_analyzer is not None:
                self.timeline = build_timeline_from_analyzer(self)
                if not self.quiet:
                    summary = self.timeline.get_summary()
                    if summary.get("has_data"):
                        date_range = summary.get("date_range", {})
                        print(f"  ✅ 时间线构建完成: {summary['total_events']} 个事件, "
                              f"{date_range.get('start', '')} ~ {date_range.get('end', '')}")
                    else:
                        print(f"  ⚠️ 无时间数据，时间线为空")
        except Exception as e:
            if not self.quiet:
                print(f"  ❌ 时间线构建失败: {e}")
            self.timeline = None

        # ==================== 9. 洞察发现 ====================
        if not self.quiet:
            print("  💡 洞察发现...")

        try:
            if InsightDiscoverer is not None:
                insight_discoverer = InsightDiscoverer()
                self.insights = insight_discoverer.discover_all(self)
                if not self.quiet:
                    print(f"  ✅ 发现 {len(self.insights)} 个洞察")
        except Exception as e:
            if not self.quiet:
                print(f"  ❌ 洞察发现失败: {e}")
            self.insights = []

        if not self.quiet:
            print("\n  🎉 BERT增强分析完成")

    def generate_full_report(self):
        """生成完整分析报告"""
        self._preprocess()
        self._compute_stats()
        self._check_quality()
        self._extract_keywords()
        self._analyze_sentiment()
        self._bert_analysis()

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
        if not self.quiet:
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
        if not self.quiet:
            print(f"✅ 清洗后文本已保存到 {output_path}")

    def save_content_texts(self, output_path: str = None):
        """保存正文文本"""
        if output_path is None:
            output_path = "content_texts.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(self.content_texts):
                if text:
                    f.write(f"=== 文本 {i+1} ===\n")
                    f.write(text[:500] + "..." if len(text) > 500 else text)
                    f.write("\n\n")
        if not self.quiet:
            print(f"✅ 正文文本已保存到 {output_path}")

    def save_filtered_texts(self, output_path: str = None):
        """保存过滤后文本"""
        if output_path is None:
            output_path = "filtered_texts.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(self.filtered_texts):
                if text:
                    f.write(f"=== 文本 {i+1} ===\n")
                    f.write(text[:500] + "..." if len(text) > 500 else text)
                    f.write("\n\n")
        if not self.quiet:
            print(f"✅ 过滤后文本已保存到 {output_path}")

    def save_templates(self, output_path: str = None):
        """保存模板词"""
        if output_path is None:
            output_path = "templates.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== 开头模板词 ===\n")
            for word in sorted(self.start_templates):
                f.write(word + "\n")
            f.write("\n=== 结尾模板词 ===\n")
            for word in sorted(self.end_templates):
                f.write(word + "\n")
        if not self.quiet:
            print(f"✅ 模板词已保存到 {output_path}")


# ==================== 便捷函数 ====================

def analyze_texts(
    texts: List[str],
    output_file: str = None,
    format: str = "html",
    quiet: bool = False,
    use_bert: bool = True
) -> TextAnalyzer:
    """便捷函数：快速分析文本列表"""
    analyzer = TextAnalyzer(texts, quiet=quiet, use_bert=use_bert)
    analyzer.generate_full_report()

    if output_file:
        if format == "html":
            analyzer.to_html(output_file)
        elif format == "json":
            analyzer.to_json(output_file)
        elif format == "md":
            analyzer.to_markdown(output_file)

    return analyzer


def analyze_file(
    file_path: str,
    text_col: str = None,
    title_col: str = None,
    time_col: str = None,
    output_file: str = None,
    format: str = "html",
    quiet: bool = False,
    use_bert: bool = True
) -> TextAnalyzer:
    """便捷函数：分析文本文件"""
    analyzer = TextAnalyzer(
        data=file_path,
        text_col=text_col,
        title_col=title_col,
        time_col=time_col,
        quiet=quiet,
        use_bert=use_bert
    )
    analyzer.generate_full_report()

    if output_file:
        if format == "html":
            analyzer.to_html(output_file)
        elif format == "json":
            analyzer.to_json(output_file)
        elif format == "md":
            analyzer.to_markdown(output_file)

    return analyzer


def analyze_folder(
    folder_path: str,
    output_file: str = None,
    format: str = "html",
    quiet: bool = False,
    use_bert: bool = True
) -> TextAnalyzer:
    """便捷函数：分析文件夹中的所有文本文件"""
    analyzer = TextAnalyzer(folder_path, quiet=quiet, use_bert=use_bert)
    analyzer.generate_full_report()

    if output_file:
        if format == "html":
            analyzer.to_html(output_file)
        elif format == "json":
            analyzer.to_json(output_file)
        elif format == "md":
            analyzer.to_markdown(output_file)

    return analyzer