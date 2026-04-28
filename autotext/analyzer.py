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

# 新增导入（带异常处理）
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
    from autotext.core.relation import RelationDiscoverer
except ImportError:
    RelationDiscoverer = None
    print("⚠️ RelationDiscoverer 导入失败")

try:
    from autotext.core.cluster import TextClusterer
except ImportError:
    TextClusterer = None
    print("⚠️ TextClusterer 导入失败")

try:
    from autotext.core.topic import TopicModeler
except ImportError:
    TopicModeler = None
    print("⚠️ TopicModeler 导入失败")

try:
    from autotext.core.event_timeline import EventTimelineAnalyzer
except ImportError:
    EventTimelineAnalyzer = None
    print("⚠️ EventTimelineAnalyzer 导入失败")

try:
    from autotext.core.insight import InsightDiscoverer
except ImportError:
    InsightDiscoverer = None
    print("⚠️ InsightDiscoverer 导入失败")


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

        参数:
        - data: DataFrame、文件路径、文件夹路径或文本列表
        - text_col: 文本列名（DataFrame模式或CSV模式）
        - title_col: 标题列名（可选）
        - time_col: 时间列名（可选）
        - metric_cols: 指标列名（可选）
        - source_name: 数据源名称
        - quiet: 静默模式
        - use_bert: 是否使用BERT增强分析
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
        self.relation_result = {}
        self.cluster_info = []
        self.topics = []
        self.event_timeline = {}
        self.insights = []

        # 模块实例
        self.topic_modeler = None
        self.clusterer = None
        self.vectorizer = None
        self.ner_model = None
        self.relation_discoverer = None

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

    def _load_data(self, data, text_col, title_col, time_col, metric_cols):
        """加载数据 - 支持 CSV 文件路径 + text_col"""

        # 情况1: 传入的是 DataFrame
        if isinstance(data, pd.DataFrame):
            loaded = TextLoader.from_dataframe(data, text_col, title_col, time_col, metric_cols)
            self.data = data
            self.texts = loaded["texts"]
            self.titles = loaded.get("titles", [])
            self.dates = loaded.get("dates", [])
            self.metrics = loaded.get("metrics", {})
            if not self.quiet:
                print(f"  ✅ 从 DataFrame 加载: {len(self.texts)} 条文本")

        # 情况2: 传入的是文件路径
        elif isinstance(data, str):
            if os.path.isdir(data):
                # 文件夹模式
                result = TextLoader.from_folder(data)
                self.texts = []
                for file_name, file_texts in result.items():
                    self.texts.extend(file_texts)
                self.source_name = self.source_name or data
                self.data = None
                if not self.quiet:
                    print(f"  ✅ 从文件夹加载: {len(self.texts)} 条文本")
            else:
                # 单文件模式
                file_ext = os.path.splitext(data)[1].lower()

                # 如果是 CSV/Excel 且指定了 text_col，按结构化数据加载
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
                            # 清理列名
                            df.columns = [str(col).strip().replace('\n', '_').replace('\r', '_') for col in df.columns]

                            # 检查 text_col 是否存在
                            if text_col not in df.columns:
                                raise ValueError(f"文本列 '{text_col}' 不存在于文件中。可用列: {list(df.columns)}")

                            # 从 DataFrame 加载
                            loaded = TextLoader.from_dataframe(df, text_col, title_col, time_col, metric_cols)
                            self.data = df
                            self.texts = loaded["texts"]
                            self.titles = loaded.get("titles", [])
                            self.dates = loaded.get("dates", [])
                            self.metrics = loaded.get("metrics", {})
                            if not self.quiet:
                                print(f"  ✅ 从 CSV/Excel 加载: {len(self.texts)} 条文本")
                                print(f"     文本列: {text_col}")
                                if title_col:
                                    print(f"     标题列: {title_col}")
                                if time_col:
                                    print(f"     时间列: {time_col}")
                            return
                    except Exception as e:
                        if not self.quiet:
                            print(f"  ⚠️ 结构化加载失败: {e}，尝试作为纯文本加载")

                # 否则按纯文本文件加载
                self.texts = TextLoader.from_file(data)
                self.source_name = self.source_name or data
                self.data = None
                if not self.quiet:
                    print(f"  ✅ 从文件加载: {len(self.texts)} 条文本")

        # 情况3: 传入的是文本列表
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

        # 第一步：基础清洗
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
            if not self.quiet and (start_tmpl or end_tmpl):
                print(f"  🧹 检测到开头模板词: {len(start_tmpl)} 个, 结尾模板词: {len(end_tmpl)} 个")
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
        self.tokens_filtered_list = []
        self.language_distribution = {}

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
            self.tokens_filtered_list.append(result.get("tokens_filtered", result["tokens_cleaned"]))

            # 统计语言
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
        """情感分析"""
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
                     '少', '新', '旧', '好', '坏', '正', '负', '涨', '跌', '不',
                     '没', '无', '非', '莫', '勿', '别', '未', '过', '很', '太',
                     '同比', '环比', '增长', '下降', '上升', '回落', '稳定'}

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
                if re.match(r'^[A-Z]{2,}[0-9]{4,}[A-Z]?$', name):
                    continue
                filtered_top.append((name, count))

            filtered[entity_type] = {
                'total': stats.get('total', 0),
                'unique': len(filtered_top),
                'top': filtered_top[:20]
            }

        return filtered

    def _bert_analysis(self):
        """BERT增强分析"""
        if not self.use_bert or len(self.texts) < 5:
            if not self.quiet:
                print("\n【BERT增强】跳过（文本数量不足或未启用）")
            return

        if not self.quiet:
            print("\n【BERT增强】深度语义分析...")
            print("  ⚠️ 首次运行会下载BERT模型（约400MB），请耐心等待...")

        # 准备有效文本
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
                for entity_type in ['per', 'org', 'loc']:
                    count = self.entity_stats.get(entity_type, {}).get('unique', 0)
                    if count > 0:
                        top_entities = self.entity_stats.get(entity_type, {}).get('top', [])[:3]
                        top_names = [name for name, _ in top_entities]
                        print(f"     - {entity_type}: {count} 个, 示例: {', '.join(top_names)}")
        except Exception as e:
            if not self.quiet:
                print(f"  ❌ 实体识别失败: {e}")
            self.entity_results = []
            self.entity_stats = {}

        # ==================== 3. 关系发现 ====================
        if not self.quiet:
            print("  🔗 关系发现...")

        try:
            if self.entity_results and RelationDiscoverer is not None:
                self.relation_discoverer = RelationDiscoverer()
                self.relation_result = self.relation_discoverer.discover(self.entity_results, valid_texts)

                if not self.quiet:
                    pairs = len(self.relation_result.get('cooccurrence_pairs', []))
                    print(f"  ✅ 发现 {pairs} 个强关联实体对")
                    for pair in self.relation_result.get('cooccurrence_pairs', [])[:3]:
                        e1 = pair['entity1'].split(':', 1)[-1] if ':' in pair['entity1'] else pair['entity1']
                        e2 = pair['entity2'].split(':', 1)[-1] if ':' in pair['entity2'] else pair['entity2']
                        print(f"     - {e1} ↔ {e2} (PMI={pair['pmi']:.2f})")
            else:
                self.relation_result = {}
                if not self.quiet:
                    print("  ⚠️ 无实体数据，跳过关系发现")
        except Exception as e:
            if not self.quiet:
                print(f"  ❌ 关系发现失败: {e}")
            self.relation_result = {}

        # ==================== 4. 文本聚类 ====================
        if not self.quiet:
            print("  🔘 文本聚类...")

        try:
            if TextClusterer is None:
                raise ImportError("TextClusterer 未导入")
            self.clusterer = TextClusterer()
            self.clusterer.fit(self.embeddings)
            self.cluster_info = self.clusterer.get_cluster_info(valid_texts, self.embeddings)

            if not self.quiet:
                n_clusters = len(self.cluster_info)
                if n_clusters > 0:
                    print(f"  ✅ 聚类完成，共 {n_clusters} 个簇")
                    for cluster in self.cluster_info[:3]:
                        keywords = ', '.join(cluster['top_words'][:5])
                        print(f"     - 簇{cluster['cluster_id']}: {cluster['size']} 条, 关键词: {keywords}")
                else:
                    print(f"  ⚠️ 未发现有效聚类")
        except Exception as e:
            if not self.quiet:
                print(f"  ❌ 聚类失败: {e}")
            self.cluster_info = []
            self.clusterer = None

        # ==================== 5. 主题建模 ====================
        if not self.quiet:
            print("  📚 主题建模...")

        try:
            if TopicModeler is None:
                raise ImportError("TopicModeler 未导入")
            self.topic_modeler = TopicModeler()
            cluster_labels = getattr(self.clusterer, 'labels', None) if self.clusterer else None
            self.topic_modeler.fit(valid_texts, cluster_labels)
            self.topics = self.topic_modeler.get_topics()

            if not self.quiet:
                print(f"  ✅ 主题建模完成，共 {len(self.topics)} 个主题")
                for topic in self.topics[:3]:
                    keywords = ', '.join(topic.get('keywords', [])[:5])
                    print(f"     - 主题{topic['topic_id']}: {keywords}")
        except Exception as e:
            if not self.quiet:
                print(f"  ❌ 主题建模失败: {e}")
            self.topics = []
            self.topic_modeler = None

        # ==================== 6. 事件脉络分析 ====================
        if not self.quiet:
            print("  📅 事件脉络分析...")

        try:
            valid_dates = [d for d in self.dates if d is not None]
            if len(valid_dates) >= 5 and self.entity_results and EventTimelineAnalyzer is not None:
                timeline = EventTimelineAnalyzer()
                self.event_timeline = timeline.analyze(
                    valid_texts, self.dates, self.entity_results, self.sentiment_results
                )

                if not self.quiet and 'error' not in self.event_timeline:
                    time_range = self.event_timeline.get('time_range', {})
                    print(f"  ✅ 事件脉络分析完成")
                    if time_range.get('start') and time_range.get('end'):
                        start_str = str(time_range['start'])[:10] if time_range['start'] else 'N/A'
                        end_str = str(time_range['end'])[:10] if time_range['end'] else 'N/A'
                        print(f"     - 时间范围: {start_str} ~ {end_str}")

                    hot_topics = self.event_timeline.get('hot_topics', [])
                    if hot_topics:
                        print(f"     - 热点话题: {len(hot_topics)} 个")
                elif not self.quiet:
                    error_msg = self.event_timeline.get('error', '时间信息不足')
                    print(f"  ⚠️ 事件脉络分析跳过: {error_msg}")
            else:
                self.event_timeline = {}
                if not self.quiet:
                    if len(valid_dates) < 5:
                        print(f"  ⚠️ 事件脉络跳过（有效时间点不足: {len(valid_dates)} < 5）")
        except Exception as e:
            if not self.quiet:
                print(f"  ❌ 事件脉络分析失败: {e}")
            self.event_timeline = {}

        # ==================== 7. 洞察发现 ====================
        if not self.quiet:
            print("  💡 洞察发现...")

        try:
            if InsightDiscoverer is None:
                raise ImportError("InsightDiscoverer 未导入")
            insight_discoverer = InsightDiscoverer()
            self.insights = insight_discoverer.discover_all(self)

            if not self.quiet:
                print(f"  ✅ 发现 {len(self.insights)} 个洞察")
                for insight_item in self.insights[:5]:
                    priority_icon = {'高': '🔴', '中': '🟠', '低': '🟢'}.get(insight_item.get('priority', '中'), '⚪')
                    print(f"     {priority_icon} {insight_item['title']}")
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
        """保存正文文本（切除模板后）"""
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
        """保存过滤后文本（去除噪音）"""
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