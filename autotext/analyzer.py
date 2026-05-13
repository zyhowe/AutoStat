# ==================== autotext/analyzer.py ====================
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
from autotext.core.keyword_extractor import KeywordExtractor
from autotext.core.sentiment import SentimentAnalyzer
from autotext.reporter import TextReporter
from autotext.core.summarizer import TextRankSummarizer, LLMSummarizer

# 导入大模型客户端
try:
    from autostat.llm_client import LLMClient
except ImportError:
    try:
        from autotext.llm_client import LLMClient
    except ImportError:
        LLMClient = None
        print("⚠️ LLMClient 导入失败")

# 新增导入
try:
    from autotext.core.vectorizer import BertVectorizer
except ImportError:
    BertVectorizer = None
    print("⚠️ BertVectorizer 导入失败")

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
    from autotext.core.topic_model import TopicModeler
except ImportError:
    TopicModeler = None
    print("⚠️ TopicModeler 导入失败")

try:
    from autotext.core.insight import InsightDiscoverer
except ImportError:
    InsightDiscoverer = None
    print("⚠️ InsightDiscoverer 导入失败")

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
                 use_bert: bool = True,
                 llm_config: Optional[Dict[str, str]] = None):
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
        self.topics = []
        self.event_timeline = {}
        self.insights = []
        self.events = []
        self.graph = None
        self.graph_insights = {}
        self.entity_profiles = []
        self.timeline = None

        # 模块实例
        self.topic_modeler = None
        self.vectorizer = None
        self.ner_model = None
        self.event_extractor = None
        self.graph_builder = None
        self.graph_analyzer = None
        self.entity_profile_builder = None
        self.textrank_summarizer = None
        self.llm_summarizer = None
        self.llm_client = None

        # 模板词
        self.start_templates = set()
        self.end_templates = set()
        self.language_distribution = {}

        # 加载数据
        self._load_data(data, text_col, title_col, time_col, metric_cols)
        self.checker = TextChecker(self.texts, self.titles, self.dates)

        # 初始化大模型客户端（使用默认配置或传入配置）
        if llm_config is None:
            llm_config = {
                "api_base": "https://api.deepseek.com/v1",
                "api_key": "sk-c0e1f1ad1a3b41429a92f29251775ecf",
                "model": "deepseek-chat"
            }

        if LLMClient is not None:
            try:
                self.llm_client = LLMClient(llm_config)
                if not self.quiet:
                    print("  ✅ 大模型客户端已初始化（使用默认配置）")
            except Exception as e:
                if not self.quiet:
                    print(f"  ⚠️ 大模型客户端初始化失败: {e}")
                self.llm_client = None
        else:
            self.llm_client = None
            if not self.quiet:
                print("  ⚠️ LLMClient 不可用，大模型功能将禁用")

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
        if isinstance(data, pd.DataFrame):
            loaded = TextLoader.from_dataframe(data, text_col, title_col, time_col, metric_cols)
            self.data = data
            self.texts = loaded["texts"]
            self.titles = loaded.get("titles", [])
            self.dates = loaded.get("dates", [])
            self.metrics = loaded.get("metrics", {})
            if not self.quiet:
                print(f"  ✅ 从 DataFrame 加载: {len(self.texts)} 条文本")
        elif isinstance(data, str):
            if os.path.isdir(data):
                result = TextLoader.from_folder(data)
                self.texts = []
                for file_name, file_texts in result.items():
                    self.texts.extend(file_texts)
                self.source_name = self.source_name or data
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
                            df.columns = [str(col).strip().replace('\n', '_') for col in df.columns]
                            if text_col not in df.columns:
                                raise ValueError(f"文本列 '{text_col}' 不存在")
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
                            print(f"  ⚠️ 结构化加载失败: {e}")
                self.texts = TextLoader.from_file(data)
                self.source_name = self.source_name or data
                if not self.quiet:
                    print(f"  ✅ 从文件加载: {len(self.texts)} 条文本")
        elif isinstance(data, list):
            self.texts = data
            if not self.quiet:
                print(f"  ✅ 从列表加载: {len(self.texts)} 条文本")
        else:
            raise ValueError("data 必须是 DataFrame、文件路径、文件夹路径或文本列表")

        if not self.texts:
            raise ValueError("没有加载到有效的文本数据")
        self.raw_texts = self.texts.copy()

    def _preprocess(self):
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
                effective_texts, min_abs=10, min_ratio=0.05,
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
        elif not self.quiet:
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
                print(
                    f"  📊 语言分布: 中文 {self.language_distribution.get('zh', 0)} 条, 英文 {self.language_distribution.get('en', 0)} 条")

    def _compute_stats(self):
        if not self.quiet:
            print("\n【阶段2】基础统计...")
        stats = TextStats(self.content_texts)
        self.stats_result = stats.compute_stats()
        if not self.quiet:
            print(f"  ✅ 总文本数: {self.stats_result['total_count']}")
            print(f"  ✅ 平均长度: {self.stats_result['char_length']['mean']:.1f} 字符")

    def _check_quality(self):
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
        if not self.quiet:
            print("\n【阶段5】情感分析...")
        analyzer = SentimentAnalyzer()
        self.sentiment_results = analyzer.analyze_batch(self.content_texts)
        self.sentiment_distribution = analyzer.get_distribution(self.sentiment_results)
        if not self.quiet:
            print(
                f"  ✅ 积极: {self.sentiment_distribution['positive_rate']:.1%}  消极: {self.sentiment_distribution['negative_rate']:.1%}  中性: {self.sentiment_distribution['neutral_rate']:.1%}")

    def _filter_valid_entities(self, entity_stats: Dict) -> Dict:
        stopwords = {'的', '了', '是', '在', '和', '与', '或', '也', '都', '还',
                     '这', '那', '有', '为', '对', '而', '并', '且', '但', '就',
                     '年', '月', '日', '时', '分', '秒', '上', '下', '中', '内',
                     '外', '前', '后', '左', '右', '高', '低', '大', '小', '多',
                     '少', '新', '旧', '好', '坏', '正', '负', '涨', '跌', '不',
                     '公司', '企业', '产品', '项目', '市场', '行业', '机构', '业务',
                     '金额', '比例', '时间', 'percent', 'money', 'date'}

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
                # 过滤掉纯数字+字母的代码串
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

    def _llm_graph_analysis(self):
        """基于大模型的信息抽取（使用 llm_extractor）"""
        if not self.llm_client:
            if not self.quiet:
                print("  ⚠️ 未配置大模型客户端")
            return

        if not self.quiet:
            print("  🤖 使用大模型进行结构化信息抽取...")

        # 获取要分析的文本（不限制数量）
        if not self.content_texts:
            if not self.quiet:
                print("  ⚠️ 没有有效的文本内容")
            return

        # 合并所有文本
        combined_text = "\n\n".join(self.content_texts)

        try:
            from autotext.llm_extractor import InfoExtractorClient

            # 从 llm_client 提取配置
            config = {
                "api_base": getattr(self.llm_client, 'api_base', ''),
                "api_key": getattr(self.llm_client, 'api_key', ''),
                "model": getattr(self.llm_client, 'model', 'deepseek-chat')
            }

            if not config["api_base"] or not config["api_key"]:
                if not self.quiet:
                    print("  ❌ 大模型配置不完整，缺少 api_base 或 api_key")
                return

            extractor = InfoExtractorClient(
                api_base=config["api_base"],
                api_key=config["api_key"],
                model=config["model"]
            )

            import time

            # 在调用 extractor.extract 之前
            if not self.quiet:
                print("  🔄 正在调用大模型抽取信息...")
                print("  ⏳ 为避免限流，等待2秒...")
                time.sleep(2)

            result = extractor.extract(combined_text)

            if not result or not result.get("entities"):
                if not self.quiet:
                    print("  ⚠️ 大模型未返回有效实体")
                return

            if not self.quiet:
                print(f"  ✅ 抽取到 {len(result.get('entities', []))} 个实体")
                print(f"  ✅ 抽取到 {len(result.get('relationships', []))} 条关系")
                print(f"  ✅ 抽取到 {len(result.get('events', []))} 个事件")
                print(f"  ✅ 抽取到 {len(result.get('themes', []))} 个主题")

            # 映射结果到 analyzer 字段
            self._map_extraction_result(result)

            # 构建增强数据
            self._build_enhanced_data_from_extraction(result)

            if not self.quiet:
                print("\n  🎉 大模型信息抽取完成")

        except ImportError as e:
            if not self.quiet:
                print(f"  ❌ 模块导入失败: {e}")
        except Exception as e:
            if not self.quiet:
                print(f"  ❌ 大模型调用失败: {e}")
                import traceback
                traceback.print_exc()

    def _map_extraction_result(self, result: Dict[str, Any]):
        """将大模型抽取结果映射到 analyzer 字段"""

        # 存储原始抽取结果
        self._extracted_entities = result.get("entities", [])
        self._extracted_relationships = result.get("relationships", [])
        self._extracted_events = result.get("events", [])
        self._extracted_themes = result.get("themes", [])
        self._categorization = result.get("categorization", {})

        # 区分静态关系和动态关系
        self._static_relationships = []
        self._dynamic_relationships = []

        for rel in result.get("relationships", []):
            rel_type = rel.get("relation_type", "dynamic")
            if rel_type == "static":
                self._static_relationships.append(rel)
            else:
                self._dynamic_relationships.append(rel)

        # ==================== 1. 实体统计 ====================
        self.entity_stats = {}
        self.entity_results = []

        # 按实体类型分组统计
        entity_by_type = {}

        for entity in result.get("entities", []):
            entity_name = entity.get("entity_name", "")
            entity_type_raw = entity.get("entity_type", "OTHER")
            entity_type = self._normalize_entity_type(entity_type_raw)

            if not entity_name or len(entity_name) < 2:
                continue

            if entity_type not in entity_by_type:
                entity_by_type[entity_type] = []

            entity_by_type[entity_type].append(entity_name)

        # 构建 entity_stats
        for etype, names in entity_by_type.items():
            from collections import Counter
            counter = Counter(names)
            self.entity_stats[etype] = {
                "total": len(names),
                "unique": len(counter),
                "top": counter.most_common(20)
            }

        # 构建 entity_results（兼容格式）
        entity_result = {}
        for etype, names in entity_by_type.items():
            unique_names = list(dict.fromkeys(names))
            entity_result[etype] = [(name, 0, len(name)) for name in unique_names]
        if entity_result:
            self.entity_results = [entity_result]

        # ==================== 2. 关系结果 ====================
        self.relation_result = {
            "cooccurrence_pairs": [],
            "entity_frequency": {}
        }

        # 统计实体频次
        for etype, stats in self.entity_stats.items():
            for name, count in stats.get("top", []):
                self.relation_result["entity_frequency"][name] = count

        # 转换关系
        for rel in result.get("relationships", []):
            subject = rel.get("subject_entity_id", "")
            predicate = rel.get("predicate", "关联")
            obj = rel.get("object_entity_id", "")

            subject_name = self._extract_entity_name_from_id(subject, result)
            obj_name = self._extract_entity_name_from_id(obj, result)

            if subject_name and obj_name:
                self.relation_result["cooccurrence_pairs"].append({
                    "entity1": subject_name,
                    "entity2": obj_name,
                    "predicate": predicate,
                    "count": 1,
                    "pmi": 1.0,
                    "contexts": []
                })

        # ==================== 3. 事件列表 ====================
        self.events = []
        events_by_type = {}

        for idx, event in enumerate(result.get("events", [])):
            event_type = event.get("event_type", "未知事件")
            event_data = {
                "event_id": event.get("event_id", f"ev_{idx}"),
                "event_type": event_type,
                "trigger": event.get("trigger_word", ""),
                "description": event.get("summary", ""),
                "timestamp": event.get("time", ""),
                "location": event.get("location", ""),
                "args": {},
                "text_index": 0,
                "participants": event.get("participants", [])
            }
            self.events.append(event_data)

            # 按类型分组
            if event_type not in events_by_type:
                events_by_type[event_type] = []
            events_by_type[event_type].append(event_data)

        self.events_by_type = events_by_type

        # ==================== 4. 主题列表 ====================
        self.topics = []
        for idx, theme in enumerate(result.get("themes", [])):
            self.topics.append({
                "topic_id": idx,
                "texts_count": 1,
                "keywords": theme.get("keywords", [])[:10],
                "weights": [1.0] * len(theme.get("keywords", [])[:10]),
                "llm_title": theme.get("theme_name", f"主题{idx}"),
                "llm_summary": theme.get("summary", ""),
                "textrank_sentences": [],
                "representative_texts": [self.content_texts[0][:300]] if self.content_texts else [],
                "parent_theme_id": theme.get("parent_theme_id"),
                "related_entity_ids": theme.get("related_entity_ids", []),
                "related_event_ids": theme.get("related_event_ids", [])
            })

        # ==================== 5. 统计信息 ====================
        self.llm_statistics = {
            "entity_count": len(result.get("entities", [])),
            "relation_count": len(result.get("relationships", [])),
            "static_relation_count": len(self._static_relationships),
            "dynamic_relation_count": len(self._dynamic_relationships),
            "event_count": len(result.get("events", [])),
            "theme_count": len(result.get("themes", []))
        }

    def _normalize_entity_type(self, entity_type: str) -> str:
        """标准化实体类型为小写简称"""
        type_map = {
            "Person": "per", "PER": "per", "人物": "per",
            "Organization": "org", "ORG": "org", "组织": "org",
            "Location": "loc", "LOC": "loc", "地点": "loc",
            "Product": "product", "PRODUCT": "product", "产品": "product",
            "EventName": "event", "Event": "event", "事件": "event",
            "LegalDoc": "legal", "Metric": "metric", "Industry": "industry",
            "Technology": "tech", "Process": "process",
            "TIME": "time", "Time": "time", "时间": "time",
            "NUMBER": "number", "Number": "number", "数值": "number"
        }
        return type_map.get(entity_type, "other")

    def _extract_entity_name_from_id(self, entity_id: str, result: Dict) -> str:
        """从 entity_id 提取实体名称"""
        if not entity_id:
            return ""
        for entity in result.get("entities", []):
            if entity.get("entity_id") == entity_id:
                return entity.get("entity_name", "")
        return entity_id

    def _extract_node_fact(self, entity: Dict) -> str:
        """从实体中提取关键事实"""
        attrs = entity.get('attributes', [])
        fact_parts = []
        for attr in attrs[:2]:
            key = attr.get('attr_name') or attr.get('attr_key', '')
            value = attr.get('attr_value', '')
            if key and value:
                fact_parts.append(f"{key}:{value}")
        if fact_parts:
            return '; '.join(fact_parts)
        evidence = entity.get('evidence', '')
        if evidence:
            return evidence[:50]
        return ''

    def _build_graph(self, entities: List[Dict], relationships: List[Dict]) -> Dict:
        """构建图谱（实体、事件、主题三类节点 + 多种边）"""
        nodes = []
        links = []
        node_ids = set()

        # 实体映射
        entity_map = {e.get("entity_id"): e for e in entities}
        entity_name_map = {e.get("entity_id"): e.get("entity_name", e.get("entity_id")) for e in entities}

        # 实体节点
        for entity in entities:
            entity_id = entity.get("entity_id")
            entity_name = entity.get("entity_name", "")
            if not entity_name or len(entity_name) < 2:
                continue
            entity_type = entity.get("entity_type", "OTHER")
            node = {
                "id": entity_id,
                "name": entity_name,
                "category": "entity",
                "type": entity_type,
                "value": len(entity.get("attributes", [])) + 1,
                "fact": self._extract_node_fact(entity),
                "symbolSize": min(35, 15 + (len(entity.get("attributes", [])) + 1) * 2)
            }
            nodes.append(node)
            node_ids.add(entity_id)

        # 事件节点
        events = getattr(self, '_extracted_events', [])
        for event in events:
            event_id = event.get("event_id")
            event_summary = event.get("summary", "")[:30]
            if not event_summary:
                event_summary = event.get("event_type", "事件")
            node = {
                "id": event_id,
                "name": event_summary,
                "category": "event",
                "type": event.get("event_type", "未知"),
                "value": 1,
                "fact": event.get("summary", "")[:50],
                "symbolSize": 25
            }
            nodes.append(node)
            node_ids.add(event_id)

        # 主题节点
        themes = getattr(self, '_extracted_themes', [])
        for theme in themes:
            theme_id = theme.get("theme_id")
            theme_name = theme.get("theme_name", f"主题{theme_id}")
            if not theme_name:
                continue
            node = {
                "id": theme_id,
                "name": theme_name[:20],
                "category": "theme",
                "type": "主题",
                "value": 1,
                "fact": theme.get("summary", "")[:50],
                "symbolSize": 30
            }
            nodes.append(node)
            node_ids.add(theme_id)

        # 实体-实体关系边
        for rel in relationships:
            subj = rel.get("subject_entity_id")
            obj = rel.get("object_entity_id")
            predicate = rel.get("predicate", "关联")
            rel_type = rel.get("relation_type", "dynamic")
            if subj and obj and subj in node_ids and obj in node_ids:
                links.append({
                    "source": subj,
                    "target": obj,
                    "type": "relation",
                    "label": predicate[:15],
                    "relation_type": rel_type,
                    "value": 1
                })

        # 实体-事件参与边
        for event in events:
            event_id = event.get("event_id")
            if event_id not in node_ids:
                continue
            for participant in event.get("participants", []):
                entity_id = participant.get("entity_id")
                role = participant.get("role", "参与")
                if entity_id and entity_id in node_ids:
                    links.append({
                        "source": entity_id,
                        "target": event_id,
                        "type": "participates",
                        "label": role[:10],
                        "relation_type": "dynamic",
                        "value": 1
                    })

        # 实体-主题归属边
        for theme in themes:
            theme_id = theme.get("theme_id")
            if theme_id not in node_ids:
                continue
            for entity_id in theme.get("related_entity_ids", []):
                if entity_id and entity_id in node_ids:
                    links.append({
                        "source": entity_id,
                        "target": theme_id,
                        "type": "belongs_to",
                        "label": "属于",
                        "relation_type": "static",
                        "value": 1
                    })
            for event_id in theme.get("related_event_ids", []):
                if event_id and event_id in node_ids:
                    links.append({
                        "source": event_id,
                        "target": theme_id,
                        "type": "belongs_to",
                        "label": "归类",
                        "relation_type": "static",
                        "value": 1
                    })

        # 事件-事件关系边
        for event in events:
            for rel_event in event.get("related_events", []):
                target_id = rel_event.get("target_event_id")
                rel_type = rel_event.get("relation_type", "相关")
                if target_id and target_id in node_ids:
                    links.append({
                        "source": event.get("event_id"),
                        "target": target_id,
                        "type": "event_relation",
                        "label": rel_type[:10],
                        "relation_type": "dynamic",
                        "value": 1
                    })

        # 去重 links
        unique_links = {}
        for link in links:
            key = f"{link['source']}|{link['target']}|{link['type']}"
            if key not in unique_links:
                unique_links[key] = link
        links = list(unique_links.values())

        return {
            "nodes": nodes,
            "links": links,
            "statistics": {
                "entity_count": len([n for n in nodes if n["category"] == "entity"]),
                "event_count": len([n for n in nodes if n["category"] == "event"]),
                "theme_count": len([n for n in nodes if n["category"] == "theme"]),
                "relation_count": len([l for l in links if l["type"] == "relation"]),
                "participation_count": len([l for l in links if l["type"] == "participates"]),
                "belongs_to_count": len([l for l in links if l["type"] == "belongs_to"])
            }
        }

    def _build_enhanced_data_from_extraction(self, result: Dict[str, Any]):
        """从大模型抽取结果构建增强数据"""

        # 1. 构建实体档案
        self.entity_profiles = self._build_entity_profiles(result)

        # 2. 构建事件链
        self.event_chains = self._build_event_chains(result)

        # 3. 构建主题层级
        self.theme_hierarchy = self._build_theme_hierarchy(result)

        # 4. 构建全局图谱（使用动态关系 + 静态关系）
        self.global_graph = self._build_graph(
            result.get("entities", []),
            result.get("relationships", [])  # 全部关系
        )

        # 5. 构建静态图谱（只使用静态关系，用于树形图）
        static_relationships = [r for r in result.get("relationships", [])
                                if r.get("relation_type") == "static"]
        self.static_graph = self._build_graph(
            result.get("entities", []),
            static_relationships
        )

    def _build_entity_profiles(self, result: Dict[str, Any]) -> List[Dict]:
        """构建实体档案（含关联关系），按重要性排序"""
        profiles = []

        # 创建实体映射
        entity_map = {e.get("entity_id"): e for e in result.get("entities", [])}

        # 创建关系索引
        relations_by_entity = {}
        for rel in result.get("relationships", []):
            subj = rel.get("subject_entity_id")
            obj = rel.get("object_entity_id")
            pred = rel.get("predicate", "关联")
            if subj:
                relations_by_entity.setdefault(subj, []).append({"target": obj, "predicate": pred, "type": "out"})
            if obj:
                relations_by_entity.setdefault(obj, []).append({"target": subj, "predicate": pred, "type": "in"})

        # 创建事件索引
        events_by_participant = {}
        for event in result.get("events", []):
            for participant in event.get("participants", []):
                entity_id = participant.get("entity_id")
                role = participant.get("role", "参与")
                if entity_id:
                    events_by_participant.setdefault(entity_id, []).append({
                        "event_id": event.get("event_id"),
                        "summary": event.get("summary", ""),
                        "role": role,
                        "time": event.get("time")
                    })

        # 创建主题索引
        themes_by_entity = {}
        for theme in result.get("themes", []):
            for entity_id in theme.get("related_entity_ids", []):
                themes_by_entity.setdefault(entity_id, []).append({
                    "theme_id": theme.get("theme_id"),
                    "name": theme.get("theme_name")
                })

        # 构建每个实体的档案
        for entity_id, entity in entity_map.items():
            entity_name = entity.get("entity_name", "")
            if not entity_name or len(entity_name) < 2:
                continue

            # 提取属性
            attributes = []
            for attr in entity.get("attributes", []):
                attributes.append({
                    "key": attr.get("attr_name") or attr.get("attr_key", ""),
                    "value": attr.get("attr_value", ""),
                    "time": attr.get("time")
                })

            # 关联实体
            related_entities = []
            for rel in relations_by_entity.get(entity_id, []):
                target_id = rel["target"]
                target_entity = entity_map.get(target_id, {})
                target_name = target_entity.get("entity_name", target_id)
                if target_name:
                    related_entities.append({
                        "name": target_name,
                        "relation": rel["predicate"],
                        "direction": rel["type"]
                    })

            # 参与的事件
            participated_events = []
            for evt in events_by_participant.get(entity_id, []):
                participated_events.append({
                    "event_id": evt["event_id"],
                    "summary": evt["summary"][:100] if evt["summary"] else "",
                    "role": evt["role"],
                    "time": evt["time"]
                })

            # 归属的主题
            belong_to_themes = []
            for theme in themes_by_entity.get(entity_id, []):
                belong_to_themes.append({
                    "theme_id": theme["theme_id"],
                    "name": theme["name"]
                })

            # 计算重要性分数：属性数量 + 关联实体数量 + 参与事件数量
            importance_score = len(attributes) + len(related_entities) + len(participated_events) + 1

            profiles.append({
                "id": entity_id,
                "name": entity_name,
                "type": entity.get("entity_type", "OTHER"),
                "attributes": attributes[:10],
                "related_entities": related_entities[:10],
                "participated_events": participated_events[:10],
                "belong_to_themes": belong_to_themes[:5],
                "mention_count": entity.get("mention_count", 1),
                "sample_context": entity.get("evidence", "")[:200],
                "importance_score": importance_score
            })

        # 按重要性分数排序
        profiles.sort(key=lambda x: x["importance_score"], reverse=True)
        return profiles

    def _build_event_chains(self, result: Dict[str, Any]) -> List[Dict]:
        """构建事件链（基于时间和因果）"""
        events = result.get("events", [])
        if len(events) < 2:
            return []

        # 解析时间
        events_with_time = []
        for event in events:
            time_str = event.get("time", "")
            # 尝试解析年份
            year = None
            if time_str:
                import re
                year_match = re.search(r'(\d{4})', time_str)
                if year_match:
                    year = int(year_match.group(1))
            events_with_time.append({
                "id": event.get("event_id"),
                "summary": event.get("summary", "")[:80],
                "time_raw": time_str,
                "year": year,
                "event_type": event.get("event_type", "未知")
            })

        # 按时间排序
        sorted_events = sorted(events_with_time, key=lambda x: x["year"] if x["year"] else 9999)

        # 构建事件链
        chains = []
        current_chain = []
        last_year = None

        for event in sorted_events:
            if event["year"] and last_year and event["year"] - last_year > 2:
                # 时间间隔超过2年，视为新链
                if len(current_chain) >= 2:
                    chains.append({
                        "chain_id": f"EC_{len(chains) + 1}",
                        "description": self._infer_chain_description(current_chain),
                        "events": current_chain.copy()
                    })
                current_chain = []

            current_chain.append(event)
            last_year = event["year"]

        if len(current_chain) >= 2:
            chains.append({
                "chain_id": f"EC_{len(chains) + 1}",
                "description": self._infer_chain_description(current_chain),
                "events": current_chain.copy()
            })

        # 添加关系
        for chain in chains:
            relations = []
            for i in range(len(chain["events"]) - 1):
                relations.append({
                    "from": chain["events"][i]["id"],
                    "to": chain["events"][i + 1]["id"],
                    "type": "时序",
                    "description": f"发生在 {chain['events'][i]['time_raw']} 之后"
                })
            chain["relations"] = relations

        return chains[:10]

    def _infer_chain_description(self, events: List[Dict]) -> str:
        """推断事件链的描述"""
        if not events:
            return ""
        types = list(set(e["event_type"] for e in events if e["event_type"]))
        if len(types) == 1:
            return f"{types[0]}事件序列"
        time_range = f"{events[0]['time_raw']} 至 {events[-1]['time_raw']}" if events[0]['time_raw'] and events[-1][
            'time_raw'] else ""
        return f"事件发展脉络 {time_range}".strip()

    def _build_theme_hierarchy(self, result: Dict[str, Any]) -> Dict:
        """构建主题层级树"""
        themes = result.get("themes", [])
        if not themes:
            return {"roots": [], "total_themes": 0}

        # 创建主题映射
        theme_map = {t.get("theme_id"): t for t in themes}

        # 构建树
        roots = []
        children_by_parent = {}

        for theme in themes:
            theme_id = theme.get("theme_id")
            parent_id = theme.get("parent_theme_id")
            node = {
                "id": theme_id,
                "name": theme.get("theme_name", f"主题{theme_id}"),
                "summary": theme.get("summary", "")[:100],
                "keywords": theme.get("keywords", [])[:5],
                "children": []
            }

            if parent_id is None:
                roots.append(node)
            else:
                children_by_parent.setdefault(parent_id, []).append(node)

            theme_map[theme_id] = theme_map.get(theme_id, {})
            theme_map[theme_id]["node"] = node

        # 组装树
        def attach_children(node):
            children = children_by_parent.get(node["id"], [])
            for child in children:
                attach_children(child)
            node["children"] = children
            return node

        for root in roots:
            attach_children(root)

        return {
            "roots": roots,
            "total_themes": len(themes)
        }

    def _bert_analysis_original(self):
        """原有的BERT增强分析（保留但暂不执行）"""
        pass

    def _bert_analysis(self):
        """BERT增强分析 - 完整版（支持大模型方式）"""
        if not self.use_bert:# or len(self.texts) < 5:
            if not self.quiet:
                print("\n【BERT增强】跳过（文本数量不足或未启用）")
            return

        if not self.quiet:
            print("\n【BERT增强】深度语义分析...")

        # 使用大模型方式
        if self.llm_client is not None:
            self._llm_graph_analysis()
            return

        # 不执行小模型回退
        if not self.quiet:
            print("  ⚠️ 未配置大模型客户端")

    def generate_full_report(self):
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
        reporter = TextReporter(self)
        return reporter.to_html(output_file, title)

    def to_json(self, output_file: str = None) -> str:
        reporter = TextReporter(self)
        return reporter.to_json(output_file)

    def to_markdown(self, output_file: str = None) -> str:
        reporter = TextReporter(self)
        return reporter.to_markdown(output_file)

    def save_raw_texts(self, output_path: str = None):
        if output_path is None:
            output_path = "raw_texts.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(self.raw_texts):
                f.write(f"=== 文本 {i + 1} ===\n")
                f.write(text[:500] + "..." if len(text) > 500 else text)
                f.write("\n\n")
        if not self.quiet:
            print(f"✅ 原始文本已保存到 {output_path}")

    def save_cleaned_texts(self, output_path: str = None):
        if output_path is None:
            output_path = "cleaned_texts.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(self.cleaned_texts):
                if text:
                    f.write(f"=== 文本 {i + 1} ===\n")
                    f.write(text[:500] + "..." if len(text) > 500 else text)
                    f.write("\n\n")
        if not self.quiet:
            print(f"✅ 清洗后文本已保存到 {output_path}")

    def save_content_texts(self, output_path: str = None):
        if output_path is None:
            output_path = "content_texts.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(self.content_texts):
                if text:
                    f.write(f"=== 文本 {i + 1} ===\n")
                    f.write(text[:500] + "..." if len(text) > 500 else text)
                    f.write("\n\n")
        if not self.quiet:
            print(f"✅ 正文文本已保存到 {output_path}")

    def save_filtered_texts(self, output_path: str = None):
        if output_path is None:
            output_path = "filtered_texts.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(self.filtered_texts):
                if text:
                    f.write(f"=== 文本 {i + 1} ===\n")
                    f.write(text[:500] + "..." if len(text) > 500 else text)
                    f.write("\n\n")
        if not self.quiet:
            print(f"✅ 过滤后文本已保存到 {output_path}")

    def save_templates(self, output_path: str = None):
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


# 便捷函数
def analyze_texts(texts: List[str], output_file: str = None, format: str = "html",
                  quiet: bool = False, use_bert: bool = True) -> TextAnalyzer:
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


def analyze_file(file_path: str, text_col: str = None, title_col: str = None,
                 time_col: str = None, output_file: str = None, format: str = "html",
                 quiet: bool = False, use_bert: bool = True) -> TextAnalyzer:
    analyzer = TextAnalyzer(data=file_path, text_col=text_col, title_col=title_col,
                            time_col=time_col, quiet=quiet, use_bert=use_bert)
    analyzer.generate_full_report()
    if output_file:
        if format == "html":
            analyzer.to_html(output_file)
        elif format == "json":
            analyzer.to_json(output_file)
        elif format == "md":
            analyzer.to_markdown(output_file)
    return analyzer


def analyze_folder(folder_path: str, output_file: str = None, format: str = "html",
                   quiet: bool = False, use_bert: bool = True) -> TextAnalyzer:
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