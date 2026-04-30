"""
文本分析报告生成器 - 生成 HTML/JSON/Markdown 报告
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from jinja2 import Template


class TextReporter:
    """文本分析报告生成器"""

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def to_html(self, output_file: Optional[str] = None, title: str = "文本分析报告") -> str:
        try:
            html = self._build_html(title)
            if output_file:
                os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(html)
                print(f"✅ HTML报告已保存到 {output_file}")
            return html
        except Exception as e:
            error_html = self._get_error_html(title, f"报告生成失败: {str(e)}")
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(error_html)
            return error_html

    def to_json(self, output_file: Optional[str] = None, indent: int = 2) -> str:
        report_data = self._build_json_data()
        report_data["analysis_time"] = datetime.now().isoformat()
        report_data["source"] = getattr(self.analyzer, "source_name", "未知")

        json_str = json.dumps(report_data, ensure_ascii=False, indent=indent, default=str)

        if output_file:
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_str)
            print(f"✅ JSON报告已保存到 {output_file}")

        return json_str

    def to_markdown(self, output_file: Optional[str] = None) -> str:
        data = self._build_json_data()

        md = f"""# 文本分析报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 数据概览

| 指标 | 数值 |
|------|------|
| 总文本数 | {data['stats']['total_count']} |
| 空文本数 | {data['stats']['empty_count']} |
| 空文本率 | {data['stats']['empty_rate']:.1%} |
| 平均长度 | {data['stats']['char_length']['mean']:.1f} 字符 |
| 重复文本对 | {data['quality'].get('duplicates', {}).get('count', 0)} |

## 语言分布

"""
        lang_dist = data.get('language_distribution', {})
        if lang_dist:
            for lang, count in lang_dist.items():
                lang_name = "中文" if lang == 'zh' else "英文" if lang == 'en' else lang.upper()
                md += f"- {lang_name}: {count} 条\n"
        else:
            md += "- 未检测到语言信息\n"

        pos = data['sentiment']['distribution']['positive_rate']
        neg = data['sentiment']['distribution']['negative_rate']
        if pos > 0.4 or neg > 0.2:
            md += f"""
## 情感分布

- 积极: {pos:.1%}
- 消极: {neg:.1%}
- 中性: {data['sentiment']['distribution']['neutral_rate']:.1%}

"""
        else:
            md += "\n## 情感分布\n整体情感中性，无明显倾向。\n"

        md += """
## 高频关键词 TOP 30

"""
        for word, count in data['keywords']['frequency'][:30]:
            md += f"- {word}: {count}\n"

        # 实体识别
        entity_stats = data.get('entity_stats_with_details', {})
        if entity_stats.get('org', {}).get('items'):
            md += "\n## 组织名实体\n"
            for item in entity_stats['org']['items'][:15]:
                md += f"- {item['name']}: {item['count']}次\n"

        if entity_stats.get('per', {}).get('items'):
            md += "\n## 人名实体\n"
            for item in entity_stats['per']['items'][:15]:
                md += f"- {item['name']}: {item['count']}次\n"

        # 事件统计
        events_by_type = data.get('events_by_type', {})
        if events_by_type:
            md += "\n## 事件统计\n"
            for et, events in events_by_type.items():
                md += f"- {et}: {len(events)}次\n"

        # 主题建模
        topics_enhanced = data.get('topics_enhanced', [])
        if topics_enhanced:
            md += "\n## 主题建模\n"
            for topic in topics_enhanced:
                md += f"\n### 主题 {topic.get('topic_id', 0)}: {topic.get('llm_title', '')}\n"
                md += f"{topic.get('llm_summary', '')}\n\n"
                md += f"**关键词**: {', '.join(topic.get('keywords', [])[:8])}\n\n"

        # 清洗建议
        if data.get('cleaning_suggestions'):
            md += "\n## 数据清洗建议\n"
            for s in data['cleaning_suggestions'][:5]:
                md += f"- {s}\n"

        if output_file:
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(md)
            print(f"✅ Markdown报告已保存到 {output_file}")

        return md

    def _build_event_description(self, event_type: str, args: Dict, fallback_desc: str) -> str:
        """构建更好的事件描述"""
        company = args.get("公司", "")
        person = args.get("人物", "")
        money = args.get("金额", "")
        percent = args.get("比例", "")

        if event_type == "发布财报":
            if company and money:
                return f"{company}发布财报，{money}"
            elif company and percent:
                return f"{company}发布财报，同比增长{percent}"
            elif company:
                return f"{company}发布财报"
            else:
                return fallback_desc[:50] if fallback_desc else "发布财报"
        elif event_type == "收购":
            if company and money:
                return f"{company}收购，交易金额{money}"
            elif company:
                return f"{company}收购"
            else:
                return fallback_desc[:50] if fallback_desc else "收购"
        elif event_type == "投资":
            if company and money:
                return f"{company}投资{money}"
            elif company:
                return f"{company}投资"
            else:
                return fallback_desc[:50] if fallback_desc else "投资"
        elif event_type == "高管任命":
            if company and person:
                return f"{company}任命{person}"
            elif company:
                return f"{company}高管任命"
            else:
                return fallback_desc[:50] if fallback_desc else "高管任命"
        elif event_type == "高管离职":
            if company and person:
                return f"{company}{person}离职"
            elif company:
                return f"{company}高管离职"
            else:
                return fallback_desc[:50] if fallback_desc else "高管离职"
        elif event_type in ["涨价", "降价"]:
            if company and percent:
                return f"{company}{event_type}{percent}"
            elif company:
                return f"{company}{event_type}"
            else:
                return fallback_desc[:50] if fallback_desc else event_type
        else:
            return fallback_desc[:60] if fallback_desc else event_type

    def _build_json_data(self) -> Dict[str, Any]:
        """构建完整的 JSON 数据"""

        # 处理关键词格式
        keywords_frequency = []
        for item in self.analyzer.keywords.get("frequency", [])[:50]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                keywords_frequency.append([item[0], item[1]])

        # ========== 1. 获取实体统计 ==========
        entity_stats_raw = getattr(self.analyzer, 'entity_stats', {})

        entity_stats = {
            "org": {"unique": 0, "top": []},
            "per": {"unique": 0, "top": []},
            "loc": {"unique": 0, "top": []}
        }

        if entity_stats_raw:
            for key in ["org", "per", "loc"]:
                if key in entity_stats_raw:
                    stats = entity_stats_raw[key]
                    entity_stats[key] = {
                        "unique": stats.get("unique", 0),
                        "top": stats.get("top", [])
                    }

        # ========== 2. 获取事件 ==========
        events = getattr(self.analyzer, 'events', [])[:200]

        # 获取主题标签
        topic_labels = []
        if hasattr(self.analyzer, 'topic_modeler') and self.analyzer.topic_modeler:
            topic_labels = self.analyzer.topic_modeler.get_topic_labels()

        # 获取主题信息
        topics = getattr(self.analyzer, 'topics', [])[:10]

        # 构建主题名称映射
        topic_name_map = {}
        for topic in topics:
            tid = topic.get("topic_id", 0)
            name = topic.get("llm_title", f"主题{tid}")
            topic_name_map[tid] = name

        # 为每个事件添加所属主题
        for idx, event in enumerate(events):
            if idx < len(topic_labels):
                topic_id = topic_labels[idx]
                event["topic_id"] = topic_id
                event["topic_name"] = topic_name_map.get(topic_id, f"主题{topic_id}")
            else:
                event["topic_id"] = -1
                event["topic_name"] = ""

        # 按事件类型分组
        events_by_type = {}
        for event in events:
            et = event.get("event_type", "未知")
            if et not in events_by_type:
                events_by_type[et] = []
            events_by_type[et].append(event)

        # ========== 3. 获取主题增强数据 ==========
        content_texts = getattr(self.analyzer, 'content_texts', [])
        topics_enhanced = []
        for topic in topics:
            tid = topic.get("topic_id", 0)
            sample_indices = []
            for idx, label in enumerate(topic_labels):
                if label == tid and idx < len(content_texts):
                    sample_indices.append(idx)

            sample_texts = []
            for idx in sample_indices[:3]:
                text = content_texts[idx] if idx < len(content_texts) else ""
                if text:
                    sample = text[:200] + "..." if len(text) > 200 else text
                    sample_texts.append({"index": idx, "text": sample})

            topics_enhanced.append({
                "topic_id": tid,
                "texts_count": topic.get("texts_count", 0),
                "keywords": topic.get("keywords", [])[:8],
                "llm_title": topic.get("llm_title", f"主题{tid}"),
                "llm_summary": topic.get("llm_summary", ""),
                "sample_texts": sample_texts,
                "timeline_events": []
            })

        # ========== 4. 构建带详情的实体数据 ==========
        relation_result = getattr(self.analyzer, 'relation_result', {})
        pairs = relation_result.get('cooccurrence_pairs', [])

        # 构建实体详情映射
        entity_details_map = {}

        for entity_type in ["org", "per"]:
            if entity_type in entity_stats_raw:
                for name, count in entity_stats_raw[entity_type].get("top", [])[:15]:
                    if len(name) < 2:
                        continue

                    # 查找相关事件
                    events_list = []
                    for event in events[:80]:
                        event_desc = event.get("description", "")
                        event_type = event.get("event_type", "")
                        args = event.get("args", {})

                        is_related = (name in event_desc or name in event_type)
                        if not is_related:
                            for arg_value in args.values():
                                if arg_value and name in str(arg_value):
                                    is_related = True
                                    break

                        if is_related:
                            desc = self._build_event_description(event_type, args, event_desc)
                            events_list.append({
                                "type": event_type,
                                "description": desc,
                                "topic_name": event.get("topic_name", "")
                            })

                    # 查找关联实体
                    related = []
                    for pair in pairs[:50]:
                        e1 = pair.get("entity1", "").split(":", 1)[-1]
                        e2 = pair.get("entity2", "").split(":", 1)[-1]
                        if name == e1 and e2 not in related:
                            related.append(e2)
                        elif name == e2 and e1 not in related:
                            related.append(e1)

                    # 查找所属主题
                    topic_names = []
                    for topic in topics[:10]:
                        for kw in topic.get("keywords", [])[:5]:
                            if kw and len(kw) >= 2 and (kw in name or name in kw):
                                tname = topic.get("llm_title", f"主题{topic.get('topic_id', 0)}")
                                if tname not in topic_names:
                                    topic_names.append(tname)
                                break

                    has_details = len(events_list) > 0 or len(related) > 0 or len(topic_names) > 0

                    entity_details_map[(name, entity_type)] = {
                        "name": name,
                        "type": "组织" if entity_type == "org" else "人物",
                        "count": count,
                        "has_details": has_details,
                        "events": events_list[:5],
                        "related_entities": related[:5],
                        "topics": topic_names[:3]
                    }

        # 构建带详情的实体统计
        entity_stats_with_details = {
            "org": {"unique": entity_stats.get("org", {}).get("unique", 0), "items": []},
            "per": {"unique": entity_stats.get("per", {}).get("unique", 0), "items": []},
            "loc": {"unique": entity_stats.get("loc", {}).get("unique", 0), "items": []}
        }

        for entity_type in ["org", "per"]:
            if entity_type in entity_stats_raw:
                for name, count in entity_stats_raw[entity_type].get("top", [])[:15]:
                    if len(name) < 2:
                        continue
                    key = (name, entity_type)
                    if key in entity_details_map:
                        entity_stats_with_details[entity_type]["items"].append(entity_details_map[key])
                    else:
                        entity_stats_with_details[entity_type]["items"].append({
                            "name": name,
                            "type": "组织" if entity_type == "org" else "人物",
                            "count": count,
                            "has_details": False,
                            "events": [],
                            "related_entities": [],
                            "topics": []
                        })

        # 添加地名
        if "loc" in entity_stats_raw:
            for name, count in entity_stats_raw["loc"].get("top", [])[:15]:
                if len(name) >= 2:
                    entity_stats_with_details["loc"]["items"].append({
                        "name": name,
                        "type": "地名",
                        "count": count,
                        "has_details": False,
                        "events": [],
                        "related_entities": [],
                        "topics": []
                    })

        # ========== 5. 获取图谱数据 ==========
        graph_viz = {}
        if hasattr(self.analyzer, 'graph') and self.analyzer.graph:
            graph_viz = self.analyzer.graph.export_for_visualization()

        # ========== 6. 获取质量详情 ==========
        quality_data = self.analyzer.quality_report
        cleaning_suggestions = self.analyzer.cleaning_suggestions
        duplicates_detail = quality_data.get('duplicates', {})
        similarity_detail = quality_data.get('similarity', {})
        length_detail = quality_data.get('length_anomalies', {})

        # ========== 7. 构建返回数据 ==========
        data = {
            'stats': self.analyzer.stats_result,
            'language_distribution': getattr(self.analyzer, 'language_distribution', {}),
            'quality': quality_data,
            'cleaning_suggestions': cleaning_suggestions[:10],
            'keywords': {'frequency': keywords_frequency, 'tfidf': []},
            'sentiment': {
                'distribution': self.analyzer.sentiment_distribution,
                'is_polarized': self.analyzer.sentiment_distribution.get('positive_rate', 0) > 0.4 or
                                self.analyzer.sentiment_distribution.get('negative_rate', 0) > 0.2
            },
            'templates': {
                'start': list(self.analyzer.start_templates)[:20],
                'end': list(self.analyzer.end_templates)[:20]
            },
            'insights': getattr(self.analyzer, 'insights', [])[:15],
            'sample_texts': self.analyzer.content_texts[:10],
            'entity_stats_with_details': entity_stats_with_details,
            'events_by_type': events_by_type,
            'topics_enhanced': topics_enhanced,
            'graph_viz': graph_viz,
            'quality_details': {
                'duplicates': {
                    'count': duplicates_detail.get('count', 0),
                    'rate': duplicates_detail.get('rate', 0),
                    'duplicate_summary': duplicates_detail.get('duplicate_summary', [])
                },
                'similarity': {
                    'count': len(similarity_detail.get('high_similarity_pairs', [])),
                    'similarity_summary': similarity_detail.get('similarity_summary', [])
                },
                'length': {
                    'short_count': len(length_detail.get('short', [])),
                    'long_count': len(length_detail.get('long', [])),
                    'short_summary': length_detail.get('short_summary', []),
                    'long_summary': length_detail.get('long_summary', [])
                }
            }
        }

        return data

    def _get_error_html(self, title: str, error_msg: str) -> str:
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #d32f2f; border-bottom: 3px solid #d32f2f; }}
        .error {{ background: #ffebee; padding: 15px; border-radius: 8px; margin: 20px 0; }}
    </style>
</head>
<body>
<div class="container">
    <h1>❌ 报告生成失败</h1>
    <div class="error">
        <p><strong>错误信息：</strong></p>
        <p>{error_msg}</p>
    </div>
</div>
</body>
</html>"""

    def _build_html(self, title: str) -> str:
        data = self._build_json_data()

        template_path = os.path.join(os.path.dirname(__file__), '..', 'templates', 'report_text.html')
        if not os.path.exists(template_path):
            template_path = os.path.join(os.path.dirname(__file__), 'templates', 'report_text.html')
        if not os.path.exists(template_path):
            return self._get_error_html(title, f"模板文件不存在: {template_path}")

        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            template = Template(template_content)

            stats = data['stats']
            sentiment = data['sentiment']
            keywords = data['keywords']['frequency']
            quality = data['quality']
            cleaning_suggestions = data['cleaning_suggestions']
            language_dist = data.get('language_distribution', {})
            templates = data.get('templates', {})
            insights = data.get('insights', [])
            sample_texts = data.get('sample_texts', [])
            quality_details = data.get('quality_details', {})
            graph_viz = data.get('graph_viz', {})

            entity_stats_with_details = data.get('entity_stats_with_details', {})
            events_by_type = data.get('events_by_type', {})
            topics_enhanced = data.get('topics_enhanced', [])

            return template.render(
                title=title,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                total_count=stats['total_count'],
                empty_count=stats['empty_count'],
                empty_rate=f"{stats['empty_rate']:.2%}",
                avg_length=f"{stats['char_length']['mean']:.1f}",
                positive_rate=f"{sentiment['distribution']['positive_rate']:.1%}",
                negative_rate=f"{sentiment['distribution']['negative_rate']:.1%}",
                neutral_rate=f"{sentiment['distribution']['neutral_rate']:.1%}",
                sentiment_is_polarized=sentiment.get('is_polarized', False),
                keywords=keywords,
                duplicate_count=quality.get('duplicates', {}).get('count', 0),
                cleaning_suggestions=cleaning_suggestions[:10],
                language_distribution=language_dist,
                start_templates=list(templates.get('start', []))[:20],
                end_templates=list(templates.get('end', []))[:20],
                insights=insights,
                sample_texts=sample_texts,
                quality_details=quality_details,
                graph_viz=graph_viz,
                entity_stats_with_details=entity_stats_with_details,
                events_by_type=events_by_type,
                topics_enhanced=topics_enhanced
            )
        except Exception as e:
            import traceback
            print(f"模板渲染失败: {traceback.format_exc()}")
            return self._get_error_html(title, f"模板渲染失败: {str(e)}")