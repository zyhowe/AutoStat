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
        entity_stats = data.get('entity_stats', {})
        if entity_stats.get('org', {}).get('top'):
            md += "\n## 组织名实体\n"
            for name, count in entity_stats['org']['top'][:20]:
                md += f"- {name}: {count}次\n"

        if entity_stats.get('per', {}).get('top'):
            md += "\n## 人名实体\n"
            for name, count in entity_stats['per']['top'][:20]:
                md += f"- {name}: {count}次\n"

        # 事件统计（按类型分组）
        events_by_type = data.get('events_by_type', {})
        if events_by_type:
            md += "\n## 事件统计\n"
            for event_type, events in events_by_type.items():
                md += f"\n### {event_type} ({len(events)})\n"
                for event in events[:10]:
                    desc = event.get('description', event.get('trigger', event_type))
                    md += f"- {desc}\n"
                    if event.get('timestamp'):
                        md += f"  📅 {event['timestamp']}\n"

        # 主题建模
        topics = data.get('topics', [])
        if topics:
            md += "\n## 主题建模\n"
            for topic in topics:
                md += f"\n### 主题 {topic.get('topic_id', 0)}\n"
                if topic.get('llm_title'):
                    md += f"**🤖 {topic['llm_title']}**\n\n"
                if topic.get('llm_summary'):
                    md += f"{topic['llm_summary']}\n\n"
                kw_list = ', '.join(topic.get('keywords', [])[:10])
                md += f"- 关键词: {kw_list}\n"
                md += f"- 文本数量: {topic.get('texts_count', 0)} 条\n"

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

    def _build_json_data(self) -> Dict[str, Any]:
        """构建完整的 JSON 数据"""

        # 处理关键词格式
        keywords_frequency = []
        for item in self.analyzer.keywords.get("frequency", [])[:50]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                keywords_frequency.append([item[0], item[1]])

        # ========== 1. 实体统计 ==========
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

        # ========== 2. 事件数据（不关联主题，按类型分组） ==========
        events = getattr(self.analyzer, 'events', [])[:200]

        # 按事件类型分组
        events_by_type = {}
        for event in events:
            et = event.get("event_type", "未知")
            if et not in events_by_type:
                events_by_type[et] = []
            events_by_type[et].append(event)

        # ========== 3. 主题数据 ==========
        topics = []
        for topic in getattr(self.analyzer, 'topics', [])[:10]:
            topics.append({
                "topic_id": topic.get("topic_id", 0),
                "texts_count": topic.get("texts_count", 0),
                "keywords": topic.get("keywords", [])[:10],
                "llm_title": topic.get("llm_title", f"主题{topic.get('topic_id', 0)}"),
                "llm_summary": topic.get("llm_summary", ""),
                "textrank_sentences": topic.get("textrank_sentences", [])[:2]
            })

        # ========== 4. 获取质量详情 ==========
        quality_data = self.analyzer.quality_report
        cleaning_suggestions = self.analyzer.cleaning_suggestions
        duplicates_detail = quality_data.get('duplicates', {})
        similarity_detail = quality_data.get('similarity', {})
        length_detail = quality_data.get('length_anomalies', {})

        # ========== 5. 获取图谱数据 ==========
        graph_viz = {}
        if hasattr(self.analyzer, 'graph') and self.analyzer.graph:
            graph_viz = self.analyzer.graph.export_for_visualization()

        # ========== 6. 构建返回数据 ==========
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
            'entity_stats': entity_stats,
            'events_by_type': events_by_type,
            'topics': topics,
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

            entity_stats = data.get('entity_stats', {})
            events_by_type = data.get('events_by_type', {})
            topics = data.get('topics', [])

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
                entity_stats=entity_stats,
                events_by_type=events_by_type,
                topics=topics
            )
        except Exception as e:
            import traceback
            print(f"模板渲染失败: {traceback.format_exc()}")
            return self._get_error_html(title, f"模板渲染失败: {str(e)}")