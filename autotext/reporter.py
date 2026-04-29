"""
文本分析报告生成器 - 生成 HTML/JSON/Markdown 报告
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from jinja2 import Template


class TextReporter:
    """文本分析报告生成器"""

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def to_html(self, output_file: Optional[str] = None, title: str = "文本分析报告") -> str:
        """生成 HTML 报告"""
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
        """生成 JSON 报告"""
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
        """生成 Markdown 报告"""
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

        # 情感分布（仅在极化时展示）
        sentiment = data.get('sentiment', {})
        dist = sentiment.get('distribution', {})
        pos = dist.get('positive_rate', 0)
        neg = dist.get('negative_rate', 0)
        if pos > 0.4 or neg > 0.2:
            md += f"""
## 情感分布

- 积极: {pos:.1%}
- 消极: {neg:.1%}
- 中性: {dist.get('neutral_rate', 0):.1%}

"""
        else:
            md += "\n## 情感分布\n整体情感中性，无明显倾向。\n"

        md += """
## 高频关键词 TOP 30

"""
        for word, count in data['keywords']['frequency'][:30]:
            md += f"- {word}: {count}\n"

        # 实体统计
        entity = data.get('entity', {})
        if entity.get('org', {}).get('unique', 0) > 0:
            md += "\n## 组织名实体\n"
            for name, count in entity.get('org', {}).get('top', [])[:20]:
                md += f"- {name}: {count}次\n"

        if entity.get('per', {}).get('unique', 0) > 0:
            md += "\n## 人名实体\n"
            for name, count in entity.get('per', {}).get('top', [])[:20]:
                md += f"- {name}: {count}次\n"

        # 事件统计
        events = data.get('events', [])
        if events:
            event_types = {}
            for event in events:
                et = event.get('event_type', '未知')
                event_types[et] = event_types.get(et, 0) + 1
            md += "\n## 事件统计\n"
            for et, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                md += f"- {et}: {count}次\n"

        # 主题结果
        topics = data.get('topics', [])
        if topics:
            md += "\n## 主题建模\n"
            for topic in topics:
                md += f"\n### 主题 {topic.get('topic_id', 0)}\n"
                kw_list = ', '.join(topic.get('keywords', [])[:10])
                md += f"- 关键词: {kw_list}\n"
                md += f"- 文本数量: {topic.get('texts_count', 0)} 条\n"

        # 实体档案
        entity_profiles = data.get('entity_profiles', [])
        if entity_profiles:
            md += "\n## 实体档案\n"
            for profile in entity_profiles[:10]:
                md += f"\n### {profile['name']} ({profile['type']})\n"
                md += f"- 提及次数: {profile['mention_count']}\n"
                if profile.get('topics'):
                    topic_str = ', '.join([t['topic_name'] for t in profile['topics'][:3]])
                    md += f"- 相关主题: {topic_str}\n"
                if profile.get('events'):
                    md += f"- 相关事件: {len(profile['events'])} 个\n"
                if profile.get('related_entities'):
                    related_str = ', '.join([r['name'] for r in profile['related_entities'][:5]])
                    md += f"- 关联实体: {related_str}\n"

        # 洞察发现
        insights = data.get('insights', [])
        if insights:
            md += "\n## 洞察发现\n"
            for insight in insights[:10]:
                priority_icon = {'高': '🔴', '中': '🟠', '低': '🟢'}.get(insight.get('priority', '中'), '⚪')
                md += f"\n### {priority_icon} {insight.get('title', '')}\n"
                md += f"{insight.get('description', '')}\n"

        # 图分析洞察
        graph_insights = data.get('graph_insights', {})
        center_nodes = graph_insights.get('center_nodes', [])
        if center_nodes:
            md += "\n## 核心节点（PageRank）\n"
            for node in center_nodes[:5]:
                md += f"- **{node.get('name', '')}** ({node.get('type', '')}): 中心性分数 {node.get('score', 0):.4f}\n"

        bridge_nodes = graph_insights.get('bridge_nodes', [])
        if bridge_nodes:
            md += "\n## 桥梁节点（Betweenness）\n"
            for node in bridge_nodes[:5]:
                md += f"- **{node.get('name', '')}** ({node.get('type', '')}): 介数 {node.get('score', 0):.4f}\n"

        # 时间线
        timeline = data.get('timeline', {})
        if timeline.get('has_data'):
            md += "\n## 事件时间线\n"
            date_range = timeline.get('date_range', {})
            md += f"- 时间范围: {date_range.get('start', '')} ~ {date_range.get('end', '')}\n"
            md += f"- 总事件数: {timeline.get('total_events', 0)}\n"

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
            elif isinstance(item, dict):
                keywords_frequency.append([item.get('word', ''), item.get('count', 0)])

        # 处理实体信息
        entity = getattr(self.analyzer, 'entity_stats', {})

        # 处理主题信息
        topics = []
        for topic in getattr(self.analyzer, 'topics', [])[:10]:
            topics.append({
                "id": topic.get("topic_id", 0),
                "texts_count": topic.get("texts_count", 0),
                "keywords": topic.get("keywords", [])[:10],
                "weights": topic.get("weights", [])[:10] if topic.get("weights") else []
            })

        # 处理洞察信息
        insights = []
        for insight in getattr(self.analyzer, 'insights', [])[:15]:
            insights.append({
                "type": insight.get("type", ""),
                "title": insight.get("title", ""),
                "description": insight.get("description", ""),
                "priority": insight.get("priority", "中"),
                "score": insight.get("score", 0),
                "data": insight.get("data", {})
            })

        # 获取质量详情
        quality_data = self.analyzer.quality_report
        cleaning_suggestions = self.analyzer.cleaning_suggestions

        # 提取重复、相似、长度异常的详情
        duplicates_detail = quality_data.get('duplicates', {})
        similarity_detail = quality_data.get('similarity', {})
        length_detail = quality_data.get('length_anomalies', {})

        # 获取事件数据
        events = getattr(self.analyzer, 'events', [])[:100]

        # 获取实体档案
        entity_profiles = []
        if hasattr(self.analyzer, 'entity_profiles'):
            for profile in self.analyzer.entity_profiles[:20]:
                entity_profiles.append(profile.to_dict())

        # 获取图分析洞察
        graph_insights = getattr(self.analyzer, 'graph_insights', {})

        # 获取时间线摘要
        timeline_summary = {}
        if hasattr(self.analyzer, 'timeline') and self.analyzer.timeline:
            timeline_summary = self.analyzer.timeline.get_summary()

        # 获取图谱可视化数据
        graph_viz = {}
        if hasattr(self.analyzer, 'graph') and self.analyzer.graph:
            graph_viz = self.analyzer.graph.export_for_visualization()

        data = {
            'stats': self.analyzer.stats_result,
            'language_distribution': getattr(self.analyzer, 'language_distribution', {}),
            'quality': quality_data,
            'cleaning_suggestions': cleaning_suggestions[:10],
            'keywords': {
                'frequency': keywords_frequency,
                'tfidf': []
            },
            'sentiment': {
                'distribution': self.analyzer.sentiment_distribution,
                'is_polarized': self.analyzer.sentiment_distribution.get('positive_rate', 0) > 0.4 or
                                self.analyzer.sentiment_distribution.get('negative_rate', 0) > 0.2
            },
            'entity': entity,
            'topics': topics,
            'templates': {
                'start': list(self.analyzer.start_templates)[:20],
                'end': list(self.analyzer.end_templates)[:20]
            },
            'insights': insights,
            'sample_texts': self.analyzer.content_texts[:10],
            # 新增数据
            'events': events[:50],
            'entity_profiles': entity_profiles,
            'graph_insights': graph_insights,
            'timeline': timeline_summary,
            'graph_viz': graph_viz,
            # 质量详情区块数据
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
        """生成错误 HTML"""
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
        """构建 HTML 内容"""
        data = self._build_json_data()

        # 获取模板路径
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
            topics = data['topics']
            quality = data['quality']
            cleaning_suggestions = data['cleaning_suggestions']
            entity = data.get('entity', {})
            language_dist = data.get('language_distribution', {})
            templates = data.get('templates', {})
            insights = data.get('insights', [])
            sample_texts = data.get('sample_texts', [])
            quality_details = data.get('quality_details', {})
            events = data.get('events', [])
            entity_profiles = data.get('entity_profiles', [])
            graph_insights = data.get('graph_insights', {})
            timeline = data.get('timeline', {})
            graph_viz = data.get('graph_viz', {})

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
                topics=topics,
                duplicate_count=quality.get('duplicates', {}).get('count', 0),
                cleaning_suggestions=cleaning_suggestions[:10],
                entity=entity,
                language_distribution=language_dist,
                start_templates=list(templates.get('start', []))[:20],
                end_templates=list(templates.get('end', []))[:20],
                insights=insights,
                sample_texts=sample_texts,
                quality_details=quality_details,
                events=events,
                entity_profiles=entity_profiles,
                graph_insights=graph_insights,
                timeline=timeline,
                graph_viz=graph_viz
            )
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"模板渲染失败: {error_detail}")
            return self._get_error_html(title, f"模板渲染失败: {str(e)}")