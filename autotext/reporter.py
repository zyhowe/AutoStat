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
        from autotext.core.report_data import ReportDataBuilder
        self.builder = ReportDataBuilder(analyzer)

    def to_html(self, output_file: Optional[str] = None, title: str = "文本分析报告") -> str:
        """生成 HTML 报告"""
        try:
            html = self._build_html(title)
            if output_file:
                # 确保输出目录存在
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

        md += f"""
## 情感分布

- 积极: {data['sentiment']['distribution']['positive_rate']:.1%}
- 消极: {data['sentiment']['distribution']['negative_rate']:.1%}
- 中性: {data['sentiment']['distribution']['neutral_rate']:.1%}

## 高频关键词 TOP 30

"""
        for word, count in data['keywords']['frequency'][:30]:
            md += f"- {word}: {count}\n"

        # 实体统计
        entity = data.get('entity', {})
        if entity.get('per', {}).get('unique', 0) > 0:
            md += "\n## 人名实体\n"
            for name, count in entity.get('per', {}).get('top', [])[:20]:
                md += f"- {name}: {count}次\n"

        if entity.get('org', {}).get('unique', 0) > 0:
            md += "\n## 组织名实体\n"
            for name, count in entity.get('org', {}).get('top', [])[:20]:
                md += f"- {name}: {count}次\n"

        if entity.get('loc', {}).get('unique', 0) > 0:
            md += "\n## 地名实体\n"
            for name, count in entity.get('loc', {}).get('top', [])[:20]:
                md += f"- {name}: {count}次\n"

        # 聚类结果
        clusters = data.get('clusters', [])
        if clusters:
            md += "\n## 文本聚类\n"
            for cluster in clusters:
                md += f"\n### 簇 {cluster.get('id', 0)}\n"
                md += f"- 数量: {cluster.get('size', 0)} 条 ({cluster.get('percentage', 0):.1%})\n"
                md += f"- 关键词: {', '.join(cluster.get('top_words', [])[:10])}\n"

        # 主题结果
        topics = data.get('topics', [])
        if topics:
            md += "\n## 主题建模\n"
            for topic in topics:
                md += f"\n### 主题 {topic.get('id', 0)}\n"
                kw_list = ', '.join(topic.get('keywords', [])[:10])
                md += f"- 关键词: {kw_list}\n"

        # 洞察发现
        insights = data.get('insights', [])
        if insights:
            md += "\n## 洞察发现\n"
            for insight in insights[:10]:
                priority_icon = {'高': '🔴', '中': '🟠', '低': '🟢'}.get(insight.get('priority', '中'), '⚪')
                md += f"\n### {priority_icon} {insight.get('title', '')}\n"
                md += f"{insight.get('description', '')}\n"

        # 清洗建议
        if data.get('cleaning_suggestions'):
            md += "\n## 清洗建议\n"
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

        # 处理关键词格式：保持 [(word, count), ...] 格式
        keywords_frequency = []
        for item in self.analyzer.keywords.get("frequency", [])[:50]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                keywords_frequency.append([item[0], item[1]])
            elif isinstance(item, dict):
                keywords_frequency.append([item.get('word', ''), item.get('count', 0)])

        # 处理聚类信息
        clusters = []
        for cluster in self.analyzer.cluster_info[:10]:
            clusters.append({
                "id": cluster.get("cluster_id", 0),
                "size": cluster.get("size", 0),
                "percentage": cluster.get("percentage", 0),
                "top_words": cluster.get("top_words", [])[:10],
                "center_text": cluster.get("center_text", "")[:200] if cluster.get("center_text") else ""
            })

        # 处理主题信息
        topics = []
        for topic in self.analyzer.topics[:10]:
            topics.append({
                "id": topic.get("topic_id", 0),
                "texts_count": topic.get("texts_count", 0),
                "keywords": topic.get("keywords", [])[:10],
                "weights": topic.get("weights", [])[:10] if topic.get("weights") else []
            })

        # 处理实体信息（已过滤）
        entity = getattr(self.analyzer, 'entity_stats', {})

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

        data = {
            'stats': self.analyzer.stats_result,
            'language_distribution': getattr(self.analyzer, 'language_distribution', {}),
            'quality': self.analyzer.quality_report,
            'cleaning_suggestions': self.analyzer.cleaning_suggestions[:10],
            'keywords': {
                'frequency': keywords_frequency,
                'tfidf': []
            },
            'sentiment': {
                'distribution': self.analyzer.sentiment_distribution,
                'results': self.analyzer.sentiment_results[:20]
            },
            'entity': entity,
            'clusters': clusters,
            'topics': topics,
            'trend': getattr(self.analyzer, 'trend_info', {}),
            'event_timeline': getattr(self.analyzer, 'event_timeline', {}),
            'templates': {
                'start': list(self.analyzer.start_templates)[:20],
                'end': list(self.analyzer.end_templates)[:20]
            },
            'insights': insights,
            'sample_texts': self.analyzer.content_texts[:10]
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
            # 尝试另一种路径
            template_path = os.path.join(os.path.dirname(__file__), 'templates', 'report_text.html')

        if not os.path.exists(template_path):
            return self._get_error_html(title, f"模板文件不存在: {template_path}")

        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            template = Template(template_content)

            stats = data['stats']
            sentiment = data['sentiment']
            keywords = data['keywords']['frequency']  # 保持 [(word, count), ...] 格式
            clusters = data['clusters']
            topics = data['topics']
            quality = data['quality']
            cleaning_suggestions = data['cleaning_suggestions']
            entity = data.get('entity', {})
            language_dist = data.get('language_distribution', {})
            templates = data.get('templates', {})
            insights = data.get('insights', [])
            sample_texts = data.get('sample_texts', [])

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
                keywords=keywords,
                clusters=clusters,
                topics=topics,
                duplicate_count=quality.get('duplicates', {}).get('count', 0),
                cleaning_suggestions=cleaning_suggestions[:10],
                entity=entity,
                language_distribution=language_dist,
                start_templates=list(templates.get('start', []))[:20],
                end_templates=list(templates.get('end', []))[:20],
                insights=insights,
                sample_texts=sample_texts
            )
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"模板渲染失败: {error_detail}")
            return self._get_error_html(title, f"模板渲染失败: {str(e)}")