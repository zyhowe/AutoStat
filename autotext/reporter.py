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
        """
        初始化报告生成器

        参数:
        - analyzer: TextAnalyzer 实例
        """
        self.analyzer = analyzer
        from autotext.core.report_data import ReportDataBuilder
        self.builder = ReportDataBuilder(analyzer)

    def to_html(self, output_file: Optional[str] = None, title: str = "文本分析报告") -> str:
        """生成 HTML 报告"""
        try:
            html = self._build_html(title)
            if output_file:
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
        report_data = self.builder.build()
        report_data["analysis_time"] = datetime.now().isoformat()
        report_data["source"] = getattr(self.analyzer, "source_name", "未知")

        json_str = json.dumps(report_data, ensure_ascii=False, indent=indent, default=str)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_str)
            print(f"✅ JSON报告已保存到 {output_file}")

        return json_str

    def to_markdown(self, output_file: Optional[str] = None) -> str:
        """生成 Markdown 报告"""
        data = self.builder.build()

        md = f"""# 文本分析报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 数据概览

| 指标 | 数值 |
|------|------|
| 总文本数 | {data['stats']['total_count']} |
| 空文本数 | {data['stats']['empty_count']} |
| 平均长度 | {data['stats']['char_length']['mean']:.1f} 字符 |

## 情感分布

- 积极: {data['sentiment']['distribution']['positive_rate']:.1%}
- 消极: {data['sentiment']['distribution']['negative_rate']:.1%}
- 中性: {data['sentiment']['distribution']['neutral_rate']:.1%}

## 高频关键词

"""
        for word, score in data['keywords']['frequency'][:20]:
            md += f"- {word}: {score}\n"

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(md)
            print(f"✅ Markdown报告已保存到 {output_file}")

        return md

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
        data = self.builder.build()

        # 获取模板路径
        template_path = os.path.join(os.path.dirname(__file__), '..', 'templates', 'report_text.html')

        if not os.path.exists(template_path):
            return self._get_error_html(title, f"模板文件不存在: {template_path}")

        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            template = Template(template_content)

            # 准备模板变量
            stats = data['stats']
            sentiment = data['sentiment']
            print("keywords1:",data['keywords'])
            keywords = data['keywords']['frequency'][:30]
            print("keywords2:", keywords)
            clusters = data['clusters']
            topics = data['topics']['list']
            quality = data['quality']
            cleaning_suggestions = data['cleaning_suggestions']
            llm_insights = data.get('llm_insights', {})

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
                cleaning_suggestions=cleaning_suggestions[:5],
                llm_insight=llm_insights.get('overall', '')
            )
        except Exception as e:
            return self._get_error_html(title, f"模板渲染失败: {str(e)}")