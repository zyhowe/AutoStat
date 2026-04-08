"""
报告生成器模块
生成HTML格式的分析报告
"""

import json
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import base64
from io import BytesIO
import matplotlib.pyplot as plt


class Reporter:
    """报告生成器"""

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def to_html(self, output_file=None, title="数据分析报告"):
        """
        生成HTML报告

        参数:
        - output_file: 输出文件路径，为None时返回HTML字符串
        - title: 报告标题
        """
        html = self._build_html(title)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"✅ 报告已保存到 {output_file}")
            return output_file
        return html

    def to_json(self, output_file=None, indent=2):
        """
        生成JSON格式结果

        参数:
        - output_file: 输出文件路径
        - indent: 缩进空格数
        """
        result = self._build_json()
        json_str = json.dumps(result, indent=indent, ensure_ascii=False, default=str)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_str)
            print(f"✅ 分析结果已保存到 {output_file}")
            return output_file
        return json_str

    def _build_html(self, title):
        """构建HTML内容"""
        # 获取数据
        data_shape = self.analyzer.data.shape
        variable_types = self.analyzer.variable_types
        quality_report = getattr(self.analyzer, 'quality_report', {})
        cleaning_suggestions = getattr(self.analyzer, 'cleaning_suggestions', [])

        # 统计变量类型
        type_counts = {}
        for typ in variable_types.values():
            type_counts[typ] = type_counts.get(typ, 0) + 1

        type_desc = {
            'continuous': '连续变量',
            'categorical': '分类变量',
            'categorical_numeric': '数值型分类',
            'ordinal': '有序分类',
            'datetime': '日期时间',
            'identifier': '标识符',
            'text': '文本',
            'other': '其他'
        }

        # 构建HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; border-left: 4px solid #4CAF50; padding-left: 15px; }}
        .summary {{ background: #e8f5e9; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
        .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; }}
        .good {{ background: #d4edda; border-left: 4px solid #28a745; padding: 10px; margin: 10px 0; }}
        .info {{ background: #d1ecf1; border-left: 4px solid #17a2b8; padding: 10px; margin: 10px 0; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 12px; }}
        .badge-continuous {{ background: #2196F3; color: white; }}
        .badge-categorical {{ background: #FF9800; color: white; }}
        .badge-datetime {{ background: #9C27B0; color: white; }}
        .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #999; }}
    </style>
</head>
<body>
<div class="container">
    <h1>📊 {title}</h1>
    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="summary">
        <h2>📈 数据概览</h2>
        <table>
            <tr><td><strong>总行数</strong></td><td>{data_shape[0]:,}</td></tr>
            <tr><td><strong>总列数</strong></td><td>{data_shape[1]}</td></tr>
            <tr><td><strong>缺失率>20%字段</strong></td><td>{len([m for m in quality_report.get('missing', []) if m['percent'] > 20])}</td></tr>
            <tr><td><strong>重复记录数</strong></td><td>{quality_report.get('duplicates', {}).get('count', 0)}</td></tr>
        </table>
    </div>

    <h2>📋 变量类型分布</h2>
    <table>
        <tr><th>类型</th><th>数量</th></tr>
"""
        for typ, count in type_counts.items():
            desc = type_desc.get(typ, typ)
            html += f"<tr><td>{desc}</td><td>{count}</td></tr>"

        html += """
    </table>

    <h2>📝 变量详情</h2>
    <table>
        <tr><th>变量名</th><th>类型</th><th>样本量</th><th>缺失数</th></tr>
"""
        for col, typ in list(variable_types.items())[:20]:
            desc = type_desc.get(typ, typ)
            missing = self.analyzer.data[col].isna().sum()
            n = len(self.analyzer.data[col].dropna())
            html += f"<tr><td>{col}</td><td>{desc}</td><td>{n:,}</td><td>{missing:,}</td></tr>"

        if len(variable_types) > 20:
            html += f"<tr><td colspan='4'>... 还有 {len(variable_types) - 20} 列</td></tr>"

        if cleaning_suggestions:
            html += """
    </table>

    <h2>💡 清洗建议</h2>
    <div class="warning">
        <ul>
"""
            for suggestion in cleaning_suggestions[:5]:
                html += f"<li>{suggestion}</li>"
            html += """
        </ul>
    </div>
"""

        html += f"""
    <div class="footer">
        <p>AutoStat 智能统计分析工具 | 报告自动生成</p>
    </div>
</div>
</body>
</html>
"""
        return html

    def _build_json(self):
        """构建JSON内容"""
        result = {
            'analysis_time': datetime.now().isoformat(),
            'source_table': self.analyzer.source_table_name,
            'data_shape': {
                'rows': len(self.analyzer.data),
                'columns': len(self.analyzer.data.columns)
            },
            'variable_types': {},
            'quality_report': {},
            'cleaning_suggestions': self.analyzer.cleaning_suggestions
        }

        for col, var_type in self.analyzer.variable_types.items():
            result['variable_types'][col] = {
                'type': var_type,
                'type_desc': self.analyzer._get_type_description(var_type)
            }

        quality = self.analyzer.quality_report
        result['quality_report']['missing'] = quality.get('missing', [])[:10]
        result['quality_report']['outliers'] = {
            col: {'count': info['count'], 'percent': info['percent']}
            for col, info in list(quality.get('outliers', {}).items())[:5]
        }
        result['quality_report']['duplicates'] = quality.get('duplicates', {})

        return result