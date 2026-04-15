"""
报告生成器模块
生成HTML格式的分析报告和JSON格式结果
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

from jinja2 import Template
from autostat.core.base import BaseAnalyzer


class Reporter:
    """报告生成器 - 生成HTML和JSON格式的分析报告"""

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def to_html(self, output_file=None, title="数据分析报告"):
        """生成HTML报告"""
        try:
            html = self._build_html(title)

            if html is None:
                html = self._get_error_html(title, "报告生成失败：返回值为None")

            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(html)
                print(f"✅ 报告已保存到 {output_file}")
                return output_file
            return html
        except Exception as e:
            error_html = self._get_error_html(title, f"报告生成失败: {str(e)}")
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(error_html)
                return output_file
            return error_html

    def to_json(self, output_file=None, indent=2, ensure_ascii=False):
        """
        生成JSON格式结果

        直接复用 analyzer.to_json() 方法，确保输出内容完整
        """
        return self.analyzer.to_json(output_file, indent, ensure_ascii)

    def to_markdown(self, output_file=None):
        """生成Markdown格式报告"""
        data = self.analyzer.data
        quality = self.analyzer.quality_report

        md = f"""# 数据分析报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 数据概览

| 指标 | 数值 |
|------|------|
| 总行数 | {len(data):,} |
| 总列数 | {len(data.columns)} |
| 重复记录数 | {quality.get('duplicates', {}).get('count', 0)} |

## 变量类型分布

| 类型 | 数量 |
|------|------|
"""
        type_counts = {}
        for typ in self.analyzer.variable_types.values():
            type_counts[typ] = type_counts.get(typ, 0) + 1
        for typ, count in type_counts.items():
            md += f"| {BaseAnalyzer.get_type_description(typ)} | {count} |\n"

        if self.analyzer.cleaning_suggestions:
            md += "\n## 清洗建议\n\n"
            for s in self.analyzer.cleaning_suggestions[:5]:
                md += f"- {s}\n"

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(md)
            print(f"✅ Markdown报告已保存到 {output_file}")
            return output_file
        return md

    def to_excel(self, output_file):
        """生成Excel格式报告"""
        data = self.analyzer.data
        quality = self.analyzer.quality_report

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Sheet 1: 数据概览
            summary_df = pd.DataFrame({
                '指标': ['总行数', '总列数', '缺失率>20%字段', '重复记录数'],
                '数值': [
                    len(data),
                    len(data.columns),
                    len([m for m in quality.get('missing', []) if m.get('percent', 0) > 20]),
                    quality.get('duplicates', {}).get('count', 0)
                ]
            })
            summary_df.to_excel(writer, sheet_name='数据概览', index=False)

            # Sheet 2: 变量详情
            var_data = []
            for col, typ in self.analyzer.variable_types.items():
                var_data.append({
                    '变量名': col,
                    '类型': BaseAnalyzer.get_type_description(typ),
                    '非空数': len(data[col].dropna()),
                    '缺失数': data[col].isna().sum()
                })
            pd.DataFrame(var_data).to_excel(writer, sheet_name='变量详情', index=False)

            # Sheet 3: 清洗建议
            if self.analyzer.cleaning_suggestions:
                pd.DataFrame({'建议': self.analyzer.cleaning_suggestions}).to_excel(
                    writer, sheet_name='清洗建议', index=False)

        print(f"✅ Excel报告已保存到 {output_file}")
        return output_file

    def _get_error_html(self, title, error_msg):
        """生成错误HTML"""
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
    <p>请检查数据或联系技术支持。</p>
</div>
</body>
</html>"""

    def _build_html(self, title):
        """构建HTML内容（包含图表）"""
        data = self.analyzer.data
        quality = self.analyzer.quality_report
        source_table = getattr(self.analyzer, 'source_table_name', '未知')
        dup_info = quality.get('duplicates', {})
        date_cols = [col for col, typ in self.analyzer.variable_types.items() if typ == 'datetime']

        # 生成图表（调用 analyzer 的方法）
        plots = {}

        try:
            # 连续变量图表
            for col, typ in self.analyzer.variable_types.items():
                if typ == 'continuous':
                    img = self.analyzer.get_plot_base64('continuous', col)
                    if img:
                        plots[f'{col}_continuous'] = img

            # 分类变量图表
            for col, typ in self.analyzer.variable_types.items():
                if typ in ['categorical', 'categorical_numeric', 'ordinal']:
                    img = self.analyzer.get_plot_base64('categorical', col)
                    if img:
                        plots[f'{col}_categorical'] = img

            # 时间序列图表 - 为每个数值变量生成，使用日期列
            if date_cols:
                date_col = date_cols[0]
                numeric_cols = [col for col, typ in self.analyzer.variable_types.items() if typ == 'continuous']
                for col in numeric_cols:
                    try:
                        # 按日期聚合
                        ts_data = data.groupby(date_col)[col].mean().reset_index()
                        ts_data = ts_data.dropna()
                        if len(ts_data) >= 3:
                            img = self.analyzer.get_plot_base64('timeseries', col)
                            if img:
                                plots[f'{col}_timeseries'] = img
                    except Exception as e:
                        if not hasattr(self.analyzer, 'quiet') or not self.analyzer.quiet:
                            print(f"⚠️ 生成 {col} 时间序列图失败: {e}")

            # 数值变量相关性热力图
            img = self.analyzer.get_numeric_correlation_base64()
            if img:
                plots['numeric_correlation'] = img

            # 分类变量关联热力图
            img = self.analyzer.get_categorical_correlation_base64()
            if img:
                plots['categorical_correlation'] = img

            # 数值-分类变量关联热力图
            img = self.analyzer.get_numeric_categorical_eta_base64()
            if img:
                plots['numeric_categorical_eta'] = img

        except Exception as e:
            print(f"⚠️ 生成图表时出错: {e}")

        # 日期范围
        date_range = None
        if date_cols:
            min_date = data[date_cols[0]].min()
            max_date = data[date_cols[0]].max()
            if pd.notna(min_date) and pd.notna(max_date):
                date_range = {'start': str(min_date.date()), 'end': str(max_date.date())}

        # 变量类型统计（使用中文描述）
        type_counts = {}
        type_display_map = {
            'continuous': '连续变量',
            'categorical': '分类变量',
            'categorical_numeric': '数值型分类',
            'ordinal': '有序分类',
            'datetime': '日期时间',
            'identifier': '标识符',
            'text': '文本',
            'other': '其他',
            'empty': '空变量'
        }
        for typ in self.analyzer.variable_types.values():
            display_name = type_display_map.get(typ, typ)
            type_counts[display_name] = type_counts.get(display_name, 0) + 1

        # 变量详情列表
        variables = []
        for col in list(data.columns)[:30]:
            var_type = self.analyzer.variable_types.get(col, 'unknown')
            series = data[col].dropna()
            if var_type == 'continuous':
                center = f"{series.mean():.2f}"
                spread = f"{series.std():.2f}"
            elif var_type in ['categorical', 'ordinal']:
                mode_val = series.mode().iloc[0] if not series.mode().empty else '-'
                center = str(mode_val)
                spread = f"{len(series)}条"
            else:
                center = '-'
                spread = '-'

            # 确定图表key
            plot_key = None
            if var_type == 'continuous':
                plot_key = f'{col}_continuous'
            elif var_type in ['categorical', 'categorical_numeric', 'ordinal']:
                plot_key = f'{col}_categorical'

            variables.append({
                'name': col,
                'type': var_type,
                'type_desc': BaseAnalyzer.get_type_description(var_type),
                'n': len(series),
                'missing': data[col].isna().sum(),
                'missing_pct': round(data[col].isna().mean() * 100, 1),
                'center': center,
                'spread': spread,
                'plot_key': plot_key
            })

        # 质量告警
        quality_alerts = []
        for item in quality.get('missing', [])[:5]:
            if item['percent'] > 20:
                quality_alerts.append(f"⚠️ {item['column']} 缺失率 {item['percent']:.1f}%")
        for col, info in quality.get('outliers', {}).items():
            if info.get('percent', 0) > 5:
                quality_alerts.append(f"⚠️ {col} 异常值比例 {info['percent']:.1f}%")

        # 偏态变量 - 使用 BaseAnalyzer
        skewed_vars = BaseAnalyzer.get_skewed_vars(data, self.analyzer.variable_types, threshold=2)

        # 不平衡分类变量 - 使用 BaseAnalyzer
        imbalanced_vars = BaseAnalyzer.get_imbalanced_vars(data, self.analyzer.variable_types, threshold=0.8)

        # 强相关对 - 使用 BaseAnalyzer，阈值 0.7
        numeric_vars = [col for col, typ in self.analyzer.variable_types.items() if typ == 'continuous']
        high_correlations = BaseAnalyzer.get_high_correlations(data, numeric_vars, threshold=0.7) if numeric_vars else []

        # 时间序列洞察
        time_series_insight = None
        time_series_diagnostics = getattr(self.analyzer, 'time_series_diagnostics', {})
        if time_series_diagnostics:
            has_auto = any(d.get('has_autocorrelation') for d in time_series_diagnostics.values())
            if has_auto:
                time_series_insight = "✅ 检测到显著自相关性，适合时间序列预测"
            else:
                time_series_insight = "⚠️ 未检测到显著自相关性，时间序列预测可能效果不佳"
        elif date_cols and numeric_vars:
            time_series_insight = "📊 检测到日期和数值变量，建议进行时间序列分析"

        # 获取模型推荐
        model_recommendations = self._get_model_recommendations()

        # 清洗建议
        cleaning_suggestions = []
        for item in quality.get('missing', []):
            if item['percent'] > 20:
                cleaning_suggestions.append({
                    'priority': '高' if item['percent'] > 50 else '中',
                    'operation': '处理缺失值',
                    'columns': item['column'],
                    'method': '删除列' if item['percent'] > 80 else '中位数/众数填充'
                })
        for col, info in quality.get('outliers', {}).items():
            if info.get('percent', 0) > 10:
                cleaning_suggestions.append({
                    'priority': '高',
                    'operation': '处理异常值',
                    'columns': col,
                    'method': '截尾或对数变换'
                })
        if dup_info.get('count', 0) > 0:
            cleaning_suggestions.append({
                'priority': '中',
                'operation': '删除重复记录',
                'columns': '全部',
                'method': f'删除 {dup_info["count"]} 条重复记录'
            })

        # 核心洞察
        key_insights = []
        if skewed_vars:
            skewed_names = ', '.join([v['name'] for v in skewed_vars[:3]])
            key_insights.append(f"发现{len(skewed_vars)}个偏态变量（{skewed_names}），建议使用中位数描述")
        if imbalanced_vars:
            imb_names = ', '.join([v['name'] for v in imbalanced_vars[:3]])
            key_insights.append(f"发现{len(imbalanced_vars)}个不平衡分类变量（{imb_names}），分析时需注意")
        if high_correlations:
            key_insights.append(f"发现{len(high_correlations)}对强相关变量，可考虑特征选择")
        if time_series_insight:
            key_insights.append(time_series_insight)
        if not key_insights:
            key_insights.append("数据分布较为均匀，可直接进行分析")

        # 下一步行动
        next_actions = []
        if quality.get('missing'):
            next_actions.append("优先处理高缺失率字段")
        if quality.get('outliers'):
            next_actions.append("检查异常值是否为数据错误")
        if numeric_vars and [c for c in self.analyzer.variable_types if self.analyzer.variable_types[c] in ['categorical', 'categorical_numeric', 'ordinal']]:
            next_actions.append("探索数值变量与分类变量的关系")
        if date_cols and numeric_vars:
            next_actions.append("进行时间序列趋势分析")
        if not next_actions:
            next_actions.append("数据质量良好，可直接进行建模分析")

        # 使用Jinja2模板渲染
        template_path = os.path.join(os.path.dirname(__file__), '..', 'templates', 'report.html')

        if not os.path.exists(template_path):
            return self._get_error_html(title, f"模板文件不存在: {template_path}")

        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            template = Template(template_content)

            return template.render(
                title=title,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                source_table=source_table,
                rows=f"{len(data):,}",
                columns=len(data.columns),
                missing_high_count=len([m for m in quality.get('missing', []) if m.get('percent', 0) > 20]),
                outlier_count=len(quality.get('outliers', {})),
                duplicate_count=dup_info.get('count', 0),
                duplicate_rate=f"{dup_info.get('percent', 0):.1f}",
                date_range=date_range,
                type_counts=type_counts,
                variables=variables,
                variables_truncated=len(data.columns) - 30 if len(data.columns) > 30 else 0,
                quality_alerts=quality_alerts,
                skewed_vars=skewed_vars[:5],
                imbalanced_vars=imbalanced_vars[:5],
                high_correlations=high_correlations[:5],
                time_series_insight=time_series_insight,
                model_recommendations=model_recommendations,
                cleaning_suggestions=cleaning_suggestions[:8],
                key_insights=key_insights[:3],
                next_actions=next_actions[:3],
                plots=plots,
                time_series_diagnostics=time_series_diagnostics
            )
        except Exception as e:
            return self._get_error_html(title, f"模板渲染失败: {str(e)}")

    def _get_model_recommendations(self):
        """获取基于实际字段的模型推荐（调用 recommendation_analyzer）"""
        if hasattr(self.analyzer, 'recommendation_analyzer'):
            numeric_vars = [col for col, typ in self.analyzer.variable_types.items() if typ == 'continuous']
            categorical_vars = [col for col, typ in self.analyzer.variable_types.items()
                               if typ in ['categorical', 'categorical_numeric', 'ordinal']]
            datetime_vars = [col for col, typ in self.analyzer.variable_types.items() if typ == 'datetime']
            return self.analyzer.recommendation_analyzer._get_model_recommendations(
                numeric_vars, categorical_vars, datetime_vars
            )
        return []

