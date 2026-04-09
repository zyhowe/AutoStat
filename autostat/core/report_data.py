"""
报告数据构建模块：为Reporter提供完整的报告数据
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional


class ReportDataBuilder:
    """报告数据构建器"""

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def build(self) -> Dict[str, Any]:
        """构建报告所需的所有数据"""
        data = self.analyzer.data
        quality = self.analyzer.quality_report
        dup_info = quality.get('duplicates', {})

        # 日期范围
        date_range = None
        date_cols = [col for col, typ in self.analyzer.variable_types.items() if typ == 'datetime']
        if date_cols:
            min_date = data[date_cols[0]].min()
            max_date = data[date_cols[0]].max()
            if pd.notna(min_date) and pd.notna(max_date):
                date_range = {'start': str(min_date.date()), 'end': str(max_date.date())}

        # 变量类型统计
        type_counts = {}
        for typ in self.analyzer.variable_types.values():
            type_counts[typ] = type_counts.get(typ, 0) + 1

        # 变量详情
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
            variables.append({
                'name': col,
                'type': var_type,
                'type_desc': self.analyzer._get_type_description(var_type),
                'n': len(series),
                'missing': data[col].isna().sum(),
                'missing_pct': round(data[col].isna().mean() * 100, 1),
                'center': center,
                'spread': spread
            })

        # 质量告警
        quality_alerts = []
        for item in quality.get('missing', [])[:5]:
            if item['percent'] > 20:
                quality_alerts.append(f"⚠️ {item['column']} 缺失率 {item['percent']:.1f}%")
        for col, info in quality.get('outliers', {}).items():
            if info.get('percent', 0) > 5:
                quality_alerts.append(f"⚠️ {col} 异常值比例 {info['percent']:.1f}%")

        # 偏态变量
        skewed_vars = []
        for col, typ in self.analyzer.variable_types.items():
            if typ == 'continuous':
                skew = data[col].dropna().skew()
                if abs(skew) > 2:
                    skewed_vars.append({'name': col, 'skew': round(skew, 2)})

        # 不平衡分类变量
        imbalanced_vars = []
        for col, typ in self.analyzer.variable_types.items():
            if typ in ['categorical', 'categorical_numeric', 'ordinal']:
                vc = data[col].value_counts(normalize=True)
                if len(vc) > 0 and vc.max() > 0.8:
                    imbalanced_vars.append({
                        'name': col,
                        'top_category': str(vc.index[0]),
                        'top_pct': round(vc.max() * 100, 1)
                    })

        # 强相关对
        numeric_vars = [col for col, typ in self.analyzer.variable_types.items() if typ == 'continuous']
        high_correlations = self._get_high_correlations(numeric_vars, threshold=0.7) if numeric_vars else []

        # 时间序列洞察
        time_series_insight = None
        if hasattr(self.analyzer, 'time_series_diagnostics') and self.analyzer.time_series_diagnostics:
            has_auto = any(d.get('has_autocorrelation') for d in self.analyzer.time_series_diagnostics.values())
            if has_auto:
                time_series_insight = "检测到显著自相关性，适合时间序列预测"

        # 模型推荐
        model_recs = self._get_model_recommendations()

        # 清洗建议
        cleaning_suggestions = []
        for item in quality.get('missing', []):
            if item['percent'] > 20:
                cleaning_suggestions.append({
                    'priority': '高',
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
            key_insights.append(f"发现{len(skewed_vars)}个偏态变量，建议使用中位数描述")
        if imbalanced_vars:
            key_insights.append(f"发现{len(imbalanced_vars)}个不平衡分类变量，分析时需注意")
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

        return {
            'source_table': getattr(self.analyzer, 'source_table_name', '未知'),
            'rows': f"{len(data):,}",
            'columns': len(data.columns),
            'missing_high_count': len([m for m in quality.get('missing', []) if m.get('percent', 0) > 20]),
            'outlier_count': len(quality.get('outliers', {})),
            'duplicate_count': dup_info.get('count', 0),
            'duplicate_rate': f"{dup_info.get('percent', 0):.1f}",
            'date_range': date_range,
            'type_counts': type_counts,
            'variables': variables,
            'variables_truncated': len(data.columns) - 30 if len(data.columns) > 30 else 0,
            'quality_alerts': quality_alerts,
            'skewed_vars': skewed_vars[:5],
            'imbalanced_vars': imbalanced_vars[:5],
            'high_correlations': high_correlations[:5],
            'time_series_insight': time_series_insight,
            'model_recommendations': model_recs,
            'cleaning_suggestions': cleaning_suggestions[:8],
            'key_insights': key_insights[:3],
            'next_actions': next_actions[:3]
        }

    def _get_high_correlations(self, numeric_vars, threshold=0.7):
        """获取强相关对"""
        correlations = []
        if len(numeric_vars) >= 2:
            corr_data = self.analyzer.data[numeric_vars].corr()
            for i in range(len(corr_data.columns)):
                for j in range(i + 1, len(corr_data.columns)):
                    val = corr_data.iloc[i, j]
                    if abs(val) >= threshold:
                        correlations.append({
                            'var1': corr_data.columns[i],
                            'var2': corr_data.columns[j],
                            'value': round(val, 3)
                        })
        return sorted(correlations, key=lambda x: abs(x['value']), reverse=True)

    def _get_model_recommendations(self):
        """获取模型推荐"""
        from autostat.core.recommendation import RecommendationAnalyzer

        rec_analyzer = RecommendationAnalyzer(
            self.analyzer.data,
            self.analyzer.variable_types,
            self.analyzer.quality_report,
            getattr(self.analyzer, 'time_series_diagnostics', {})
        )
        return rec_analyzer._get_model_recommendations(
            [col for col, typ in self.analyzer.variable_types.items() if typ == 'continuous'],
            [col for col, typ in self.analyzer.variable_types.items() if typ in ['categorical', 'categorical_numeric', 'ordinal']],
            [col for col, typ in self.analyzer.variable_types.items() if typ == 'datetime']
        )