"""洞察服务 - 从 web/services/insight_service.py 提取核心逻辑"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import re


class InsightService:
    """洞察生成服务 - 独立实现，不依赖 web/"""

    @staticmethod
    def extract_top_conclusions(analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从分析结果中提取核心结论"""
        conclusions = []

        # 用于收集同类结论
        ts_vars = []
        high_corr_pairs = []
        classification_targets = []
        high_missing_cols = []
        outlier_cols = []

        # 1. 收集时间序列变量
        ts_diagnostics = analysis_result.get('time_series_diagnostics', {})
        for col, diag in ts_diagnostics.items():
            if diag.get('has_autocorrelation'):
                if '_' in col:
                    base = col.rsplit('_', 1)[0]
                    if base not in ts_vars:
                        ts_vars.append(base)
                else:
                    if col not in ts_vars:
                        ts_vars.append(col)

        if ts_vars:
            display_vars = ts_vars[:3]
            display_str = '、'.join(display_vars)
            if len(ts_vars) > 3:
                display_str += f'等{len(ts_vars)}个字段'
            conclusions.append({
                "icon": "📈",
                "title": f"{len(ts_vars)}个序列具有时间规律",
                "description": f"{display_str} 检测到显著自相关性，可用自身历史值预测未来走势"
            })

        # 2. 收集强相关对
        high_corrs = analysis_result.get('correlations', {}).get('high_correlations', [])
        for corr in high_corrs:
            if abs(corr.get('value', 0)) >= 0.7:
                high_corr_pairs.append(f"{corr.get('var1', '')}↔{corr.get('var2', '')}")

        if high_corr_pairs:
            display_pairs = high_corr_pairs[:3]
            display_str = '、'.join(display_pairs)
            if len(high_corr_pairs) > 3:
                display_str += f'等{len(high_corr_pairs)}对'
            conclusions.append({
                "icon": "🔗",
                "title": f"发现{len(high_corr_pairs)}对强相关关系",
                "description": f"{display_str} 相关系数|r|≥0.7，建议重点关注"
            })

        # 3. 聚类机会
        variable_types = analysis_result.get('variable_types', {})
        numeric_vars = [col for col, info in variable_types.items() if info.get('type') == 'continuous']
        n_samples = analysis_result.get('data_shape', {}).get('rows', 0)
        if len(numeric_vars) >= 3 and n_samples >= 100:
            conclusions.append({
                "icon": "🔘",
                "title": "适合进行聚类分析",
                "description": f"{len(numeric_vars)}个数值指标，{n_samples}个样本，可识别用户/患者分群"
            })

        # 4. 收集可预测目标
        model_recs = analysis_result.get('model_recommendations', [])
        for rec in model_recs:
            target = rec.get('target_column', '')
            if target and not target.endswith('_year') and not target.endswith('_month') and not target.endswith(
                    '_quarter'):
                if target not in classification_targets:
                    classification_targets.append(target)

        if classification_targets:
            display_targets = classification_targets[:3]
            display_str = '、'.join(display_targets)
            if len(classification_targets) > 3:
                display_str += f'等{len(classification_targets)}个字段'
            conclusions.append({
                "icon": "📊",
                "title": f"{len(classification_targets)}个字段可预测",
                "description": f"{display_str} 基于关联特征可建立预测模型"
            })

        # 5. 收集数据质量问题
        quality = analysis_result.get('quality_report', {})
        missing = quality.get('missing', [])
        for m in missing:
            if m.get('percent', 0) > 20:
                high_missing_cols.append(m.get('column', ''))

        if high_missing_cols:
            display_cols = high_missing_cols[:3]
            display_str = '、'.join(display_cols)
            if len(high_missing_cols) > 3:
                display_str += f'等{len(high_missing_cols)}个字段'
            conclusions.append({
                "icon": "⚠️",
                "title": f"{len(high_missing_cols)}个字段缺失率较高",
                "description": f"{display_str} 缺失率超过20%，建议处理"
            })

        # 6. 收集异常值字段
        outliers = quality.get('outliers', {})
        for col in outliers.keys():
            outlier_cols.append(col)

        if outlier_cols:
            display_cols = outlier_cols[:3]
            display_str = '、'.join(display_cols)
            if len(outlier_cols) > 3:
                display_str += f'等{len(outlier_cols)}个字段'
            conclusions.append({
                "icon": "🚨",
                "title": f"发现{len(outlier_cols)}个字段存在异常值",
                "description": f"{display_str} 存在异常值，建议检查数据来源"
            })

        # 7. 关联规则机会
        categorical_vars = [col for col, info in variable_types.items()
                            if info.get('type') in ['categorical', 'categorical_numeric', 'ordinal']]
        if len(categorical_vars) >= 3:
            conclusions.append({
                "icon": "🔗",
                "title": "可挖掘关联规则",
                "description": f"{len(categorical_vars)}个分类变量，可发现「如果A则B」的关联模式"
            })

        return conclusions

    @staticmethod
    def generate_rule_based_insights(analysis_result: Dict[str, Any]) -> List[str]:
        """基于规则生成洞察（无需大模型）"""
        insights = []

        rows = analysis_result.get('data_shape', {}).get('rows', 0)
        if rows > 100000:
            insights.append("📊 数据量较大，建议使用采样分析以获得更快响应")
        elif rows < 100:
            insights.append("📊 数据量较小，分析结果可能存在统计偏差")

        quality = analysis_result.get('quality_report', {})
        missing = quality.get('missing', [])
        high_missing = [m for m in missing if m.get('percent', 0) > 20]
        if high_missing:
            fields = ', '.join([m['column'] for m in high_missing])
            insights.append(f"⚠️ 发现{len(high_missing)}个字段缺失率超过20%（{fields}），建议填充或删除")

        high_corrs = analysis_result.get('correlations', {}).get('high_correlations', [])
        if high_corrs:
            top_corr = high_corrs[0]
            insights.append(
                f"🔗 发现强相关：{top_corr['var1']} 与 {top_corr['var2']} 相关系数 {top_corr['value']}，可考虑特征选择")

        skewed = analysis_result.get('distribution_insights', {}).get('skewed_variables', [])
        if skewed:
            insights.append(f"📈 发现{len(skewed)}个偏态变量，建议使用中位数描述或进行对数变换")

        imbalanced = analysis_result.get('distribution_insights', {}).get('imbalanced_categoricals', [])
        if imbalanced:
            insights.append(f"⚖️ 发现{len(imbalanced)}个不平衡分类变量，分析时需注意类别分布")

        ts_forecastable = analysis_result.get('time_series_forecastable', False)
        if ts_forecastable:
            insights.append(f"📅 检测到可预测的时间序列数据，建议进行趋势分析和预测")

        return insights