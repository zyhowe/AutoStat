"""
洞察生成服务 - 价值预览、核心结论、自然语言解读
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class InsightService:
    """洞察生成服务"""

    @staticmethod
    def generate_value_preview(df: pd.DataFrame) -> Dict[str, Any]:
        """
        生成价值预览
        """
        result = {
            "predictable": {"has": False},
            "timeseries": {"has": False},
            "needs_cleaning": {"has": False},
            "clustering": {"has": False},
            "recommended_analysis": ""
        }

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if len(numeric_cols) >= 2:
            variances = df[numeric_cols].var()
            best_target = variances.idxmax() if not variances.empty else numeric_cols[0]

            result["predictable"] = {
                "has": True,
                "target": best_target,
                "feature_count": len(numeric_cols) - 1,
                "description": f"可用{len(numeric_cols) - 1}个特征预测{best_target}"
            }

        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if date_cols and numeric_cols:
            result["timeseries"] = {
                "has": True,
                "columns": date_cols,
                "numeric_columns": numeric_cols[:3],
                "description": f"检测到时间序列数据，可预测{', '.join(numeric_cols[:2])}的未来走势"
            }

        total_cells = len(df) * len(df.columns)
        missing_cells = df.isna().sum().sum()
        missing_rate = missing_cells / total_cells if total_cells > 0 else 0

        high_missing_cols = df.columns[df.isna().sum() / len(df) > 0.2].tolist()

        if missing_rate > 0.05 or high_missing_cols:
            result["needs_cleaning"] = {
                "has": True,
                "missing_rate": missing_rate,
                "high_missing_cols": high_missing_cols[:3],
                "description": f"缺失率{missing_rate:.1%}，{len(high_missing_cols)}个字段缺失率>20%"
            }

        if len(numeric_cols) >= 3 and len(df) >= 100:
            result["clustering"] = {
                "has": True,
                "feature_count": len(numeric_cols),
                "description": f"可用{len(numeric_cols)}个数值特征进行用户分群"
            }

        recommendations = []
        if result["predictable"]["has"]:
            recommendations.append(f"{result['predictable']['target']}预测")
        if result["timeseries"]["has"]:
            recommendations.append("时间序列趋势分析")
        if result["clustering"]["has"]:
            recommendations.append("用户/客户分群")

        result["recommended_analysis"] = " + ".join(recommendations[:2]) if recommendations else "数据探索性分析"

        return result

    @staticmethod
    def generate_smart_summary(df: pd.DataFrame, preview: Dict[str, Any]) -> str:
        """生成智能摘要"""
        rows, cols = df.shape

        parts = [f"您的数据包含 {rows:,} 行 × {cols} 列"]

        dimensions = []
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        date_cols = df.select_dtypes(include=['datetime64']).columns

        if len(numeric_cols) > 0:
            dimensions.append(f"{len(numeric_cols)} 个数值字段")
        if len(cat_cols) > 0:
            dimensions.append(f"{len(cat_cols)} 个分类字段")
        if len(date_cols) > 0:
            dimensions.append(f"{len(date_cols)} 个时间字段")

        if dimensions:
            parts.append(f"，涉及 {', '.join(dimensions)}")

        if preview.get("predictable", {}).get("has"):
            parts.append(f"。可预测 {preview['predictable']['target']}")

        if preview.get("timeseries", {}).get("has"):
            parts.append("，可做趋势分析")

        needs_cleaning = preview.get("needs_cleaning", {})
        if needs_cleaning.get("has"):
            high_missing_count = len(needs_cleaning.get("high_missing_cols", []))
            parts.append(f"。注意：{high_missing_count} 个字段需要清洗")

        return "".join(parts)

    @staticmethod
    def extract_top_conclusions(analysis_result: Dict[str, Any], n: int = None) -> List[Dict[str, Any]]:
        """
        从分析结果中提取核心结论（全部列出，不截断）
        """
        conclusions = []

        # 用于收集同类结论
        ts_vars = []           # 时间序列变量
        high_corr_pairs = []   # 强相关对
        classification_targets = []  # 可预测目标
        high_missing_cols = []  # 高缺失字段
        outlier_cols = []       # 异常值字段

        # 1. 收集时间序列变量（全部列出）
        ts_diagnostics = analysis_result.get('time_series_diagnostics', {})
        for col, diag in ts_diagnostics.items():
            if diag.get('has_autocorrelation'):
                if '_' in col:
                    parts = col.rsplit('_', 1)
                    base_var = parts[0]
                    group = parts[1]
                    ts_vars.append(f"{base_var}({group}组)")
                else:
                    ts_vars.append(col)

        if ts_vars:
            # 全部列出，不截断
            var_str = '、'.join(ts_vars)
            conclusions.append({
                "icon": "📈",
                "title": f"{len(ts_vars)}个序列具有时间规律",
                "description": f"{var_str} 检测到显著自相关性，可用自身历史值预测未来走势"
            })

        # 2. 收集强相关对（相关系数 ≥ 0.5）
        high_corrs = analysis_result.get('correlations', {}).get('high_correlations', [])
        for corr in high_corrs:
            if abs(corr.get('value', 0)) >= 0.5:
                direction = "正" if corr.get('value', 0) > 0 else "负"
                high_corr_pairs.append(f"{corr['var1']}↔{corr['var2']}({direction}r={corr['value']:.2f})")

        if high_corr_pairs:
            # 全部列出，不截断
            pair_str = '、'.join(high_corr_pairs)
            conclusions.append({
                "icon": "🔗",
                "title": f"发现{len(high_corr_pairs)}对强相关关系",
                "description": pair_str
            })

        # 3. 聚类机会
        numeric_vars = [col for col, info in analysis_result.get('variable_types', {}).items()
                        if info.get('type') == 'continuous']
        n_samples = analysis_result.get('data_shape', {}).get('rows', 0)
        if len(numeric_vars) >= 3 and n_samples >= 100:
            conclusions.append({
                "icon": "🔘",
                "title": f"适合进行聚类分析",
                "description": f"{len(numeric_vars)}个数值指标，{n_samples}个样本，可识别用户/患者分群"
            })

        # 4. 收集可预测目标（分类/回归）
        model_recs = analysis_result.get('model_recommendations', [])
        for rec in model_recs:
            target = rec.get('target_column', '')
            # 排除派生列
            if target and not target.endswith('_year') and not target.endswith('_month') and not target.endswith('_quarter'):
                if target not in classification_targets:
                    classification_targets.append(target)

        if classification_targets:
            # 全部列出，不截断
            target_str = '、'.join(classification_targets)
            conclusions.append({
                "icon": "📊",
                "title": f"可预测 {target_str}",
                "description": "基于关联特征可建立预测模型"
            })

        # 5. 收集数据质量问题
        quality = analysis_result.get('quality_report', {})
        missing = quality.get('missing', [])
        for m in missing:
            if m.get('percent', 0) > 20:
                high_missing_cols.append(f"{m['column']}({m['percent']:.0f}%)")

        if high_missing_cols:
            # 全部列出，不截断
            missing_str = '、'.join(high_missing_cols)
            conclusions.append({
                "icon": "⚠️",
                "title": f"{len(high_missing_cols)}个字段缺失率较高",
                "description": missing_str
            })

        outliers = quality.get('outliers', {})
        for col, info in outliers.items():
            outlier_cols.append(f"{col}({info.get('percent', 0):.1f}%)")

        if outlier_cols:
            # 全部列出，不截断
            outlier_str = '、'.join(outlier_cols)
            conclusions.append({
                "icon": "🚨",
                "title": f"发现{len(outlier_cols)}个字段存在异常值",
                "description": outlier_str
            })

        # 6. 关联规则机会
        categorical_vars = [col for col, info in analysis_result.get('variable_types', {}).items()
                            if info.get('type') in ['categorical', 'categorical_numeric', 'ordinal']]
        if len(categorical_vars) >= 3:
            conclusions.append({
                "icon": "🔗",
                "title": f"可挖掘关联规则",
                "description": f"{len(categorical_vars)}个分类变量，可发现「如果A则B」的关联模式"
            })

        return conclusions

    @staticmethod
    def generate_natural_language_insight(chart_type: str, data: Dict[str, Any]) -> str:
        """生成自然语言解读"""
        templates = {
            "continuous": InsightService._insight_continuous,
            "categorical": InsightService._insight_categorical,
            "correlation": InsightService._insight_correlation,
            "timeseries": InsightService._insight_timeseries
        }

        generator = templates.get(chart_type, InsightService._insight_default)
        return generator(data)

    @staticmethod
    def _insight_continuous(data: Dict) -> str:
        """连续变量解读"""
        name = data.get('name', '变量')
        mean = data.get('mean', 0)
        median = data.get('median', 0)
        skew = data.get('skew', 0)
        min_val = data.get('min', 0)
        max_val = data.get('max', 0)

        if abs(skew) < 0.5:
            distribution = "近似正态分布"
        elif skew > 0:
            distribution = "右偏分布（有较大异常值）"
        else:
            distribution = "左偏分布（有较小异常值）"

        insight = f"**{name}** 集中在 {min_val:.1f}~{max_val:.1f} 之间，呈{distribution}。"
        insight += f"均值 {mean:.1f}，中位数 {median:.1f}。"

        if abs(mean - median) / (max_val - min_val + 0.01) > 0.1:
            insight += f" 均值与中位数差异较大，建议使用中位数描述中心趋势。"

        return insight

    @staticmethod
    def _insight_categorical(data: Dict) -> str:
        """分类变量解读"""
        name = data.get('name', '变量')
        value_counts = data.get('value_counts', {})

        if not value_counts:
            return f"**{name}** 是分类变量。"

        items = list(value_counts.items())
        items.sort(key=lambda x: x[1], reverse=True)

        top_item, top_count = items[0]
        total = sum(value_counts.values())
        top_pct = top_count / total * 100

        if len(items) == 1:
            insight = f"**{name}** 只有一个类别：{top_item}。"
        elif len(items) <= 5:
            others = ", ".join([f"{k}({v})" for k, v in items[1:3]])
            insight = f"**{name}** 中，**{top_item}** 占比最高（{top_pct:.1f}%），"
            insight += f"其他类别包括 {others}。"
        else:
            insight = f"**{name}** 有 {len(items)} 个类别，其中 **{top_item}** 占比最高（{top_pct:.1f}%）。"

        if top_pct > 80:
            insight += " ⚠️ 该变量高度不平衡，分析时需注意。"

        return insight

    @staticmethod
    def _insight_correlation(data: Dict) -> str:
        """相关性解读"""
        var1 = data.get('var1', '变量A')
        var2 = data.get('var2', '变量B')
        value = data.get('value', 0)

        strength = "强" if abs(value) > 0.7 else "中" if abs(value) > 0.3 else "弱"
        direction = "正" if value > 0 else "负"

        return f"**{var1}** 与 **{var2}** 呈{strength}{direction}相关（r={value:.3f}）。"

    @staticmethod
    def _insight_timeseries(data: Dict) -> str:
        """时间序列解读"""
        name = data.get('name', '变量')
        trend = data.get('trend', '平稳')
        seasonality = data.get('seasonality', False)

        insight = f"**{name}** 呈现{trend}趋势"
        if seasonality:
            insight += "，具有明显的季节性规律"

        return insight + "。"

    @staticmethod
    def _insight_default(data: Dict) -> str:
        """默认解读"""
        return f"数据显示了 {data.get('name', '变量')} 的分布特征。"

    @staticmethod
    def generate_rule_based_insights(analysis_result: Dict[str, Any]) -> List[str]:
        """基于规则生成洞察（无需大模型）"""
        insights = []

        rows = analysis_result.get('data_shape', {}).get('rows', 0)
        if rows > 100000:
            insights.append("📊 数据量较大，建议使用采样分析以获得更快响应")
        elif rows < 100:
            insights.append("📊 数据量较小，分析结果可能存在统计偏差")

        missing = analysis_result.get('quality_report', {}).get('missing', [])
        high_missing = [m for m in missing if m.get('percent', 0) > 20]
        if high_missing:
            fields = ', '.join([m['column'] for m in high_missing])
            insights.append(f"⚠️ 发现{len(high_missing)}个字段缺失率超过20%（{fields}），建议填充或删除")

        high_corrs = analysis_result.get('correlations', {}).get('high_correlations', [])
        if high_corrs:
            top_corr = high_corrs[0]
            insights.append(f"🔗 发现强相关：{top_corr['var1']} 与 {top_corr['var2']} 相关系数 {top_corr['value']}，可考虑特征选择")

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