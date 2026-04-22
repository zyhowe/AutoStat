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

        参数:
        - df: 数据框

        返回:
        {
            "predictable": {...},
            "timeseries": {...},
            "needs_cleaning": {...},  # 注意：这里是 needs_cleaning
            "clustering": {...},
            "recommended_analysis": str
        }
        """
        result = {
            "predictable": {"has": False},
            "timeseries": {"has": False},
            "needs_cleaning": {"has": False},  # 修正拼写
            "clustering": {"has": False},
            "recommended_analysis": ""
        }

        # 1. 可预测性分析
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if len(numeric_cols) >= 2:
            # 找出最佳预测目标（方差最大的数值列）
            variances = df[numeric_cols].var()
            best_target = variances.idxmax() if not variances.empty else numeric_cols[0]

            result["predictable"] = {
                "has": True,
                "target": best_target,
                "feature_count": len(numeric_cols) - 1,
                "description": f"可用{len(numeric_cols) - 1}个特征预测{best_target}"
            }

        # 2. 时间序列分析
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if date_cols and numeric_cols:
            result["timeseries"] = {
                "has": True,
                "columns": date_cols,
                "numeric_columns": numeric_cols[:3],
                "description": f"检测到时间序列数据，可预测{', '.join(numeric_cols[:2])}的未来走势"
            }

        # 3. 数据质量分析
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

        # 4. 聚类分析
        if len(numeric_cols) >= 3 and len(df) >= 100:
            result["clustering"] = {
                "has": True,
                "feature_count": len(numeric_cols),
                "description": f"可用{len(numeric_cols)}个数值特征进行用户分群"
            }

        # 5. 推荐分析
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
        """
        生成智能摘要（一句话总结）
        """
        rows, cols = df.shape

        parts = [f"您的数据包含 {rows:,} 行 × {cols} 列"]

        # 添加维度信息
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

        # 添加价值信息
        if preview.get("predictable", {}).get("has"):
            parts.append(f"。可预测 {preview['predictable']['target']}")

        if preview.get("timeseries", {}).get("has"):
            parts.append("，可做趋势分析")

        # 修复：使用 needs_cleaning 而不是 needs_cleaning
        needs_cleaning = preview.get("needs_cleaning", {})
        if needs_cleaning.get("has"):
            high_missing_count = len(needs_cleaning.get("high_missing_cols", []))
            parts.append(f"。注意：{high_missing_count} 个字段需要清洗")

        return "".join(parts)

    @staticmethod
    def extract_top_conclusions(analysis_result: Dict[str, Any], n: int = 5) -> List[Dict[str, Any]]:
        """
        从分析结果中提取核心结论
        """
        conclusions = []

        # 1. 强相关性结论
        high_corrs = analysis_result.get('correlations', {}).get('high_correlations', [])
        for corr in high_corrs[:2]:
            conclusions.append({
                "icon": "🔗",
                "title": f"{corr['var1']} 与 {corr['var2']} 高度相关",
                "description": f"相关系数 {corr['value']:.3f}，属于{'正' if corr['value'] > 0 else '负'}相关",
                "action": f"可考虑用{corr['var1']}预测{corr['var2']}，或进行特征选择",
                "priority": "high"
            })

        # 2. 时间序列结论
        ts_diagnostics = analysis_result.get('time_series_diagnostics', {})
        for col, diag in list(ts_diagnostics.items())[:1]:
            if diag.get('has_autocorrelation'):
                conclusions.append({
                    "icon": "📈",
                    "title": f"{col} 具有时间序列规律",
                    "description": "检测到显著的自相关性，历史数据可预测未来",
                    "action": "建议使用ARIMA或LSTM进行时序预测",
                    "priority": "high"
                })

        # 3. 数据质量结论
        quality = analysis_result.get('quality_report', {})
        missing = quality.get('missing', [])
        high_missing = [m for m in missing if m.get('percent', 0) > 20]
        if high_missing:
            conclusions.append({
                "icon": "⚠️",
                "title": f"{len(high_missing)}个字段缺失率较高",
                "description": f"最高缺失率: {high_missing[0]['column']} ({high_missing[0]['percent']:.1f}%)",
                "action": "建议删除缺失率>80%的字段，其余用中位数/众数填充",
                "priority": "medium"
            })

        # 4. 建模机会
        recommendations = analysis_result.get('model_recommendations', [])
        for rec in recommendations[:1]:
            conclusions.append({
                "icon": "🤖",
                "title": f"适合进行{rec.get('task_type', '预测分析')}",
                "description": rec.get('reason', '基于数据特征，可建立预测模型'),
                "action": f"推荐使用: {rec.get('ml', '随机森林')}",
                "priority": "high"
            })

        # 5. 异常值结论
        outliers = quality.get('outliers', {})
        if outliers:
            outlier_cols = list(outliers.keys())[:2]
            outlier_info = outliers.get(outlier_cols[0], {}) if outlier_cols else {}
            conclusions.append({
                "icon": "🚨",
                "title": f"发现{len(outliers)}个字段存在异常值",
                "description": f"涉及字段: {', '.join(outlier_cols)}，最高比例: {outlier_info.get('percent', 0):.1f}%",
                "action": "建议检查异常值是否为数据错误，必要时进行截尾处理",
                "priority": "medium"
            })

        # 6. 偏态结论
        skewed = analysis_result.get('distribution_insights', {}).get('skewed_variables', [])
        if skewed:
            skewed_names = [s['name'] for s in skewed[:2]]
            conclusions.append({
                "icon": "📊",
                "title": f"发现{len(skewed)}个偏态变量",
                "description": f"偏态变量: {', '.join(skewed_names)}",
                "action": "建议使用中位数描述或进行对数变换",
                "priority": "low"
            })

        return conclusions[:n]

    @staticmethod
    def generate_natural_language_insight(chart_type: str, data: Dict[str, Any]) -> str:
        """
        生成自然语言解读
        """
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
        """
        基于规则生成洞察（无需大模型）
        """
        insights = []

        # 数据规模洞察
        rows = analysis_result.get('data_shape', {}).get('rows', 0)
        if rows > 100000:
            insights.append("📊 数据量较大，建议使用采样分析以获得更快响应")
        elif rows < 100:
            insights.append("📊 数据量较小，分析结果可能存在统计偏差")

        # 缺失值洞察
        missing = analysis_result.get('quality_report', {}).get('missing', [])
        high_missing = [m for m in missing if m.get('percent', 0) > 20]
        if high_missing:
            fields = ', '.join([m['column'] for m in high_missing[:3]])
            insights.append(f"⚠️ 发现{len(high_missing)}个字段缺失率超过20%（{fields}），建议填充或删除")

        # 相关性洞察
        high_corrs = analysis_result.get('correlations', {}).get('high_correlations', [])
        if high_corrs:
            top_corr = high_corrs[0]
            insights.append(f"🔗 发现强相关：{top_corr['var1']} 与 {top_corr['var2']} 相关系数 {top_corr['value']}，可考虑特征选择")

        # 偏态洞察
        skewed = analysis_result.get('distribution_insights', {}).get('skewed_variables', [])
        if skewed:
            insights.append(f"📈 发现{len(skewed)}个偏态变量，建议使用中位数描述或进行对数变换")

        # 不平衡分类洞察
        imbalanced = analysis_result.get('distribution_insights', {}).get('imbalanced_categoricals', [])
        if imbalanced:
            insights.append(f"⚖️ 发现{len(imbalanced)}个不平衡分类变量，分析时需注意类别分布")

        # 时间序列洞察
        ts_forecastable = analysis_result.get('time_series_forecastable', False)
        if ts_forecastable:
            insights.append(f"📅 检测到可预测的时间序列数据，建议进行趋势分析和预测")

        return insights