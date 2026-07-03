"""洞察生成服务 - 核心分析模块，不依赖 web/ 或 api_server/"""
import math
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, date


class InsightService:
    """洞察生成服务 - 所有返回值均为 JSON 安全类型"""

    @staticmethod
    def extract_top_conclusions(analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从分析结果中提取核心结论"""
        conclusions = []

        # 1. 时间序列变量
        ts_diagnostics = analysis_result.get('time_series_diagnostics', {})
        ts_vars = []
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
                display_str += f'等{len(ts_vars)}个'
            conclusions.append({
                "icon": "📈",
                "title": f"{len(ts_vars)}个序列具有时间规律",
                "description": f"{display_str} 检测到显著自相关性，可用自身历史值预测未来走势"
            })

        # 2. 强相关对
        high_corrs = analysis_result.get('correlations', {}).get('high_correlations', [])
        high_corr_pairs = []
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
                "description": f"{len(numeric_vars)}个数值指标，{n_samples}个样本，可识别分群"
            })

        # 4. 可预测目标
        model_recs = analysis_result.get('model_recommendations', [])
        classification_targets = []
        for rec in model_recs:
            target = rec.get('target_column', '')
            if target and not target.endswith('_year') and not target.endswith('_month') and not target.endswith('_quarter'):
                if target not in classification_targets:
                    classification_targets.append(target)

        if classification_targets:
            display_targets = classification_targets[:3]
            display_str = '、'.join(display_targets)
            if len(classification_targets) > 3:
                display_str += f'等{len(classification_targets)}个'
            conclusions.append({
                "icon": "📊",
                "title": f"{len(classification_targets)}个字段可预测",
                "description": f"{display_str} 基于关联特征可建立预测模型"
            })

        # 5. 数据质量问题
        quality = analysis_result.get('quality_report', {})
        missing = quality.get('missing', [])
        high_missing_cols = []
        for m in missing:
            if m.get('percent', 0) > 20:
                high_missing_cols.append(m.get('column', ''))

        if high_missing_cols:
            display_cols = high_missing_cols[:3]
            display_str = '、'.join(display_cols)
            if len(high_missing_cols) > 3:
                display_str += f'等{len(high_missing_cols)}个'
            conclusions.append({
                "icon": "⚠️",
                "title": f"{len(high_missing_cols)}个字段缺失率较高",
                "description": f"{display_str} 缺失率超过20%，建议处理"
            })

        # 6. 异常值字段
        outliers = quality.get('outliers', {})
        outlier_cols = list(outliers.keys())
        if outlier_cols:
            display_cols = outlier_cols[:3]
            display_str = '、'.join(display_cols)
            if len(outlier_cols) > 3:
                display_str += f'等{len(outlier_cols)}个'
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

        # 确保所有值都是 JSON 安全类型
        for c in conclusions:
            for k, v in c.items():
                if isinstance(v, (datetime, date, pd.Timestamp)):
                    c[k] = v.strftime('%Y-%m-%d')
                elif v is None:
                    c[k] = ''

        return conclusions

    @staticmethod
    def generate_rule_based_insights(analysis_result: Dict[str, Any]) -> List[str]:
        """基于规则生成洞察"""
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
            insights.append(f"🔗 发现强相关：{top_corr['var1']} 与 {top_corr['var2']} 相关系数 {top_corr['value']}，可考虑特征选择")

        return insights

    # ==================== 图表解读方法 ====================

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
        name = data.get('name', '变量')
        mean = data.get('mean', 0)
        median = data.get('median', 0)
        skew = data.get('skew', 0)
        min_val = data.get('min', 0)
        max_val = data.get('max', 0)

        # 安全转换
        try:
            mean = float(mean)
            median = float(median)
            skew = float(skew)
            min_val = float(min_val)
            max_val = float(max_val)
        except (ValueError, TypeError):
            pass

        if abs(skew) < 0.5:
            distribution = "近似正态分布"
        elif skew > 0:
            distribution = "右偏分布（有较大异常值）"
        else:
            distribution = "左偏分布（有较小异常值）"

        insight = f"**{name}** 集中在 {min_val:.1f}~{max_val:.1f} 之间，呈{distribution}。"
        insight += f"均值 {mean:.1f}，中位数 {median:.1f}。"

        if abs(mean - median) / (max_val - min_val + 0.01) > 0.1:
            insight += " 均值与中位数差异较大，建议使用中位数描述中心趋势。"

        return insight

    @staticmethod
    def _insight_categorical(data: Dict) -> str:
        name = data.get('name', '变量')
        value_counts = data.get('value_counts', {})

        if not value_counts:
            return f"**{name}** 是分类变量。"

        items = list(value_counts.items())
        items.sort(key=lambda x: x[1], reverse=True)

        if not items:
            return f"**{name}** 无有效数据。"

        top_item, top_count = items[0]
        total = sum(value_counts.values())
        top_pct = top_count / total * 100 if total > 0 else 0

        if len(items) == 1:
            return f"**{name}** 只有一个类别：{top_item}。"
        elif len(items) <= 5:
            others = ", ".join([f"{k}({v})" for k, v in items[1:3]])
            return f"**{name}** 中，**{top_item}** 占比最高（{top_pct:.1f}%），其他类别包括 {others}。"
        else:
            return f"**{name}** 有 {len(items)} 个类别，其中 **{top_item}** 占比最高（{top_pct:.1f}%）。"

    @staticmethod
    def _insight_correlation(data: Dict) -> str:
        var1 = data.get('var1', '变量A')
        var2 = data.get('var2', '变量B')
        value = data.get('value', 0)

        try:
            value = float(value)
        except (ValueError, TypeError):
            value = 0

        strength = "强" if abs(value) > 0.7 else "中" if abs(value) > 0.3 else "弱"
        direction = "正" if value > 0 else "负"

        return f"**{var1}** 与 **{var2}** 呈{strength}{direction}相关（r={value:.3f}）。"

    @staticmethod
    def _insight_timeseries(data: Dict) -> str:
        name = data.get('name', '变量')
        trend = data.get('trend', '平稳')
        seasonality = data.get('seasonality', False)

        insight = f"**{name}** 呈现{trend}趋势"
        if seasonality:
            insight += "，具有明显的季节性规律"

        return insight + "。"

    @staticmethod
    def _insight_default(data: Dict) -> str:
        return f"数据显示了 {data.get('name', '变量')} 的分布特征。"