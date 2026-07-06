"""推荐问题生成服务 - 基于分析结果生成个性化推荐问题"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime


class RecommendationService:
    """推荐问题生成器"""

    # 问题场景分类
    SCENES = {
        'report_summary': '报告总览',
        'data_overview': '数据概览',
        'quality': '质量看板',
        'data_validation': '数据核验',
        'pattern_discovery': '规律发现',
        'smart_prediction': '智能预测'
    }

    def __init__(self):
        self.questions = {}

    def generate(self, analysis_result: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """
        生成所有场景的推荐问题

        Args:
            analysis_result: 分析结果 JSON

        Returns:
            {
                "report_summary": [{"icon": "📊", "text": "..."}],
                "data_overview": [...],
                ...
            }
        """
        result = {}

        # 提取关键数据
        data_shape = analysis_result.get('data_shape', {})
        variable_types = analysis_result.get('variable_types', {})
        variable_summaries = analysis_result.get('variable_summaries', {})
        quality_report = analysis_result.get('quality_report', {})
        correlations = analysis_result.get('correlations', {})
        ts_diagnostics = analysis_result.get('time_series_diagnostics', {})
        audit_rules = quality_report.get('audit_rules', {})
        cleaning_suggestions = analysis_result.get('cleaning_suggestions', [])
        model_recommendations = analysis_result.get('model_recommendations', [])

        # 统计变量类型
        continuous_vars = [k for k, v in variable_types.items() if v.get('type') == 'continuous']
        categorical_vars = [k for k, v in variable_types.items()
                           if v.get('type') in ['categorical', 'categorical_numeric', 'ordinal']]
        datetime_vars = [k for k, v in variable_types.items() if v.get('type') == 'datetime']

        # 提取关键值（安全转换）
        rows = data_shape.get('rows', 0)
        cols = data_shape.get('columns', 0)
        quality_score = quality_report.get('overall_score')
        alerts = quality_report.get('alerts', [])
        missing_list = quality_report.get('missing', [])
        outliers = quality_report.get('outliers', {})
        duplicates = quality_report.get('duplicates', {})

        # ✅ 安全转换：duplicates 的值可能是字符串
        dup_count_raw = duplicates.get('count', 0)
        try:
            dup_count = int(dup_count_raw) if dup_count_raw is not None else 0
        except (ValueError, TypeError):
            dup_count = 0

        dup_rate_raw = duplicates.get('percent', 0)
        try:
            dup_rate = float(dup_rate_raw) if dup_rate_raw is not None else 0
        except (ValueError, TypeError):
            dup_rate = 0

        high_correlations = correlations.get('high_correlations', [])

        # 获取质量维度得分
        dimensions = quality_report.get('dimensions', {})
        dim_names = {'completeness': '完整性', 'accuracy': '准确性',
                    'consistency': '一致性', 'timeliness': '及时性',
                    'uniqueness': '唯一性'}
        sorted_dims = sorted([(k, v) for k, v in dimensions.items() if k in dim_names],
                           key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
        worst_dim = (dim_names.get(sorted_dims[0][0], sorted_dims[0][0]), sorted_dims[0][1]) if sorted_dims else None
        best_dim = (dim_names.get(sorted_dims[-1][0], sorted_dims[-1][0]), sorted_dims[-1][1]) if sorted_dims else None

        # ==================== 生成各场景问题 ====================

        # 1. 报告总览
        report_summary = []

        # 数据规模
        report_summary.append({
            "icon": "📊",
            "text": f"数据共 {rows:,} 行 {cols} 列，从整体看数据规模如何影响分析策略？"
        })

        # 质量评分
        if quality_score is not None and isinstance(quality_score, (int, float)):
            grade = '优秀' if quality_score >= 90 else '良好' if quality_score >= 80 else '一般' if quality_score >= 70 else '需关注'
            report_summary.append({
                "icon": "🔍",
                "text": f"综合质量评分 {quality_score:.1f} 分（{grade}），哪些维度拉低了整体得分？"
            })

        # 强相关
        if high_correlations:
            top_corr = high_correlations[0]
            report_summary.append({
                "icon": "🔗",
                "text": f"发现「{top_corr.get('var1', '')}」与「{top_corr.get('var2', '')}」强相关 (r={top_corr.get('value', 0):.2f})，这对业务意味着什么？"
            })

        # 异常值
        if outliers:
            top_field = max(outliers.items(), key=lambda x: x[1].get('percent', 0) if isinstance(x[1].get('percent', 0), (int, float)) else 0)
            top_pct = top_field[1].get('percent', 0)
            if isinstance(top_pct, str):
                try:
                    top_pct = float(top_pct)
                except:
                    top_pct = 0
            report_summary.append({
                "icon": "🚨",
                "text": f"检测到 {len(outliers)} 个字段存在异常值，最高是「{top_field[0]}」({top_pct:.1f}%)，需要检查数据来源吗？"
            })

        # 重复记录
        if dup_count > 0:
            report_summary.append({
                "icon": "📋",
                "text": f"发现 {dup_count} 条重复记录，是否需要去重处理？"
            })

        # 模型推荐
        if model_recommendations:
            top_rec = model_recommendations[0]
            report_summary.append({
                "icon": "🤖",
                "text": f"根据数据特征推荐「{top_rec.get('ml', top_rec.get('task_type', ''))}」做「{top_rec.get('task_type', '预测')}」，这个建议合理吗？"
            })

        result['report_summary'] = report_summary[:5]

        # ==================== 2. 数据概览 ====================
        data_overview = []

        # 连续变量
        if continuous_vars:
            top_cont = continuous_vars[0]
            summary = variable_summaries.get(top_cont, {})
            mean_val = summary.get('mean')
            std_val = summary.get('std')
            if mean_val is not None and std_val is not None and isinstance(mean_val, (int, float)):
                data_overview.append({
                    "icon": "📊",
                    "text": f"数值字段「{top_cont}」的均值 {mean_val:.2f}、标准差 {std_val:.2f}，分布是否正常？"
                })

        # 分类变量
        if categorical_vars:
            top_cat = categorical_vars[0]
            n_unique = variable_types.get(top_cat, {}).get('n_unique', 0)
            if n_unique and isinstance(n_unique, (int, float)) and n_unique > 0:
                data_overview.append({
                    "icon": "📋",
                    "text": f"分类字段「{top_cat}」有 {int(n_unique)} 个类别，需要关注哪些高频类别？"
                })

        # 日期字段
        if datetime_vars:
            date_col = datetime_vars[0]
            summary = variable_summaries.get(date_col, {})
            min_date = summary.get('min_date')
            max_date = summary.get('max_date')
            if min_date and max_date:
                data_overview.append({
                    "icon": "📅",
                    "text": f"日期范围从 {min_date} 到 {max_date}，适合做时间序列分析吗？"
                })

        # 缺失值
        if missing_list:
            top_missing = max(missing_list, key=lambda x: x.get('percent', 0) if isinstance(x.get('percent', 0), (int, float)) else 0)
            top_pct = top_missing.get('percent', 0)
            if isinstance(top_pct, str):
                try:
                    top_pct = float(top_pct)
                except:
                    top_pct = 0
            if top_pct > 5:
                data_overview.append({
                    "icon": "⚠️",
                    "text": f"「{top_missing.get('column', '')}」缺失率 {top_pct:.1f}%，建议填充还是删除？"
                })

        # 查询类推荐
        data_overview.append({"icon": "🔍", "text": "查询最近7天的数据"})
        data_overview.append({"icon": "📊", "text": "统计各分类的分布情况"})
        data_overview.append({"icon": "📈", "text": "查询销售额最高的前10条记录"})
        data_overview.append({"icon": "🔍", "text": "分析各月的变化趋势"})

        result['data_overview'] = data_overview[:6]

        # ==================== 3. 质量看板 ====================
        quality_questions = []

        if quality_score is not None and isinstance(quality_score, (int, float)):
            quality_questions.append({
                "icon": "📊",
                "text": f"综合质量评分 {quality_score:.1f} 分，各维度得分如何解读？"
            })

        if worst_dim:
            score_val = worst_dim[1] if isinstance(worst_dim[1], (int, float)) else 0
            quality_questions.append({
                "icon": "🔍",
                "text": f"「{worst_dim[0]}」得分 {score_val:.1f} 分，是什么原因导致偏低？"
            })

        if best_dim and isinstance(best_dim[1], (int, float)) and best_dim[1] >= 90:
            quality_questions.append({
                "icon": "✅",
                "text": f"「{best_dim[0]}」得分 {best_dim[1]:.1f} 分，做得好的原因是什么？"
            })

        # 告警统计
        warn_count = len([a for a in alerts if a.get('level') == 'warning'])
        err_count = len([a for a in alerts if a.get('level') == 'error'])
        if warn_count > 0 or err_count > 0:
            quality_questions.append({
                "icon": "🚨",
                "text": f"发现 {warn_count} 个警告、{err_count} 个错误，最严重的问题是什么？"
            })

        result['quality'] = quality_questions[:5]

        # ==================== 4. 数据核验 ====================
        validation_questions = []

        # 勾稽规则
        arithmetic_rules = audit_rules.get('arithmetic_rules', [])
        rules_with_violation = [r for r in arithmetic_rules if r.get('violation_count', 0) > 0]
        if rules_with_violation:
            top_rule = max(rules_with_violation, key=lambda x: x.get('violation_count', 0) if isinstance(x.get('violation_count', 0), (int, float)) else 0)
            vc = top_rule.get('violation_count', 0)
            if isinstance(vc, str):
                try:
                    vc = int(vc)
                except:
                    vc = 0
            validation_questions.append({
                "icon": "🔗",
                "text": f"规则「{top_rule.get('rule', '')[:30]}...」有 {vc} 条违反记录，可能是数据录入错误吗？"
            })

        # 异常值
        if outliers:
            top_out = max(outliers.items(), key=lambda x: x[1].get('percent', 0) if isinstance(x[1].get('percent', 0), (int, float)) else 0)
            out_count = top_out[1].get('count', 0)
            out_pct = top_out[1].get('percent', 0)
            if isinstance(out_count, str):
                try:
                    out_count = int(out_count)
                except:
                    out_count = 0
            if isinstance(out_pct, str):
                try:
                    out_pct = float(out_pct)
                except:
                    out_pct = 0
            validation_questions.append({
                "icon": "🚨",
                "text": f"「{top_out[0]}」有 {out_count} 个异常值（{out_pct:.1f}%），如何处理这些异常？"
            })

        # 缺失值
        if missing_list:
            top_miss = max(missing_list, key=lambda x: x.get('percent', 0) if isinstance(x.get('percent', 0), (int, float)) else 0)
            miss_count = top_miss.get('count', 0)
            miss_pct = top_miss.get('percent', 0)
            if isinstance(miss_count, str):
                try:
                    miss_count = int(miss_count)
                except:
                    miss_count = 0
            if isinstance(miss_pct, str):
                try:
                    miss_pct = float(miss_pct)
                except:
                    miss_pct = 0
            validation_questions.append({
                "icon": "⚠️",
                "text": f"「{top_miss.get('column', '')}」缺失 {miss_count} 条（{miss_pct:.1f}%），建议删除还是填充？"
            })

        # 重复记录
        if dup_count > 0:
            validation_questions.append({
                "icon": "📋",
                "text": f"发现 {dup_count} 条重复记录（占比 {dup_rate:.1f}%），删除前需要先确认哪些字段重复吗？"
            })

        # 清洗建议
        if cleaning_suggestions:
            validation_questions.append({
                "icon": "🧹",
                "text": f"清洗建议：{cleaning_suggestions[0][:40]}...，按什么优先级执行？"
            })

        result['data_validation'] = validation_questions[:5]

        # ==================== 5. 规律发现 ====================
        pattern_questions = []

        if high_correlations:
            top_corr = high_correlations[0]
            pattern_questions.append({
                "icon": "🔗",
                "text": f"「{top_corr.get('var1', '')}」与「{top_corr.get('var2', '')}」相关系数 {top_corr.get('value', 0):.2f}，它们之间存在因果关系吗？"
            })
            if len(high_correlations) >= 2:
                second = high_correlations[1]
                pattern_questions.append({
                    "icon": "🔗",
                    "text": f"「{second.get('var1', '')}」与「{second.get('var2', '')}」也强相关 (r={second.get('value', 0):.2f})，这组关联有什么业务含义？"
                })

        # 时间序列
        ts_vars = [k for k, v in ts_diagnostics.items() if v.get('has_autocorrelation')]
        if ts_vars:
            ts_var = ts_vars[0]
            p_value = ts_diagnostics.get(ts_var, {}).get('lb_p', 0.001)
            if isinstance(p_value, str):
                try:
                    p_value = float(p_value)
                except:
                    p_value = 0.001
            pattern_questions.append({
                "icon": "📈",
                "text": f"「{ts_var}」检测到自相关（p={p_value:.4f}），适合用自身历史值预测未来吗？"
            })

        non_stationary = [k for k, v in ts_diagnostics.items() if not v.get('is_stationary') and v.get('has_autocorrelation')]
        if non_stationary:
            pattern_questions.append({
                "icon": "📈",
                "text": f"「{non_stationary[0]}」非平稳且存在趋势，建议先做差分处理吗？"
            })

        # 时间序列中季节性的
        seasonal_vars = [k for k, v in ts_diagnostics.items() if v.get('has_seasonality')]
        if seasonal_vars:
            pattern_questions.append({
                "icon": "📅",
                "text": f"「{seasonal_vars[0]}」检测到季节性规律，适合用 SARIMA 建模吗？"
            })

        result['pattern_discovery'] = pattern_questions[:5]

        # ==================== 6. 智能预测 ====================
        prediction_questions = []

        # 基于模型推荐生成预测类问题
        if model_recommendations:
            for rec in model_recommendations[:2]:
                target = rec.get('target_column', '')
                task_type = rec.get('task_type', '预测')
                if target:
                    prediction_questions.append({
                        "icon": "🔮",
                        "text": f"预测「{target}」的未来趋势，用什么模型比较好？"
                    })
                    prediction_questions.append({
                        "icon": "🤖",
                        "text": f"基于当前数据，预测下个月的「{target}」"
                    })
                    break

        # 如果有时间序列，增加预测建议
        if ts_vars:
            prediction_questions.append({
                "icon": "📈",
                "text": f"对「{ts_vars[0]}」做未来7天的预测"
            })

        # 通用预测问题
        if not prediction_questions:
            prediction_questions = [
                {"icon": "🔮", "text": "根据当前数据，有什么可以预测的业务指标？"},
                {"icon": "🤖", "text": "推荐适合的预测模型并解释原因"},
                {"icon": "📊", "text": "对主要指标做未来趋势预测"}
            ]

        result['smart_prediction'] = prediction_questions[:5]

        return result

    def format_for_frontend(self, questions: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        格式化输出，确保每个问题包含 icon 和 text

        Args:
            questions: 生成的问题字典

        Returns:
            格式化后的问题字典
        """
        result = {}
        for scene, q_list in questions.items():
            formatted = []
            for q in q_list:
                if isinstance(q, dict):
                    formatted.append({
                        "icon": q.get("icon", "💬"),
                        "text": q.get("text", "")
                    })
                elif isinstance(q, str):
                    formatted.append({"icon": "💬", "text": q})
            result[scene] = formatted
        return result