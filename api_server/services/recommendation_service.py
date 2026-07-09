"""
推荐问题生成服务 - 基于分析结果生成个性化推荐问题
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime


class RecommendationService:
    """推荐问题生成器 - 6大场景 × 26+子项全覆盖"""

    # 场景名称映射
    SCENE_NAMES = {
        'report_summary': '报告总览',
        'data_overview': '数据概览',
        'quality': '质量看板',
        'data_validation': '数据核验',
        'pattern_discovery': '规律发现',
        'smart_prediction': '智能预测'
    }

    # 子项名称映射
    SUB_SCENE_NAMES = {
        # report_summary
        'overview': '数据概览',
        'conclusions': '核心结论',
        'insights': '业务洞察',
        # data_overview
        'distribution': '分布特征',
        'categorical': '分类变量',
        'continuous': '连续变量',
        'datetime': '日期时间',
        'missing': '缺失值',
        'identifier': '标识符',
        'text': '文本字段',
        'natural_query': '自然语言查数',
        'generate_sql': '生成SQL',
        # quality
        'overall': '综合评分',
        'completeness': '完整性',
        'accuracy': '准确性',
        'consistency': '一致性',
        'uniqueness': '唯一性',
        # data_validation
        'audit_rules': '勾稽规则',
        'outliers': '异常值',
        'missing_detail': '缺失详情',
        'duplicates': '重复记录',
        'cleaning': '清洗建议',
        # pattern_discovery
        'correlation': '相关性分析',
        'timeseries': '时间序列',
        'trend': '趋势分析',
        'categorical_pattern': '分类关联',
        'distribution_insight': '分布洞察',
        # smart_prediction
        'model_recommend': '模型推荐',
        'target_select': '目标选择',
        'feature_select': '特征选择',
        'forecast': '未来预测'
    }

    def __init__(self):
        self.questions = {}
        self._analysis_result = None

    # ============================================================
    # 主入口：生成所有推荐问题
    # ============================================================

    def generate(self, analysis_result: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """
        生成所有场景的推荐问题

        Args:
            analysis_result: 分析结果 JSON

        Returns:
            {
                "report_summary": {
                    "overview": [{"icon": "📊", "text": "...", "prompt": "...", "dataKey": "..."}],
                    ...
                },
                ...
            }
        """
        self._analysis_result = analysis_result

        result = {}

        # 1. 报告总览
        result["report_summary"] = self._generate_report_summary_questions()

        # 2. 数据概览
        result["data_overview"] = self._generate_data_overview_questions()

        # 3. 质量看板
        result["quality"] = self._generate_quality_questions()

        # 4. 数据核验
        result["data_validation"] = self._generate_data_validation_questions()

        # 5. 规律发现
        result["pattern_discovery"] = self._generate_pattern_discovery_questions()

        # 6. 智能预测
        result["smart_prediction"] = self._generate_smart_prediction_questions()

        return result

    # ============================================================
    # 安全类型转换辅助方法
    # ============================================================

    def _safe_int(self, value: Any, default: int = 0) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _format_num(self, value: Any, decimals: int = 2) -> str:
        if value is None:
            return '—'
        try:
            v = float(value)
            if abs(v) >= 1000000:
                return f"{v / 10000:.1f}万"
            elif abs(v) >= 10000:
                return f"{v / 10000:.2f}万"
            elif abs(v) >= 1000:
                return f"{v:.1f}"
            else:
                return f"{v:.{decimals}f}"
        except (ValueError, TypeError):
            return str(value)

    def _get_high_correlations(self, threshold: float = 0.7) -> List[Dict]:
        """获取强相关对 - 从 analysis_result 中读取"""
        if not self._analysis_result:
            return []

        correlations = self._analysis_result.get('correlations', {})
        high_corrs = correlations.get('high_correlations', [])
        return high_corrs

    def _get_model_recommendations(self) -> List[Dict]:
        """获取模型推荐 - 从 analysis_result 中读取"""
        if not self._analysis_result:
            return []
        return self._analysis_result.get('model_recommendations', [])

    # ============================================================
    # 1. 报告总览
    # ============================================================

    def _generate_report_summary_questions(self) -> Dict[str, List[Dict]]:
        """生成报告总览推荐问题"""
        questions = {
            "overview": [],
            "conclusions": [],
            "insights": []
        }

        if not self._analysis_result:
            return questions

        data_shape = self._analysis_result.get('data_shape', {})
        rows = data_shape.get('rows', 0)
        cols = data_shape.get('columns', 0)
        variable_types = self._analysis_result.get('variable_types', {})
        summaries = self._analysis_result.get('variable_summaries', {})
        quality = self._analysis_result.get('quality_report', {})

        # ---- overview ----
        questions["overview"].append({
            "icon": "📊",
            "text": f"数据共 {rows:,} 行 {cols} 列，整体数据规模如何？",
            "prompt": f"请从整体上总结数据的核心特征和分布规律，包括数据规模、变量类型、关键统计指标等",
            "dataKey": "report_summary.overview"
        })

        continuous_vars = [col for col, info in variable_types.items() if info.get('type') == 'continuous']
        if continuous_vars:
            top_var = continuous_vars[0]
            info = summaries.get(top_var, {})
            mean_val = info.get('mean')
            median_val = info.get('median')
            if mean_val is not None:
                questions["overview"].append({
                    "icon": "📊",
                    "text": f"「{top_var}」的均值 {self._format_num(mean_val)}，中位数 {self._format_num(median_val)}，分布是否正常？",
                    "prompt": f"请结合数据源中的实际数据，回答：「{top_var}」的均值 {self._format_num(mean_val)}，中位数 {self._format_num(median_val)}，分布是否正常？",
                    "dataKey": "report_summary.overview"
                })

        # ---- conclusions ----
        high_corrs = self._get_high_correlations()
        if high_corrs:
            top = high_corrs[0]
            questions["conclusions"].append({
                "icon": "🔗",
                "text": f"「{top.get('var1', '')}」与「{top.get('var2', '')}」强相关 (r={top.get('value', 0):.2f})，建议重点关注",
                "prompt": f"请结合数据源中的实际数据，回答：「{top.get('var1', '')}」与「{top.get('var2', '')}」强相关 (r={top.get('value', 0):.2f})，建议重点关注",
                "dataKey": "report_summary.conclusions"
            })

        outliers = quality.get('outliers', {})
        if outliers:
            top_field = max(outliers.items(), key=lambda x: self._safe_float(x[1].get('percent', 0)))
            questions["conclusions"].append({
                "icon": "🚨",
                "text": f"「{top_field[0]}」存在 {self._safe_int(top_field[1].get('count', 0))} 个异常值 ({self._safe_float(top_field[1].get('percent', 0)):.1f}%)，建议检查",
                "prompt": f"请结合数据源中的实际数据，回答：「{top_field[0]}」存在 {self._safe_int(top_field[1].get('count', 0))} 个异常值 ({self._safe_float(top_field[1].get('percent', 0)):.1f}%)，建议检查",
                "dataKey": "report_summary.conclusions"
            })

        # ---- insights ----
        insights_list = []

        # 时间序列洞察
        ts_diag = self._analysis_result.get('time_series_diagnostics', {})
        has_auto = any(v.get('has_autocorrelation') for v in ts_diag.values())
        if has_auto:
            auto_vars = [k for k, v in ts_diag.items() if v.get('has_autocorrelation')]
            if auto_vars:
                insights_list.append({
                    "icon": "📈",
                    "text": f"「{auto_vars[0]}」检测到自相关性，适合进行时间序列预测",
                    "prompt": f"请结合数据源中的实际数据，回答：「{auto_vars[0]}」检测到自相关性，适合进行时间序列预测",
                    "dataKey": "report_summary.insights"
                })

        # 数据规模洞察
        if rows > 100000:
            insights_list.append({
                "icon": "📊",
                "text": f"数据量较大（{rows:,} 行），建议采样分析",
                "prompt": f"请结合数据源中的实际数据，回答：数据量较大（{rows:,} 行），建议采样分析",
                "dataKey": "report_summary.insights"
            })
        elif rows < 100:
            insights_list.append({
                "icon": "📊",
                "text": f"数据量较小（{rows} 行），分析结果可能存在偏差",
                "prompt": f"请结合数据源中的实际数据，回答：数据量较小（{rows} 行），分析结果可能存在偏差",
                "dataKey": "report_summary.insights"
            })

        # 强相关数量洞察
        if len(high_corrs) > 10:
            insights_list.append({
                "icon": "🔗",
                "text": f"发现 {len(high_corrs)} 对强相关关系，可能存在多重共线性",
                "prompt": f"请结合数据源中的实际数据，回答：发现 {len(high_corrs)} 对强相关关系，可能存在多重共线性",
                "dataKey": "report_summary.insights"
            })

        # 缺失值洞察
        missing_list = quality.get('missing', [])
        if missing_list:
            high_missing = [m for m in missing_list if self._safe_float(m.get('percent', 0)) > 20]
            if high_missing:
                insights_list.append({
                    "icon": "⚠️",
                    "text": f"发现 {len(high_missing)} 个字段缺失率超过 20%，建议处理",
                    "prompt": f"请结合数据源中的实际数据，回答：发现 {len(high_missing)} 个字段缺失率超过 20%，建议处理",
                    "dataKey": "report_summary.insights"
                })

        questions["insights"] = insights_list[:3]

        return questions

    # ============================================================
    # 2. 数据概览
    # ============================================================

    def _generate_data_overview_questions(self) -> Dict[str, List[Dict]]:
        """生成数据概览推荐问题"""
        questions = {
            "distribution": [],
            "categorical": [],
            "continuous": [],
            "datetime": [],
            "missing": [],
            "identifier": [],
            "text": [],
            "natural_query": [],
            "generate_sql": []
        }

        if not self._analysis_result:
            return questions

        variable_types = self._analysis_result.get('variable_types', {})
        summaries = self._analysis_result.get('variable_summaries', {})
        quality = self._analysis_result.get('quality_report', {})
        data_shape = self._analysis_result.get('data_shape', {})
        rows = data_shape.get('rows', 0)

        continuous_vars = [col for col, info in variable_types.items() if info.get('type') == 'continuous']
        categorical_vars = [col for col, info in variable_types.items()
                           if info.get('type') in ['categorical', 'categorical_numeric', 'ordinal']]
        datetime_vars = [col for col, info in variable_types.items() if info.get('type') == 'datetime']

        # ---- distribution ----
        for col in continuous_vars[:3]:
            info = summaries.get(col, {})
            mean_val = info.get('mean')
            median_val = info.get('median')
            std_val = info.get('std')
            skew_val = info.get('skew')
            if mean_val is not None:
                if skew_val and abs(self._safe_float(skew_val)) > 2:
                    text = f"「{col}」偏度 {self._safe_float(skew_val):.2f}，分布严重偏斜，建议使用中位数描述"
                    prompt = f"请结合数据源中的实际数据，回答：{text}"
                    questions["distribution"].append({
                        "icon": "📊",
                        "text": text,
                        "prompt": prompt,
                        "dataKey": "data_overview.distribution"
                    })
                else:
                    text = f"「{col}」均值 {self._format_num(mean_val)}，中位数 {self._format_num(median_val)}，标准差 {self._format_num(std_val)}"
                    prompt = f"请结合数据源中的实际数据，回答：{text}"
                    questions["distribution"].append({
                        "icon": "📊",
                        "text": text,
                        "prompt": prompt,
                        "dataKey": "data_overview.distribution"
                    })

        # ---- categorical ----
        for col in categorical_vars[:3]:
            info = summaries.get(col, {})
            n_unique = info.get('n_unique', 0)
            mode_pct = info.get('mode_pct', 0)
            mode_val = info.get('mode')
            if n_unique and self._safe_int(n_unique) > 0:
                mode_pct_float = self._safe_float(mode_pct)
                if mode_pct_float > 80:
                    text = f"「{col}」中「{mode_val}」占比 {mode_pct_float:.1f}%，存在严重不平衡"
                    prompt = f"请结合数据源中的实际数据，回答：{text}"
                    questions["categorical"].append({
                        "icon": "🏷️",
                        "text": text,
                        "prompt": prompt,
                        "dataKey": "data_overview.categorical"
                    })
                else:
                    text = f"「{col}」共 {self._safe_int(n_unique)} 个类别，占比最高的类别为「{mode_val}」({mode_pct_float:.1f}%)"
                    prompt = f"请结合数据源中的实际数据，回答：{text}"
                    questions["categorical"].append({
                        "icon": "🏷️",
                        "text": text,
                        "prompt": prompt,
                        "dataKey": "data_overview.categorical"
                    })

        # ---- continuous ----
        for col in continuous_vars[:2]:
            info = summaries.get(col, {})
            min_val = info.get('min')
            max_val = info.get('max')
            mean_val = info.get('mean')
            if min_val is not None:
                text = f"「{col}」范围 [{self._format_num(min_val)}, {self._format_num(max_val)}]，均值 {self._format_num(mean_val)}"
                prompt = f"请结合数据源中的实际数据，回答：{text}"
                questions["continuous"].append({
                    "icon": "📈",
                    "text": text,
                    "prompt": prompt,
                    "dataKey": "data_overview.continuous"
                })

        # ---- datetime ----
        for col in datetime_vars[:1]:
            info = summaries.get(col, {})
            min_date = info.get('min_date')
            max_date = info.get('max_date')
            if min_date and max_date:
                text = f"「{col}」时间范围 {min_date} 至 {max_date}"
                prompt = f"请结合数据源中的实际数据，回答：{text}"
                questions["datetime"].append({
                    "icon": "📅",
                    "text": text,
                    "prompt": prompt,
                    "dataKey": "data_overview.datetime"
                })

        # ---- missing ----
        missing_list = quality.get('missing', [])
        for item in missing_list[:2]:
            pct = self._safe_float(item.get('percent', 0))
            if pct > 5:
                text = f"「{item.get('column', '')}」缺失率 {pct:.1f}%，建议处理"
                prompt = f"请结合数据源中的实际数据，回答：{text}"
                questions["missing"].append({
                    "icon": "⚠️",
                    "text": text,
                    "prompt": prompt,
                    "dataKey": "data_overview.missing"
                })

        # ---- identifier ----
        identifier_vars = [col for col, info in variable_types.items() if info.get('type') == 'identifier']
        for col in identifier_vars[:1]:
            info = summaries.get(col, {})
            n_unique = info.get('n_unique', 0)
            text = f"「{col}」有 {self._safe_int(n_unique)} 个唯一值，可作为主键"
            prompt = f"请结合数据源中的实际数据，回答：{text}"
            questions["identifier"].append({
                "icon": "🆔",
                "text": text,
                "prompt": prompt,
                "dataKey": "data_overview.identifier"
            })

        # ---- text ----
        text_vars = [col for col, info in variable_types.items() if info.get('type') == 'text']
        for col in text_vars[:1]:
            info = summaries.get(col, {})
            n_unique = info.get('n_unique', 0)
            text = f"「{col}」有 {self._safe_int(n_unique)} 个唯一值，属于文本类型"
            prompt = f"请结合数据源中的实际数据，回答：{text}"
            questions["text"].append({
                "icon": "📝",
                "text": text,
                "prompt": prompt,
                "dataKey": "data_overview.text"
            })

        # ---- natural_query ----
        natural_list = []

        if datetime_vars:
            date_col = datetime_vars[0]
            natural_list.append({
                "icon": "🔍",
                "text": f"查询「{date_col}」在最近 7 天的数据",
                "prompt": f"请查询「{date_col}」在最近 7 天的数据",
                "dataKey": "data_overview.natural_query"
            })
            natural_list.append({
                "icon": "🔍",
                "text": f"查询「{date_col}」在指定时间范围内的数据",
                "prompt": f"请查询「{date_col}」在指定时间范围内的数据",
                "dataKey": "data_overview.natural_query"
            })

        if continuous_vars:
            top_num = continuous_vars[0]
            natural_list.append({
                "icon": "🔍",
                "text": f"查询「{top_num}」大于平均值的记录",
                "prompt": f"请查询「{top_num}」大于平均值的记录",
                "dataKey": "data_overview.natural_query"
            })

        if categorical_vars:
            top_cat = categorical_vars[0]
            natural_list.append({
                "icon": "🔍",
                "text": f"按「{top_cat}」分组统计记录数",
                "prompt": f"请按「{top_cat}」分组统计记录数",
                "dataKey": "data_overview.natural_query"
            })

        # 去重
        seen = set()
        for q in natural_list:
            key = q.get("text", "")
            if key not in seen:
                seen.add(key)
                questions["natural_query"].append(q)
        questions["natural_query"] = questions["natural_query"][:5]

        # ---- generate_sql ----
        sql_list = []

        if datetime_vars:
            date_col = datetime_vars[0]
            sql_list.append({
                "icon": "📝",
                "text": f"生成按「{date_col}」分组统计的 SQL",
                "prompt": f"请生成按「{date_col}」分组统计的 SQL 语句",
                "dataKey": "data_overview.generate_sql"
            })
            sql_list.append({
                "icon": "📝",
                "text": f"生成查询「{date_col}」在指定时间范围的 SQL",
                "prompt": f"请生成查询「{date_col}」在指定时间范围的 SQL 语句",
                "dataKey": "data_overview.generate_sql"
            })

        if continuous_vars:
            top_num = continuous_vars[0]
            sql_list.append({
                "icon": "📝",
                "text": f"生成查询「{top_num}」大于指定值的 SQL",
                "prompt": f"请生成查询「{top_num}」大于指定值的 SQL 语句",
                "dataKey": "data_overview.generate_sql"
            })

        if categorical_vars:
            top_cat = categorical_vars[0]
            sql_list.append({
                "icon": "📝",
                "text": f"生成按「{top_cat}」分组统计的 SQL",
                "prompt": f"请生成按「{top_cat}」分组统计的 SQL 语句",
                "dataKey": "data_overview.generate_sql"
            })

        # 去重
        seen = set()
        for q in sql_list:
            key = q.get("text", "")
            if key not in seen:
                seen.add(key)
                questions["generate_sql"].append(q)
        questions["generate_sql"] = questions["generate_sql"][:5]

        return questions

    # ============================================================
    # 3. 质量看板
    # ============================================================

    def _generate_quality_questions(self) -> Dict[str, List[Dict]]:
        """生成质量看板推荐问题"""
        questions = {
            "overall": [],
            "completeness": [],
            "accuracy": [],
            "consistency": [],
            "uniqueness": []
        }

        if not self._analysis_result:
            return questions

        quality = self._analysis_result.get('quality_report', {})
        variable_types = self._analysis_result.get('variable_types', {})
        data_shape = self._analysis_result.get('data_shape', {})
        rows = data_shape.get('rows', 0)
        cols = len(variable_types)

        # ---- overall ----
        overall_score = quality.get('overall_score')
        if overall_score is not None:
            score = self._safe_float(overall_score)
            grade = '优秀' if score >= 90 else '良好' if score >= 80 else '一般' if score >= 70 else '需关注'
            text = f"综合质量评分 {score:.1f} 分（{grade}），各维度得分是否均衡？"
            prompt = f"请结合数据源中的实际数据，回答：{text}"
            questions["overall"].append({
                "icon": "⭐",
                "text": text,
                "prompt": prompt,
                "dataKey": "quality.overall"
            })
        else:
            # 如果没有 overall_score，生成一条通用说明
            questions["overall"].append({
                "icon": "⭐",
                "text": "当前数据质量评分暂未计算，建议先执行质量评估",
                "prompt": "请结合数据源中的实际数据，回答：当前数据质量评分暂未计算，建议先执行质量评估",
                "dataKey": "quality.overall"
            })

        # ---- completeness ----
        missing_list = quality.get('missing', [])
        if missing_list:
            total_missing = sum(self._safe_int(m.get('count', 0)) for m in missing_list)
            overall_missing_rate = (total_missing / (rows * cols)) * 100 if rows > 0 else 0
            text = f"整体缺失率 {overall_missing_rate:.1f}%，共 {len(missing_list)} 个字段存在缺失"
            prompt = f"请结合数据源中的实际数据，回答：{text}"
            questions["completeness"].append({
                "icon": "📊",
                "text": text,
                "prompt": prompt,
                "dataKey": "quality.completeness"
            })
            top = max(missing_list, key=lambda x: self._safe_float(x.get('percent', 0)))
            top_pct = self._safe_float(top.get('percent', 0))
            if top_pct > 20:
                text = f"「{top.get('column', '')}」缺失率 {top_pct:.1f}%，建议优先处理"
                prompt = f"请结合数据源中的实际数据，回答：{text}"
                questions["completeness"].append({
                    "icon": "⚠️",
                    "text": text,
                    "prompt": prompt,
                    "dataKey": "quality.completeness"
                })
        else:
            questions["completeness"].append({
                "icon": "✅",
                "text": "未发现缺失值，数据完整性良好",
                "prompt": "请结合数据源中的实际数据，回答：未发现缺失值，数据完整性良好",
                "dataKey": "quality.completeness"
            })

        # ---- accuracy ----
        outliers = quality.get('outliers', {})
        if outliers:
            total_outliers = sum(self._safe_int(info.get('count', 0)) for info in outliers.values())
            text = f"共发现 {len(outliers)} 个字段存在异常值，总计 {total_outliers} 条"
            prompt = f"请结合数据源中的实际数据，回答：{text}"
            questions["accuracy"].append({
                "icon": "🎯",
                "text": text,
                "prompt": prompt,
                "dataKey": "quality.accuracy"
            })
            top = max(outliers.items(), key=lambda x: self._safe_float(x[1].get('percent', 0)))
            top_pct = self._safe_float(top[1].get('percent', 0))
            if top_pct > 5:
                text = f"「{top[0]}」异常值 {self._safe_int(top[1].get('count', 0))} 条 ({top_pct:.1f}%)"
                prompt = f"请结合数据源中的实际数据，回答：{text}"
                questions["accuracy"].append({
                    "icon": "🚨",
                    "text": text,
                    "prompt": prompt,
                    "dataKey": "quality.accuracy"
                })
        else:
            questions["accuracy"].append({
                "icon": "✅",
                "text": "未发现异常值，数据准确性良好",
                "prompt": "请结合数据源中的实际数据，回答：未发现异常值，数据准确性良好",
                "dataKey": "quality.accuracy"
            })

        # ---- consistency ----
        audit_rules = quality.get('audit_rules', {})
        arithmetic_count = len(audit_rules.get('arithmetic_rules', []))
        if arithmetic_count > 0:
            text = f"发现 {arithmetic_count} 条数值关系规则，数据一致性如何？"
            prompt = f"请结合数据源中的实际数据，回答：{text}"
            questions["consistency"].append({
                "icon": "🔗",
                "text": text,
                "prompt": prompt,
                "dataKey": "quality.consistency"
            })
        else:
            questions["consistency"].append({
                "icon": "✅",
                "text": "未发现数值关系规则，数据一致性良好",
                "prompt": "请结合数据源中的实际数据，回答：未发现数值关系规则，数据一致性良好",
                "dataKey": "quality.consistency"
            })

        # ---- uniqueness ----
        dup_info = quality.get('duplicates', {})
        dup_count = self._safe_int(dup_info.get('count', 0))
        dup_pct = self._safe_float(dup_info.get('percent', 0))
        if dup_count > 0:
            text = f"发现 {dup_count} 条重复记录，占比 {dup_pct:.1f}%"
            prompt = f"请结合数据源中的实际数据，回答：{text}"
            questions["uniqueness"].append({
                "icon": "🆔",
                "text": text,
                "prompt": prompt,
                "dataKey": "quality.uniqueness"
            })
        else:
            text = "未发现重复记录，数据唯一性良好"
            prompt = "请结合数据源中的实际数据，回答：未发现重复记录，数据唯一性良好"
            questions["uniqueness"].append({
                "icon": "✅",
                "text": text,
                "prompt": prompt,
                "dataKey": "quality.uniqueness"
            })

        return questions

    # ============================================================
    # 4. 数据核验
    # ============================================================

    def _generate_data_validation_questions(self) -> Dict[str, List[Dict]]:
        """生成数据核验推荐问题"""
        questions = {
            "audit_rules": [],
            "outliers": [],
            "missing_detail": [],
            "duplicates": [],
            "cleaning": []
        }

        if not self._analysis_result:
            return questions

        quality = self._analysis_result.get('quality_report', {})
        audit_rules = quality.get('audit_rules', {})
        arithmetic = audit_rules.get('arithmetic_rules', [])
        functional = audit_rules.get('functional_dependencies', [])
        temporal = audit_rules.get('temporal_rules', [])
        total = len(arithmetic) + len(functional) + len(temporal)

        # ---- audit_rules ----
        if total > 0:
            text = f"共发现 {total} 条勾稽规则（数值关系 {len(arithmetic)} 条、函数依赖 {len(functional)} 条、时序约束 {len(temporal)} 条）"
            prompt = f"请结合数据源中的实际数据，回答：{text}"
            questions["audit_rules"].append({
                "icon": "🔗",
                "text": text,
                "prompt": prompt,
                "dataKey": "data_validation.audit_rules"
            })
            violations = [r for r in arithmetic if self._safe_int(r.get('violation_count', 0)) > 0]
            if violations:
                top = max(violations, key=lambda x: self._safe_int(x.get('violation_count', 0)))
                text = f"规则「{top.get('rule', '')[:30]}...」有 {self._safe_int(top.get('violation_count', 0))} 条违反记录"
                prompt = f"请结合数据源中的实际数据，回答：{text}"
                questions["audit_rules"].append({
                    "icon": "⚠️",
                    "text": text,
                    "prompt": prompt,
                    "dataKey": "data_validation.audit_rules"
                })
        else:
            questions["audit_rules"].append({
                "icon": "✅",
                "text": "未发现勾稽规则，数据一致性良好",
                "prompt": "请结合数据源中的实际数据，回答：未发现勾稽规则，数据一致性良好",
                "dataKey": "data_validation.audit_rules"
            })

        # ---- outliers ----
        outliers = quality.get('outliers', {})
        if outliers:
            for field, info in list(outliers.items())[:2]:
                text = f"「{field}」有 {self._safe_int(info.get('count', 0))} 个异常值 ({self._safe_float(info.get('percent', 0)):.1f}%)"
                prompt = f"请结合数据源中的实际数据，回答：{text}"
                questions["outliers"].append({
                    "icon": "🚨",
                    "text": text,
                    "prompt": prompt,
                    "dataKey": "data_validation.outliers"
                })
        else:
            questions["outliers"].append({
                "icon": "✅",
                "text": "未发现异常值",
                "prompt": "请结合数据源中的实际数据，回答：未发现异常值",
                "dataKey": "data_validation.outliers"
            })

        # ---- missing_detail ----
        missing_list = quality.get('missing', [])
        if missing_list:
            for item in missing_list[:2]:
                pct = self._safe_float(item.get('percent', 0))
                if pct > 5:
                    text = f"「{item.get('column', '')}」缺失 {self._safe_int(item.get('count', 0))} 条 ({pct:.1f}%)"
                    prompt = f"请结合数据源中的实际数据，回答：{text}"
                    questions["missing_detail"].append({
                        "icon": "⚠️",
                        "text": text,
                        "prompt": prompt,
                        "dataKey": "data_validation.missing"
                    })
        else:
            questions["missing_detail"].append({
                "icon": "✅",
                "text": "未发现缺失值",
                "prompt": "请结合数据源中的实际数据，回答：未发现缺失值",
                "dataKey": "data_validation.missing"
            })

        # ---- duplicates ----
        dup_info = quality.get('duplicates', {})
        dup_count = self._safe_int(dup_info.get('count', 0))
        if dup_count > 0:
            text = f"发现 {dup_count} 条重复记录，建议去重"
            prompt = f"请结合数据源中的实际数据，回答：{text}"
            questions["duplicates"].append({
                "icon": "📋",
                "text": text,
                "prompt": prompt,
                "dataKey": "data_validation.duplicates"
            })
        else:
            questions["duplicates"].append({
                "icon": "✅",
                "text": "未发现重复记录",
                "prompt": "请结合数据源中的实际数据，回答：未发现重复记录",
                "dataKey": "data_validation.duplicates"
            })

        # ---- cleaning ----
        cleaning = self._analysis_result.get('cleaning_suggestions', [])
        if cleaning:
            for suggestion in cleaning[:3]:
                text = suggestion
                prompt = f"请结合数据源中的实际数据，回答：{text}"
                questions["cleaning"].append({
                    "icon": "🧹",
                    "text": text,
                    "prompt": prompt,
                    "dataKey": "data_validation.cleaning"
                })
        else:
            questions["cleaning"].append({
                "icon": "✅",
                "text": "数据质量良好，无明显清洗建议",
                "prompt": "请结合数据源中的实际数据，回答：数据质量良好，无明显清洗建议",
                "dataKey": "data_validation.cleaning"
            })

        return questions

    # ============================================================
    # 5. 规律发现
    # ============================================================

    def _generate_pattern_discovery_questions(self) -> Dict[str, List[Dict]]:
        """生成规律发现推荐问题"""
        questions = {
            "correlation": [],
            "timeseries": [],
            "trend": [],
            "categorical_pattern": [],
            "distribution_insight": []
        }

        if not self._analysis_result:
            return questions

        variable_types = self._analysis_result.get('variable_types', {})
        summaries = self._analysis_result.get('variable_summaries', {})
        correlations = self._analysis_result.get('correlations', {})

        high_corrs = correlations.get('high_correlations', [])

        # ---- correlation ----
        if high_corrs:
            for corr in high_corrs[:3]:
                text = f"「{corr.get('var1', '')}」与「{corr.get('var2', '')}」相关系数 {corr.get('value', 0):.3f}"
                prompt = f"请结合数据源中的实际数据，回答：{text}"
                questions["correlation"].append({
                    "icon": "🔗",
                    "text": text,
                    "prompt": prompt,
                    "dataKey": "pattern_discovery.correlation"
                })
        else:
            questions["correlation"].append({
                "icon": "ℹ️",
                "text": "未发现强相关关系（|r| ≥ 0.7），变量间可能独立",
                "prompt": "请结合数据源中的实际数据，回答：未发现强相关关系（|r| ≥ 0.7），变量间可能独立",
                "dataKey": "pattern_discovery.correlation"
            })

        # ---- timeseries ----
        ts_diag = self._analysis_result.get('time_series_diagnostics', {})
        if ts_diag:
            for key, diag in list(ts_diag.items())[:2]:
                if diag.get('has_autocorrelation'):
                    text = f"「{key}」检测到自相关，适合时序预测"
                    prompt = f"请结合数据源中的实际数据，回答：{text}"
                    questions["timeseries"].append({
                        "icon": "📈",
                        "text": text,
                        "prompt": prompt,
                        "dataKey": "pattern_discovery.timeseries"
                    })
                else:
                    text = f"「{key}」未检测到自相关，可能为白噪声"
                    prompt = f"请结合数据源中的实际数据，回答：{text}"
                    questions["timeseries"].append({
                        "icon": "📊",
                        "text": text,
                        "prompt": prompt,
                        "dataKey": "pattern_discovery.timeseries"
                    })
        else:
            questions["timeseries"].append({
                "icon": "ℹ️",
                "text": "未检测到时间序列数据，无法进行时序分析",
                "prompt": "请结合数据源中的实际数据，回答：未检测到时间序列数据，无法进行时序分析",
                "dataKey": "pattern_discovery.timeseries"
            })

        # ---- trend ----
        datetime_vars = [col for col, info in variable_types.items() if info.get('type') == 'datetime']
        if datetime_vars:
            continuous_vars = [col for col, info in variable_types.items() if info.get('type') == 'continuous']
            for col in continuous_vars[:2]:
                try:
                    info = summaries.get(col, {})
                    mean_val = info.get('mean')
                    if mean_val is not None:
                        text = f"「{col}」均值 {self._format_num(mean_val)}，建议结合时间维度分析趋势"
                        prompt = f"请结合数据源中的实际数据，回答：{text}"
                        questions["trend"].append({
                            "icon": "📈",
                            "text": text,
                            "prompt": prompt,
                            "dataKey": "pattern_discovery.trend"
                        })
                except:
                    pass
        if not questions["trend"]:
            questions["trend"].append({
                "icon": "ℹ️",
                "text": "数据中无日期时间字段，无法进行趋势分析",
                "prompt": "请结合数据源中的实际数据，回答：数据中无日期时间字段，无法进行趋势分析",
                "dataKey": "pattern_discovery.trend"
            })

        # ---- categorical_pattern ----
        categorical_vars = [col for col, info in variable_types.items()
                           if info.get('type') in ['categorical', 'categorical_numeric', 'ordinal']]
        if len(categorical_vars) >= 2:
            text = f"发现 {len(categorical_vars)} 个分类变量，可探索分类间的关联模式"
            prompt = f"请结合数据源中的实际数据，回答：{text}"
            questions["categorical_pattern"].append({
                "icon": "🏷️",
                "text": text,
                "prompt": prompt,
                "dataKey": "pattern_discovery.categorical_pattern"
            })
        else:
            questions["categorical_pattern"].append({
                "icon": "ℹ️",
                "text": "分类变量数量不足（<2），无法进行关联分析",
                "prompt": "请结合数据源中的实际数据，回答：分类变量数量不足（<2），无法进行关联分析",
                "dataKey": "pattern_discovery.categorical_pattern"
            })

        # ---- distribution_insight ----
        continuous_vars = [col for col, info in variable_types.items() if info.get('type') == 'continuous']
        if continuous_vars:
            for col in continuous_vars[:2]:
                info = summaries.get(col, {})
                skew_val = info.get('skew')
                if skew_val and abs(self._safe_float(skew_val)) > 1:
                    direction = '右' if self._safe_float(skew_val) > 0 else '左'
                    text = f"「{col}」偏度 {self._safe_float(skew_val):.2f}，呈{direction}偏分布"
                    prompt = f"请结合数据源中的实际数据，回答：{text}"
                    questions["distribution_insight"].append({
                        "icon": "📊",
                        "text": text,
                        "prompt": prompt,
                        "dataKey": "pattern_discovery.distribution_insight"
                    })
        if not questions["distribution_insight"]:
            questions["distribution_insight"].append({
                "icon": "ℹ️",
                "text": "未发现明显的偏态分布特征",
                "prompt": "请结合数据源中的实际数据，回答：未发现明显的偏态分布特征",
                "dataKey": "pattern_discovery.distribution_insight"
            })

        return questions

    # ============================================================
    # 6. 智能预测
    # ============================================================

    def _generate_smart_prediction_questions(self) -> Dict[str, List[Dict]]:
        """生成智能预测推荐问题"""
        questions = {
            "model_recommend": [],
            "target_select": [],
            "feature_select": [],
            "forecast": []
        }

        if not self._analysis_result:
            return questions

        variable_types = self._analysis_result.get('variable_types', {})
        ts_diag = self._analysis_result.get('time_series_diagnostics', {})

        continuous_vars = [col for col, info in variable_types.items() if info.get('type') == 'continuous']
        categorical_vars = [col for col, info in variable_types.items()
                           if info.get('type') in ['categorical', 'categorical_numeric', 'ordinal']]

        model_recs = self._get_model_recommendations()

        # ---- model_recommend ----
        if model_recs:
            for rec in model_recs[:2]:
                text = f"{rec.get('task_type', '')}：推荐使用 {rec.get('ml', '')}"
                prompt = f"请结合数据源中的实际数据，回答：{text}"
                questions["model_recommend"].append({
                    "icon": "🤖",
                    "text": text,
                    "prompt": prompt,
                    "dataKey": "smart_prediction.model_recommend"
                })
        else:
            questions["model_recommend"].append({
                "icon": "ℹ️",
                "text": "当前数据特征无法生成模型推荐",
                "prompt": "请结合数据源中的实际数据，回答：当前数据特征无法生成模型推荐",
                "dataKey": "smart_prediction.model_recommend"
            })

        # ---- target_select ----
        if continuous_vars:
            top_target = continuous_vars[0]
            text = f"「{top_target}」适合作为回归预测的目标变量"
            prompt = f"请结合数据源中的实际数据，回答：{text}"
            questions["target_select"].append({
                "icon": "🎯",
                "text": text,
                "prompt": prompt,
                "dataKey": "smart_prediction.target_select"
            })
        if categorical_vars:
            top_cat = categorical_vars[0]
            text = f"「{top_cat}」适合作为分类预测的目标变量"
            prompt = f"请结合数据源中的实际数据，回答：{text}"
            questions["target_select"].append({
                "icon": "🎯",
                "text": text,
                "prompt": prompt,
                "dataKey": "smart_prediction.target_select"
            })
        if not questions["target_select"]:
            questions["target_select"].append({
                "icon": "ℹ️",
                "text": "未找到适合作为预测目标的变量",
                "prompt": "请结合数据源中的实际数据，回答：未找到适合作为预测目标的变量",
                "dataKey": "smart_prediction.target_select"
            })

        # ---- feature_select ----
        all_vars = continuous_vars + categorical_vars
        if len(all_vars) >= 3:
            feature_str = '、'.join(all_vars[:3])
            text = f"推荐特征：{feature_str}{'等' if len(all_vars) > 3 else ''}"
            prompt = f"请结合数据源中的实际数据，回答：{text}"
            questions["feature_select"].append({
                "icon": "📊",
                "text": text,
                "prompt": prompt,
                "dataKey": "smart_prediction.feature_select"
            })
        else:
            questions["feature_select"].append({
                "icon": "ℹ️",
                "text": "特征数量不足（<3），建议增加更多特征",
                "prompt": "请结合数据源中的实际数据，回答：特征数量不足（<3），建议增加更多特征",
                "dataKey": "smart_prediction.feature_select"
            })

        # ---- forecast ----
        auto_vars = [k for k, v in ts_diag.items() if v.get('has_autocorrelation')]
        if auto_vars:
            text = f"对「{auto_vars[0]}」进行未来趋势预测"
            prompt = f"请结合数据源中的实际数据，回答：{text}"
            questions["forecast"].append({
                "icon": "🔮",
                "text": text,
                "prompt": prompt,
                "dataKey": "smart_prediction.forecast"
            })
        elif continuous_vars:
            text = f"对「{continuous_vars[0]}」进行未来趋势预测"
            prompt = f"请结合数据源中的实际数据，回答：{text}"
            questions["forecast"].append({
                "icon": "🔮",
                "text": text,
                "prompt": prompt,
                "dataKey": "smart_prediction.forecast"
            })
        else:
            questions["forecast"].append({
                "icon": "ℹ️",
                "text": "未找到适合预测的变量",
                "prompt": "请结合数据源中的实际数据，回答：未找到适合预测的变量",
                "dataKey": "smart_prediction.forecast"
            })

        return questions

    # ============================================================
    # 对外接口
    # ============================================================

    def get_recommended_questions(self, session_id: str) -> Optional[Dict]:
        """从 session 读取已生成的推荐问题"""
        from pathlib import Path

        data_dir = Path.home() / ".autostat" / "data" / session_id
        questions_file = data_dir / "recommended_questions.json"

        if questions_file.exists():
            try:
                with open(questions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 读取推荐问题失败: {e}")
                return None
        return None

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
                        "text": q.get("text", ""),
                        "prompt": q.get("prompt", ""),
                        "dataKey": q.get("dataKey", "")
                    })
                elif isinstance(q, str):
                    formatted.append({"icon": "💬", "text": q, "prompt": q, "dataKey": ""})
            result[scene] = formatted
        return result