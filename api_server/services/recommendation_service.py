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

    def generate(self, analysis_result: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """
        生成所有场景的推荐问题

        Args:
            analysis_result: 分析结果 JSON

        Returns:
            {
                "report_summary": {
                    "overview": [{"icon": "📊", "text": "..."}],
                    "conclusions": [...],
                    "insights": [...]
                },
                ...
            }
        """
        result = {}

        # 1. 报告总览
        result["report_summary"] = self._generate_report_summary(analysis_result)

        # 2. 数据概览
        result["data_overview"] = self._generate_data_overview(analysis_result)

        # 3. 质量看板
        result["quality"] = self._generate_quality(analysis_result)

        # 4. 数据核验
        result["data_validation"] = self._generate_data_validation(analysis_result)

        # 5. 规律发现
        result["pattern_discovery"] = self._generate_pattern_discovery(analysis_result)

        # 6. 智能预测
        result["smart_prediction"] = self._generate_smart_prediction(analysis_result)

        return result

    # ==================== 安全类型转换辅助方法 ====================

    def _safe_int(self, value: Any, default: int = 0) -> int:
        """安全转换为整数"""
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """安全转换为浮点数"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _format_num(self, value: Any, decimals: int = 2) -> str:
        """格式化数值"""
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

    # ==================== 1. 报告总览 ====================

    def _generate_report_summary(self, result: Dict) -> Dict[str, List[Dict]]:
        """生成报告总览推荐问题"""
        questions = {
            "overview": [],
            "conclusions": [],
            "insights": []
        }

        data_shape = result.get('data_shape', {})
        rows = data_shape.get('rows', 0)
        cols = data_shape.get('columns', 0)

        variable_types = result.get('variable_types', {})
        type_counts = {}
        for info in variable_types.values():
            typ = info.get('type', 'unknown')
            type_counts[typ] = type_counts.get(typ, 0) + 1

        type_display = {
            'continuous': '连续变量',
            'categorical': '分类变量',
            'categorical_numeric': '数值型分类',
            'ordinal': '有序分类',
            'datetime': '日期时间',
            'identifier': '标识符',
            'text': '文本'
        }
        type_summary = '、'.join([f"{type_display.get(t, t)} {c}个" for t, c in type_counts.items() if t in type_display])

        # overview
        questions["overview"].append({
            "icon": "📊",
            "text": f"数据共 {rows:,} 行 {cols} 列，包含 {type_summary}"
        })

        # 如果有连续变量，添加分布描述
        continuous_vars = [col for col, info in variable_types.items() if info.get('type') == 'continuous']
        if continuous_vars:
            summaries = result.get('variable_summaries', {})
            for col in continuous_vars[:2]:
                info = summaries.get(col, {})
                mean_val = info.get('mean')
                median_val = info.get('median')
                if mean_val is not None:
                    questions["overview"].append({
                        "icon": "📊",
                        "text": f"「{col}」均值 {self._format_num(mean_val)}，中位数 {self._format_num(median_val)}"
                    })

        # conclusions：有强相关时生成
        high_corrs = result.get('correlations', {}).get('high_correlations', [])
        if high_corrs:
            for corr in high_corrs[:2]:
                questions["conclusions"].append({
                    "icon": "🔗",
                    "text": f"「{corr.get('var1', '')}」与「{corr.get('var2', '')}」强相关 (r={corr.get('value', 0):.2f})"
                })

        # conclusions：有异常值时生成
        outliers = result.get('quality_report', {}).get('outliers', {})
        if outliers:
            top_field = max(outliers.items(), key=lambda x: self._safe_float(x[1].get('percent', 0)))
            pct = self._safe_float(top_field[1].get('percent', 0))
            count = self._safe_int(top_field[1].get('count', 0))
            questions["conclusions"].append({
                "icon": "🚨",
                "text": f"「{top_field[0]}」存在 {count} 个异常值 ({pct:.1f}%)"
            })

        # insights：有时间序列时生成
        ts_diag = result.get('time_series_diagnostics', {})
        auto_vars = [k for k, v in ts_diag.items() if v.get('has_autocorrelation')]
        if auto_vars:
            questions["insights"].append({
                "icon": "📈",
                "text": f"「{auto_vars[0]}」检测到自相关性，适合时间序列预测"
            })

        return questions

    # ==================== 2. 数据概览 ====================

    def _generate_data_overview(self, result: Dict) -> Dict[str, List[Dict]]:
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

        # 🔥 修复：获取 data_shape
        data_shape = result.get('data_shape', {})
        variable_types = result.get('variable_types', {})
        summaries = result.get('variable_summaries', {})
        quality = result.get('quality_report', {})
        table_name = result.get('source_table', '数据表')

        # ===== distribution：连续变量分布 =====
        continuous_vars = [col for col, info in variable_types.items() if info.get('type') == 'continuous']
        for col in continuous_vars[:3]:
            info = summaries.get(col, {})
            mean_val = info.get('mean')
            median_val = info.get('median')
            std_val = info.get('std')
            skew_val = info.get('skew')
            if mean_val is not None:
                if skew_val and abs(self._safe_float(skew_val)) > 2:
                    questions["distribution"].append({
                        "icon": "📊",
                        "text": f"「{col}」偏度 {self._safe_float(skew_val):.2f}，分布严重偏斜，建议使用中位数描述"
                    })
                else:
                    questions["distribution"].append({
                        "icon": "📊",
                        "text": f"「{col}」均值 {self._format_num(mean_val)}，中位数 {self._format_num(median_val)}，标准差 {self._format_num(std_val)}"
                    })

        # ===== categorical：分类变量 =====
        categorical_vars = [col for col, info in variable_types.items()
                            if info.get('type') in ['categorical', 'categorical_numeric', 'ordinal']]
        for col in categorical_vars[:3]:
            info = summaries.get(col, {})
            n_unique = info.get('n_unique', 0)
            mode_pct = info.get('mode_pct', 0)
            mode_val = info.get('mode')
            if n_unique and self._safe_int(n_unique) > 0:
                mode_pct_float = self._safe_float(mode_pct)
                if mode_pct_float > 80:
                    questions["categorical"].append({
                        "icon": "🏷️",
                        "text": f"「{col}」中「{mode_val}」占比 {mode_pct_float:.1f}%，存在严重不平衡"
                    })
                else:
                    questions["categorical"].append({
                        "icon": "🏷️",
                        "text": f"「{col}」共 {self._safe_int(n_unique)} 个类别，占比最高的类别为「{mode_val}」({mode_pct_float:.1f}%)"
                    })

        # ===== continuous：连续变量详情 =====
        for col in continuous_vars[:2]:
            info = summaries.get(col, {})
            min_val = info.get('min')
            max_val = info.get('max')
            mean_val = info.get('mean')
            if min_val is not None:
                questions["continuous"].append({
                    "icon": "📈",
                    "text": f"「{col}」范围 [{self._format_num(min_val)}, {self._format_num(max_val)}]，均值 {self._format_num(mean_val)}"
                })

        # ===== datetime：日期列 =====
        datetime_vars = [col for col, info in variable_types.items() if info.get('type') == 'datetime']
        date_col = None
        min_date = None
        max_date = None
        for col in datetime_vars[:1]:
            info = summaries.get(col, {})
            min_date = info.get('min_date')
            max_date = info.get('max_date')
            if min_date and max_date:
                date_col = col
                try:
                    from datetime import datetime as dt
                    min_d = dt.strptime(str(min_date), '%Y-%m-%d') if isinstance(min_date, str) else min_date
                    max_d = dt.strptime(str(max_date), '%Y-%m-%d') if isinstance(max_date, str) else max_date
                    days = (max_d - min_d).days
                    questions["datetime"].append({
                        "icon": "📅",
                        "text": f"「{col}」时间范围 {min_date} 至 {max_date}，跨度 {days} 天"
                    })
                except:
                    questions["datetime"].append({
                        "icon": "📅",
                        "text": f"「{col}」时间范围 {min_date} 至 {max_date}"
                    })

        # ===== missing：缺失值 =====
        missing_list = quality.get('missing', [])
        for item in missing_list[:2]:
            pct = self._safe_float(item.get('percent', 0))
            if pct > 5:
                questions["missing"].append({
                    "icon": "⚠️",
                    "text": f"「{item.get('column', '')}」缺失率 {pct:.1f}%，建议处理"
                })

        # ===== identifier：标识符 =====
        identifier_vars = [col for col, info in variable_types.items() if info.get('type') == 'identifier']
        for col in identifier_vars[:1]:
            info = summaries.get(col, {})
            n_unique = info.get('n_unique', 0)
            questions["identifier"].append({
                "icon": "🆔",
                "text": f"「{col}」有 {self._safe_int(n_unique)} 个唯一值，可作为主键"
            })

        # ===== text：文本字段 =====
        text_vars = [col for col, info in variable_types.items() if info.get('type') == 'text']
        for col in text_vars[:1]:
            info = summaries.get(col, {})
            n_unique = info.get('n_unique', 0)
            questions["text"].append({
                "icon": "📝",
                "text": f"「{col}」有 {self._safe_int(n_unique)} 个唯一值，属于文本类型"
            })

        # ===== natural_query：根据数据特征生成自然语言查询问题 =====
        rows = data_shape.get('rows', 0)

        if date_col and min_date and max_date:
            questions["natural_query"].append({
                "icon": "🔍",
                "text": f"查询「{date_col}」在 {min_date} 至 {max_date} 之间的数据"
            })
            questions["natural_query"].append({
                "icon": "🔍",
                "text": f"查询「{date_col}」为最近 7 天的数据"
            })

        if categorical_vars:
            top_cat = categorical_vars[0]
            questions["natural_query"].append({
                "icon": "🔍",
                "text": f"按「{top_cat}」分组统计记录数"
            })
            if len(categorical_vars) >= 2:
                questions["natural_query"].append({
                    "icon": "🔍",
                    "text": f"按「{categorical_vars[0]}」和「{categorical_vars[1]}」交叉统计"
                })

        if continuous_vars:
            top_num = continuous_vars[0]
            questions["natural_query"].append({
                "icon": "🔍",
                "text": f"查询「{top_num}」大于平均值的记录"
            })
            if len(continuous_vars) >= 2:
                questions["natural_query"].append({
                    "icon": "🔍",
                    "text": f"查询「{continuous_vars[0]}」大于「{continuous_vars[1]}」的记录"
                })

        if rows > 10000:
            questions["natural_query"].append({
                "icon": "🔍",
                "text": f"查询前 100 条数据"
            })
        elif rows > 1000:
            questions["natural_query"].append({
                "icon": "🔍",
                "text": f"随机抽取 100 条数据查看"
            })

        # 去重
        seen = set()
        unique_natural = []
        for q in questions["natural_query"]:
            key = q.get("text", "")
            if key not in seen:
                seen.add(key)
                unique_natural.append(q)
        questions["natural_query"] = unique_natural[:6]

        # ===== generate_sql：根据数据特征生成 SQL 推荐问题 =====
        if date_col:
            questions["generate_sql"].append({
                "icon": "📝",
                "text": f"生成查询「{date_col}」在指定时间范围的 SQL 语句"
            })
            questions["generate_sql"].append({
                "icon": "📝",
                "text": f"生成按「{date_col}」分组统计的 SQL 语句"
            })

        if categorical_vars:
            cat_col = categorical_vars[0]
            questions["generate_sql"].append({
                "icon": "📝",
                "text": f"生成按「{cat_col}」分组统计记录数的 SQL 语句"
            })
            if len(categorical_vars) >= 2:
                questions["generate_sql"].append({
                    "icon": "📝",
                    "text": f"生成按「{categorical_vars[0]}」和「{categorical_vars[1]}」分组统计的 SQL 语句"
                })

        if continuous_vars:
            num_col = continuous_vars[0]
            questions["generate_sql"].append({
                "icon": "📝",
                "text": f"生成查询「{num_col}」大于指定值的 SQL 语句"
            })

        tables_info = result.get('tables_info', {})
        if tables_info and len(tables_info) > 1:
            table_names = list(tables_info.keys())
            questions["generate_sql"].append({
                "icon": "📝",
                "text": f"生成关联查询「{table_names[0]}」和「{table_names[1]}」的 SQL 语句"
            })

        if identifier_vars:
            id_col = identifier_vars[0]
            questions["generate_sql"].append({
                "icon": "📝",
                "text": f"生成根据「{id_col}」查询单条记录的 SQL 语句"
            })

        # 去重
        seen = set()
        unique_sql = []
        for q in questions["generate_sql"]:
            key = q.get("text", "")
            if key not in seen:
                seen.add(key)
                unique_sql.append(q)
        questions["generate_sql"] = unique_sql[:6]

        return questions

    # ==================== 3. 质量看板 ====================

    def _generate_quality(self, result: Dict) -> Dict[str, List[Dict]]:
        """生成质量看板推荐问题"""
        questions = {
            "overall": [],
            "completeness": [],
            "accuracy": [],
            "consistency": [],
            "uniqueness": []
        }

        quality = result.get('quality_report', {})
        dimensions = quality.get('dimensions', {})
        outliers = quality.get('outliers', {})
        missing_list = quality.get('missing', [])
        dup_info = quality.get('duplicates', {})
        audit_rules = quality.get('audit_rules', {})

        # overall：综合评分
        overall_score = quality.get('overall_score')
        if overall_score is not None:
            score = self._safe_float(overall_score)
            grade = '优秀' if score >= 90 else '良好' if score >= 80 else '一般' if score >= 70 else '需关注'
            questions["overall"].append({
                "icon": "⭐",
                "text": f"综合质量评分 {score:.1f} 分（{grade}），各维度得分是否均衡？"
            })

        # completeness：完整性
        if missing_list:
            total_missing = 0
            for m in missing_list:
                total_missing += self._safe_int(m.get('count', 0))
            total_rows = result.get('data_shape', {}).get('rows', 1)
            total_cols = len(result.get('variable_types', {}))
            if total_rows > 0 and total_cols > 0:
                overall_missing_rate = (total_missing / (total_rows * total_cols)) * 100
            else:
                overall_missing_rate = 0
            questions["completeness"].append({
                "icon": "📊",
                "text": f"整体缺失率 {overall_missing_rate:.1f}%，共 {len(missing_list)} 个字段存在缺失"
            })
            top = max(missing_list, key=lambda x: self._safe_float(x.get('percent', 0)))
            top_pct = self._safe_float(top.get('percent', 0))
            if top_pct > 20:
                questions["completeness"].append({
                    "icon": "⚠️",
                    "text": f"「{top.get('column', '')}」缺失率 {top_pct:.1f}%，建议优先处理"
                })

        # accuracy：准确性
        if outliers:
            total_outliers = 0
            for info in outliers.values():
                total_outliers += self._safe_int(info.get('count', 0))
            questions["accuracy"].append({
                "icon": "🎯",
                "text": f"共发现 {len(outliers)} 个字段存在异常值，总计 {total_outliers} 条"
            })
            top = max(outliers.items(), key=lambda x: self._safe_float(x[1].get('percent', 0)))
            top_pct = self._safe_float(top[1].get('percent', 0))
            if top_pct > 5:
                questions["accuracy"].append({
                    "icon": "🚨",
                    "text": f"「{top[0]}」异常值 {self._safe_int(top[1].get('count', 0))} 条 ({top_pct:.1f}%)"
                })

        # consistency：一致性
        arithmetic_count = len(audit_rules.get('arithmetic_rules', []))
        if arithmetic_count > 0:
            questions["consistency"].append({
                "icon": "🔗",
                "text": f"发现 {arithmetic_count} 条数值关系规则，数据一致性如何？"
            })

        # uniqueness：唯一性
        dup_count = self._safe_int(dup_info.get('count', 0))
        dup_pct = self._safe_float(dup_info.get('percent', 0))
        if dup_count > 0:
            questions["uniqueness"].append({
                "icon": "🆔",
                "text": f"发现 {dup_count} 条重复记录，占比 {dup_pct:.1f}%"
            })
        else:
            questions["uniqueness"].append({
                "icon": "✅",
                "text": "未发现重复记录，数据唯一性良好"
            })

        return questions

    # ==================== 4. 数据核验 ====================

    def _generate_data_validation(self, result: Dict) -> Dict[str, List[Dict]]:
        """生成数据核验推荐问题"""
        questions = {
            "audit_rules": [],
            "outliers": [],
            "missing_detail": [],
            "duplicates": [],
            "cleaning": []
        }

        quality = result.get('quality_report', {})
        audit_rules = quality.get('audit_rules', {})
        outliers = quality.get('outliers', {})
        missing_list = quality.get('missing', [])
        dup_info = quality.get('duplicates', {})
        cleaning = result.get('cleaning_suggestions', [])

        # audit_rules：勾稽规则
        arithmetic = audit_rules.get('arithmetic_rules', [])
        functional = audit_rules.get('functional_dependencies', [])
        temporal = audit_rules.get('temporal_rules', [])
        total = len(arithmetic) + len(functional) + len(temporal)

        if total > 0:
            questions["audit_rules"].append({
                "icon": "🔗",
                "text": f"共发现 {total} 条勾稽规则（数值关系 {len(arithmetic)} 条、函数依赖 {len(functional)} 条、时序约束 {len(temporal)} 条）"
            })
            violations = [r for r in arithmetic if self._safe_int(r.get('violation_count', 0)) > 0]
            if violations:
                top = max(violations, key=lambda x: self._safe_int(x.get('violation_count', 0)))
                questions["audit_rules"].append({
                    "icon": "⚠️",
                    "text": f"规则「{top.get('rule', '')[:30]}...」有 {self._safe_int(top.get('violation_count', 0))} 条违反记录"
                })

        # outliers：异常值
        for field, info in list(outliers.items())[:2]:
            questions["outliers"].append({
                "icon": "🚨",
                "text": f"「{field}」有 {self._safe_int(info.get('count', 0))} 个异常值 ({self._safe_float(info.get('percent', 0)):.1f}%)"
            })

        # missing_detail：缺失详情
        for item in missing_list[:2]:
            pct = self._safe_float(item.get('percent', 0))
            if pct > 5:
                questions["missing_detail"].append({
                    "icon": "⚠️",
                    "text": f"「{item.get('column', '')}」缺失 {self._safe_int(item.get('count', 0))} 条 ({pct:.1f}%)"
                })

        # duplicates：重复记录
        dup_count = self._safe_int(dup_info.get('count', 0))
        if dup_count > 0:
            questions["duplicates"].append({
                "icon": "📋",
                "text": f"发现 {dup_count} 条重复记录，建议去重"
            })

        # cleaning：清洗建议
        for suggestion in cleaning[:3]:
            questions["cleaning"].append({
                "icon": "🧹",
                "text": suggestion
            })

        return questions

    # ==================== 5. 规律发现 ====================

    def _generate_pattern_discovery(self, result: Dict) -> Dict[str, List[Dict]]:
        """生成规律发现推荐问题"""
        questions = {
            "correlation": [],
            "timeseries": [],
            "trend": [],
            "categorical_pattern": [],
            "distribution_insight": []
        }

        variable_types = result.get('variable_types', {})
        summaries = result.get('variable_summaries', {})
        high_corrs = result.get('correlations', {}).get('high_correlations', [])
        ts_diag = result.get('time_series_diagnostics', {})

        # correlation：相关性
        for corr in high_corrs[:3]:
            questions["correlation"].append({
                "icon": "🔗",
                "text": f"「{corr.get('var1', '')}」与「{corr.get('var2', '')}」相关系数 {corr.get('value', 0):.3f}"
            })

        # timeseries：时间序列
        for key, diag in list(ts_diag.items())[:2]:
            if diag.get('has_autocorrelation'):
                questions["timeseries"].append({
                    "icon": "📈",
                    "text": f"「{key}」检测到自相关，适合时序预测"
                })
            else:
                questions["timeseries"].append({
                    "icon": "📊",
                    "text": f"「{key}」未检测到自相关，可能为白噪声"
                })

        # trend：趋势分析
        datetime_vars = [col for col, info in variable_types.items() if info.get('type') == 'datetime']
        if datetime_vars:
            continuous_vars = [col for col, info in variable_types.items() if info.get('type') == 'continuous']
            for col in continuous_vars[:2]:
                try:
                    info = summaries.get(col, {})
                    mean_val = info.get('mean')
                    if mean_val is not None:
                        questions["trend"].append({
                            "icon": "📈",
                            "text": f"「{col}」均值 {self._format_num(mean_val)}，建议结合时间维度分析趋势"
                        })
                except:
                    pass

        # categorical_pattern：分类关联
        categorical_vars = [col for col, info in variable_types.items()
                           if info.get('type') in ['categorical', 'categorical_numeric', 'ordinal']]
        if len(categorical_vars) >= 2:
            questions["categorical_pattern"].append({
                "icon": "🏷️",
                "text": f"发现 {len(categorical_vars)} 个分类变量，可探索分类间的关联模式"
            })

        # distribution_insight：分布洞察
        continuous_vars = [col for col, info in variable_types.items() if info.get('type') == 'continuous']
        for col in continuous_vars[:2]:
            info = summaries.get(col, {})
            skew_val = info.get('skew')
            if skew_val and abs(self._safe_float(skew_val)) > 1:
                direction = '右' if self._safe_float(skew_val) > 0 else '左'
                questions["distribution_insight"].append({
                    "icon": "📊",
                    "text": f"「{col}」偏度 {self._safe_float(skew_val):.2f}，呈{direction}偏分布"
                })

        return questions

    # ==================== 6. 智能预测 ====================

    def _generate_smart_prediction(self, result: Dict) -> Dict[str, List[Dict]]:
        """生成智能预测推荐问题"""
        questions = {
            "model_recommend": [],
            "target_select": [],
            "feature_select": [],
            "forecast": []
        }

        variable_types = result.get('variable_types', {})
        model_recs = result.get('model_recommendations', [])
        ts_diag = result.get('time_series_diagnostics', {})

        continuous_vars = [col for col, info in variable_types.items() if info.get('type') == 'continuous']
        categorical_vars = [col for col, info in variable_types.items()
                           if info.get('type') in ['categorical', 'categorical_numeric', 'ordinal']]

        # model_recommend：模型推荐
        for rec in model_recs[:2]:
            questions["model_recommend"].append({
                "icon": "🤖",
                "text": f"{rec.get('task_type', '')}：推荐使用 {rec.get('ml', '')}"
            })

        # target_select：目标选择
        if continuous_vars:
            questions["target_select"].append({
                "icon": "🎯",
                "text": f"「{continuous_vars[0]}」适合作为回归预测的目标变量"
            })
        if categorical_vars:
            questions["target_select"].append({
                "icon": "🎯",
                "text": f"「{categorical_vars[0]}」适合作为分类预测的目标变量"
            })

        # feature_select：特征选择
        all_vars = continuous_vars + categorical_vars
        if len(all_vars) >= 3:
            feature_str = '、'.join(all_vars[:3])
            questions["feature_select"].append({
                "icon": "📊",
                "text": f"推荐特征：{feature_str}{'等' if len(all_vars) > 3 else ''}"
            })

        # forecast：未来预测
        auto_vars = [k for k, v in ts_diag.items() if v.get('has_autocorrelation')]
        if auto_vars:
            questions["forecast"].append({
                "icon": "🔮",
                "text": f"对「{auto_vars[0]}」进行未来趋势预测"
            })
        elif continuous_vars:
            questions["forecast"].append({
                "icon": "🔮",
                "text": f"对「{continuous_vars[0]}」进行未来趋势预测"
            })

        return questions

    # ==================== 对外接口 ====================

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
                        "text": q.get("text", "")
                    })
                elif isinstance(q, str):
                    formatted.append({"icon": "💬", "text": q})
            result[scene] = formatted
        return result