"""
技术事实清单构建器
从 analysis_result 中提取 F01-F18 技术事实
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


class TechFactSheet:
    """技术事实清单"""

    def __init__(self, session_id: str, analysis_result: Dict[str, Any]):
        self.session_id = session_id
        self.analysis_result = analysis_result
        self.facts = {}
        self._build()

    # ==================== 安全类型转换辅助方法 ====================

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """安全转换为整数"""
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value)) if value else default
            except (ValueError, TypeError):
                return default
        return default

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """安全转换为浮点数"""
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value) if value else default
            except (ValueError, TypeError):
                return default
        return default

    @staticmethod
    def _safe_bool(value: Any, default: bool = False) -> bool:
        """安全转换为布尔值"""
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return default

    @staticmethod
    def _safe_list(value: Any, default: Optional[List] = None) -> List:
        """安全转换为列表"""
        if default is None:
            default = []
        if value is None:
            return default
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return default

    @staticmethod
    def _safe_dict(value: Any, default: Optional[Dict] = None) -> Dict:
        """安全转换为字典"""
        if default is None:
            default = {}
        if value is None:
            return default
        if isinstance(value, dict):
            return value
        return default

    @staticmethod
    def _safe_get(data: Any, key: str, default: Any = None) -> Any:
        """安全从字典中获取值"""
        if not data or not isinstance(data, dict):
            return default
        return data.get(key, default)

    # ==================== 构建方法 ====================

    def _build(self):
        """构建所有技术事实"""
        self.facts["F01"] = self._extract_F01()
        self.facts["F02"] = self._extract_F02()
        self.facts["F03"] = self._extract_F03()
        self.facts["F04"] = self._extract_F04()
        self.facts["F05"] = self._extract_F05()
        self.facts["F06"] = self._extract_F06()
        self.facts["F07"] = self._extract_F07()
        self.facts["F08"] = self._extract_F08()
        self.facts["F09"] = self._extract_F09()
        self.facts["F10"] = self._extract_F10()
        self.facts["F11"] = self._extract_F11()
        self.facts["F12"] = self._extract_F12()
        self.facts["F13"] = self._extract_F13()
        self.facts["F14"] = self._extract_F14()
        self.facts["F15"] = self._extract_F15()
        self.facts["F16"] = self._extract_F16()
        self.facts["F17"] = self._extract_F17()
        self.facts["F18"] = self._extract_F18()

    def to_dict(self) -> Dict[str, Any]:
        """输出为字典"""
        return {
            "session_id": self.session_id,
            "generated_at": datetime.now().isoformat(),
            "facts": self.facts
        }

    def get(self, fact_id: str) -> Any:
        """获取单个技术事实"""
        return self.facts.get(fact_id)

    # ==================== F01: 数据规模 ====================

    def _extract_F01(self) -> Dict[str, Any]:
        """F01: 数据规模"""
        shape = self._safe_dict(self.analysis_result.get("data_shape"))
        multi_info = self._safe_dict(self.analysis_result.get("multi_table_info"))
        tables = self._safe_dict(multi_info.get("tables"))

        return {
            "rows": self._safe_int(shape.get("rows")),
            "columns": self._safe_int(shape.get("columns")),
            "table_count": len(tables) if tables else 1,
            "table_names": list(tables.keys()) if tables else ["数据表"]
        }

    # ==================== F02: 变量类型分布 ====================

    def _extract_F02(self) -> Dict[str, int]:
        """F02: 变量类型分布"""
        var_types = self._safe_dict(self.analysis_result.get("variable_types"))
        type_counts = {
            "continuous": 0,
            "categorical": 0,
            "categorical_numeric": 0,
            "ordinal": 0,
            "datetime": 0,
            "identifier": 0,
            "text": 0,
            "other": 0
        }

        for col, info in var_types.items():
            if isinstance(info, dict):
                typ = info.get("type", "other")
            else:
                typ = info if isinstance(info, str) else "other"
            if typ in type_counts:
                type_counts[typ] += 1
            else:
                type_counts["other"] += 1

        return type_counts

    # ==================== F03: 表间关系 ====================

    def _extract_F03(self) -> List[Dict[str, Any]]:
        """F03: 表间关系"""
        multi_info = self._safe_dict(self.analysis_result.get("multi_table_info"))
        return self._safe_list(multi_info.get("relationships"))

    # ==================== F04: 数值-数值相关性 ====================

    def _extract_F04(self) -> Dict[str, Any]:
        """F04: 数值-数值相关性"""
        correlations = self._safe_dict(self.analysis_result.get("correlations"))
        matrix = self._safe_dict(correlations.get("matrix"))
        high_corrs = self._safe_list(correlations.get("high_correlations"))
        all_corrs = []

        # 提取所有相关性对
        var_names = list(matrix.keys())
        for i in range(len(var_names)):
            for j in range(i + 1, len(var_names)):
                v1, v2 = var_names[i], var_names[j]
                val = matrix.get(v1, {}).get(v2)
                if val is not None and not isinstance(val, str) and abs(val) > 0.1:
                    all_corrs.append({
                        "var1": v1,
                        "var2": v2,
                        "value": round(float(val), 4)
                    })

        all_corrs.sort(key=lambda x: abs(x["value"]), reverse=True)

        return {
            "high_correlations": high_corrs,
            "all_correlations": all_corrs,
            "count_high": len(high_corrs),
            "count_total": len(all_corrs)
        }

    # ==================== F05: 分类-分类关联 ====================

    def _extract_F05(self) -> Dict[str, Any]:
        """F05: 分类-分类关联 (Cramer's V)"""
        categorical_corr = self._safe_dict(self.analysis_result.get("categorical_correlation"))
        matrix = self._safe_dict(categorical_corr.get("matrix"))
        return {
            "significant_pairs": self._safe_list(categorical_corr.get("significant_pairs")),
            "matrix": matrix,
            "count_significant": len(categorical_corr.get("significant_pairs", []))
        }

    # ==================== F06: 数值-分类关联 ====================

    def _extract_F06(self) -> Dict[str, Any]:
        """F06: 数值-分类关联 (Eta-squared)"""
        eta_corr = self._safe_dict(self.analysis_result.get("eta_correlation"))
        matrix = self._safe_dict(eta_corr.get("matrix"))
        return {
            "significant_pairs": self._safe_list(eta_corr.get("significant_pairs")),
            "matrix": matrix,
            "count_significant": len(eta_corr.get("significant_pairs", []))
        }

    # ==================== F07: 时间序列诊断 ====================

    def _extract_F07(self) -> Dict[str, Any]:
        """F07: 时间序列诊断"""
        ts_diag = self._safe_dict(self.analysis_result.get("time_series_diagnostics"))
        if not ts_diag:
            return {
                "has_autocorrelation": False,
                "has_stationary": False,
                "has_seasonality": False,
                "has_trend": False,
                "count": 0,
                "series": {}
            }

        auto_vars = []
        stationary_vars = []
        seasonal_vars = []
        trend_vars = []

        for key, diag in ts_diag.items():
            if self._safe_bool(diag.get("has_autocorrelation")):
                auto_vars.append(key)
            if self._safe_bool(diag.get("is_stationary")):
                stationary_vars.append(key)
            if self._safe_bool(diag.get("has_seasonality")):
                seasonal_vars.append(key)
            # 趋势判断：如果有自相关且非平稳，可能有趋势
            if self._safe_bool(diag.get("has_autocorrelation")) and not self._safe_bool(diag.get("is_stationary")):
                trend_vars.append(key)

        return {
            "has_autocorrelation": len(auto_vars) > 0,
            "has_stationary": len(stationary_vars) > 0,
            "has_seasonality": len(seasonal_vars) > 0,
            "has_trend": len(trend_vars) > 0,
            "count": len(ts_diag),
            "auto_vars": auto_vars[:5],
            "stationary_vars": stationary_vars[:5],
            "seasonal_vars": seasonal_vars[:5],
            "trend_vars": trend_vars[:5],
            "series": {k: {"n_samples": self._safe_int(v.get("n_samples"))} for k, v in list(ts_diag.items())[:10]}
        }

    # ==================== F08: 异常值检测 ====================

    def _extract_F08(self) -> Dict[str, Any]:
        """F08: 异常值检测"""
        quality = self._safe_dict(self.analysis_result.get("quality_report"))
        outliers = self._safe_dict(quality.get("outliers"))

        outlier_list = []
        total_count = 0
        for col, info in outliers.items():
            count = self._safe_int(info.get("count"))
            percent = self._safe_float(info.get("percent"))
            outlier_list.append({
                "field": col,
                "count": count,
                "percent": round(percent, 2),
                "lower_bound": info.get("lower_bound"),
                "upper_bound": info.get("upper_bound")
            })
            total_count += count

        outlier_list.sort(key=lambda x: x["percent"], reverse=True)

        return {
            "fields": outlier_list,
            "count_fields": len(outlier_list),
            "total_count": total_count,
            "has_outliers": len(outlier_list) > 0
        }

    # ==================== F09: 缺失值分析 ====================

    def _extract_F09(self) -> Dict[str, Any]:
        """F09: 缺失值分析"""
        quality = self._safe_dict(self.analysis_result.get("quality_report"))
        missing = self._safe_list(quality.get("missing"))

        missing_list = []
        for item in missing:
            missing_list.append({
                "field": item.get("column", ""),
                "count": self._safe_int(item.get("count")),
                "percent": round(self._safe_float(item.get("percent")), 2)
            })

        missing_list.sort(key=lambda x: x["percent"], reverse=True)

        return {
            "fields": missing_list,
            "count_fields": len(missing_list),
            "has_missing": len(missing_list) > 0,
            "high_missing_fields": [m for m in missing_list if m["percent"] > 20]
        }

    # ==================== F10: 重复记录 ====================

    def _extract_F10(self) -> Dict[str, Any]:
        """F10: 重复记录"""
        quality = self._safe_dict(self.analysis_result.get("quality_report"))
        duplicates = self._safe_dict(quality.get("duplicates"))
        count = self._safe_int(duplicates.get("count"))
        percent = self._safe_float(duplicates.get("percent"))

        return {
            "count": count,
            "percent": round(percent, 2),
            "has_duplicates": count > 0,
            "based_on": duplicates.get("based_on", "all_columns")
        }

    # ==================== F11: 勾稽规则 ====================

    def _extract_F11(self) -> Dict[str, Any]:
        """F11: 勾稽规则"""
        quality = self._safe_dict(self.analysis_result.get("quality_report"))
        audit_rules = self._safe_dict(quality.get("audit_rules"))

        arithmetic = self._safe_list(audit_rules.get("arithmetic_rules"))
        functional = self._safe_list(audit_rules.get("functional_dependencies"))
        temporal = self._safe_list(audit_rules.get("temporal_rules"))
        date_rules = self._safe_list(audit_rules.get("date_rules"))

        all_rules = arithmetic + functional + temporal + date_rules

        return {
            "arithmetic_count": len(arithmetic),
            "functional_count": len(functional),
            "temporal_count": len(temporal),
            "date_rules_count": len(date_rules),
            "total_count": len(all_rules),
            "has_rules": len(all_rules) > 0,
            "rules": all_rules[:20]
        }

    # ==================== F12: 分布特征 ====================

    def _extract_F12(self) -> Dict[str, Any]:
        """F12: 分布特征"""
        summaries = self._safe_dict(self.analysis_result.get("variable_summaries"))
        skewed_vars = []
        imbalanced_vars = []

        for col, info in summaries.items():
            # 偏态变量
            skew = info.get("skew")
            if skew is not None:
                skew_float = self._safe_float(skew)
                if abs(skew_float) > 1:
                    skewed_vars.append({
                        "name": col,
                        "skew": round(skew_float, 3),
                        "type": info.get("type", "unknown")
                    })

            # 不平衡分类变量
            if info.get("type") in ["categorical", "categorical_numeric", "ordinal"]:
                mode_pct = self._safe_float(info.get("mode_pct"))
                if mode_pct > 80:
                    imbalanced_vars.append({
                        "name": col,
                        "mode_pct": round(mode_pct, 2),
                        "n_unique": self._safe_int(info.get("n_unique"))
                    })

        skewed_vars.sort(key=lambda x: abs(x["skew"]), reverse=True)
        imbalanced_vars.sort(key=lambda x: x["mode_pct"], reverse=True)

        return {
            "skewed_vars": skewed_vars[:10],
            "skewed_count": len(skewed_vars),
            "imbalanced_vars": imbalanced_vars[:10],
            "imbalanced_count": len(imbalanced_vars)
        }

    # ==================== F13: 聚类条件 ====================

    def _extract_F13(self) -> Dict[str, Any]:
        """F13: 聚类条件"""
        var_types = self._safe_dict(self.analysis_result.get("variable_types"))
        numeric_count = 0
        for info in var_types.values():
            typ = info.get("type") if isinstance(info, dict) else info
            if typ == "continuous":
                numeric_count += 1

        rows = self._safe_int(self.analysis_result.get("data_shape", {}).get("rows"))

        return {
            "numeric_count": numeric_count,
            "sample_count": rows,
            "is_ready": numeric_count >= 3 and rows >= 100,
            "condition": f"数值变量{numeric_count}个，样本量{rows}行"
        }

    # ==================== F14: 预测条件 ====================

    def _extract_F14(self) -> Dict[str, Any]:
        """F14: 预测条件"""
        ts_diag = self._safe_dict(self.analysis_result.get("time_series_diagnostics"))
        has_auto = any(self._safe_bool(v.get("has_autocorrelation")) for v in ts_diag.values())

        correlations = self._safe_dict(self.analysis_result.get("correlations"))
        high_corrs = self._safe_list(correlations.get("high_correlations"))

        return {
            "has_autocorrelation": has_auto,
            "has_strong_correlations": len(high_corrs) > 0,
            "strong_correlation_count": len(high_corrs),
            "can_predict": has_auto or len(high_corrs) > 0
        }

    # ==================== F15: 实体共现 ====================

    def _extract_F15(self) -> Dict[str, Any]:
        """F15: 实体共现"""
        # 从实体分析中提取
        entity_analysis = self._safe_dict(self.analysis_result.get("entity_analysis"))

        # 如果有实体共现数据，直接使用
        if entity_analysis:
            return {
                "has_entity": True,
                "cooccurrence_pairs": self._safe_list(entity_analysis.get("cooccurrence_pairs")),
                "entity_count": self._safe_int(entity_analysis.get("entity_count"))
            }

        # 否则通过启发式推断：identifier 类型且唯一值较多的列可能是实体列
        var_types = self._safe_dict(self.analysis_result.get("variable_types"))
        summaries = self._safe_dict(self.analysis_result.get("variable_summaries"))
        entity_cols = []

        for col, info in var_types.items():
            typ = info.get("type") if isinstance(info, dict) else info
            if typ == "identifier":
                n_unique = self._safe_int(summaries.get(col, {}).get("n_unique"))
                if n_unique > 10:
                    entity_cols.append({"name": col, "n_unique": n_unique})

        return {
            "has_entity": len(entity_cols) > 0,
            "entity_columns": entity_cols,
            "entity_count": len(entity_cols)
        }

    # ==================== F16: 文本分析 ====================

    def _extract_F16(self) -> Dict[str, Any]:
        """F16: 文本分析"""
        var_types = self._safe_dict(self.analysis_result.get("variable_types"))
        text_cols = []

        for col, info in var_types.items():
            typ = info.get("type") if isinstance(info, dict) else info
            if typ == "text":
                text_cols.append(col)

        return {
            "text_column_count": len(text_cols),
            "text_columns": text_cols[:10],
            "has_text": len(text_cols) > 0
        }

    # ==================== F17: 唯一标识符 ====================

    def _extract_F17(self) -> Dict[str, Any]:
        """F17: 唯一标识符"""
        var_types = self._safe_dict(self.analysis_result.get("variable_types"))
        summaries = self._safe_dict(self.analysis_result.get("variable_summaries"))
        shape = self._safe_dict(self.analysis_result.get("data_shape"))
        rows = self._safe_int(shape.get("rows"))
        id_cols = []

        for col, info in var_types.items():
            typ = info.get("type") if isinstance(info, dict) else info
            if typ == "identifier":
                n_unique = self._safe_int(summaries.get(col, {}).get("n_unique"))
                uniqueness = n_unique / rows if rows > 0 else 0
                id_cols.append({
                    "name": col,
                    "n_unique": n_unique,
                    "uniqueness": round(uniqueness * 100, 2)
                })

        id_cols.sort(key=lambda x: x["uniqueness"], reverse=True)

        return {
            "id_columns": id_cols,
            "count": len(id_cols)
        }

    # ==================== F18: 质量评分 ====================

    def _extract_F18(self) -> Dict[str, Any]:
        """F18: 质量评分"""
        quality = self._safe_dict(self.analysis_result.get("quality_report"))
        dimensions = self._safe_dict(quality.get("dimensions"))

        return {
            "overall_score": self._safe_float(quality.get("overall_score")),
            "grade": quality.get("grade", "未知"),
            "grade_icon": quality.get("grade_icon", "⚪"),
            "dimensions": {
                "completeness": self._safe_float(dimensions.get("completeness")),
                "accuracy": self._safe_float(dimensions.get("accuracy")),
                "consistency": self._safe_float(dimensions.get("consistency")),
                "uniqueness": self._safe_float(dimensions.get("uniqueness"))
            }
        }


def build_tech_fact_sheet(session_id: str, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    构建技术事实清单

    参数:
    - session_id: 会话ID
    - analysis_result: 分析结果字典

    返回:
    - 技术事实清单字典
    """
    sheet = TechFactSheet(session_id, analysis_result)
    return sheet.to_dict()