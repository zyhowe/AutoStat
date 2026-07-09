"""
五维数据质量评分器

评分维度：
1. 完整性 (Completeness) - 字段非空率
2. 准确性 (Accuracy) - 异常值比例
3. 一致性 (Consistency) - 勾稽规则满足率
4. 及时性 (Timeliness) - 数据新鲜度
5. 唯一性 (Uniqueness) - 重复记录比例
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum


class QualityGrade(Enum):
    EXCELLENT = (90, 100, "优秀", "🟢")
    GOOD = (80, 89, "良好", "🟢")
    FAIR = (70, 79, "一般", "🟡")
    POOR = (60, 69, "较差", "🟠")
    BAD = (0, 59, "差", "🔴")

    def __init__(self, min_score: int, max_score: int, label: str, icon: str):
        self.min_score = min_score
        self.max_score = max_score
        self.label = label
        self.icon = icon

    @classmethod
    def from_score(cls, score: float) -> "QualityGrade":
        for grade in cls:
            if grade.min_score <= score <= grade.max_score:
                return grade
        return cls.BAD


QUALITY_GRADES = [
    {"min": 90, "max": 100, "label": "优秀", "icon": "🟢"},
    {"min": 80, "max": 89, "label": "良好", "icon": "🟢"},
    {"min": 70, "max": 79, "label": "一般", "icon": "🟡"},
    {"min": 60, "max": 69, "label": "较差", "icon": "🟠"},
    {"min": 0, "max": 59, "label": "差", "icon": "🔴"},
]


@dataclass
class QualityScore:
    timestamp: str
    table_name: str
    overall_score: float
    grade: str
    grade_icon: str
    dimensions: Dict[str, float]
    field_scores: Dict[str, Dict[str, float]]
    alerts: List[Dict] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class QualityScorer:
    WEIGHTS = {
        "completeness": 0.25,
        "accuracy": 0.25,
        "consistency": 0.20,
        "timeliness": 0.15,
        "uniqueness": 0.15,
    }
    TIMELINESS_THRESHOLD = 7

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.WEIGHTS.copy()
        total = sum(self.weights.values())
        if total != 1.0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def score(self, df: pd.DataFrame, table_name: str = "unknown",
              variable_types: Optional[Dict[str, str]] = None,
              audit_rules: Optional[Dict[str, Any]] = None,
              last_updated: Optional[datetime] = None,
              id_columns: Optional[List[str]] = None) -> QualityScore:
        # 保持不变（调用的子方法会被向量化）
        if variable_types is None:
            from autostat.core.base import BaseAnalyzer
            base = BaseAnalyzer(df, quiet=True)
            base._infer_variable_types()
            variable_types = base.variable_types

        completeness = self._score_completeness(df)
        accuracy = self._score_accuracy(df, variable_types)  # 已向量化
        consistency = self._score_consistency(df, audit_rules)
        timeliness = self._score_timeliness(last_updated)
        uniqueness = self._score_uniqueness(df, id_columns)

        dimensions = {
            "completeness": completeness,
            "accuracy": accuracy,
            "consistency": consistency,
            "timeliness": timeliness,
            "uniqueness": uniqueness,
        }

        overall = (completeness * self.weights["completeness"] +
                   accuracy * self.weights["accuracy"] +
                   consistency * self.weights["consistency"] +
                   timeliness * self.weights["timeliness"] +
                   uniqueness * self.weights["uniqueness"]) * 100

        grade = QualityGrade.from_score(overall)
        field_scores = self._score_fields(df, variable_types)
        alerts = self._generate_alerts(dimensions, field_scores, audit_rules)

        return QualityScore(
            timestamp=datetime.now().isoformat(),
            table_name=table_name,
            overall_score=round(overall, 1),
            grade=grade.label,
            grade_icon=grade.icon,
            dimensions={k: round(v * 100, 1) for k, v in dimensions.items()},
            field_scores=field_scores,
            alerts=alerts,
            details={
                "rows": len(df),
                "columns": len(df.columns),
                "weights": self.weights,
                "thresholds": self._get_thresholds(),
            }
        )

    def _score_completeness(self, df: pd.DataFrame) -> float:
        # 保持不变
        if df.empty:
            return 0.0
        nonnull_rates = df.notna().mean().values  # 已向量化，无需改动
        return float(np.mean(nonnull_rates)) if nonnull_rates.size > 0 else 0.0

    # ==================== 向量化改造的 _score_accuracy ====================
    def _score_accuracy(self, df: pd.DataFrame, variable_types: Dict[str, str]) -> float:
        """
        准确性得分：基于异常值检测（向量化计算所有连续变量）
        """
        if df.empty:
            return 0.0

        # 只处理连续变量
        continuous_cols = [col for col, typ in variable_types.items() if typ == 'continuous' and col in df.columns]
        if not continuous_cols:
            return 1.0  # 无连续变量，得满分

        # 提取连续变量数据
        cont_data = df[continuous_cols]
        # 计算每列的 Q1, Q3, IQR（向量化）
        q1 = cont_data.quantile(0.25, numeric_only=True)
        q3 = cont_data.quantile(0.75, numeric_only=True)
        iqr = q3 - q1

        # 计算每列的异常值比例
        scores = []
        for col in continuous_cols:
            series = cont_data[col].dropna()
            if len(series) < 3:
                scores.append(1.0)
                continue
            lower = q1[col] - 1.5 * iqr[col]
            upper = q3[col] + 1.5 * iqr[col]
            # 如果 IQR 为 0，则全部视为正常
            if iqr[col] == 0:
                scores.append(1.0)
            else:
                outliers = ((series < lower) | (series > upper)).sum()
                col_score = 1 - (outliers / len(series))
                scores.append(max(0, col_score))

        return float(np.mean(scores)) if scores else 1.0

    def _score_consistency(self, df: pd.DataFrame, audit_rules: Optional[Dict[str, Any]]) -> float:
        """
        一致性得分：勾稽规则满足率

        返回: 0-1 之间的分数
        """
        if not audit_rules:
            # 没有规则时，一致性得分为 1.0（满分）
            return 1.0

        # 收集所有规则
        all_rules = []
        all_rules.extend(audit_rules.get("arithmetic_rules", []))
        all_rules.extend(audit_rules.get("temporal_rules", []))
        all_rules.extend(audit_rules.get("foreign_keys", []))

        if not all_rules:
            return 1.0

        total_satisfied = 0
        total_rules = 0

        for rule in all_rules:
            fields = rule.get("fields", [])
            if not fields:
                continue

            # 检查所有字段是否都在 df 中
            valid_fields = [f for f in fields if f in df.columns]
            if len(valid_fields) != len(fields):
                continue

            # 获取有效行
            valid_mask = df[fields].notna().all(axis=1)
            valid_rows = valid_mask.sum()

            if valid_rows == 0:
                continue

            total_rules += 1

            # 解析并验证规则
            rule_str = rule.get("rule", "")
            satisfied = self._verify_rule(df[valid_mask], rule_str, fields)
            if satisfied:
                total_satisfied += 1

        return total_satisfied / total_rules if total_rules > 0 else 1.0

    def _verify_rule(self, df_subset: pd.DataFrame, rule_str: str, fields: List[str]) -> bool:
        """
        验证单条规则是否满足

        简化实现：检查规则中的字段是否满足基本的相等/不等关系
        """
        if "=" in rule_str:
            left, right = rule_str.split("=")
            left_fields = [f.strip() for f in left.split("+") if f.strip()]
            right_fields = [f.strip() for f in right.split("+") if f.strip()]

            # 如果左右都是单个字段，检查是否相等
            if len(left_fields) == 1 and len(right_fields) == 1:
                f1, f2 = left_fields[0], right_fields[0]
                if f1 in df_subset.columns and f2 in df_subset.columns:
                    # 数值比较
                    if pd.api.types.is_numeric_dtype(df_subset[f1]) and pd.api.types.is_numeric_dtype(df_subset[f2]):
                        diff = (df_subset[f1] - df_subset[f2]).abs()
                        return (diff < 1e-6).all()
                    # 字符串比较
                    return (df_subset[f1] == df_subset[f2]).all()

        # 默认：无法验证则视为满足
        return True

    def _score_timeliness(self, last_updated: Optional[datetime]) -> float:
        """
        及时性得分：基于数据最后更新时间

        返回: 0-1 之间的分数
        """
        if last_updated is None:
            # 无时间信息时，给 0.8 分（假设基本及时）
            return 0.8

        days_ago = (datetime.now() - last_updated).days
        if days_ago <= 1:
            return 1.0
        elif days_ago <= self.TIMELINESS_THRESHOLD:
            return 1.0 - (days_ago / self.TIMELINESS_THRESHOLD) * 0.5
        else:
            return max(0, 0.5 - (days_ago - self.TIMELINESS_THRESHOLD) / 30)

    def _score_uniqueness(self, df: pd.DataFrame, id_columns: Optional[List[str]]) -> float:
        """
        唯一性得分：基于重复记录比例

        返回: 0-1 之间的分数
        """
        if df.empty:
            return 0.0

        if id_columns:
            # 使用指定的标识符列
            valid_ids = [col for col in id_columns if col in df.columns]
            if valid_ids:
                dup_count = df.duplicated(subset=valid_ids).sum()
                return 1 - (dup_count / len(df))
            else:
                # 回退到全列检查
                dup_count = df.duplicated().sum()
                return 1 - (dup_count / len(df))
        else:
            # 全列检查
            dup_count = df.duplicated().sum()
            return 1 - (dup_count / len(df))

    def _score_fields(self, df: pd.DataFrame, variable_types: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """计算每个字段的得分"""
        field_scores = {}

        for col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                field_scores[col] = {"completeness": 0.0, "accuracy": 1.0}
                continue

            # 完整性
            completeness = df[col].notna().mean()

            # 准确性（简化）
            var_type = variable_types.get(col, "unknown")
            if var_type == "continuous" and len(series) > 3:
                Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                    outliers = ((series < lower) | (series > upper)).sum()
                    accuracy = 1 - (outliers / len(series))
                else:
                    accuracy = 1.0
            else:
                accuracy = 1.0

            field_scores[col] = {
                "completeness": round(completeness * 100, 1),
                "accuracy": round(accuracy * 100, 1),
            }

        return field_scores

    def _generate_alerts(
        self,
        dimensions: Dict[str, float],
        field_scores: Dict[str, Dict[str, float]],
        audit_rules: Optional[Dict[str, Any]]
    ) -> List[Dict]:
        """生成告警"""
        alerts = []

        # 维度告警
        for dim, score in dimensions.items():
            if score < 0.6:
                alerts.append({
                    "level": "error",
                    "dimension": dim,
                    "message": f"{dim} 得分偏低: {score * 100:.1f}%",
                    "threshold": 60,
                    "current": score * 100,
                })
            elif score < 0.8:
                alerts.append({
                    "level": "warning",
                    "dimension": dim,
                    "message": f"{dim} 得分一般: {score * 100:.1f}%",
                    "threshold": 80,
                    "current": score * 100,
                })

        # 字段告警
        for col, scores in field_scores.items():
            if scores.get("completeness", 100) < 50:
                alerts.append({
                    "level": "error",
                    "field": col,
                    "message": f"{col} 缺失率过高: {100 - scores['completeness']:.1f}%",
                    "threshold": 50,
                    "current": scores["completeness"],
                })
            elif scores.get("accuracy", 100) < 60:
                alerts.append({
                    "level": "warning",
                    "field": col,
                    "message": f"{col} 异常值比例过高: {100 - scores['accuracy']:.1f}%",
                    "threshold": 40,
                    "current": scores["accuracy"],
                })

        # 规则告警
        if audit_rules:
            rules = audit_rules.get("arithmetic_rules", [])
            for rule in rules[:5]:
                if rule.get("violation_count", 0) > 0:
                    alerts.append({
                        "level": "warning",
                        "rule": rule.get("rule", ""),
                        "message": f"勾稽规则违反: {rule.get('rule', '')}",
                        "violations": rule.get("violation_count", 0),
                    })

        return alerts[:10]

    def _get_thresholds(self) -> Dict[str, float]:
        """获取各维度阈值"""
        return {
            "completeness": 0.8,
            "accuracy": 0.8,
            "consistency": 0.7,
            "timeliness": 0.7,
            "uniqueness": 0.9,
        }


def quick_score(
    df: pd.DataFrame,
    table_name: str = "unknown",
    **kwargs
) -> QualityScore:
    """
    快速评分（便捷函数）

    参数:
    - df: 数据框
    - table_name: 表名
    - **kwargs: 传递给 QualityScorer.score 的参数

    返回: QualityScore 对象
    """
    scorer = QualityScorer()
    return scorer.score(df, table_name=table_name, **kwargs)