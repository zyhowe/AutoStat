"""质量报告服务 - 不依赖 web/"""
from typing import Dict, Any, List
import pandas as pd

from autostat.core.scorer import QualityScorer


class QualityService:
    """质量报告服务"""

    @staticmethod
    def get_score(df: pd.DataFrame, variable_types: Dict[str, str]) -> Dict[str, Any]:
        """获取质量评分"""
        scorer = QualityScorer()
        result = scorer.score(df, variable_types=variable_types)

        return {
            "overall_score": result.overall_score,
            "grade": result.grade,
            "grade_icon": result.grade_icon,
            "dimensions": result.dimensions,
            "alerts": result.alerts,
            "field_scores": result.field_scores,
            "details": result.details
        }

    @staticmethod
    def get_issues(quality_result: Dict) -> List[Dict]:
        """获取问题清单"""
        alerts = quality_result.get("alerts", [])
        issues = []

        for alert in alerts:
            if alert.get("level") in ["error", "warning"]:
                issues.append({
                    "level": alert.get("level"),
                    "dimension": alert.get("dimension"),
                    "field": alert.get("field"),
                    "message": alert.get("message"),
                    "threshold": alert.get("threshold"),
                    "current": alert.get("current")
                })

        return issues

    @staticmethod
    def get_suggestions(quality_result: Dict) -> List[str]:
        """获取清洗建议"""
        suggestions = []
        alerts = quality_result.get("alerts", [])

        for alert in alerts:
            if alert.get("level") == "error":
                if alert.get("dimension") == "completeness":
                    suggestions.append(f"处理 {alert.get('field', '未知字段')} 的缺失值")
                elif alert.get("dimension") == "accuracy":
                    suggestions.append(f"检查 {alert.get('field', '未知字段')} 的异常值")
                elif alert.get("dimension") == "consistency":
                    suggestions.append("检查数据一致性")
                elif alert.get("dimension") == "uniqueness":
                    suggestions.append("去重处理")

        if not suggestions:
            suggestions.append("数据质量良好，无需清洗")

        return suggestions