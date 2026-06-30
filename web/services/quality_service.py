"""
质量监控服务
"""

from typing import Dict, Any, Optional, List

from autostat.core.quality import QualityScorer, QualityMonitor, QualityAlert
from autostat.core.quality.alert import console_notifier
from autostat.shared.storage import SharedStorage
from autostat.shared.schemas import QualityScore, QualitySnapshot


class QualityService:
    """质量监控服务"""

    @staticmethod
    def score_data(data, session_id: str, table_name: str = "unknown") -> Optional[QualityScore]:
        """评分数据"""
        scorer = QualityScorer()
        result = scorer.score(data, table_name=table_name)

        if result:
            SharedStorage.save_quality_score(session_id, result)

        return result

    @staticmethod
    def get_latest_score(session_id: str) -> Optional[QualityScore]:
        """获取最新评分"""
        return SharedStorage.load_quality_score(session_id)

    @staticmethod
    def get_history(session_id: str) -> List[Dict]:
        """获取历史趋势"""
        return SharedStorage.load_quality_history(session_id)

    @staticmethod
    def get_trend_data(session_id: str) -> Dict[str, Any]:
        """获取趋势数据"""
        history = SharedStorage.load_quality_history(session_id)

        if not history:
            return {"has_data": False}

        scores = [h.get("overall_score", 0) for h in history]

        return {
            "has_data": True,
            "current": scores[-1] if scores else 0,
            "history": history,
            "min": min(scores) if scores else 0,
            "max": max(scores) if scores else 0,
            "trend": "up" if len(scores) > 1 and scores[-1] > scores[-2] else "down"
        }

    @staticmethod
    def check_alerts(session_id: str, score_data: Dict) -> List:
        """检查告警"""
        alert = QualityAlert()
        alert.add_notifier(console_notifier())

        events = alert.check(score_data)

        if events:
            SharedStorage.save_alerts(session_id, events)

        return events

    @staticmethod
    def get_alerts(session_id: str) -> List:
        """获取告警"""
        return SharedStorage.load_alerts(session_id)