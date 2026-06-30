"""
预测预警服务
"""

from typing import Dict, Any, Optional, List
import pandas as pd

from autostat.core.forecast import Forecaster, AlertEngine, ForecastMonitor
from autostat.core.forecast.alert import console_alert_notifier
from autostat.shared.storage import SharedStorage
from autostat.shared.schemas import ForecastResult, AlertEvent


class ForecastService:
    """预测预警服务"""

    @staticmethod
    def forecast(
        data: pd.DataFrame,
        target: str,
        periods: int = 12,
        time_col: Optional[str] = None
    ) -> ForecastResult:
        """执行预测"""
        forecaster = Forecaster()
        return forecaster.forecast(data, target, time_col, periods)

    @staticmethod
    def save_forecast(session_id: str, target: str, result: ForecastResult):
        """保存预测结果"""
        SharedStorage.save_forecast(session_id, target, result)

    @staticmethod
    def get_forecast(session_id: str, target: str) -> Optional[ForecastResult]:
        """获取预测结果"""
        return SharedStorage.load_forecast(session_id, target)

    @staticmethod
    def check_alerts(
        data: Dict[str, Any],
        session_id: str
    ) -> List[AlertEvent]:
        """检查预警"""
        engine = AlertEngine()
        engine.add_notifier(console_alert_notifier())

        events = engine.check(data)

        if events:
            SharedStorage.save_alerts(session_id, events)

        return events

    @staticmethod
    def get_alerts(session_id: str) -> List[AlertEvent]:
        """获取预警"""
        return SharedStorage.load_alerts(session_id)

    @staticmethod
    def resolve_alert(session_id: str, alert_id: str):
        """解决预警"""
        alerts = SharedStorage.load_alerts(session_id)
        for alert in alerts:
            if alert.id == alert_id:
                alert.resolved = True
                break
        SharedStorage.save_alerts(session_id, alerts)