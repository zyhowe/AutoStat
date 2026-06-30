"""
决策支持服务
"""

from typing import Dict, Any, Optional, List
import pandas as pd

from autostat.core.decision import AnomalyDetector, RootCauseAnalyzer, ActionRecommender
from autostat.shared.storage import SharedStorage
from autostat.shared.schemas import Anomaly, RootCause, ActionSuggestion


class DecisionService:
    """决策支持服务"""

    @staticmethod
    def detect_anomalies(
        analysis_result: Dict[str, Any]
    ) -> List[Anomaly]:
        """检测异常"""
        detector = AnomalyDetector()
        return detector.detect_from_analysis_result(analysis_result)

    @staticmethod
    def save_anomalies(session_id: str, anomalies: List[Anomaly]):
        """保存异常"""
        SharedStorage.save_anomalies(session_id, anomalies)

    @staticmethod
    def get_anomalies(session_id: str) -> List[Anomaly]:
        """获取异常"""
        return SharedStorage.load_anomalies(session_id)

    @staticmethod
    def analyze_root_cause(
        anomaly: Anomaly,
        data: pd.DataFrame,
        dimension_cols: List[str],
        metric_cols: List[str]
    ) -> Dict[str, Any]:
        """根因分析"""
        analyzer = RootCauseAnalyzer()
        result = analyzer.analyze(
            anomaly.__dict__,
            data,
            dimension_cols,
            metric_cols
        )
        return result.__dict__

    @staticmethod
    def generate_suggestions(
        anomaly: Anomaly,
        root_causes: List[RootCause],
        llm_client=None
    ) -> List[ActionSuggestion]:
        """生成建议"""
        recommender = ActionRecommender(llm_client)
        return recommender.recommend(
            anomaly.__dict__,
            [rc.__dict__ for rc in root_causes],
            {}
        )

    @staticmethod
    def save_suggestions(session_id: str, suggestions: List[ActionSuggestion]):
        """保存建议"""
        SharedStorage.save_suggestions(session_id, suggestions)

    @staticmethod
    def get_suggestions(session_id: str) -> List[ActionSuggestion]:
        """获取建议"""
        return SharedStorage.load_suggestions(session_id)