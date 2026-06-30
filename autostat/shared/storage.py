"""
统一存储扩展
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from autostat.shared.schemas import (
    QualityScore,
    QualitySnapshot,
    Anomaly,
    RootCause,
    ActionSuggestion,
    ForecastResult,
    AlertEvent
)


class SharedStorage:
    """
    跨模块共享存储

    用于存储质量评分、异常事件、预测结果等
    """

    BASE_PATH = Path.home() / ".autostat" / "modules"

    @classmethod
    def _ensure_dir(cls):
        cls.BASE_PATH.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _get_session_path(cls, session_id: str) -> Path:
        cls._ensure_dir()
        path = cls.BASE_PATH / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ==================== 质量评分存储 ====================

    @classmethod
    def save_quality_score(cls, session_id: str, score: QualityScore):
        """保存质量评分"""
        path = cls._get_session_path(session_id)
        with open(path / "quality_score.json", "w", encoding="utf-8") as f:
            json.dump(score.__dict__, f, ensure_ascii=False, indent=2, default=str)

        # 追加到历史
        history_path = path / "quality_history.json"
        history = cls.load_quality_history(session_id)
        history.append(score.__dict__)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2, default=str)

    @classmethod
    def load_quality_score(cls, session_id: str) -> Optional[QualityScore]:
        """加载最新质量评分"""
        path = cls._get_session_path(session_id)
        file_path = path / "quality_score.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return QualityScore(**data)
        return None

    @classmethod
    def load_quality_history(cls, session_id: str) -> List[Dict]:
        """加载质量历史"""
        path = cls._get_session_path(session_id)
        file_path = path / "quality_history.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    # ==================== 异常存储 ====================

    @classmethod
    def save_anomalies(cls, session_id: str, anomalies: List[Anomaly]):
        """保存异常事件"""
        path = cls._get_session_path(session_id)
        data = [a.__dict__ for a in anomalies]
        with open(path / "anomalies.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    @classmethod
    def load_anomalies(cls, session_id: str) -> List[Anomaly]:
        """加载异常事件"""
        path = cls._get_session_path(session_id)
        file_path = path / "anomalies.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [Anomaly(**item) for item in data]
        return []

    # ==================== 决策建议存储 ====================

    @classmethod
    def save_suggestions(cls, session_id: str, suggestions: List[ActionSuggestion]):
        """保存行动建议"""
        path = cls._get_session_path(session_id)
        data = [s.__dict__ for s in suggestions]
        with open(path / "suggestions.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    @classmethod
    def load_suggestions(cls, session_id: str) -> List[ActionSuggestion]:
        """加载行动建议"""
        path = cls._get_session_path(session_id)
        file_path = path / "suggestions.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [ActionSuggestion(**item) for item in data]
        return []

    # ==================== 预测结果存储 ====================

    @classmethod
    def save_forecast(cls, session_id: str, target: str, result: ForecastResult):
        """保存预测结果"""
        path = cls._get_session_path(session_id)
        data = result.__dict__
        with open(path / f"forecast_{target}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    @classmethod
    def load_forecast(cls, session_id: str, target: str) -> Optional[ForecastResult]:
        """加载预测结果"""
        path = cls._get_session_path(session_id)
        file_path = path / f"forecast_{target}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return ForecastResult(**data)
        return None

    # ==================== 预警事件存储 ====================

    @classmethod
    def save_alerts(cls, session_id: str, alerts: List[AlertEvent]):
        """保存预警事件"""
        path = cls._get_session_path(session_id)
        data = []
        for a in alerts:
            item = a.__dict__.copy()
            item["level"] = a.level  # AlertLevel 是元组
            data.append(item)
        with open(path / "alerts.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    @classmethod
    def load_alerts(cls, session_id: str) -> List[AlertEvent]:
        """加载预警事件"""
        path = cls._get_session_path(session_id)
        file_path = path / "alerts.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [AlertEvent(**item) for item in data]
        return []

    # ==================== 通用存储 ====================

    @classmethod
    def save_data(cls, session_id: str, key: str, data: Any):
        """保存任意数据（pickle）"""
        path = cls._get_session_path(session_id)
        with open(path / f"{key}.pkl", "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load_data(cls, session_id: str, key: str) -> Optional[Any]:
        """加载任意数据"""
        path = cls._get_session_path(session_id)
        file_path = path / f"{key}.pkl"
        if file_path.exists():
            with open(file_path, "rb") as f:
                return pickle.load(f)
        return None

    @classmethod
    def list_sessions(cls) -> List[str]:
        """列出所有会话"""
        cls._ensure_dir()
        return [p.name for p in cls.BASE_PATH.iterdir() if p.is_dir()]