"""
质量趋势监控

监控数据质量的历史变化，检测趋势异常和突变
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque


@dataclass
class QualitySnapshot:
    """质量快照"""
    timestamp: str
    overall_score: float
    dimensions: Dict[str, float]
    alert_count: int


@dataclass
class TrendResult:
    """趋势分析结果"""
    direction: str  # "up", "down", "stable"
    change_pct: float
    is_anomaly: bool
    anomaly_type: Optional[str]  # "spike", "drop"
    message: str


@dataclass
class MonitorResult:
    """监控结果"""
    current: QualitySnapshot
    history: List[QualitySnapshot]
    trend: TrendResult
    anomalies: List[Dict]
    summary: str


class QualityMonitor:
    """
    质量趋势监控

    使用方式:
        monitor = QualityMonitor()
        monitor.add_snapshot(snapshot)
        result = monitor.analyze(window=30)
    """

    def __init__(self, max_history: int = 365):
        """
        初始化监控器

        参数:
        - max_history: 最大历史记录数
        """
        self.history: List[QualitySnapshot] = []
        self.max_history = max_history

    def add_snapshot(self, snapshot: QualitySnapshot):
        """添加快照"""
        self.history.append(snapshot)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def add_score(self, score_result) -> QualitySnapshot:
        """从评分结果创建快照并添加"""
        snapshot = QualitySnapshot(
            timestamp=score_result.timestamp,
            overall_score=score_result.overall_score,
            dimensions=score_result.dimensions,
            alert_count=len(score_result.alerts)
        )
        self.add_snapshot(snapshot)
        return snapshot

    def analyze(self, window: int = 30) -> MonitorResult:
        """
        分析趋势

        参数:
        - window: 分析窗口（天）

        返回: MonitorResult
        """
        if len(self.history) < 3:
            return self._empty_result("数据不足，至少需要3个快照")

        # 获取窗口数据
        recent = self.history[-window:] if window < len(self.history) else self.history
        current = recent[-1]

        # 计算趋势
        trend = self._analyze_trend(recent)

        # 检测异常
        anomalies = self._detect_anomalies(recent)

        # 生成摘要
        summary = self._generate_summary(current, trend, anomalies)

        return MonitorResult(
            current=current,
            history=recent,
            trend=trend,
            anomalies=anomalies,
            summary=summary
        )

    def _analyze_trend(self, snapshots: List[QualitySnapshot]) -> TrendResult:
        """分析趋势方向"""
        if len(snapshots) < 3:
            return TrendResult(
                direction="stable",
                change_pct=0.0,
                is_anomaly=False,
                anomaly_type=None,
                message="数据不足"
            )

        scores = [s.overall_score for s in snapshots]
        n = len(scores)

        # 计算变化率
        start = scores[0]
        end = scores[-1]
        change_pct = ((end - start) / start * 100) if start > 0 else 0

        # 判断方向
        if change_pct > 5:
            direction = "up"
        elif change_pct < -5:
            direction = "down"
        else:
            direction = "stable"

        # 检测是否为异常
        is_anomaly = False
        anomaly_type = None

        if len(scores) >= 7:
            # 检测突变
            recent_avg = np.mean(scores[-3:])
            prev_avg = np.mean(scores[-7:-3])
            if recent_avg > prev_avg * 1.15:
                is_anomaly = True
                anomaly_type = "spike"
            elif recent_avg < prev_avg * 0.85:
                is_anomaly = True
                anomaly_type = "drop"

        # 检测持续恶化（连续5期下降）
        if len(scores) >= 5:
            recent_scores = scores[-5:]
            if all(recent_scores[i] > recent_scores[i+1] for i in range(len(recent_scores)-1)):
                is_anomaly = True
                anomaly_type = "continuous_drop"

        message = self._format_trend_message(direction, change_pct, is_anomaly, anomaly_type)

        return TrendResult(
            direction=direction,
            change_pct=round(change_pct, 1),
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            message=message
        )

    def _detect_anomalies(self, snapshots: List[QualitySnapshot]) -> List[Dict]:
        """检测异常点"""
        anomalies = []
        if len(snapshots) < 5:
            return anomalies

        scores = [s.overall_score for s in snapshots]
        mean = np.mean(scores)
        std = np.std(scores)

        if std == 0:
            return anomalies

        for i, s in enumerate(snapshots):
            z_score = (s.overall_score - mean) / std
            if abs(z_score) > 2.5:
                anomalies.append({
                    "timestamp": s.timestamp,
                    "score": s.overall_score,
                    "z_score": round(z_score, 2),
                    "type": "spike" if z_score > 0 else "drop",
                    "severity": "high" if abs(z_score) > 3 else "medium",
                })

        return anomalies

    def _generate_summary(
        self,
        current: QualitySnapshot,
        trend: TrendResult,
        anomalies: List[Dict]
    ) -> str:
        """生成摘要"""
        parts = [
            f"当前质量评分: {current.overall_score:.1f}",
            f"趋势: {trend.message}"
        ]

        if anomalies:
            parts.append(f"发现 {len(anomalies)} 个异常点")

        return " | ".join(parts)

    def _format_trend_message(
        self,
        direction: str,
        change_pct: float,
        is_anomaly: bool,
        anomaly_type: Optional[str]
    ) -> str:
        """格式化趋势消息"""
        if is_anomaly:
            if anomaly_type == "spike":
                return f"突增 {change_pct:.1f}% ⚠️"
            elif anomaly_type == "drop":
                return f"突降 {abs(change_pct):.1f}% 🚨"
            elif anomaly_type == "continuous_drop":
                return f"持续下降 {abs(change_pct):.1f}% 🚨"
            else:
                return f"异常波动 {change_pct:.1f}% ⚠️"

        if direction == "up":
            return f"上升 {change_pct:.1f}% 📈"
        elif direction == "down":
            return f"下降 {abs(change_pct):.1f}% 📉"
        else:
            return f"稳定 ({change_pct:.1f}%) ✅"

    def _empty_result(self, message: str) -> MonitorResult:
        """空结果"""
        return MonitorResult(
            current=QualitySnapshot(
                timestamp=datetime.now().isoformat(),
                overall_score=0,
                dimensions={},
                alert_count=0
            ),
            history=[],
            trend=TrendResult(
                direction="stable",
                change_pct=0.0,
                is_anomaly=False,
                anomaly_type=None,
                message=message
            ),
            anomalies=[],
            summary=message
        )

    def get_trend_data(self, dimension: Optional[str] = None) -> List[Dict]:
        """获取趋势数据（用于可视化）"""
        if dimension:
            return [
                {"timestamp": s.timestamp, "value": s.dimensions.get(dimension, 0)}
                for s in self.history
            ]
        else:
            return [
                {"timestamp": s.timestamp, "value": s.overall_score}
                for s in self.history
            ]


def create_snapshot_from_scorer(score_result) -> QualitySnapshot:
    """从评分结果创建快照"""
    return QualitySnapshot(
        timestamp=score_result.timestamp,
        overall_score=score_result.overall_score,
        dimensions=score_result.dimensions,
        alert_count=len(score_result.alerts)
    )