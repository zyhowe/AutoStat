"""
预测监控模块

监控预测效果，检测模型漂移
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class ForecastSnapshot:
    """预测快照"""
    timestamp: str
    target: str
    actual: float
    predicted: float
    error: float
    error_pct: float


@dataclass
class MonitorResult:
    """监控结果"""
    status: str  # "good", "warning", "critical"
    message: str
    metrics: Dict[str, float]
    recent_errors: List[float]
    drift_detected: bool


class ForecastMonitor:
    """
    预测监控器

    使用方式:
        monitor = ForecastMonitor()
        result = monitor.check(predictions, actuals)
    """

    def __init__(
        self,
        max_history: int = 100,
        error_threshold: float = 0.1,
        drift_threshold: float = 0.2
    ):
        """
        初始化

        参数:
        - max_history: 最大历史记录数
        - error_threshold: 错误率阈值
        - drift_threshold: 漂移检测阈值
        """
        self.max_history = max_history
        self.error_threshold = error_threshold
        self.drift_threshold = drift_threshold
        self.history: List[ForecastSnapshot] = []

    def add_snapshot(
        self,
        target: str,
        actual: float,
        predicted: float
    ):
        """添加快照"""
        error = actual - predicted
        error_pct = error / (abs(actual) + 1e-6)

        snapshot = ForecastSnapshot(
            timestamp=datetime.now().isoformat(),
            target=target,
            actual=actual,
            predicted=predicted,
            error=error,
            error_pct=error_pct
        )

        self.history.append(snapshot)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def check(self) -> MonitorResult:
        """
        检查监控状态

        返回: MonitorResult
        """
        if len(self.history) < 3:
            return MonitorResult(
                status="good",
                message="数据不足，需要至少3个快照",
                metrics={},
                recent_errors=[],
                drift_detected=False
            )

        # 计算指标
        errors = [s.error_pct for s in self.history]
        recent_errors = errors[-10:] if len(errors) >= 10 else errors

        mean_error = np.mean(recent_errors) * 100
        std_error = np.std(recent_errors) * 100
        mape = np.mean(np.abs(recent_errors)) * 100

        # 检测漂移
        drift_detected = self._detect_drift()

        # 判断状态
        if mape > self.error_threshold * 200:
            status = "critical"
            message = f"预测误差过高: MAPE={mape:.1f}%"
        elif mape > self.error_threshold * 100:
            status = "warning"
            message = f"预测误差偏高: MAPE={mape:.1f}%"
        elif drift_detected:
            status = "warning"
            message = "检测到模型漂移，建议重新训练"
        else:
            status = "good"
            message = f"预测效果良好: MAPE={mape:.1f}%"

        return MonitorResult(
            status=status,
            message=message,
            metrics={
                "mape": mape,
                "mean_error": mean_error,
                "std_error": std_error,
                "n_samples": len(self.history)
            },
            recent_errors=recent_errors,
            drift_detected=drift_detected
        )

    def _detect_drift(self) -> bool:
        """
        检测模型漂移

        通过比较近期误差与历史误差的分布差异
        """
        if len(self.history) < 20:
            return False

        errors = [s.error_pct for s in self.history]
        n = len(errors)

        # 前70% vs 后30%
        split = int(n * 0.7)
        early_errors = errors[:split]
        late_errors = errors[split:]

        if len(early_errors) < 5 or len(late_errors) < 5:
            return False

        # KS检验或简单比较
        early_mean = np.mean(np.abs(early_errors))
        late_mean = np.mean(np.abs(late_errors))

        # 如果近期误差比历史误差高20%以上，视为漂移
        if late_mean > early_mean * (1 + self.drift_threshold):
            return True

        return False

    def get_trend_data(self) -> List[Dict[str, Any]]:
        """获取趋势数据（用于可视化）"""
        return [
            {
                "timestamp": s.timestamp[:19],
                "actual": s.actual,
                "predicted": s.predicted,
                "error": s.error_pct * 100
            }
            for s in self.history[-50:]
        ]

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        if not self.history:
            return {"has_data": False}

        errors = [s.error_pct for s in self.history]
        recent_errors = errors[-10:] if len(errors) >= 10 else errors

        return {
            "has_data": True,
            "total_checks": len(self.history),
            "recent_checks": len(recent_errors),
            "mape": np.mean(np.abs(recent_errors)) * 100,
            "mean_error": np.mean(recent_errors) * 100,
            "max_error": np.max(np.abs(recent_errors)) * 100,
            "drift_detected": self._detect_drift()
        }


def check_forecast(
    actuals: List[float],
    predictions: List[float],
    **kwargs
) -> MonitorResult:
    """便捷函数：检查预测"""
    if len(actuals) != len(predictions):
        raise ValueError("实际值和预测值长度不一致")

    monitor = ForecastMonitor(**kwargs)
    for a, p in zip(actuals, predictions):
        monitor.add_snapshot("target", a, p)

    return monitor.check()