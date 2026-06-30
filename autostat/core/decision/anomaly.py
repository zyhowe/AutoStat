"""
异常发现模块

自动检测数据中的异常模式：
- 突增/突降 (Spike/Drop)
- 趋势反转 (Trend Reversal)
- 分布偏移 (Distribution Shift)
- 规则违反 (Rule Violation)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class AnomalyType(Enum):
    """异常类型"""
    SPIKE = "spike"              # 突增
    DROP = "drop"                # 突降
    TREND_REVERSAL = "trend_reversal"  # 趋势反转
    DISTRIBUTION_SHIFT = "distribution_shift"  # 分布偏移
    RULE_VIOLATION = "rule_violation"  # 规则违反
    OUTLIER = "outlier"          # 异常值


@dataclass
class Anomaly:
    """异常事件"""
    id: str
    type: AnomalyType
    severity: str  # "critical", "high", "medium", "low"
    target: str    # 目标列名
    value: Any
    expected: Any
    message: str
    timestamp: str
    data: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)


class AnomalyDetector:
    """
    异常发现器

    使用方式:
        detector = AnomalyDetector()
        anomalies = detector.detect(df, variable_types, metrics)
    """

    def __init__(
        self,
        spike_threshold: float = 3.0,
        trend_window: int = 7,
        shift_threshold: float = 0.2
    ):
        """
        初始化异常发现器

        参数:
        - spike_threshold: 突增/突降阈值（标准差倍数）
        - trend_window: 趋势检测窗口
        - shift_threshold: 分布偏移阈值
        """
        self.spike_threshold = spike_threshold
        self.trend_window = trend_window
        self.shift_threshold = shift_threshold

    def detect(
        self,
        df: pd.DataFrame,
        variable_types: Dict[str, str],
        metrics: Optional[List[str]] = None,
        time_col: Optional[str] = None
    ) -> List[Anomaly]:
        """
        检测异常

        参数:
        - df: 数据框
        - variable_types: 变量类型
        - metrics: 要检测的指标列（默认所有数值列）
        - time_col: 时间列（用于趋势检测）

        返回: 异常列表
        """
        anomalies = []

        # 获取数值列
        numeric_cols = [col for col, typ in variable_types.items()
                        if typ == "continuous" and col in df.columns]

        if metrics:
            numeric_cols = [col for col in numeric_cols if col in metrics]

        if not numeric_cols:
            return anomalies

        # 1. 检测突增/突降（基于时间序列）
        if time_col and time_col in df.columns:
            anomalies.extend(self._detect_spikes_drop(df, numeric_cols, time_col))

        # 2. 检测趋势反转
        if time_col and time_col in df.columns and len(df) >= 30:
            anomalies.extend(self._detect_trend_reversal(df, numeric_cols, time_col))

        # 3. 检测分布偏移
        anomalies.extend(self._detect_distribution_shift(df, numeric_cols))

        # 4. 检测异常值
        anomalies.extend(self._detect_outliers(df, numeric_cols))

        return anomalies

    def _detect_spikes_drop(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        time_col: str
    ) -> List[Anomaly]:
        """检测突增/突降"""
        anomalies = []

        if len(df) < 10:
            return anomalies

        df_sorted = df.sort_values(time_col)

        for col in numeric_cols:
            series = df_sorted[col].values

            if len(series) < 10:
                continue

            # 计算移动平均和标准差
            window = min(7, len(series) // 4)
            if window < 2:
                continue

            # 计算最后几个点的变化
            recent = series[-window:]
            prev = series[-2*window:-window] if len(series) >= 2*window else series[:window]

            if len(prev) == 0:
                continue

            mean_prev = np.mean(prev)
            std_prev = np.std(prev)

            if std_prev == 0:
                continue

            for i, val in enumerate(recent):
                if abs(val - mean_prev) > self.spike_threshold * std_prev:
                    anomaly_type = AnomalyType.SPIKE if val > mean_prev else AnomalyType.DROP
                    anomalies.append(Anomaly(
                        id=f"{col}_{datetime.now().timestamp()}_{i}",
                        type=anomaly_type,
                        severity="high" if abs(val - mean_prev) > 4 * std_prev else "medium",
                        target=col,
                        value=val,
                        expected=mean_prev,
                        message=f"{col} 出现{'突增' if val > mean_prev else '突降'}",
                        timestamp=datetime.now().isoformat(),
                        data={
                            "current": val,
                            "mean": mean_prev,
                            "std": std_prev,
                            "z_score": (val - mean_prev) / std_prev,
                        }
                    ))

        return anomalies

    def _detect_trend_reversal(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        time_col: str
    ) -> List[Anomaly]:
        """检测趋势反转"""
        anomalies = []

        if len(df) < 30:
            return anomalies

        df_sorted = df.sort_values(time_col)

        for col in numeric_cols:
            series = df_sorted[col].values

            if len(series) < 30:
                continue

            # 计算趋势（最近N期的斜率）
            n = 10
            if len(series) < n:
                continue

            # 最近趋势
            recent = series[-n:]
            x_recent = np.arange(n)
            slope_recent = np.polyfit(x_recent, recent, 1)[0]

            # 之前趋势
            if len(series) < 2 * n:
                continue
            prev = series[-2*n:-n]
            x_prev = np.arange(n)
            slope_prev = np.polyfit(x_prev, prev, 1)[0]

            # 判断是否反转
            if slope_prev > 0.01 * np.std(series) and slope_recent < -0.01 * np.std(series):
                anomalies.append(Anomaly(
                    id=f"trend_{col}_{datetime.now().timestamp()}",
                    type=AnomalyType.TREND_REVERSAL,
                    severity="medium",
                    target=col,
                    value=slope_recent,
                    expected=slope_prev,
                    message=f"{col} 趋势反转：由涨转跌",
                    timestamp=datetime.now().isoformat(),
                    data={
                        "slope_prev": slope_prev,
                        "slope_recent": slope_recent,
                    }
                ))
            elif slope_prev < -0.01 * np.std(series) and slope_recent > 0.01 * np.std(series):
                anomalies.append(Anomaly(
                    id=f"trend_{col}_{datetime.now().timestamp()}",
                    type=AnomalyType.TREND_REVERSAL,
                    severity="medium",
                    target=col,
                    value=slope_recent,
                    expected=slope_prev,
                    message=f"{col} 趋势反转：由跌转涨",
                    timestamp=datetime.now().isoformat(),
                    data={
                        "slope_prev": slope_prev,
                        "slope_recent": slope_recent,
                    }
                ))

        return anomalies

    def _detect_distribution_shift(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str]
    ) -> List[Anomaly]:
        """检测分布偏移"""
        anomalies = []

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 20:
                continue

            # 将数据分为前后两半
            n = len(series)
            half = n // 2
            first_half = series[:half]
            second_half = series[half:]

            if len(first_half) < 5 or len(second_half) < 5:
                continue

            # 比较均值
            mean1, mean2 = first_half.mean(), second_half.mean()
            if mean1 != 0:
                shift_pct = abs(mean2 - mean1) / abs(mean1)
            else:
                shift_pct = abs(mean2 - mean1)

            if shift_pct > self.shift_threshold:
                anomalies.append(Anomaly(
                    id=f"shift_{col}_{datetime.now().timestamp()}",
                    type=AnomalyType.DISTRIBUTION_SHIFT,
                    severity="medium" if shift_pct > 0.3 else "low",
                    target=col,
                    value=mean2,
                    expected=mean1,
                    message=f"{col} 分布偏移 {shift_pct:.1%}",
                    timestamp=datetime.now().isoformat(),
                    data={
                        "mean_before": mean1,
                        "mean_after": mean2,
                        "shift_pct": shift_pct,
                    }
                ))

        return anomalies

    def _detect_outliers(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str]
    ) -> List[Anomaly]:
        """检测异常值"""
        anomalies = []

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 5:
                continue

            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0:
                continue

            lower = Q1 - 3 * IQR  # 使用 3*IQR 更严格
            upper = Q3 + 3 * IQR

            outliers = series[(series < lower) | (series > upper)]
            if len(outliers) > 0:
                # 只报告最极端的几个
                for idx, val in outliers.head(3).items():
                    anomalies.append(Anomaly(
                        id=f"outlier_{col}_{idx}",
                        type=AnomalyType.OUTLIER,
                        severity="high" if len(outliers) > len(series) * 0.05 else "medium",
                        target=col,
                        value=val,
                        expected=(Q1 + Q3) / 2,
                        message=f"{col} 存在极端异常值: {val:.2f}",
                        timestamp=datetime.now().isoformat(),
                        data={
                            "lower_bound": lower,
                            "upper_bound": upper,
                            "Q1": Q1,
                            "Q3": Q3,
                            "IQR": IQR,
                        }
                    ))

        return anomalies

    def detect_from_analysis_result(self, analysis_result: Dict[str, Any]) -> List[Anomaly]:
        """
        从分析结果中检测异常

        参数:
        - analysis_result: 来自 AutoStatisticalAnalyzer.to_json() 的结果

        返回: 异常列表
        """
        anomalies = []

        # 1. 从时间序列诊断中检测
        ts_diag = analysis_result.get("time_series_diagnostics", {})
        for key, diag in ts_diag.items():
            # 如果检测到非平稳，可能意味着趋势变化
            if not diag.get("is_stationary", True):
                anomalies.append(Anomaly(
                    id=f"ts_{key}_{datetime.now().timestamp()}",
                    type=AnomalyType.TREND_REVERSAL,
                    severity="medium",
                    target=key,
                    value="non_stationary",
                    expected="stationary",
                    message=f"{key} 时间序列不平稳，可能存在趋势变化",
                    timestamp=datetime.now().isoformat(),
                    data={"diagnostic": diag}
                ))

        # 2. 从异常值报告中检测
        outliers = analysis_result.get("quality_report", {}).get("outliers", {})
        for col, info in outliers.items():
            if info.get("percent", 0) > 5:
                anomalies.append(Anomaly(
                    id=f"outlier_{col}_{datetime.now().timestamp()}",
                    type=AnomalyType.OUTLIER,
                    severity="high" if info.get("percent", 0) > 10 else "medium",
                    target=col,
                    value=info.get("count", 0),
                    expected=0,
                    message=f"{col} 异常值比例 {info.get('percent', 0):.1f}%",
                    timestamp=datetime.now().isoformat(),
                    data=info
                ))

        # 3. 从勾稽规则中检测
        audit_rules = analysis_result.get("quality_report", {}).get("audit_rules", {})
        for rule in audit_rules.get("arithmetic_rules", []):
            if rule.get("violation_count", 0) > 0:
                anomalies.append(Anomaly(
                    id=f"rule_{datetime.now().timestamp()}",
                    type=AnomalyType.RULE_VIOLATION,
                    severity="medium",
                    target=rule.get("rule", ""),
                    value=rule.get("violation_count", 0),
                    expected=0,
                    message=f"勾稽规则违反: {rule.get('rule', '')}",
                    timestamp=datetime.now().isoformat(),
                    data=rule
                ))

        return anomalies


def detect_anomalies(
    df: pd.DataFrame,
    variable_types: Dict[str, str],
    **kwargs
) -> List[Anomaly]:
    """便捷函数：检测异常"""
    detector = AnomalyDetector(**kwargs)
    return detector.detect(df, variable_types)