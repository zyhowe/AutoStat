"""
时序预测模块

自动选择并执行时间序列预测
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class ForecastResult:
    """预测结果"""
    target: str
    model_name: str
    values: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    confidence: float
    metrics: Dict[str, float]
    periods: int
    timestamp: str


class Forecaster:
    """
    时序预测器

    使用方式:
        forecaster = Forecaster()
        result = forecaster.forecast(df, target, periods=12)
    """

    def __init__(
        self,
        models: Optional[List[str]] = None,
        confidence_level: float = 0.95,
        auto_select: bool = True
    ):
        """
        初始化

        参数:
        - models: 要使用的模型列表（默认自动选择）
        - confidence_level: 置信区间水平
        - auto_select: 是否自动选择最佳模型
        """
        self.models = models or ["prophet", "arima", "lstm"]
        self.confidence_level = confidence_level
        self.auto_select = auto_select

    def forecast(
        self,
        df: pd.DataFrame,
        target: str,
        time_col: Optional[str] = None,
        periods: int = 12,
        freq: str = "D"
    ) -> ForecastResult:
        """
        执行预测

        参数:
        - df: 数据框
        - target: 目标列
        - time_col: 时间列
        - periods: 预测周期数
        - freq: 频率

        返回: ForecastResult
        """
        # 准备数据
        if time_col is None:
            time_col = self._detect_time_col(df)

        if time_col is None:
            raise ValueError("未检测到时间列")

        # 提取时间序列
        data = df[[time_col, target]].dropna()
        data[time_col] = pd.to_datetime(data[time_col])
        data = data.sort_values(time_col)

        if len(data) < 10:
            raise ValueError(f"数据不足，需要至少10个时间点，当前{len(data)}")

        # 尝试使用 Prophet
        try:
            return self._forecast_with_prophet(data, time_col, target, periods)
        except Exception as e:
            print(f"Prophet预测失败: {e}")

        # 降级到ARIMA
        try:
            return self._forecast_with_arima(data, time_col, target, periods)
        except Exception as e:
            print(f"ARIMA预测失败: {e}")

        # 最终降级：移动平均
        return self._forecast_with_ma(data, target, periods)

    def _detect_time_col(self, df: pd.DataFrame) -> Optional[str]:
        """检测时间列"""
        time_cols = df.select_dtypes(include=['datetime64']).columns
        if len(time_cols) > 0:
            return time_cols[0]

        # 尝试转换
        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                return col
            except:
                continue

        return None

    def _forecast_with_prophet(
        self,
        data: pd.DataFrame,
        time_col: str,
        target: str,
        periods: int
    ) -> ForecastResult:
        """使用 Prophet 预测"""
        try:
            from prophet import Prophet

            # 准备数据
            df_prophet = data.rename(columns={time_col: "ds", target: "y"})

            # 创建模型
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=self.confidence_level
            )

            # 拟合
            model.fit(df_prophet)

            # 预测
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)

            # 提取结果
            pred_values = forecast["yhat"].values[-periods:]
            lower = forecast["yhat_lower"].values[-periods:]
            upper = forecast["yhat_upper"].values[-periods:]

            # 计算指标（使用历史数据）
            historical = forecast["yhat"].values[:-periods]
            actual = df_prophet["y"].values
            mape = np.mean(np.abs((actual - historical[:len(actual)]) / (actual + 1e-6))) * 100

            return ForecastResult(
                target=target,
                model_name="Prophet",
                values=pred_values,
                lower_bound=lower,
                upper_bound=upper,
                confidence=self.confidence_level,
                metrics={"mape": mape},
                periods=periods,
                timestamp=datetime.now().isoformat()
            )

        except ImportError:
            raise ImportError("Prophet未安装，请运行: pip install prophet")

    def _forecast_with_arima(
        self,
        data: pd.DataFrame,
        time_col: str,
        target: str,
        periods: int
    ) -> ForecastResult:
        """使用 ARIMA 预测"""
        try:
            from statsmodels.tsa.arima.model import ARIMA

            series = data[target].values

            # 自动选择参数
            # 简单实现：尝试几个组合
            best_aic = float('inf')
            best_order = (1, 1, 1)

            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            model = ARIMA(series, order=(p, d, q))
                            result = model.fit()
                            if result.aic < best_aic:
                                best_aic = result.aic
                                best_order = (p, d, q)
                        except:
                            continue

            model = ARIMA(series, order=best_order)
            result = model.fit()

            # 预测
            forecast_result = result.forecast(steps=periods)

            return ForecastResult(
                target=target,
                model_name=f"ARIMA{best_order}",
                values=forecast_result.values,
                lower_bound=forecast_result.values - 1.96 * np.std(forecast_result.values),
                upper_bound=forecast_result.values + 1.96 * np.std(forecast_result.values),
                confidence=self.confidence_level,
                metrics={"aic": best_aic},
                periods=periods,
                timestamp=datetime.now().isoformat()
            )

        except ImportError:
            raise ImportError("statsmodels未安装，请运行: pip install statsmodels")

    def _forecast_with_ma(
        self,
        data: pd.DataFrame,
        target: str,
        periods: int
    ) -> ForecastResult:
        """简单移动平均预测（降级方案）"""
        series = data[target].values

        # 使用最后3个点的平均变化
        if len(series) >= 4:
            diffs = np.diff(series[-4:])
            avg_change = np.mean(diffs)
            last_value = series[-1]

            pred_values = [last_value + avg_change * (i + 1) for i in range(periods)]
            std = np.std(diffs) if len(diffs) > 0 else 1

            return ForecastResult(
                target=target,
                model_name="移动平均",
                values=np.array(pred_values),
                lower_bound=np.array(pred_values) - 1.96 * std,
                upper_bound=np.array(pred_values) + 1.96 * std,
                confidence=0.8,
                metrics={},
                periods=periods,
                timestamp=datetime.now().isoformat()
            )
        else:
            # 用平均值
            mean_val = np.mean(series)
            return ForecastResult(
                target=target,
                model_name="均值预测",
                values=np.full(periods, mean_val),
                lower_bound=np.full(periods, mean_val * 0.8),
                upper_bound=np.full(periods, mean_val * 1.2),
                confidence=0.6,
                metrics={},
                periods=periods,
                timestamp=datetime.now().isoformat()
            )

    def get_forecast_chart_data(self, result: ForecastResult) -> Dict[str, Any]:
        """获取图表数据"""
        return {
            "target": result.target,
            "model_name": result.model_name,
            "periods": result.periods,
            "values": result.values.tolist(),
            "lower_bound": result.lower_bound.tolist(),
            "upper_bound": result.upper_bound.tolist(),
            "confidence": result.confidence,
            "metrics": result.metrics,
            "timestamp": result.timestamp
        }


def forecast(
    df: pd.DataFrame,
    target: str,
    **kwargs
) -> ForecastResult:
    """便捷函数：执行预测"""
    forecaster = Forecaster(**kwargs)
    return forecaster.forecast(df, target)