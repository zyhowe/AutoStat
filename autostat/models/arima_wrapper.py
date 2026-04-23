"""
ARIMA 模型包装器 - 使其符合 sklearn 风格的 fit(X, y) 接口
"""

import numpy as np
import pandas as pd
from typing import Optional, Union


class ARIMAWrapper:
    """
    ARIMA 模型包装器，提供与 sklearn 兼容的 fit/predict 接口

    使用 statsmodels.tsa.arima.model.ARIMA 作为后端
    """

    def __init__(self, p: int = 1, d: int = 1, q: int = 1):
        """
        初始化 ARIMA 模型

        参数:
        - p: 自回归阶数 (AR)
        - d: 差分阶数 (I)
        - q: 移动平均阶数 (MA)
        """
        self.p = p
        self.d = d
        self.q = q
        self.model = None
        self.result = None
        self._fitted = False
        self._last_series = None
        self._order = (p, d, q)

    def fit(self, X: Union[np.ndarray, pd.Series, pd.DataFrame], y: Optional[np.ndarray] = None):
        """
        拟合 ARIMA 模型

        参数:
        - X: 时间序列数据（一维）
        - y: 忽略（为保持接口一致）

        返回:
        - self
        """
        # 转换为一维数组
        if isinstance(X, pd.DataFrame):
            # 取第一列
            series = X.iloc[:, 0].values
        elif isinstance(X, pd.Series):
            series = X.values
        else:
            series = np.array(X).flatten()

        # 删除 NaN
        series = series[~np.isnan(series)]

        if len(series) < 10:
            raise ValueError(f"时间序列样本量不足: {len(series)} < 10")

        self._last_series = series

        try:
            from statsmodels.tsa.arima.model import ARIMA

            self.model = ARIMA(series, order=self._order)
            self.result = self.model.fit()
            self._fitted = True

        except ImportError:
            raise ImportError("statsmodels 未安装，请运行: pip install statsmodels")

        return self

    def predict(self, X: Union[np.ndarray, pd.Series, pd.DataFrame, int]) -> np.ndarray:
        """
        预测未来值

        参数:
        - X: 预测步数（整数）或 特征（忽略实际值）

        返回:
        - 预测值数组
        """
        if not self._fitted or self.result is None:
            raise ValueError("模型尚未拟合，请先调用 fit()")

        # 如果 X 是整数，表示预测步数
        if isinstance(X, (int, np.integer)):
            steps = int(X)
            forecast = self.result.forecast(steps=steps)
            return np.array(forecast)

        # 如果 X 是数组，取其长度作为预测步数
        if hasattr(X, '__len__') and not isinstance(X, str):
            steps = len(X)
            forecast = self.result.forecast(steps=steps)
            return np.array(forecast)

        # 默认预测 1 步
        forecast = self.result.forecast(steps=1)
        return np.array(forecast)

    def predict_proba(self, X=None):
        """ARIMA 不支持概率预测"""
        raise NotImplementedError("ARIMA 不支持 predict_proba")

    def forecast(self, steps: int = 1) -> np.ndarray:
        """
        预测未来 steps 步

        参数:
        - steps: 预测步数

        返回:
        - 预测值数组
        """
        return self.predict(steps)

    def get_params(self, deep: bool = True) -> dict:
        """获取模型参数"""
        return {
            'p': self.p,
            'd': self.d,
            'q': self.q
        }

    def set_params(self, **params):
        """设置模型参数"""
        for key, value in params.items():
            if key in ['p', 'd', 'q']:
                setattr(self, key, value)
        self._order = (self.p, self.d, self.q)
        return self

    def summary(self) -> str:
        """返回模型摘要"""
        if not self._fitted or self.result is None:
            return "模型尚未拟合"

        return str(self.result.summary())

    @property
    def aic(self) -> float:
        """AIC 信息准则"""
        if not self._fitted or self.result is None:
            return np.nan
        return self.result.aic

    @property
    def bic(self) -> float:
        """BIC 信息准则"""
        if not self._fitted or self.result is None:
            return np.nan
        return self.result.bic