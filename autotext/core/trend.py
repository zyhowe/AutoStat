"""
时间趋势模块 - 时间序列分析
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


class TrendAnalyzer:
    """时间趋势分析器"""

    def __init__(self, texts: List[str], dates: List[Any]):
        """
        初始化趋势分析器

        参数:
        - texts: 文本列表
        - dates: 对应的时间列表（datetime 或可转换的字符串）
        """
        self.texts = texts
        self.dates = self._parse_dates(dates)

        # 按时间排序
        sorted_pairs = sorted(zip(self.dates, self.texts), key=lambda x: x[0])
        self.dates = [p[0] for p in sorted_pairs]
        self.texts = [p[1] for p in sorted_pairs]

    def _parse_dates(self, dates: List[Any]) -> List[datetime]:
        """解析日期"""
        parsed = []
        for d in dates:
            if d is None or pd.isna(d):
                parsed.append(None)
                continue
            if isinstance(d, datetime):
                parsed.append(d)
            else:
                try:
                    parsed.append(pd.to_datetime(d).to_pydatetime())
                except:
                    parsed.append(None)
        return parsed

    def get_time_range(self) -> Dict[str, Optional[str]]:
        """获取时间范围"""
        valid_dates = [d for d in self.dates if d is not None]
        if not valid_dates:
            return {"start": None, "end": None, "days": 0}

        return {
            "start": min(valid_dates).strftime("%Y-%m-%d"),
            "end": max(valid_dates).strftime("%Y-%m-%d"),
            "days": (max(valid_dates) - min(valid_dates)).days
        }

    def aggregate_by_period(self, period: str = "day") -> pd.DataFrame:
        """
        按周期聚合文本数量

        参数:
        - period: 'day', 'week', 'month', 'quarter', 'year'

        返回: DataFrame with columns ['date', 'count']
        """
        if period == "day":
            freq = "D"
            date_format = "%Y-%m-%d"
        elif period == "week":
            freq = "W"
            date_format = "%Y-W%W"
        elif period == "month":
            freq = "M"
            date_format = "%Y-%m"
        elif period == "quarter":
            freq = "Q"
            date_format = "%Y-Q%q"
        elif period == "year":
            freq = "Y"
            date_format = "%Y"
        else:
            freq = "D"
            date_format = "%Y-%m-%d"

        # 创建 DataFrame
        df = pd.DataFrame({"date": self.dates, "text": self.texts})
        df = df.dropna(subset=["date"])
        df["date"] = pd.to_datetime(df["date"])

        # 按周期分组统计
        df["period"] = df["date"].dt.to_period(freq)
        counts = df.groupby("period").size().reset_index(name="count")
        counts["period_str"] = counts["period"].astype(str)

        return counts[["period_str", "count"]]

    def get_seasonal_pattern(self) -> Dict[str, Any]:
        """
        检测周期性模式

        返回:
        {
            "has_weekly": bool,      # 是否有周周期
            "has_monthly": bool,     # 是否有月周期
            "weekly_pattern": Dict,  # 各星期几的平均数量
            "monthly_pattern": Dict  # 各月份的平均数量
        }
        """
        df = pd.DataFrame({"date": self.dates, "text": self.texts})
        df = df.dropna(subset=["date"])
        df["date"] = pd.to_datetime(df["date"])

        result = {"has_weekly": False, "has_monthly": False, "weekly_pattern": {}, "monthly_pattern": {}}

        # 周模式
        if len(df) >= 14:  # 至少两周数据
            df["weekday"] = df["date"].dt.dayofweek
            weekday_counts = df.groupby("weekday").size()
            if len(weekday_counts) > 0:
                avg = weekday_counts.mean()
                std = weekday_counts.std()
                # 如果某天偏离均值超过 30%，认为有周模式
                if (abs(weekday_counts - avg) > avg * 0.3).any():
                    result["has_weekly"] = True
                result["weekly_pattern"] = {int(k): int(v) for k, v in weekday_counts.to_dict().items()}

        # 月模式
        if len(df) >= 24:  # 至少两年数据
            df["month"] = df["date"].dt.month
            month_counts = df.groupby("month").size()
            if len(month_counts) > 0:
                avg = month_counts.mean()
                std = month_counts.std()
                if (abs(month_counts - avg) > avg * 0.3).any():
                    result["has_monthly"] = True
                result["monthly_pattern"] = {int(k): int(v) for k, v in month_counts.to_dict().items()}

        return result

    def get_trend_line(self) -> Dict[str, Any]:
        """
        获取趋势线（线性回归）

        返回:
        {
            "slope": float,          # 斜率（正=上升，负=下降）
            "intercept": float,
            "r2": float,             # 拟合度
            "direction": str         # 'up', 'down', 'stable'
        }
        """
        from scipy import stats

        df = self.aggregate_by_period("day")
        if len(df) < 3:
            return {"slope": 0, "intercept": 0, "r2": 0, "direction": "stable"}

        x = np.arange(len(df))
        y = df["count"].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r2 = r_value ** 2

        if slope > 0.05 * y.mean():
            direction = "up"
        elif slope < -0.05 * y.mean():
            direction = "down"
        else:
            direction = "stable"

        return {
            "slope": slope,
            "intercept": intercept,
            "r2": r2,
            "p_value": p_value,
            "direction": direction
        }

    def get_top_keywords_by_period(self, keywords_func, period: str = "month", top_n: int = 10) -> Dict[str, List[str]]:
        """
        按时间段获取关键词

        参数:
        - keywords_func: 关键词提取函数，接受文本列表返回关键词列表
        - period: 时间周期
        - top_n: 每个时间段的关键词数量

        返回: {时间段: [关键词列表]}
        """
        df = self.aggregate_by_period(period)
        result = {}

        for period_str in df["period_str"].values:
            # 获取该时间段内的文本
            period_texts = self._get_texts_by_period(period_str, period)
            if period_texts:
                keywords = keywords_func(period_texts)
                result[period_str] = [k for k, _ in keywords[:top_n]]

        return result

    def _get_texts_by_period(self, period_str: str, period: str) -> List[str]:
        """获取指定时间段内的文本"""
        # 简化实现：根据 period 格式匹配
        texts_in_period = []
        for date, text in zip(self.dates, self.texts):
            if date is None:
                continue
            if period == "day" and date.strftime("%Y-%m-%d") == period_str:
                texts_in_period.append(text)
            elif period == "month" and date.strftime("%Y-%m") == period_str:
                texts_in_period.append(text)
            elif period == "year" and date.strftime("%Y") == period_str:
                texts_in_period.append(text)
        return texts_in_period

    def detect_anomalies(self, threshold: float = 2.0) -> List[Dict]:
        """
        检测异常时间段（文本量突变）

        参数:
        - threshold: 标准差倍数

        返回: [{"date": str, "count": int, "deviation": float, "type": "spike"/"drop"}]
        """
        df = self.aggregate_by_period("day")
        if len(df) < 7:
            return []

        counts = df["count"].values
        mean = counts.mean()
        std = counts.std()

        anomalies = []
        for i, row in df.iterrows():
            deviation = (row["count"] - mean) / std if std > 0 else 0
            if abs(deviation) > threshold:
                anomalies.append({
                    "date": row["period_str"],
                    "count": int(row["count"]),
                    "deviation": deviation,
                    "type": "spike" if deviation > 0 else "drop"
                })

        return anomalies