"""
文本检查器模块 - 统一管理各分析的适用条件
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class TextChecker:
    """文本分析条件检查器"""

    def __init__(self, texts: List[str], titles: Optional[List[str]] = None,
                 dates: Optional[List[Any]] = None):
        """
        初始化检查器

        参数:
        - texts: 文本列表
        - titles: 标题列表（可选）
        - dates: 时间列表（可选）
        """
        self.texts = texts
        self.titles = titles
        self.dates = dates

    def check_sample_size(self) -> Dict[str, Any]:
        """检查样本量是否充足"""
        n = len(self.texts)

        if n < 10:
            return {"suitable": False, "reason": f"样本量不足 (n={n}<10)", "method": "仅基础统计"}
        elif n < 50:
            return {"suitable": True, "reason": f"样本量较小 (n={n})", "method": "基础统计 + 关键词", "caution": "聚类和主题建模可能效果不佳"}
        elif n < 200:
            return {"suitable": True, "reason": f"样本量适中 (n={n})", "method": "全部分析", "caution": "聚类和主题建模结果仅供参考"}
        else:
            return {"suitable": True, "reason": f"样本量充足 (n={n})", "method": "全部分析", "caution": None}

    def check_text_length(self) -> Dict[str, Any]:
        """检查文本长度是否适合"""
        lengths = [len(t) for t in self.texts if t]
        if not lengths:
            return {"suitable": False, "reason": "没有有效文本", "method": None}

        avg_len = np.mean(lengths)
        min_len = np.min(lengths)
        max_len = np.max(lengths)

        if avg_len < 10:
            return {"suitable": False, "reason": f"文本过短 (平均{avg_len:.0f}字符)", "method": "仅统计"}
        elif avg_len < 50:
            return {"suitable": True, "reason": f"文本较短 (平均{avg_len:.0f}字符)", "method": "关键词 + 情感", "caution": "主题建模可能效果不佳"}
        else:
            return {"suitable": True, "reason": f"文本长度适中 (平均{avg_len:.0f}字符)", "method": "全部分析"}

    def check_time_series(self) -> Dict[str, Any]:
        """检查时间序列分析条件"""
        if self.dates is None or len(self.dates) == 0:
            return {"suitable": False, "reason": "无时间信息", "method": None}

        valid_dates = [d for d in self.dates if d is not None and not pd.isna(d)]
        if len(valid_dates) < 10:
            return {"suitable": False, "reason": f"有效时间点不足 (n={len(valid_dates)}<10)", "method": None}

        unique_dates = len(set(str(d) for d in valid_dates))
        if unique_dates < 5:
            return {"suitable": False, "reason": f"唯一时间点过少 (n={unique_dates}<5)", "method": None}

        return {"suitable": True, "reason": "适合时间序列分析", "method": "趋势分析 + 异常检测"}

    def check_clustering(self) -> Dict[str, Any]:
        """检查聚类分析条件"""
        n = len(self.texts)
        if n < 20:
            return {"suitable": False, "reason": f"样本量不足 (n={n}<20)", "method": None}

        avg_len = np.mean([len(t) for t in self.texts if t])
        if avg_len < 20:
            return {"suitable": False, "reason": f"文本过短 (平均{avg_len:.0f}字符)", "method": None}

        return {"suitable": True, "reason": f"适合聚类分析 (n={n})", "method": "K-Means 聚类"}

    def check_topic_modeling(self) -> Dict[str, Any]:
        """检查主题建模条件"""
        n = len(self.texts)
        if n < 50:
            return {"suitable": False, "reason": f"样本量不足 (n={n}<50)", "method": None}

        avg_len = np.mean([len(t) for t in self.texts if t])
        if avg_len < 50:
            return {"suitable": False, "reason": f"文本过短 (平均{avg_len:.0f}字符)", "method": None}

        return {"suitable": True, "reason": f"适合主题建模 (n={n})", "method": "LDA"}

    def get_all_check_results(self) -> Dict[str, Any]:
        """获取所有检查结果"""
        return {
            "sample_size": self.check_sample_size(),
            "text_length": self.check_text_length(),
            "time_series": self.check_time_series(),
            "clustering": self.check_clustering(),
            "topic_modeling": self.check_topic_modeling()
        }

    def get_suitable_methods(self) -> List[str]:
        """获取适合的分析方法"""
        results = self.get_all_check_results()
        methods = []
        for key, result in results.items():
            if result.get("suitable"):
                methods.append(result.get("method"))
        return methods