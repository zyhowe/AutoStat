"""
基础统计模块 - 长度、词数、句数、分布
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class TextStats:
    """文本基础统计"""

    def __init__(self, texts: List[str], preprocessed_data: Optional[List[Dict]] = None):
        """
        初始化统计器

        参数:
        - texts: 原始文本列表
        - preprocessed_data: 预处理结果列表（可选，如提供则直接使用）
        """
        self.texts = texts
        self.preprocessed_data = preprocessed_data

    def compute_stats(self) -> Dict[str, Any]:
        """
        计算基础统计

        返回:
        {
            "total_count": int,           # 总文本数
            "empty_count": int,           # 空文本数
            "empty_rate": float,          # 空文本率
            "char_length": {
                "min": int,
                "max": int,
                "mean": float,
                "median": float,
                "std": float,
                "distribution": List[int]  # 直方图数据
            },
            "word_count": {
                "min": int,
                "max": int,
                "mean": float,
                "median": float,
                "std": float
            },
            "sentence_count": {
                "min": int,
                "max": int,
                "mean": float,
                "median": float
            },
            "language_distribution": Dict[str, int]  # 语言分布
        }
        """
        # 获取或计算预处理数据
        if self.preprocessed_data is None:
            from autotext.core.preprocessor import TextPreprocessor
            preprocessor = TextPreprocessor()
            self.preprocessed_data = preprocessor.process_batch(self.texts)

        # 统计空文本
        empty_count = sum(1 for d in self.preprocessed_data if len(d.get("cleaned", "")) == 0)

        # 字符长度
        char_lengths = [len(d.get("cleaned", "")) for d in self.preprocessed_data if len(d.get("cleaned", "")) > 0]

        # 词数（有效词，停用词后）
        word_counts = [d.get("token_cleaned_count", 0) for d in self.preprocessed_data if d.get("token_cleaned_count", 0) > 0]

        # 句子数
        sentence_counts = [d.get("sentence_count", 0) for d in self.preprocessed_data]

        # 语言分布
        lang_dist = {}
        for d in self.preprocessed_data:
            lang = d.get("language", "unknown")
            lang_dist[lang] = lang_dist.get(lang, 0) + 1

        # 字符长度直方图（20个区间）
        if char_lengths:
            hist, bins = np.histogram(char_lengths, bins=20)
            distribution = hist.tolist()
        else:
            distribution = []

        return {
            "total_count": len(self.texts),
            "empty_count": empty_count,
            "empty_rate": empty_count / len(self.texts) if self.texts else 0,
            "char_length": {
                "min": min(char_lengths) if char_lengths else 0,
                "max": max(char_lengths) if char_lengths else 0,
                "mean": np.mean(char_lengths) if char_lengths else 0,
                "median": np.median(char_lengths) if char_lengths else 0,
                "std": np.std(char_lengths) if char_lengths else 0,
                "distribution": distribution
            },
            "word_count": {
                "min": min(word_counts) if word_counts else 0,
                "max": max(word_counts) if word_counts else 0,
                "mean": np.mean(word_counts) if word_counts else 0,
                "median": np.median(word_counts) if word_counts else 0,
                "std": np.std(word_counts) if word_counts else 0
            },
            "sentence_count": {
                "min": min(sentence_counts) if sentence_counts else 0,
                "max": max(sentence_counts) if sentence_counts else 0,
                "mean": np.mean(sentence_counts) if sentence_counts else 0,
                "median": np.median(sentence_counts) if sentence_counts else 0
            },
            "language_distribution": lang_dist
        }

    def get_summary_table(self) -> pd.DataFrame:
        """生成统计摘要表格"""
        stats = self.compute_stats()
        rows = [
            ["总文本数", stats["total_count"]],
            ["空文本数", f"{stats['empty_count']} ({stats['empty_rate']:.2%})"],
            ["字符数 - 最小值", stats["char_length"]["min"]],
            ["字符数 - 最大值", stats["char_length"]["max"]],
            ["字符数 - 均值", f"{stats['char_length']['mean']:.2f}"],
            ["字符数 - 中位数", stats["char_length"]["median"]],
            ["词数 - 均值", f"{stats['word_count']['mean']:.2f}"],
            ["词数 - 中位数", stats["word_count"]["median"]],
            ["句子数 - 均值", f"{stats['sentence_count']['mean']:.2f}"],
            ["句子数 - 中位数", stats["sentence_count"]["median"]],
        ]
        return pd.DataFrame(rows, columns=["指标", "数值"])

    def get_outlier_texts(self, threshold: float = 2.0) -> List[Dict]:
        """
        获取异常文本（过短或过长）

        参数:
        - threshold: 标准差倍数，超过 mean ± threshold * std 视为异常

        返回: [{"index": int, "text": str, "length": int, "type": "short"/"long"}]
        """
        stats = self.compute_stats()
        char_lengths = [len(t) for t in self.texts]
        mean = stats["char_length"]["mean"]
        std = stats["char_length"]["std"]

        lower_bound = max(0, mean - threshold * std)
        upper_bound = mean + threshold * std

        outliers = []
        for i, text in enumerate(self.texts):
            length = len(text)
            if length < lower_bound:
                outliers.append({"index": i, "text": text[:200], "length": length, "type": "short"})
            elif length > upper_bound:
                outliers.append({"index": i, "text": text[:200], "length": length, "type": "long"})

        return outliers