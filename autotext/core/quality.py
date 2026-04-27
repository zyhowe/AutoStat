"""
数据质量模块 - 空值、重复、异常检测
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple  # 添加 Tuple
from collections import Counter


class TextQuality:
    """文本数据质量检查"""

    def __init__(self, texts: List[str]):
        """
        初始化质量检查器

        参数:
        - texts: 原始文本列表
        """
        self.texts = texts
        self._results = None

    def check(self) -> Dict[str, Any]:
        """
        执行质量检查

        返回:
        {
            "total": int,
            "empty": {
                "count": int,
                "rate": float,
                "indices": List[int]
            },
            "duplicates": {
                "count": int,
                "rate": float,
                "duplicate_groups": List[Dict]
            },
            "length_anomalies": {
                "short": List[Dict],  # 过短文本
                "long": List[Dict]    # 过长文本
            },
            "similarity": {
                "high_similarity_pairs": List[Tuple[int, int, float]]
            }
        }
        """
        if self._results is not None:
            return self._results

        n = len(self.texts)

        # 空值检测
        empty_indices = [i for i, t in enumerate(self.texts) if not t or not isinstance(t, str) or len(t.strip()) == 0]
        empty_count = len(empty_indices)

        # 重复检测
        seen = {}
        duplicate_groups = []
        for i, text in enumerate(self.texts):
            if not text:
                continue
            key = text[:200]  # 用前200字符作为键
            if key in seen:
                duplicate_groups.append({"original": seen[key], "duplicate": i, "text": text[:100]})
            else:
                seen[key] = i

        # 长度异常（调用 TextStats）
        from autotext.core.stats import TextStats
        stats = TextStats(self.texts)
        stats_result = stats.compute_stats()
        outliers = stats.get_outlier_texts(threshold=2.0)

        short_outliers = [o for o in outliers if o["type"] == "short"]
        long_outliers = [o for o in outliers if o["type"] == "long"]

        # 简单相似度检测（基于公共词比例）
        high_similarity_pairs = self._detect_similar_pairs(threshold=0.8)

        self._results = {
            "total": n,
            "empty": {
                "count": empty_count,
                "rate": empty_count / n if n > 0 else 0,
                "indices": empty_indices[:10]  # 只返回前10个
            },
            "duplicates": {
                "count": len(duplicate_groups),
                "rate": len(duplicate_groups) / n if n > 0 else 0,
                "duplicate_groups": duplicate_groups[:10]
            },
            "length_anomalies": {
                "short": short_outliers[:10],
                "long": long_outliers[:10]
            },
            "similarity": {
                "high_similarity_pairs": high_similarity_pairs[:10]
            }
        }

        return self._results

    def _detect_similar_pairs(self, threshold: float = 0.8) -> List[Tuple[int, int, float]]:
        """
        检测相似文本对（基于 Jaccard 相似度）

        注意：O(n^2) 复杂度，对于大文本集需要采样
        """
        from autotext.core.preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()

        # 对文本进行预处理，获取词集合
        token_sets = []
        for text in self.texts:
            if not text:
                token_sets.append(set())
                continue
            cleaned = preprocessor.clean_text(text)
            tokens = preprocessor.tokenize(cleaned)
            tokens_cleaned = preprocessor.remove_stopwords(tokens)
            token_sets.append(set(tokens_cleaned))

        # 采样检测（如果文本太多，只检测前500条）
        n = len(token_sets)
        max_sample = 500
        if n > max_sample:
            indices = list(range(n))
            import random
            random.seed(42)
            sampled = random.sample(indices, max_sample)
            token_sets = [token_sets[i] for i in sampled]
            texts = [self.texts[i] for i in sampled]
        else:
            sampled = list(range(n))
            texts = self.texts

        pairs = []
        n_sample = len(token_sets)
        for i in range(n_sample):
            for j in range(i + 1, n_sample):
                set_i = token_sets[i]
                set_j = token_sets[j]
                if not set_i or not set_j:
                    continue
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                similarity = intersection / union if union > 0 else 0
                if similarity >= threshold:
                    pairs.append((sampled[i], sampled[j], round(similarity, 3)))

        return pairs

    def get_summary(self) -> Dict:
        """获取质量摘要"""
        results = self.check()
        return {
            "empty_count": results["empty"]["count"],
            "empty_rate": results["empty"]["rate"],
            "duplicate_count": results["duplicates"]["count"],
            "duplicate_rate": results["duplicates"]["rate"],
            "short_count": len(results["length_anomalies"]["short"]),
            "long_count": len(results["length_anomalies"]["long"]),
            "similar_pairs_count": len(results["similarity"]["high_similarity_pairs"])
        }

    def get_cleaning_suggestions(self) -> List[str]:
        """生成清洗建议"""
        results = self.check()
        suggestions = []

        if results["empty"]["count"] > 0:
            suggestions.append(f"发现 {results['empty']['count']} 条空文本 (占比 {results['empty']['rate']:.1%})，建议删除")

        if results["duplicates"]["count"] > 0:
            suggestions.append(f"发现 {results['duplicates']['count']} 对重复文本 (占比 {results['duplicates']['rate']:.1%})，建议去重")

        short_count = len(results["length_anomalies"]["short"])
        long_count = len(results["length_anomalies"]["long"])

        if short_count > 0:
            suggestions.append(f"发现 {short_count} 条过短文本，可能是无意义内容，建议检查")

        if long_count > 0:
            suggestions.append(f"发现 {long_count} 条过长文本，已截断处理")

        if results["similarity"]["high_similarity_pairs"]:
            suggestions.append(f"发现 {len(results['similarity']['high_similarity_pairs'])} 对高相似文本，可能存在重复或抄袭")

        if not suggestions:
            suggestions.append("数据质量良好，无明显问题")

        return suggestions