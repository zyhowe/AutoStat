"""
数据质量模块 - 空值、重复、异常检测（增强版，提供详情数据）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from difflib import SequenceMatcher


class TextQuality:
    """文本数据质量检查（增强版）"""

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
            "empty": {...},
            "duplicates": {
                "count": int,
                "rate": float,
                "duplicate_groups": [...],  # 重复组详情
                "duplicate_summary": [...]   # 重复文本摘要
            },
            "similarity": {
                "high_similarity_pairs": [...],  # 高相似对详情
                "similarity_summary": [...]       # 高相似摘要
            },
            "length_anomalies": {
                "short": [...],
                "long": [...],
                "short_summary": [...],   # 过短文本摘要
                "long_summary": [...]     # 过长文本摘要
            }
        }
        """
        if self._results is not None:
            return self._results

        n = len(self.texts)

        # 空值检测
        empty_indices = [i for i, t in enumerate(self.texts) if not t or not isinstance(t, str) or len(t.strip()) == 0]
        empty_count = len(empty_indices)

        # 重复检测（增强版）
        duplicate_groups = self._detect_duplicate_groups()
        duplicate_summary = self._format_duplicate_summary(duplicate_groups)

        # 长度异常
        from autotext.core.stats import TextStats
        stats = TextStats(self.texts)
        stats_result = stats.compute_stats()
        outliers = stats.get_outlier_texts(threshold=2.0)

        short_outliers = [o for o in outliers if o["type"] == "short"]
        long_outliers = [o for o in outliers if o["type"] == "long"]

        short_summary = self._format_length_summary(short_outliers, "过短")
        long_summary = self._format_length_summary(long_outliers, "过长")

        # 相似度检测（增强版）
        high_similarity_pairs = self._detect_similar_pairs(threshold=0.7)
        similarity_summary = self._format_similarity_summary(high_similarity_pairs)

        self._results = {
            "total": n,
            "empty": {
                "count": empty_count,
                "rate": empty_count / n if n > 0 else 0,
                "indices": empty_indices[:10]
            },
            "duplicates": {
                "count": len(duplicate_groups),
                "rate": len(duplicate_groups) / n if n > 0 else 0,
                "duplicate_groups": duplicate_groups[:20],
                "duplicate_summary": duplicate_summary
            },
            "similarity": {
                "high_similarity_pairs": high_similarity_pairs[:20],
                "similarity_summary": similarity_summary
            },
            "length_anomalies": {
                "short": short_outliers[:20],
                "long": long_outliers[:20],
                "short_summary": short_summary,
                "long_summary": long_summary
            }
        }

        return self._results

    def _detect_duplicate_groups(self) -> List[Dict]:
        """检测重复文本组（完全相同）"""
        seen = {}
        groups = []

        for i, text in enumerate(self.texts):
            if not text:
                continue
            # 使用完整文本作为键（或前500字符）
            key = text[:500] if len(text) > 500 else text
            if key in seen:
                groups.append({
                    "original_index": seen[key],
                    "duplicate_index": i,
                    "text_preview": text[:200] + "..." if len(text) > 200 else text,
                    "length": len(text)
                })
            else:
                seen[key] = i

        return groups

    def _detect_similar_pairs(self, threshold: float = 0.7, max_samples: int = 200) -> List[Dict]:
        """
        检测高相似文本对（使用 Jaccard 相似度）

        参数:
        - threshold: 相似度阈值（0-1）
        - max_samples: 最大采样数（避免 O(n^2) 爆炸）
        """
        from autotext.core.preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()

        # 预处理文本，获取词集合
        token_sets = []
        valid_indices = []
        valid_texts = []

        for i, text in enumerate(self.texts):
            if not text or len(text) < 20:
                continue
            cleaned = preprocessor.clean_text(text)
            tokens = preprocessor.tokenize(cleaned)
            tokens_cleaned = preprocessor.remove_stopwords(tokens)
            if len(tokens_cleaned) > 3:
                token_sets.append(set(tokens_cleaned))
                valid_indices.append(i)
                valid_texts.append(text)

        n = len(token_sets)
        if n > max_samples:
            # 采样
            import random
            random.seed(42)
            sampled_indices = random.sample(range(n), max_samples)
            token_sets = [token_sets[i] for i in sampled_indices]
            valid_indices = [valid_indices[i] for i in sampled_indices]
            valid_texts = [valid_texts[i] for i in sampled_indices]
            n = max_samples

        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                set_i = token_sets[i]
                set_j = token_sets[j]
                if not set_i or not set_j:
                    continue

                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                similarity = intersection / union if union > 0 else 0

                if similarity >= threshold:
                    # 计算具体相似度分数
                    text1 = valid_texts[i][:200]
                    text2 = valid_texts[j][:200]
                    # 使用 SequenceMatcher 计算更精确的相似度
                    seq_similarity = SequenceMatcher(None, text1, text2).ratio()

                    pairs.append({
                        "index1": valid_indices[i],
                        "index2": valid_indices[j],
                        "similarity": round(seq_similarity, 3),
                        "jaccard": round(similarity, 3),
                        "text1_preview": text1 + ("..." if len(valid_texts[i]) > 200 else ""),
                        "text2_preview": text2 + ("..." if len(valid_texts[j]) > 200 else "")
                    })

        # 按相似度排序
        pairs.sort(key=lambda x: x["similarity"], reverse=True)
        return pairs

    def _format_duplicate_summary(self, duplicate_groups: List[Dict]) -> List[Dict]:
        """格式化重复文本摘要"""
        if not duplicate_groups:
            return []

        # 按文本分组统计
        text_counts = {}
        for group in duplicate_groups:
            text_preview = group["text_preview"]
            if text_preview not in text_counts:
                text_counts[text_preview] = {
                    "text": text_preview,
                    "count": 1,
                    "indices": [group["original_index"], group["duplicate_index"]]
                }
            else:
                text_counts[text_preview]["count"] += 1
                text_counts[text_preview]["indices"].append(group["duplicate_index"])

        # 转换为列表，按重复次数排序
        summary = []
        for text_preview, info in text_counts.items():
            summary.append({
                "text": info["text"],
                "count": info["count"],
                "indices": sorted(set(info["indices"]))[:10]  # 最多显示10个索引
            })

        summary.sort(key=lambda x: x["count"], reverse=True)
        return summary[:10]

    def _format_similarity_summary(self, pairs: List[Dict]) -> List[Dict]:
        """格式化高相似文本摘要"""
        if not pairs:
            return []

        summary = []
        for pair in pairs[:10]:
            summary.append({
                "index1": pair["index1"],
                "index2": pair["index2"],
                "similarity": pair["similarity"],
                "text1": pair["text1_preview"],
                "text2": pair["text2_preview"]
            })
        return summary

    def _format_length_summary(self, outliers: List[Dict], type_name: str) -> List[Dict]:
        """格式化长度异常摘要"""
        if not outliers:
            return []

        summary = []
        for outlier in outliers[:10]:
            summary.append({
                "index": outlier["index"],
                "length": outlier["length"],
                "text_preview": outlier["text"][:200] + ("..." if len(outlier["text"]) > 200 else ""),
                "type": type_name
            })
        return summary

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

        dup_count = results["duplicates"]["count"]
        if dup_count > 0:
            suggestions.append(f"发现 {dup_count} 对重复文本 (占比 {results['duplicates']['rate']:.1%})，建议去重")

        short_count = len(results["length_anomalies"]["short"])
        long_count = len(results["length_anomalies"]["long"])

        if short_count > 0:
            suggestions.append(f"发现 {short_count} 条过短文本，可能是无意义内容，建议检查")

        if long_count > 0:
            suggestions.append(f"发现 {long_count} 条过长文本，已截断处理")

        sim_count = len(results["similarity"]["high_similarity_pairs"])
        if sim_count > 0:
            suggestions.append(f"发现 {sim_count} 对高相似文本 (相似度≥70%)，可能存在重复或抄袭")

        if not suggestions:
            suggestions.append("数据质量良好，无明显问题")

        return suggestions