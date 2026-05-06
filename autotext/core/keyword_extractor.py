"""
关键词提取模块 - TF-IDF、TextRank
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter
import math


class KeywordExtractor:
    """关键词提取器"""

    def __init__(self, language: str = "auto"):
        """
        初始化关键词提取器

        参数:
        - language: 语言 ('auto', 'zh', 'en')
        """
        self.language = language
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._vocabulary = None

    def extract_tfidf(self, texts: List[str], top_n: int = 30, max_features: int = 5000) -> List[Tuple[str, float]]:
        """
        使用 TF-IDF 提取全局关键词

        参数:
        - texts: 文本列表（已预处理，空格分隔的词序列）
        - top_n: 返回的关键词数量
        - max_features: 最大特征数

        返回: [(关键词, TF-IDF 分数), ...]
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        # 将文本转换为空格分隔的词序列
        documents = [' '.join(text.split()) if isinstance(text, str) else str(text) for text in texts]

        self._tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            token_pattern=r'(?u)\b\w+\b',
            stop_words=None
        )

        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(documents)

        # 计算所有文档的词的平均 TF-IDF
        feature_names = self._tfidf_vectorizer.get_feature_names_out()
        avg_tfidf = self._tfidf_matrix.mean(axis=0).A1

        # 按分数排序
        word_scores = [(feature_names[i], avg_tfidf[i]) for i in range(len(feature_names))]
        word_scores.sort(key=lambda x: x[1], reverse=True)

        return word_scores[:top_n]

    def extract_frequency(self, tokens_list: List[List[str]], top_n: int = 50) -> List[Tuple[str, int]]:
        """
        使用词频统计提取高频词

        参数:
        - tokens_list: 分词后的词列表（每个文档一个列表）
        - top_n: 返回的关键词数量

        返回: [(词, 频率), ...]
        """
        counter = Counter()
        for tokens in tokens_list:
            counter.update(tokens)

        return counter.most_common(top_n)

    def extract_tfidf_by_doc(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        提取单篇文本的 TF-IDF 关键词（需要先调用 extract_tfidf 训练全局模型）

        参数:
        - text: 单篇文本
        - top_n: 返回的关键词数量

        返回: [(关键词, 分数), ...]
        """
        if self._tfidf_vectorizer is None or self._tfidf_matrix is None:
            raise ValueError("请先调用 extract_tfidf 训练全局模型")

        if not isinstance(text, str):
            text = str(text)

        doc_matrix = self._tfidf_vectorizer.transform([text])
        feature_names = self._tfidf_vectorizer.get_feature_names_out()
        scores = doc_matrix.toarray()[0]

        word_scores = [(feature_names[i], scores[i]) for i in range(len(feature_names)) if scores[i] > 0]
        word_scores.sort(key=lambda x: x[1], reverse=True)

        return word_scores[:top_n]

    def extract_textrank(self, text: str, top_n: int = 10, window: int = 5) -> List[Tuple[str, float]]:
        """
        使用 TextRank 提取关键词（基于共现图）

        参数:
        - text: 文本
        - top_n: 返回的关键词数量
        - window: 共现窗口大小

        返回: [(关键词, TextRank 分数), ...]
        """
        import re
        from collections import defaultdict

        # 预处理
        if isinstance(text, list):
            text = ' '.join(text)

        # 简单分词（按非字母数字字符分割）
        words = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbfa-zA-Z]+', text)
        words = [w.lower() for w in words if len(w) > 1]

        # 去重保留顺序
        unique_words = []
        seen = set()
        for w in words:
            if w not in seen:
                seen.add(w)
                unique_words.append(w)

        if len(unique_words) < 3:
            return [(w, 1.0) for w in unique_words[:top_n]]

        # 构建共现图
        graph = defaultdict(lambda: defaultdict(float))
        n_words = len(unique_words)

        for i in range(n_words):
            for j in range(i + 1, min(i + window + 1, n_words)):
                w1, w2 = unique_words[i], unique_words[j]
                graph[w1][w2] += 1.0
                graph[w2][w1] += 1.0

        # 计算 TextRank 分数
        scores = {w: 1.0 for w in unique_words}
        damping = 0.85
        max_iter = 50
        tol = 1e-6

        for _ in range(max_iter):
            new_scores = {}
            for w in unique_words:
                sum_score = 0.0
                for neighbor, weight in graph[w].items():
                    out_sum = sum(graph[neighbor].values())
                    if out_sum > 0:
                        sum_score += scores[neighbor] * weight / out_sum
                new_scores[w] = (1 - damping) + damping * sum_score

            # 检查收敛
            diff = sum(abs(new_scores[w] - scores[w]) for w in unique_words)
            scores = new_scores
            if diff < tol:
                break

        # 排序
        word_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return word_scores[:top_n]

    def get_keyword_cloud_data(self, tokens_list: List[List[str]], top_n: int = 100) -> List[Dict]:
        """
        获取词云数据

        参数:
        - tokens_list: 分词后的词列表
        - top_n: 返回的词数量

        返回: [{"text": str, "value": int}, ...]
        """
        freqs = self.extract_frequency(tokens_list, top_n)
        return [{"text": word, "value": count} for word, count in freqs]

    def extract_ngrams(self, tokens_list: List[List[str]], n: int = 2, top_n: int = 20) -> List[Tuple[str, int]]:
        """
        提取 N-gram 短语

        参数:
        - tokens_list: 分词后的词列表
        - n: N-gram 长度
        - top_n: 返回数量

        返回: [(短语, 频率), ...]
        """
        ngram_counter = Counter()

        for tokens in tokens_list:
            for i in range(len(tokens) - n + 1):
                ngram = ''.join(tokens[i:i + n])
                ngram_counter[ngram] += 1

        return ngram_counter.most_common(top_n)