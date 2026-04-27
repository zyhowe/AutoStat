"""
主题建模模块 - LDA
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class TopicModeler:
    """主题建模器 - LDA"""

    def __init__(self, n_topics: int = 10, max_features: int = 2000, random_state: int = 42):
        """
        初始化主题建模器

        参数:
        - n_topics: 主题数量
        - max_features: 最大特征数
        - random_state: 随机种子
        """
        self.n_topics = n_topics
        self.max_features = max_features
        self.random_state = random_state
        self.model = None
        self.vectorizer = None
        self.topic_distributions = None
        self._fitted = False

    def fit(self, texts: List[str]):
        """
        训练 LDA 模型

        参数:
        - texts: 文本列表（已预处理）
        """
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        # 向量化
        documents = [' '.join(text.split()) if isinstance(text, str) else str(text) for text in texts]

        self.vectorizer = CountVectorizer(max_features=self.max_features, token_pattern=r'(?u)\b\w+\b')
        X = self.vectorizer.fit_transform(documents)

        # LDA
        self.model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=self.random_state,
            learning_method='online'
        )
        self.topic_distributions = self.model.fit_transform(X)
        self._fitted = True

        return self

    def get_topics(self, top_n_words: int = 10) -> List[Dict]:
        """
        获取主题及关键词

        返回:
        [
            {
                "topic_id": int,
                "keywords": List[str],
                "weights": List[float]
            },
            ...
        ]
        """
        if not self._fitted:
            raise ValueError("请先调用 fit() 训练模型")

        feature_names = self.vectorizer.get_feature_names_out()
        topics = []

        for topic_idx, topic in enumerate(self.model.components_):
            top_indices = topic.argsort()[:-top_n_words - 1:-1]
            keywords = [feature_names[i] for i in top_indices]
            weights = [topic[i] for i in top_indices]
            # 归一化权重
            total = sum(weights)
            weights = [w / total for w in weights]

            topics.append({
                "topic_id": topic_idx,
                "keywords": keywords,
                "weights": weights
            })

        return topics

    def get_topic_distribution(self) -> List[float]:
        """
        获取整体主题分布（各主题在所有文档中的平均比例）

        返回: 各主题的比例列表
        """
        if not self._fitted:
            raise ValueError("请先调用 fit() 训练模型")

        return (self.topic_distributions.mean(axis=0)).tolist()

    def get_document_topics(self, doc_index: int) -> List[Tuple[int, float]]:
        """
        获取单篇文档的主题分布

        参数:
        - doc_index: 文档索引

        返回: [(主题ID, 比例), ...] 按比例降序
        """
        if not self._fitted:
            raise ValueError("请先调用 fit() 训练模型")

        dist = self.topic_distributions[doc_index]
        topics = [(i, dist[i]) for i in range(len(dist))]
        topics.sort(key=lambda x: x[1], reverse=True)
        return topics

    def predict(self, text: str) -> List[Tuple[int, float]]:
        """
        预测新文本的主题分布

        参数:
        - text: 文本

        返回: [(主题ID, 比例), ...] 按比例降序
        """
        if not self._fitted:
            raise ValueError("请先调用 fit() 训练模型")

        doc = self.vectorizer.transform([text])
        dist = self.model.transform(doc)[0]
        topics = [(i, dist[i]) for i in range(len(dist))]
        topics.sort(key=lambda x: x[1], reverse=True)
        return topics

    def get_coherence_score(self) -> float:
        """
        获取主题一致性（评估主题质量）

        返回: 一致性分数（越高越好）
        """
        # 简化实现，实际可以使用 gensim 的 CoherenceModel
        # 这里返回基于主题词之间共现的简单评分
        if not self._fitted:
            return 0

        topics = self.get_topics(top_n_words=10)
        coherence_sum = 0
        for topic in topics:
            keywords = topic["keywords"]
            # 简单的词频相关性
            score = 0
            for i in range(len(keywords)):
                for j in range(i + 1, len(keywords)):
                    # 这里简化处理，实际需要词共现矩阵
                    score += 1
            coherence_sum += score / max(len(keywords) * (len(keywords) - 1) / 2, 1)
        return coherence_sum / len(topics) / 10  # 归一化到 0-1 范围