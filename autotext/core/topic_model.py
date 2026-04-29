"""
主题建模模块 - 统一封装 LDA（替代原有 topic.py）
"""

import numpy as np
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class TopicModeler:
    """主题建模器 - 基于 LDA（与原有实现保持一致）"""

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
        self._feature_names = []
        self._original_X = None
        self.topic_labels = None  # 每条文本的主题标签
        self.labels = None  # 兼容 cluster_labels

    def _build_dicts(self, token_lists: List[List[str]]) -> List[Dict[str, int]]:
        """将 token 列表转换为词频字典"""
        from collections import Counter
        dicts = []
        for tokens in token_lists:
            if tokens:
                dicts.append(Counter(tokens))
            else:
                dicts.append({})
        return dicts

    def fit(self, token_lists: List[List[str]], cluster_labels: List[int] = None):
        """
        训练 LDA 模型

        参数:
        - token_lists: 已经分词好的 token 列表
        - cluster_labels: 聚类标签（可选，用于兼容）
        """
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        # 存储标签
        if cluster_labels is not None:
            self.labels = cluster_labels.copy()
        else:
            self.labels = [0] * len(token_lists)

        dicts = self._build_dicts(token_lists)

        # 过滤空字典
        non_empty = [i for i, d in enumerate(dicts) if d]
        if len(non_empty) < 10:
            print(f"  ⚠️ 主题建模: 有效文本不足 ({len(non_empty)} < 10)")
            self._fitted = True
            self.topic_labels = [-1] * len(token_lists)
            return self

        filtered_dicts = [dicts[i] for i in non_empty]
        self._original_indices = non_empty

        # 使用 DictVectorizer 转换
        self.vectorizer = DictVectorizer()
        X = self.vectorizer.fit_transform(filtered_dicts)
        self._feature_names = self.vectorizer.get_feature_names_out()

        # 限制特征数量
        if X.shape[1] > self.max_features:
            col_sums = X.sum(axis=0).A1
            top_indices = col_sums.argsort()[-self.max_features:][::-1]
            X = X[:, top_indices]
            self._feature_names = self._feature_names[top_indices]

        self._original_X = X

        # 确定主题数
        if self.n_topics is None or self.n_topics > X.shape[0] - 1:
            self.n_topics = min(10, max(2, X.shape[0] // 20))

        # LDA
        self.model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=self.random_state,
            learning_method='online',
            max_iter=100
        )
        self.topic_distributions = self.model.fit_transform(X)

        # 生成主题标签
        self.topic_labels = [-1] * len(dicts)
        for i, label in zip(non_empty, np.argmax(self.topic_distributions, axis=1)):
            self.topic_labels[i] = int(label)

        self._fitted = True
        self.labels = self.topic_labels

        return self

    def get_topics(self, top_n_words: int = 10) -> List[Dict]:
        """获取主题列表"""
        if not self._fitted:
            return []

        topics = []
        for topic_idx, topic in enumerate(self.model.components_):
            n = min(top_n_words, len(topic))
            top_indices = topic.argsort()[:-n - 1:-1]
            keywords = [self._feature_names[i] for i in top_indices]
            weights = [topic[i] for i in top_indices]

            # 归一化权重
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]

            topics.append({
                "topic_id": topic_idx,
                "texts_count": sum(1 for l in self.topic_labels if l == topic_idx),
                "keywords": keywords,
                "weights": weights,
                "representative_text": self._get_representative_text(topic_idx)
            })

        # 按文本数量排序
        topics.sort(key=lambda x: x["texts_count"], reverse=True)

        # 重新编号
        for i, topic in enumerate(topics):
            topic["topic_id"] = i

        return topics

    def _get_representative_text(self, topic_id: int) -> str:
        """获取主题的代表性文本"""
        # 简化实现，返回空字符串
        return ""

    def get_topic_distribution(self) -> List[float]:
        """获取主题分布"""
        if not self._fitted:
            return []
        if self.topic_distributions is None:
            return []
        return self.topic_distributions.mean(axis=0).tolist()

    def get_document_topics(self, doc_index: int) -> List[Tuple[int, float]]:
        """获取文档的主题分布"""
        if not self._fitted or self.topic_distributions is None:
            return []
        if doc_index >= len(self.topic_distributions):
            return []
        dist = self.topic_distributions[doc_index]
        topics = [(i, dist[i]) for i in range(len(dist))]
        topics.sort(key=lambda x: x[1], reverse=True)
        return topics

    def get_topic_labels(self) -> List[int]:
        """获取每条文本的主题标签"""
        return self.topic_labels if self.topic_labels else []