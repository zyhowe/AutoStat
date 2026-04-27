"""
文本聚类模块 - K-Means 聚类
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter


class TextClusterer:
    """文本聚类器 - K-Means"""

    def __init__(self, n_clusters: int = None, max_clusters: int = 10, random_state: int = 42):
        """
        初始化聚类器

        参数:
        - n_clusters: 聚类数量（None 时自动确定）
        - max_clusters: 最大聚类数（自动确定时使用）
        - random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.model = None
        self.vectorizer = None
        self.labels = None
        self._fitted = False

    def _auto_determine_clusters(self, X, max_clusters: int = 10) -> int:
        """
        使用肘部法则自动确定聚类数

        参数:
        - X: 特征矩阵
        - max_clusters: 最大聚类数

        返回: 最佳聚类数
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        best_k = 2
        best_score = -1

        for k in range(2, min(max_clusters + 1, X.shape[0])):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)

            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = k

        return best_k

    def fit(self, texts: List[str], max_features: int = 5000):
        """
        训练聚类模型

        参数:
        - texts: 文本列表（已预处理）
        - max_features: 最大特征数
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        from sklearn.decomposition import TruncatedSVD

        # 向量化
        documents = [' '.join(text.split()) if isinstance(text, str) else str(text) for text in texts]

        self.vectorizer = TfidfVectorizer(max_features=max_features, token_pattern=r'(?u)\b\w+\b')
        X = self.vectorizer.fit_transform(documents)

        # 降维（可选，提高聚类质量）
        if X.shape[1] > 100:
            svd = TruncatedSVD(n_components=min(100, X.shape[1] - 1), random_state=self.random_state)
            X = svd.fit_transform(X)

        # 确定聚类数
        if self.n_clusters is None:
            self.n_clusters = self._auto_determine_clusters(X, self.max_clusters)

        # 聚类
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.labels = self.model.fit_predict(X)
        self._fitted = True

        return self

    def predict(self, text: str) -> int:
        """
        预测新文本的簇

        参数:
        - text: 文本

        返回: 簇标签
        """
        if not self._fitted:
            raise ValueError("请先调用 fit() 训练模型")

        doc = self.vectorizer.transform([text])
        return self.model.predict(doc)[0]

    def get_cluster_info(self, texts: List[str], top_n_words: int = 10) -> List[Dict]:
        """
        获取聚类信息

        参数:
        - texts: 原始文本列表（用于提取代表性文本）
        - top_n_words: 每个簇的关键词数量

        返回:
        [
            {
                "cluster_id": int,
                "size": int,
                "percentage": float,
                "top_words": List[str],
                "sample_texts": List[str]
            },
            ...
        ]
        """
        if not self._fitted:
            raise ValueError("请先调用 fit() 训练模型")

        # 获取特征名称
        feature_names = self.vectorizer.get_feature_names_out()

        # 计算每个簇的中心词
        cluster_info = []
        for i in range(self.n_clusters):
            # 获取该簇的样本索引
            indices = [idx for idx, label in enumerate(self.labels) if label == i]
            size = len(indices)

            # 获取该簇的中心向量
            center = self.model.cluster_centers_[i]

            # 获取 top words
            if feature_names is not None:
                top_indices = np.argsort(center)[::-1][:top_n_words]
                top_words = [feature_names[idx] for idx in top_indices if center[idx] > 0]
            else:
                top_words = []

            # 获取代表性文本（取前3条）
            sample_texts = [texts[idx][:200] for idx in indices[:3]]

            cluster_info.append({
                "cluster_id": i,
                "size": size,
                "percentage": size / len(texts) if texts else 0,
                "top_words": top_words,
                "sample_texts": sample_texts
            })

        # 按大小排序
        cluster_info.sort(key=lambda x: x["size"], reverse=True)

        return cluster_info

    def get_silhouette_score(self) -> float:
        """
        获取轮廓系数（评估聚类质量）

        返回: 轮廓系数（-1 到 1）
        """
        if not self._fitted:
            raise ValueError("请先调用 fit() 训练模型")

        from sklearn.metrics import silhouette_score

        # 重新获取特征矩阵
        documents = self.vectorizer._documents if hasattr(self.vectorizer, '_documents') else None
        if documents is None:
            return 0

        X = self.vectorizer.transform(documents)
        if X.shape[0] < 2 or len(set(self.labels)) < 2:
            return 0

        return silhouette_score(X, self.labels)