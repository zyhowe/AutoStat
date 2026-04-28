"""
文本聚类模块 - 基于 sklearn（无需 umap/hdbscan）
"""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# 使用 sklearn 替代
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class TextClusterer:
    """文本聚类器 - 基于 sklearn"""

    def __init__(self, method: str = "kmeans", n_clusters: int = None,
                 max_clusters: int = 10, random_state: int = 42):
        """
        初始化聚类器

        参数:
        - method: 聚类方法 ('kmeans', 'minibatch_kmeans', 'agglomerative')
        - n_clusters: 聚类数量（None则自动确定）
        - max_clusters: 自动确定时的最大聚类数
        - random_state: 随机种子
        """
        self.method = method
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.labels = None
        self._fitted = False
        self._embeddings = None
        self._reducer = None
        self._clusterer = None

    def fit(self, embeddings: np.ndarray):
        """
        训练聚类模型

        参数:
        - embeddings: 文本向量矩阵 (N, D)
        """
        self._embeddings = embeddings
        n_samples = len(embeddings)

        if n_samples < 3:
            self.labels = np.full(n_samples, -1)
            self._fitted = True
            return self

        # 降维处理
        reduced_embeddings = self._reduce_dimensions(embeddings, n_samples)

        # 确定聚类数
        if self.n_clusters is None and n_samples >= 5:
            self.n_clusters = self._auto_determine_clusters(reduced_embeddings)

        # 确保 n_clusters 有效
        if self.n_clusters is None or self.n_clusters < 2:
            self.n_clusters = min(3, max(2, n_samples // 10))

        if self.n_clusters > n_samples - 1:
            self.n_clusters = max(2, n_samples // 2)

        # 执行聚类
        self.labels = self._do_clustering(reduced_embeddings, n_samples)

        self._fitted = True
        return self

    def _reduce_dimensions(self, embeddings: np.ndarray, n_samples: int) -> np.ndarray:
        """降维处理（使用 PCA）"""
        # 如果特征维度已经很小，直接返回
        if embeddings.shape[1] <= 20:
            return embeddings

        # 确定降维目标维度
        n_components = min(20, n_samples - 1, embeddings.shape[1])
        if n_components < 2:
            return embeddings

        try:
            # 标准化
            scaler = StandardScaler()
            scaled = scaler.fit_transform(embeddings)

            # PCA 降维
            pca = PCA(n_components=n_components, random_state=self.random_state)
            reduced = pca.fit_transform(scaled)

            # 如果 PCA 解释方差太低，尝试保留更多维度
            explained_var = pca.explained_variance_ratio_.sum()
            if explained_var < 0.6 and embeddings.shape[1] > n_components:
                n_components2 = min(50, n_samples - 1, embeddings.shape[1])
                if n_components2 > n_components:
                    pca2 = PCA(n_components=n_components2, random_state=self.random_state)
                    reduced = pca2.fit_transform(scaled)

            return reduced
        except Exception as e:
            print(f"⚠️ PCA 降维失败: {e}，使用原始向量")
            return embeddings

    def _auto_determine_clusters(self, X: np.ndarray) -> int:
        """自动确定最佳聚类数（使用轮廓系数）"""
        n_samples = len(X)
        max_k = min(self.max_clusters, n_samples - 1)

        if max_k < 2:
            return 2

        best_k = 2
        best_score = -1

        # 样本量小时，限制搜索范围
        if n_samples < 50:
            k_range = range(2, min(max_k + 1, n_samples))
        else:
            k_range = range(2, min(max_k + 1, 11))

        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(X)

                # 轮廓系数需要至少2个簇且每个簇至少2个样本
                if len(set(labels)) > 1:
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
            except Exception:
                continue

        return best_k

    def _do_clustering(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        """执行聚类"""
        try:
            if self.method == "kmeans":
                self._clusterer = KMeans(
                    n_clusters=self.n_clusters,
                    random_state=self.random_state,
                    n_init=10,
                    max_iter=300
                )
                labels = self._clusterer.fit_predict(X)

            elif self.method == "minibatch_kmeans":
                self._clusterer = MiniBatchKMeans(
                    n_clusters=self.n_clusters,
                    random_state=self.random_state,
                    batch_size=min(1000, n_samples // 10),
                    n_init=3
                )
                labels = self._clusterer.fit_predict(X)

            elif self.method == "agglomerative":
                # 层次聚类（样本量大时较慢）
                if n_samples > 2000:
                    # 大数据量降采样
                    from sklearn.cluster import AgglomerativeClustering

                    # 先 MiniBatchKMeans 预聚类
                    pre_k = min(100, n_samples // 100)
                    if pre_k > 1:
                        pre_kmeans = MiniBatchKMeans(n_clusters=pre_k, random_state=self.random_state)
                        pre_labels = pre_kmeans.fit_predict(X)
                        X_agg = pre_kmeans.cluster_centers_

                        agg = AgglomerativeClustering(
                            n_clusters=self.n_clusters,
                            linkage='ward'
                        )
                        agg_labels = agg.fit_predict(X_agg)

                        # 映射回原始标签
                        labels = np.zeros(n_samples, dtype=int)
                        for i, pl in enumerate(pre_labels):
                            labels[i] = agg_labels[pl]
                    else:
                        agg = AgglomerativeClustering(
                            n_clusters=self.n_clusters,
                            linkage='ward'
                        )
                        labels = agg.fit_predict(X)
                else:
                    from sklearn.cluster import AgglomerativeClustering
                    self._clusterer = AgglomerativeClustering(
                        n_clusters=self.n_clusters,
                        linkage='ward'
                    )
                    labels = self._clusterer.fit_predict(X)
            else:
                # 默认使用 KMeans
                self._clusterer = KMeans(
                    n_clusters=self.n_clusters,
                    random_state=self.random_state,
                    n_init=10
                )
                labels = self._clusterer.fit_predict(X)

            return labels

        except Exception as e:
            print(f"⚠️ 聚类失败: {e}，使用简单划分")
            # 降级：简单均分
            labels = np.zeros(n_samples, dtype=int)
            chunk_size = n_samples // self.n_clusters
            for k in range(self.n_clusters):
                start = k * chunk_size
                end = (k + 1) * chunk_size if k < self.n_clusters - 1 else n_samples
                labels[start:end] = k
            return labels

    def get_cluster_info(self, texts: List[str], embeddings: np.ndarray = None,
                         top_n_words: int = 10) -> List[Dict]:
        """获取聚类信息"""
        if not self._fitted:
            raise ValueError("请先调用 fit() 训练模型")

        cluster_info = []
        unique_labels = set(self.labels) - {-1}

        if not unique_labels:
            return []

        # 计算每个簇的平均向量（用于找中心）
        cluster_centers = {}
        if embeddings is not None:
            for label in unique_labels:
                mask = self.labels == label
                if np.any(mask):
                    cluster_centers[label] = embeddings[mask].mean(axis=0)

        for label in unique_labels:
            indices = [i for i, l in enumerate(self.labels) if l == label]
            size = len(indices)
            total_non_noise = len([l for l in self.labels if l != -1])
            percentage = size / total_non_noise if total_non_noise > 0 else 0

            # 获取关键词
            all_words = []
            for idx in indices[:100]:
                if idx < len(texts):
                    text = texts[idx][:200]
                    words = self._simple_tokenize(text)
                    all_words.extend(words)

            word_counter = Counter(all_words)
            top_words = [w for w, _ in word_counter.most_common(top_n_words)]

            # 找中心文本
            center_text = ""
            if embeddings is not None and label in cluster_centers:
                center_vec = cluster_centers[label]
                min_dist = float('inf')
                for idx in indices:
                    if idx < len(embeddings):
                        dist = np.linalg.norm(embeddings[idx] - center_vec)
                        if dist < min_dist:
                            min_dist = dist
                            if idx < len(texts):
                                center_text = texts[idx][:300]

            cluster_info.append({
                "cluster_id": int(label),
                "size": size,
                "percentage": round(percentage, 3),
                "top_words": top_words,
                "center_text": center_text
            })

        cluster_info.sort(key=lambda x: x["size"], reverse=True)

        # 重新编号
        for i, info in enumerate(cluster_info):
            info["cluster_id"] = i

        return cluster_info

    def _simple_tokenize(self, text: str) -> List[str]:
        """简单分词"""
        import re
        words = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
        stopwords = {'的', '了', '是', '在', '和', '与', '或', '也', '都', '还',
                     '这', '那', '有', '为', '对', '而', '并', '且', '但', '就',
                     '到', '从', '由', '于', '之', '将', '会', '能', '可', '以',
                     '一个', '我们', '他们', '你们', '这些', '那些'}
        return [w for w in words if w not in stopwords]

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """预测新文本的聚类标签"""
        if not self._fitted:
            raise ValueError("请先调用 fit() 训练模型")

        if self._clusterer is not None and hasattr(self._clusterer, 'predict'):
            return self._clusterer.predict(embeddings)
        else:
            # 简单最近邻
            distances = np.linalg.norm(embeddings[:, np.newaxis, :] - self._embeddings, axis=2)
            nearest = np.argmin(distances, axis=1)
            return self.labels[nearest]