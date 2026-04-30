"""
主题建模模块 - 基于 LDA，支持 TextRank + LLM 摘要
"""

import numpy as np
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
import re
import warnings

warnings.filterwarnings('ignore')


class TopicModeler:
    """主题建模器 - 基于 LDA，支持摘要生成"""

    def __init__(self, n_topics: int = 10, max_features: int = 2000, random_state: int = 42):
        self.n_topics = n_topics
        self.max_features = max_features
        self.random_state = random_state
        self.model = None
        self.vectorizer = None
        self.topic_distributions = None
        self._fitted = False
        self._feature_names = []
        self._original_X = None
        self.topic_labels = None
        self.labels = None
        self._original_indices = None
        self._texts = None

    def _build_dicts(self, token_lists: List[List[str]]) -> List[Dict[str, int]]:
        from collections import Counter
        dicts = []
        for tokens in token_lists:
            if tokens:
                dicts.append(Counter(tokens))
            else:
                dicts.append({})
        return dicts

    def fit(self, token_lists: List[List[str]], texts: List[str] = None, cluster_labels: List[int] = None):
        """
        训练 LDA 模型

        参数:
        - token_lists: 已经分词好的 token 列表
        - texts: 原始文本列表（用于摘要）
        - cluster_labels: 聚类标签（可选）
        """
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        self._texts = texts
        self.labels = cluster_labels

        dicts = self._build_dicts(token_lists)

        non_empty = [i for i, d in enumerate(dicts) if d]
        if len(non_empty) < 10:
            print(f"  ⚠️ 主题建模: 有效文本不足 ({len(non_empty)} < 10)")
            self._fitted = True
            self.topic_labels = [-1] * len(token_lists)
            return self

        filtered_dicts = [dicts[i] for i in non_empty]
        self._original_indices = non_empty

        self.vectorizer = DictVectorizer()
        X = self.vectorizer.fit_transform(filtered_dicts)
        self._feature_names = self.vectorizer.get_feature_names_out()

        if X.shape[1] > self.max_features:
            col_sums = X.sum(axis=0).A1
            top_indices = col_sums.argsort()[-self.max_features:][::-1]
            X = X[:, top_indices]
            self._feature_names = self._feature_names[top_indices]

        self._original_X = X

        if self.n_topics is None or self.n_topics > X.shape[0] - 1:
            self.n_topics = min(10, max(2, X.shape[0] // 20))

        self.model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=self.random_state,
            learning_method='online',
            max_iter=100
        )
        self.topic_distributions = self.model.fit_transform(X)

        self.topic_labels = [-1] * len(dicts)
        for i, label in zip(non_empty, np.argmax(self.topic_distributions, axis=1)):
            self.topic_labels[i] = int(label)

        self._fitted = True
        return self

    def get_topics(self, top_n_words: int = 10) -> List[Dict]:
        if not self._fitted:
            return []

        topics = []
        for topic_idx, topic in enumerate(self.model.components_):
            n = min(top_n_words, len(topic))
            top_indices = topic.argsort()[:-n - 1:-1]
            keywords = [self._feature_names[i] for i in top_indices]
            weights = [topic[i] for i in top_indices]
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]

            topics.append({
                "topic_id": topic_idx,
                "texts_count": sum(1 for l in self.topic_labels if l == topic_idx),
                "keywords": keywords,
                "weights": weights,
                "textrank_sentences": [],
                "llm_title": "",
                "llm_summary": "",
                "representative_texts": []
            })

        topics.sort(key=lambda x: x["texts_count"], reverse=True)
        for i, topic in enumerate(topics):
            topic["topic_id"] = i

        return topics

    def add_summaries(self, topics: List[Dict], texts: List[str], textrank_summarizer=None, llm_summarizer=None):
        """为主题添加摘要"""
        if not self._fitted or not self.topic_labels:
            return topics

        for topic in topics:
            topic_id = topic["topic_id"]

            # 获取该主题下的文本索引
            indices = [i for i, l in enumerate(self.topic_labels) if l == topic_id]
            topic_texts = [texts[i] for i in indices if i < len(texts)]

            if not topic_texts:
                continue

            # 记录代表性文本
            topic["representative_texts"] = [t[:300] for t in topic_texts[:3]]

            # TextRank 摘要
            if textrank_summarizer:
                try:
                    key_sentences = self._extract_key_sentences(topic_texts, textrank_summarizer)
                    topic["textrank_sentences"] = key_sentences
                except Exception as e:
                    print(f"  ⚠️ TextRank 摘要失败: {e}")

            # LLM 摘要
            if llm_summarizer and llm_summarizer.is_available():
                try:
                    llm_result = llm_summarizer.generate_cluster_summary(
                        topic.get("keywords", [])[:10],
                        topic_texts
                    )
                    topic["llm_title"] = llm_result.get("title", "")
                    topic["llm_summary"] = llm_result.get("summary", "")
                except Exception as e:
                    print(f"  ⚠️ LLM 摘要失败: {e}")

        return topics

    def _extract_key_sentences(self, texts: List[str], textrank_summarizer, top_n: int = 3) -> List[str]:
        """从文本列表中提取关键句子"""
        if not texts:
            return []

        combined = "\n".join([t[:500] for t in texts[:10]])
        return textrank_summarizer.extract_key_sentences(combined, top_n)

    def get_topic_distribution(self) -> List[float]:
        if not self._fitted:
            return []
        if self.topic_distributions is None:
            return []
        return self.topic_distributions.mean(axis=0).tolist()

    def get_topic_labels(self) -> List[int]:
        return self.topic_labels if self.topic_labels else []