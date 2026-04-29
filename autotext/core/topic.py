"""
主题建模模块 - 基于聚类结果的TF-IDF
"""

from collections import Counter
from typing import List, Dict, Any, Optional
import numpy as np


class TopicModeler:
    """主题建模器 - 基于聚类结果的TF-IDF"""

    def __init__(self, n_topics: int = 10):
        self.n_topics = n_topics
        self._fitted = False
        self.topics = []
        self.topic_labels = None  # 新增：存储每条文本的主题标签
        self.labels = None        # 新增：兼容 cluster_labels 参数

    def fit(self, texts: List[str], cluster_labels: List[int] = None):
        """
        训练主题模型

        参数:
        - texts: 文本列表
        - cluster_labels: 聚类标签（-1表示噪声）
        """
        # 存储标签供后续使用
        if cluster_labels is not None:
            self.topic_labels = cluster_labels.copy()
            self.labels = cluster_labels.copy()
        else:
            # 如果没有传入标签，创建默认标签（所有文本归为一类）
            self.topic_labels = [0] * len(texts)
            self.labels = [0] * len(texts)

        # 只使用有聚类标签的文本
        valid_indices = [i for i, l in enumerate(self.topic_labels) if l != -1]
        if not valid_indices:
            self._fitted = True
            return self

        valid_texts = [texts[i] for i in valid_indices]
        valid_labels = [self.topic_labels[i] for i in valid_indices]

        unique_labels = set(valid_labels)
        self.topics = []

        # 计算全局词频（用于TF-IDF）
        global_word_freq = Counter()
        for text in valid_texts:
            words = self._simple_tokenize(text)
            global_word_freq.update(words)
        total_docs = len(valid_texts)

        for label in unique_labels:
            # 获取该主题的文本
            topic_indices = [i for i, l in enumerate(valid_labels) if l == label]
            topic_texts = [valid_texts[i] for i in topic_indices]

            # 统计主题内词频
            topic_word_freq = Counter()
            for text in topic_texts:
                words = self._simple_tokenize(text)
                topic_word_freq.update(words)

            # 计算TF-IDF分数
            word_scores = []
            for word, tf in topic_word_freq.most_common(50):
                # 计算文档频率
                df = global_word_freq.get(word, 1)
                idf = np.log(total_docs / df) if df > 0 else 0
                score = tf * idf
                word_scores.append((word, score))

            word_scores.sort(key=lambda x: x[1], reverse=True)

            # 找代表性文本
            representative_text = ""
            if topic_texts:
                # 选择最长的文本作为代表
                representative_text = max(topic_texts, key=len)[:300]

            self.topics.append({
                "topic_id": int(label),
                "texts_count": len(topic_texts),
                "keywords": [w for w, _ in word_scores[:15]],
                "weights": [round(s, 3) for _, s in word_scores[:15]],
                "representative_text": representative_text
            })

        # 按文本数量排序
        self.topics.sort(key=lambda x: x["texts_count"], reverse=True)

        # 重新编号，并更新 topic_labels 中的标签
        old_to_new = {}
        for i, topic in enumerate(self.topics):
            old_id = topic["topic_id"]
            old_to_new[old_id] = i
            topic["topic_id"] = i

        # 更新 topic_labels 为新编号
        if self.topic_labels is not None:
            new_labels = []
            for label in self.topic_labels:
                if label == -1:
                    new_labels.append(-1)
                else:
                    new_labels.append(old_to_new.get(label, -1))
            self.topic_labels = new_labels
            self.labels = new_labels

        self._fitted = True
        return self

    def get_topics(self) -> List[Dict]:
        """获取主题列表"""
        if not self._fitted:
            return []
        return self.topics

    def get_topic_distribution(self) -> List[float]:
        """获取主题分布"""
        if not self._fitted:
            return []
        total = sum(t["texts_count"] for t in self.topics)
        if total == 0:
            return []
        return [t["texts_count"] / total for t in self.topics]

    def _simple_tokenize(self, text: str) -> List[str]:
        """简单分词"""
        import re
        words = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
        stopwords = {'的', '了', '是', '在', '和', '与', '或', '也', '都', '还',
                     '这', '那', '有', '为', '对', '而', '并', '且', '但', '就',
                     '到', '从', '由', '于', '之', '将', '会', '能', '可', '以'}
        return [w for w in words if w not in stopwords and len(w) >= 2]