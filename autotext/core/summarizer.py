"""
文本摘要模块 - TextRank + LLM 双方案
"""

import re
from collections import defaultdict
from typing import List, Dict, Any, Optional


class TextRankSummarizer:
    """TextRank 摘要提取器（无监督，不依赖外部模型）"""

    def __init__(self):
        self._window = 2
        self._damping = 0.85
        self._iterations = 30
        self._tol = 1e-6

    def _split_sentences(self, text: str) -> List[str]:
        """分句（简单规则）"""
        # 按句号、问号、感叹号、分号分割
        sentences = re.split(r'[。！？！;；\n]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences

    def _tokenize(self, sentence: str) -> List[str]:
        """简单分词（提取2-4字中文词）"""
        words = re.findall(r'[\u4e00-\u9fff]{2,4}', sentence)
        # 停用词过滤
        stopwords = {'的', '了', '是', '在', '和', '与', '或', '也', '都', '还',
                     '这', '那', '有', '为', '对', '而', '并', '且', '但', '就',
                     '到', '从', '由', '于', '之', '将', '会', '能', '可', '以',
                     '年', '月', '日', '时', '分', '秒', '上', '下', '中', '内',
                     '外', '前', '后', '左', '右', '高', '低', '大', '小', '多',
                     '少', '新', '旧', '好', '坏', '正', '负', '涨', '跌'}
        return [w for w in words if w not in stopwords and len(w) >= 2]

    def _build_similarity_matrix(self, sentences: List[str]) -> List[List[float]]:
        """构建句子相似度矩阵（基于公共词比例）"""
        n = len(sentences)
        if n == 0:
            return []

        # 句子->词集合
        word_sets = [set(self._tokenize(s)) for s in sentences]

        # 初始化相似度矩阵
        similarity = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                if not word_sets[i] or not word_sets[j]:
                    continue
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                sim = intersection / union if union > 0 else 0
                if sim > 0.1:  # 阈值
                    similarity[i][j] = sim
                    similarity[j][i] = sim

        return similarity

    def _textrank(self, similarity_matrix: List[List[float]]) -> List[float]:
        """TextRank 算法计算句子权重"""
        n = len(similarity_matrix)
        if n == 0:
            return []

        # 初始化分数
        scores = [1.0] * n

        for _ in range(self._iterations):
            new_scores = []
            for i in range(n):
                sum_score = 0.0
                for j in range(n):
                    if i != j and similarity_matrix[i][j] > 0:
                        # 计算出度
                        out_degree = sum(1 for k in range(n) if similarity_matrix[j][k] > 0)
                        if out_degree > 0:
                            sum_score += similarity_matrix[i][j] * scores[j] / out_degree
                new_score = (1 - self._damping) + self._damping * sum_score
                new_scores.append(new_score)

            # 检查收敛
            diff = sum(abs(new_scores[i] - scores[i]) for i in range(n))
            scores = new_scores
            if diff < self._tol:
                break

        return scores

    def extract_key_sentences(self, text: str, top_n: int = 3) -> List[str]:
        """提取关键句"""
        if not text or len(text) < 50:
            return []

        sentences = self._split_sentences(text)
        if len(sentences) <= top_n:
            return sentences

        similarity = self._build_similarity_matrix(sentences)
        scores = self._textrank(similarity)

        if not scores:
            return sentences[:top_n]

        # 按分数排序取 top_n
        indexed = list(enumerate(sentences))
        indexed.sort(key=lambda x: scores[x[0]], reverse=True)

        # 按原文顺序返回
        selected_indices = sorted([idx for idx, _ in indexed[:top_n]])
        return [sentences[i] for i in selected_indices]

    def extract_cluster_key_sentences(self, texts: List[str], top_n: int = 3) -> List[str]:
        """从簇的多个文本中提取代表性句子"""
        if not texts:
            return []

        # 合并文本（限制总长度）
        combined = "\n".join([t[:500] for t in texts[:10]])
        return self.extract_key_sentences(combined, top_n)


class LLMSummarizer:
    """大模型摘要生成器"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def is_available(self) -> bool:
        return self.llm_client is not None

    def generate_cluster_summary(self, keywords: List[str], sample_texts: List[str],
                                 max_length: int = 200, summary_type: str = "cluster") -> Dict[str, str]:
        """
        生成簇/主题的标题和摘要

        参数:
        - keywords: 关键词列表
        - sample_texts: 样本文本列表
        - max_length: 摘要最大长度
        - summary_type: 类型 ("cluster" 或 "topic")

        返回: {"title": "...", "summary": "..."}
        """
        if not self.is_available():
            return {"title": "", "summary": ""}

        # 准备输入
        keyword_str = "、".join(keywords[:12])
        sample_str = "\n".join([t[:200] for t in sample_texts[:3]])

        if summary_type == "topic":
            prompt = f"""请根据以下关键词和文本片段，为这个主题生成一个简短的主题标题（10字以内）和一段内容摘要（80字以内）。

    关键词：{keyword_str}

    文本片段：
    {sample_str}

    请按以下格式输出：
    标题：[标题内容]
    摘要：[摘要内容]"""
        else:
            prompt = f"""请根据以下关键词和文本片段，为这个文本聚类生成一个简短的主题标题（10字以内）和一段内容摘要（80字以内）。

    关键词：{keyword_str}

    文本片段：
    {sample_str}

    请按以下格式输出：
    标题：[标题内容]
    摘要：[摘要内容]"""

        try:
            response = self.llm_client.chat([{"role": "user", "content": prompt}], temperature=0.5)

            # 解析响应
            title = ""
            summary = ""
            for line in response.strip().split("\n"):
                if line.startswith("标题：") or line.startswith("标题:"):
                    title = line.replace("标题：", "").replace("标题:", "").strip()
                elif line.startswith("摘要：") or line.startswith("摘要:"):
                    summary = line.replace("摘要：", "").replace("摘要:", "").strip()

            # 截断
            if len(title) > 20:
                title = title[:20]
            if len(summary) > max_length:
                summary = summary[:max_length]

            return {"title": title, "summary": summary}
        except Exception as e:
            print(f"LLM摘要生成失败: {e}")
            return {"title": "", "summary": ""}