"""
情感分析模块 - 基于情感词典
"""

import re
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter


class SentimentAnalyzer:
    """情感分析器 - 基于情感词典"""

    # 中文正面情感词（扩展）
    CHINESE_POSITIVE = {
        '好', '不错', '满意', '喜欢', '赞', '棒', '优秀', '出色', '精彩', '完美',
        '开心', '高兴', '愉快', '幸福', '感激', '感谢', '支持', '推荐', '值得',
        '良心', '厚道', '贴心', '周到', '专业', '高效', '快速', '便宜', '实惠',
        '性价比', '超值', '给力', '牛逼', '厉害', '强大', '先进', '创新',
        '清晰', '流畅', '稳定', '耐用', '美观', '大气', '时尚', '舒适',
        '环保', '健康', '安全', '放心', '靠谱', '可信', '真诚', '热情',
    }

    # 中文负面情感词（扩展）
    CHINESE_NEGATIVE = {
        '差', '差劲', '糟糕', '失望', '讨厌', '恨', '垃圾', '烂', '恶心',
        '难过', '伤心', '痛苦', '愤怒', '生气', '抱怨', '投诉', '退款',
        '退货', '问题', '毛病', '故障', '卡顿', '闪退', '死机', '慢',
        '贵', '昂贵', '坑', '骗', '假', '劣质', '粗糙', '简陋',
        '难用', '麻烦', '繁琐', '混乱', '差评', '黑心', '无良',
        '虚假', '欺骗', '忽悠', '敷衍', '拖延', '推诿',
    }

    # 中文否定词
    CHINESE_NEGATION = {'不', '没', '无', '非', '莫', '勿', '别', '未', '不是', '没有'}

    # 英文正面词
    ENGLISH_POSITIVE = {
        'good', 'great', 'excellent', 'perfect', 'awesome', 'amazing', 'fantastic',
        'nice', 'lovely', 'beautiful', 'wonderful', 'superb', 'outstanding',
        'happy', 'pleased', 'satisfied', 'delighted', 'grateful',
        'love', 'like', 'recommend', 'support', 'value',
        'fast', 'quick', 'easy', 'simple', 'clean', 'clear', 'stable',
        'cheap', 'affordable', 'worth', 'best', 'favorite'
    }

    # 英文负面词
    ENGLISH_NEGATIVE = {
        'bad', 'terrible', 'awful', 'horrible', 'poor', 'worse', 'worst',
        'disappointed', 'disappointing', 'frustrated', 'annoying',
        'hate', 'dislike', 'complaint', 'issue', 'problem',
        'broken', 'damaged', 'defective', 'faulty',
        'slow', 'lag', 'crash', 'freeze', 'expensive', 'overpriced',
        'fake', 'fraud', 'scam', 'useless', 'waste'
    }

    # 英文否定词
    ENGLISH_NEGATION = {'no', 'not', 'never', 'none', "n't"}

    def __init__(self, language: str = "auto"):
        """
        初始化情感分析器

        参数:
        - language: 语言 ('auto', 'zh', 'en')
        """
        self.language = language

    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        if self.language != "auto":
            return self.language

        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        if chinese_chars / max(len(text), 1) > 0.2:
            return "zh"
        return "en"

    def _tokenize_sentences(self, text: str) -> List[str]:
        """分句"""
        sentences = re.split(r'[。！？!?；;]', text)
        return [s.strip() for s in sentences if s.strip()]

    def _tokenize_words(self, text: str, language: str) -> List[str]:
        """分词"""
        if language == "zh":
            try:
                import jieba
                return list(jieba.cut(text))
            except ImportError:
                return re.findall(r'[\u4e00-\u9fff]+', text)
        else:
            return re.findall(r'[a-zA-Z\']+', text.lower())

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        分析单条文本的情感

        参数:
        - text: 文本

        返回:
        {
            "sentiment": str,          # 'positive', 'negative', 'neutral'
            "confidence": float,       # 0-1
            "positive_score": float,   # 正面词得分
            "negative_score": float,   # 负面词得分
            "positive_words": List[str],
            "negative_words": List[str],
            "sentence_analysis": List[Dict]  # 分句分析
        }
        """
        if not text or not isinstance(text, str):
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "positive_score": 0.0,
                "negative_score": 0.0,
                "positive_words": [],
                "negative_words": [],
                "sentence_analysis": []
            }

        language = self._detect_language(text)
        positive_set = self.CHINESE_POSITIVE if language == "zh" else self.ENGLISH_POSITIVE
        negative_set = self.CHINESE_NEGATIVE if language == "zh" else self.ENGLISH_NEGATIVE
        negation_set = self.CHINESE_NEGATION if language == "zh" else self.ENGLISH_NEGATION

        sentences = self._tokenize_sentences(text)

        total_positive = 0
        total_negative = 0
        all_positive_words = []
        all_negative_words = []
        sentence_results = []

        for sent in sentences:
            words = self._tokenize_words(sent, language)
            pos_score = 0
            neg_score = 0
            pos_words = []
            neg_words = []
            negated = False

            for word in words:
                # 处理否定词
                if word in negation_set:
                    negated = True
                    continue

                if word in positive_set:
                    score = 1
                    if negated:
                        score = -0.5
                        pos_words.append(f"!{word}")
                    else:
                        pos_words.append(word)
                    pos_score += score
                elif word in negative_set:
                    score = 1
                    if negated:
                        score = -0.5
                        neg_words.append(f"!{word}")
                    else:
                        neg_words.append(word)
                    neg_score += score

            total_positive += max(0, pos_score)
            total_negative += max(0, neg_score)
            all_positive_words.extend([w for w in pos_words if not w.startswith('!')])
            all_negative_words.extend([w for w in neg_words if not w.startswith('!')])

            if pos_score > 0 or neg_score > 0:
                sent_sentiment = "positive" if pos_score > neg_score else "negative" if neg_score > pos_score else "neutral"
                sentence_results.append({
                    "text": sent[:100],
                    "sentiment": sent_sentiment,
                    "positive_score": pos_score,
                    "negative_score": neg_score,
                    "positive_words": pos_words[:5],
                    "negative_words": neg_words[:5]
                })

        # 计算整体情感
        total_score = total_positive - total_negative
        if total_score > 0:
            sentiment = "positive"
        elif total_score < 0:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # 计算置信度
        max_score = max(total_positive, total_negative, 1)
        confidence = min(abs(total_score) / max_score, 1.0) if max_score > 0 else 0.0

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "positive_score": total_positive,
            "negative_score": total_negative,
            "positive_words": list(set(all_positive_words))[:10],
            "negative_words": list(set(all_negative_words))[:10],
            "sentence_analysis": sentence_results[:5]
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """批量分析情感"""
        return [self.analyze(t) for t in texts]

    def get_distribution(self, results: List[Dict]) -> Dict[str, Any]:
        """
        获取情感分布

        参数:
        - results: analyze_batch 的结果列表

        返回:
        {
            "positive": int,
            "negative": int,
            "neutral": int,
            "positive_rate": float,
            "negative_rate": float,
            "neutral_rate": float
        }
        """
        counts = {"positive": 0, "negative": 0, "neutral": 0}
        for r in results:
            counts[r["sentiment"]] += 1

        total = len(results)
        return {
            "positive": counts["positive"],
            "negative": counts["negative"],
            "neutral": counts["neutral"],
            "positive_rate": counts["positive"] / total if total > 0 else 0,
            "negative_rate": counts["negative"] / total if total > 0 else 0,
            "neutral_rate": counts["neutral"] / total if total > 0 else 0
        }

    def get_summary(self, results: List[Dict]) -> Dict:
        """获取情感分析摘要"""
        distribution = self.get_distribution(results)

        # 计算平均置信度
        avg_confidence = sum(r["confidence"] for r in results) / len(results) if results else 0

        # 找出最积极和最消极的文本
        positive_texts = [(i, r["positive_score"] - r["negative_score"]) for i, r in enumerate(results) if r["sentiment"] == "positive"]
        negative_texts = [(i, r["positive_score"] - r["negative_score"]) for i, r in enumerate(results) if r["sentiment"] == "negative"]

        positive_texts.sort(key=lambda x: x[1], reverse=True)
        negative_texts.sort(key=lambda x: x[1])

        return {
            "distribution": distribution,
            "avg_confidence": avg_confidence,
            "most_positive_indices": [i for i, _ in positive_texts[:3]],
            "most_negative_indices": [i for i, _ in negative_texts[:3]]
        }