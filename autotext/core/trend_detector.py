"""
趋势检测模块 - 热点话题、情感转折点
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np


class TrendDetector:
    """趋势检测器"""

    def __init__(self):
        self.hot_topics = []
        self.sentiment_shifts = []

    def detect_hot_topics(self, entity_by_date: Dict[str, Dict[str, int]],
                           window: int = 7,
                           threshold: float = 1.0) -> List[Dict]:
        """
        检测热点话题（近期突增）

        参数:
        - entity_by_date: {"实体名": {"日期": 次数}}
        - window: 近期窗口大小（天）
        - threshold: 突增倍数阈值

        返回: [
            {
                "entity": "实体名",
                "recent_avg": 45,
                "history_avg": 12,
                "increase_pct": 275,
                "peak_date": "2026-04-25"
            }
        ]
        """
        hot_topics = []

        for entity, date_counts in entity_by_date.items():
            if len(date_counts) < window + 3:
                continue

            dates = sorted(date_counts.keys())
            counts = [date_counts[d] for d in dates]

            # 近期窗口
            recent_counts = counts[-window:] if len(counts) >= window else counts
            recent_avg = sum(recent_counts) / len(recent_counts) if recent_counts else 0

            # 历史窗口（排除近期）
            history_counts = counts[:-window] if len(counts) > window else []
            history_avg = sum(history_counts) / len(history_counts) if history_counts else recent_avg

            if history_avg == 0:
                continue

            increase_pct = (recent_avg - history_avg) / history_avg * 100

            if increase_pct > threshold * 100:
                # 找到峰值日期
                peak_idx = np.argmax(recent_counts)
                peak_date = dates[-(window - peak_idx)] if peak_idx < len(dates) else dates[-1]

                hot_topics.append({
                    "entity": entity,
                    "recent_avg": round(recent_avg, 1),
                    "history_avg": round(history_avg, 1),
                    "increase_pct": round(increase_pct, 1),
                    "peak_date": peak_date
                })

        # 按增长率排序
        hot_topics.sort(key=lambda x: x["increase_pct"], reverse=True)
        return hot_topics[:10]

    def detect_sentiment_shifts(self, entity_sentiment_by_date: Dict[str, Dict[str, float]],
                                  window: int = 3,
                                  shift_threshold: float = 0.3) -> List[Dict]:
        """
        检测情感转折点

        参数:
        - entity_sentiment_by_date: {"实体名": {"日期": 情感得分(-1到1)}}
        - window: 平滑窗口
        - shift_threshold: 转折阈值

        返回: [
            {
                "entity": "实体名",
                "from_sentiment": 0.6,
                "to_sentiment": -0.2,
                "shift_date": "2026-04-20",
                "direction": "negative"
            }
        ]
        """
        shifts = []

        for entity, date_sentiments in entity_sentiment_by_date.items():
            if len(date_sentiments) < window + 3:
                continue

            dates = sorted(date_sentiments.keys())
            sentiments = [date_sentiments[d] for d in dates]

            # 平滑
            smoothed = []
            for i in range(len(sentiments)):
                start = max(0, i - window // 2)
                end = min(len(sentiments), i + window // 2 + 1)
                smoothed.append(sum(sentiments[start:end]) / (end - start))

            # 检测转折点
            for i in range(1, len(smoothed) - 1):
                prev = smoothed[i - 1]
                curr = smoothed[i]
                nxt = smoothed[i + 1]

                # 检测峰值或谷值
                if curr > prev + shift_threshold and curr > nxt + shift_threshold:
                    direction = "positive_to_negative" if nxt < curr - shift_threshold else "peak"
                    shifts.append({
                        "entity": entity,
                        "from_sentiment": round(prev, 2),
                        "to_sentiment": round(nxt, 2),
                        "shift_date": dates[i],
                        "direction": direction,
                        "type": "peak"
                    })
                elif curr < prev - shift_threshold and curr < nxt - shift_threshold:
                    direction = "negative_to_positive" if nxt > curr + shift_threshold else "valley"
                    shifts.append({
                        "entity": entity,
                        "from_sentiment": round(prev, 2),
                        "to_sentiment": round(nxt, 2),
                        "shift_date": dates[i],
                        "direction": direction,
                        "type": "valley"
                    })

        # 按变化幅度排序
        shifts.sort(key=lambda x: abs(x["to_sentiment"] - x["from_sentiment"]), reverse=True)
        return shifts[:10]

    def build_entity_by_date(self, entity_results: List[Dict], dates: List) -> Dict[str, Dict[str, int]]:
        """从实体识别结果和时间构建实体-日期频次矩阵"""
        if not dates or len(dates) != len(entity_results):
            return {}

        entity_by_date = defaultdict(lambda: defaultdict(int))

        for idx, result in enumerate(entity_results):
            date = dates[idx]
            if date is None:
                continue
            date_str = str(date)[:10]  # YYYY-MM-DD

            for entity_type, entities in result.items():
                for entity_text, _, _ in entities:
                    key = f"{entity_type}:{entity_text}"
                    entity_by_date[key][date_str] += 1

        return dict(entity_by_date)

    def build_sentiment_by_date(self, entity_results: List[Dict], sentiment_results: List[Dict],
                                 dates: List) -> Dict[str, Dict[str, float]]:
        """构建实体-日期情感矩阵"""
        if not dates or len(dates) != len(entity_results) or not sentiment_results:
            return {}

        entity_sentiment = defaultdict(lambda: defaultdict(list))

        for idx, result in enumerate(entity_results):
            date = dates[idx]
            if date is None or idx >= len(sentiment_results):
                continue
            date_str = str(date)[:10]

            sentiment_score = 0
            sent = sentiment_results[idx].get('sentiment', 'neutral')
            if sent == 'positive':
                sentiment_score = 1
            elif sent == 'negative':
                sentiment_score = -1
            else:
                sentiment_score = 0

            for entity_type, entities in result.items():
                for entity_text, _, _ in entities:
                    key = f"{entity_type}:{entity_text}"
                    entity_sentiment[key][date_str].append(sentiment_score)

        # 取每日平均
        result = {}
        for entity, date_scores in entity_sentiment.items():
            result[entity] = {}
            for date_str, scores in date_scores.items():
                result[entity][date_str] = sum(scores) / len(scores) if scores else 0

        return result