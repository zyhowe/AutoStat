"""
事件脉络分析模块 - 按时间聚合实体，还原事件发展
"""

from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd


class EventTimelineAnalyzer:
    """事件脉络分析器"""

    def __init__(self):
        self.events = []
        self.entity_timelines = {}
        self.hot_topics = {}

    def analyze(self, texts: List[str], dates: List[Any],
                entity_results: List[Dict[str, List[Tuple[str, int, int]]]],
                sentiment_results: List[Dict] = None) -> Dict[str, Any]:
        """
        分析事件脉络

        参数:
        - texts: 文本列表
        - dates: 日期列表
        - entity_results: 实体识别结果
        - sentiment_results: 情感分析结果（可选）
        """
        if not dates or len(dates) == 0:
            return {"error": "无时间信息"}

        # 解析日期
        parsed_dates = self._parse_dates(dates)
        valid_indices = [i for i, d in enumerate(parsed_dates) if d is not None]

        if len(valid_indices) < 5:
            return {"error": "有效时间点不足"}

        # 构建实体时间线
        entity_timelines = self._build_entity_timelines(
            texts, parsed_dates, entity_results, sentiment_results, valid_indices
        )

        # 检测热点事件
        hot_topics = self._detect_hot_topics(entity_timelines)

        # 构建事件脉络
        event_chains = self._build_event_chains(entity_timelines, hot_topics)

        # 情感演变
        sentiment_evolution = self._analyze_sentiment_evolution(
            entity_timelines, sentiment_results, parsed_dates, valid_indices
        )

        return {
            "entity_timelines": entity_timelines,
            "hot_topics": hot_topics,
            "event_chains": event_chains,
            "sentiment_evolution": sentiment_evolution,
            "time_range": {
                "start": min(parsed_dates[i] for i in valid_indices),
                "end": max(parsed_dates[i] for i in valid_indices)
            },
            "total_time_points": len(valid_indices)
        }

    def _parse_dates(self, dates: List[Any]) -> List[Optional[datetime]]:
        """解析日期"""
        parsed = []
        for d in dates:
            if d is None or pd.isna(d):
                parsed.append(None)
                continue
            if isinstance(d, datetime):
                parsed.append(d)
            else:
                try:
                    parsed.append(pd.to_datetime(d).to_pydatetime())
                except Exception:
                    parsed.append(None)
        return parsed

    def _build_entity_timelines(self, texts: List[str], dates: List[Optional[datetime]],
                                 entity_results: List[Dict], sentiment_results: List[Dict],
                                 valid_indices: List[int]) -> List[Dict]:
        """构建实体时间线"""
        # 按日期排序
        sorted_indices = sorted(valid_indices, key=lambda i: dates[i])

        # 按实体聚合
        entity_by_date = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'texts': [], 'sentiments': []}))

        for idx in sorted_indices:
            date = dates[idx]
            date_str = date.strftime("%Y-%m-%d")
            text = texts[idx][:200]

            # 获取实体
            for entity_type, entities in entity_results[idx].items():
                for entity_text, _, _ in entities:
                    key = f"{entity_type}:{entity_text}"
                    entity_by_date[key][date_str]['count'] += 1
                    entity_by_date[key][date_str]['texts'].append(text)

                    # 添加情感
                    if sentiment_results and idx < len(sentiment_results):
                        sent = sentiment_results[idx].get('sentiment', 'neutral')
                        score = 1 if sent == 'positive' else (-1 if sent == 'negative' else 0)
                        entity_by_date[key][date_str]['sentiments'].append(score)

        # 转换为列表格式
        timelines = []
        for entity, date_data in entity_by_date.items():
            timeline = []
            for date_str, data in sorted(date_data.items()):
                avg_sentiment = sum(data['sentiments']) / len(data['sentiments']) if data['sentiments'] else 0
                timeline.append({
                    'date': date_str,
                    'count': data['count'],
                    'avg_sentiment': round(avg_sentiment, 2),
                    'sample_text': data['texts'][0] if data['texts'] else ''
                })
            timelines.append({
                'entity': entity,
                'total_mentions': sum(d['count'] for d in timeline),
                'timeline': timeline
            })

        # 按提及次数排序
        timelines.sort(key=lambda x: x['total_mentions'], reverse=True)

        return timelines[:20]  # 只返回前20个

    def _detect_hot_topics(self, entity_timelines: List[Dict]) -> List[Dict]:
        """检测热点话题（时间窗口内突增）"""
        hot_topics = []

        for entity_data in entity_timelines:
            timeline = entity_data['timeline']
            if len(timeline) < 3:
                continue

            # 计算平均提及次数
            avg_count = sum(d['count'] for d in timeline) / len(timeline)

            # 检测突增
            peaks = []
            for i, d in enumerate(timeline):
                if d['count'] > avg_count * 2:
                    # 检查是否是持续峰值
                    is_peak = True
                    if i > 0 and timeline[i-1]['count'] > avg_count * 1.5:
                        is_peak = False
                    if i < len(timeline) - 1 and timeline[i+1]['count'] > avg_count * 1.5:
                        is_peak = False

                    if is_peak:
                        peaks.append({
                            'date': d['date'],
                            'count': d['count'],
                            'avg_sentiment': d['avg_sentiment']
                        })

            if peaks:
                hot_topics.append({
                    'entity': entity_data['entity'],
                    'total_mentions': entity_data['total_mentions'],
                    'avg_mentions': round(avg_count, 1),
                    'peaks': peaks[:3]
                })

        # 按峰值数量排序
        hot_topics.sort(key=lambda x: len(x['peaks']), reverse=True)

        return hot_topics[:15]

    def _build_event_chains(self, entity_timelines: List[Dict],
                            hot_topics: List[Dict]) -> List[Dict]:
        """构建事件链（按时间聚合相关实体）"""
        # 简化实现：以热点话题的事件链为主
        event_chains = []

        for hot in hot_topics[:5]:
            entity = hot['entity']

            # 查找同时间段相关的其他实体
            related_entities = []
            for other in entity_timelines[:10]:
                if other['entity'] != entity:
                    # 简单的时间重叠检测
                    overlap = False
                    for peak in hot['peaks']:
                        for other_point in other['timeline']:
                            if other_point['date'] == peak['date']:
                                overlap = True
                                break
                        if overlap:
                            break
                    if overlap:
                        related_entities.append({
                            'entity': other['entity'],
                            'relation_strength': 'high'
                        })

            event_chains.append({
                'main_entity': entity,
                'peaks': hot['peaks'],
                'related_entities': related_entities[:5]
            })

        return event_chains

    def _analyze_sentiment_evolution(self, entity_timelines: List[Dict],
                                      sentiment_results: List[Dict],
                                      dates: List[Optional[datetime]],
                                      valid_indices: List[int]) -> Dict[str, Any]:
        """分析情感演变"""
        if not sentiment_results:
            return {}

        # 按日期聚合情感
        sentiment_by_date = defaultdict(list)

        for idx in valid_indices:
            if idx >= len(sentiment_results):
                continue
            date = dates[idx]
            date_str = date.strftime("%Y-%m-%d")
            sent = sentiment_results[idx].get('sentiment', 'neutral')
            score = 1 if sent == 'positive' else (-1 if sent == 'negative' else 0)
            sentiment_by_date[date_str].append(score)

        # 计算每日平均情感
        timeline = []
        for date_str, scores in sorted(sentiment_by_date.items()):
            timeline.append({
                'date': date_str,
                'avg_sentiment': round(sum(scores) / len(scores), 2),
                'positive_ratio': sum(1 for s in scores if s > 0) / len(scores),
                'negative_ratio': sum(1 for s in scores if s < 0) / len(scores)
            })

        # 检测情感转折点
        turning_points = []
        for i in range(1, len(timeline) - 1):
            prev_avg = timeline[i-1]['avg_sentiment']
            curr_avg = timeline[i]['avg_sentiment']
            next_avg = timeline[i+1]['avg_sentiment']

            # 检测趋势变化
            if (curr_avg > prev_avg + 0.3 and curr_avg > next_avg + 0.2) or \
               (curr_avg < prev_avg - 0.3 and curr_avg < next_avg - 0.2):
                turning_points.append({
                    'date': timeline[i]['date'],
                    'sentiment': curr_avg,
                    'direction': '上升' if curr_avg > prev_avg else '下降'
                })

        return {
            'timeline': timeline[:50],
            'turning_points': turning_points[:10],
            'overall_sentiment': round(sum(s['avg_sentiment'] for s in timeline) / len(timeline), 2)
        }