"""
事件时间线构建模块 - 按时间维度组织事件
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta


class TimelineBuilder:
    """事件时间线构建器"""

    def __init__(self):
        self.events = []  # [(timestamp, event_dict), ...]

    def add_event(self, timestamp: Any, event: Dict):
        """添加事件到时间线"""
        parsed_time = self._parse_timestamp(timestamp)
        if parsed_time:
            self.events.append((parsed_time, event))

    def _parse_timestamp(self, timestamp: Any) -> Optional[datetime]:
        """解析时间戳"""
        if timestamp is None:
            return None

        if isinstance(timestamp, datetime):
            return timestamp

        if isinstance(timestamp, str):
            # 尝试多种格式
            formats = [
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%Y年%m月%d日",
                "%Y-%m-%d %H:%M:%S",
                "%Y/%m/%d %H:%M:%S"
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp, fmt)
                except ValueError:
                    continue

        return None

    def get_global_timeline(self, start_date: str = None,
                            end_date: str = None) -> List[Dict]:
        """
        获取全局时间线

        返回: [
            {"date": "2024-01-15", "events": [{"event_type": "...", ...}]},
            ...
        ]
        """
        if not self.events:
            return []

        # 按日期分组
        events_by_date = defaultdict(list)
        for timestamp, event in self.events:
            date_str = timestamp.strftime("%Y-%m-%d")
            events_by_date[date_str].append(event)

        # 转换为列表并排序
        timeline = []
        for date_str in sorted(events_by_date.keys()):
            timeline.append({
                "date": date_str,
                "events": events_by_date[date_str]
            })

        # 应用日期过滤
        if start_date:
            timeline = [t for t in timeline if t["date"] >= start_date]
        if end_date:
            timeline = [t for t in timeline if t["date"] <= end_date]

        return timeline

    def get_entity_timeline(self, entity_name: str) -> List[Dict]:
        """
        获取实体时间线（该实体参与的事件）

        返回: [
            {"date": "2024-01-15", "events": [...], "role": "主语"},
            ...
        ]
        """
        entity_timeline = []

        for timestamp, event in self.events:
            args = event.get("args", {})
            # 检查事件中是否包含该实体
            for role, value in args.items():
                if isinstance(value, str) and entity_name in value:
                    entity_timeline.append({
                        "date": timestamp.strftime("%Y-%m-%d"),
                        "event_type": event.get("event_type", ""),
                        "trigger": event.get("trigger", ""),
                        "role": role,
                        "details": value
                    })
                    break

        # 按时间排序
        entity_timeline.sort(key=lambda x: x["date"])
        return entity_timeline

    def get_topic_timeline(self, topic_id: int, topic_events: Dict = None) -> List[Dict]:
        """
        获取主题时间线（属于该主题的事件）

        参数:
        - topic_id: 主题ID
        - topic_events: 事件->主题的映射 {event_id: topic_id}

        返回: 事件列表
        """
        if not topic_events:
            return []

        topic_timeline = []
        for timestamp, event in self.events:
            event_id = event.get("event_id")
            if event_id and topic_events.get(event_id) == topic_id:
                topic_timeline.append({
                    "date": timestamp.strftime("%Y-%m-%d"),
                    "event_type": event.get("event_type", ""),
                    "trigger": event.get("trigger", ""),
                    "args": event.get("args", {})
                })

        topic_timeline.sort(key=lambda x: x["date"])
        return topic_timeline

    def get_heatmap_data(self) -> Dict:
        """
        获取事件热度数据（用于热力图）

        返回: {
            "dates": ["2024-01-01", ...],
            "counts": [5, 8, 3, ...]
        }
        """
        if not self.events:
            return {"dates": [], "counts": []}

        # 按日期统计事件数量
        date_counts = defaultdict(int)
        for timestamp, _ in self.events:
            date_str = timestamp.strftime("%Y-%m-%d")
            date_counts[date_str] += 1

        # 生成连续日期范围
        dates = sorted(date_counts.keys())
        counts = [date_counts[d] for d in dates]

        return {"dates": dates, "counts": counts}

    def detect_anomaly_dates(self, threshold: float = 2.0) -> List[Dict]:
        """
        检测异常日期（事件数量突增）

        参数:
        - threshold: 标准差倍数阈值

        返回: [{"date": "2024-01-15", "count": 15, "deviation": 2.5}]
        """
        heatmap = self.get_heatmap_data()
        if len(heatmap["counts"]) < 5:
            return []

        counts = heatmap["counts"]
        mean = sum(counts) / len(counts)
        std = (sum((c - mean) ** 2 for c in counts) / len(counts)) ** 0.5

        if std == 0:
            return []

        anomalies = []
        for date, count in zip(heatmap["dates"], counts):
            deviation = (count - mean) / std
            if deviation > threshold:
                anomalies.append({
                    "date": date,
                    "count": count,
                    "deviation": round(deviation, 2),
                    "type": "peak"
                })

        anomalies.sort(key=lambda x: x["deviation"], reverse=True)
        return anomalies[:10]

    def get_summary(self) -> Dict:
        """获取时间线摘要"""
        if not self.events:
            return {
                "has_data": False,
                "total_events": 0,
                "date_range": None
            }

        timestamps = [t for t, _ in self.events]
        min_date = min(timestamps).strftime("%Y-%m-%d")
        max_date = max(timestamps).strftime("%Y-%m-%d")

        return {
            "has_data": True,
            "total_events": len(self.events),
            "date_range": {"start": min_date, "end": max_date},
            "event_types": self._get_event_type_distribution()
        }

    def _get_event_type_distribution(self) -> Dict:
        """获取事件类型分布"""
        type_counts = defaultdict(int)
        for _, event in self.events:
            event_type = event.get("event_type", "未知")
            type_counts[event_type] += 1
        return dict(type_counts)

    def build_from_events(self, events: List[Dict]):
        """从事件列表构建时间线"""
        self.events = []
        for event in events:
            timestamp = event.get("timestamp")
            if timestamp:
                self.add_event(timestamp, event)


def build_timeline_from_analyzer(analyzer) -> TimelineBuilder:
    """从分析器构建时间线"""
    builder = TimelineBuilder()

    # 从事件抽取结果构建
    events = getattr(analyzer, 'events', [])
    builder.build_from_events(events)

    return builder