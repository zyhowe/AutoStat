"""
事件时间线构建模块 - 按时间维度组织事件
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta


class TimelineBuilder:
    """事件时间线构建器"""

    def __init__(self):
        self.events = []

    def add_event(self, timestamp: Any, event: Dict):
        parsed_time = self._parse_timestamp(timestamp)
        if parsed_time:
            self.events.append((parsed_time, event))

    def _parse_timestamp(self, timestamp: Any) -> Optional[datetime]:
        if timestamp is None:
            return None
        if isinstance(timestamp, datetime):
            return timestamp
        if isinstance(timestamp, str):
            formats = ["%Y-%m-%d", "%Y/%m/%d", "%Y年%m月%d日", "%Y-%m-%d %H:%M:%S"]
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp, fmt)
                except ValueError:
                    continue
        return None

    def get_global_timeline(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """获取按日期分组的全局时间线"""
        if not self.events:
            return []

        from datetime import datetime

        # 按日期分组
        events_by_date = {}
        valid_years = range(2020, 2031)  # 只接受2020-2030年的日期

        for timestamp, event in self.events:
            if not timestamp:
                continue
            date_str = timestamp.strftime("%Y-%m-%d")
            # 过滤无效年份
            try:
                year = int(date_str[:4])
                if year not in valid_years:
                    continue
            except:
                continue

            if date_str not in events_by_date:
                events_by_date[date_str] = []
            events_by_date[date_str].append(event)

        # 转换为列表并排序
        timeline = []
        for date_str in sorted(events_by_date.keys()):
            timeline.append({
                "date": date_str,
                "events": events_by_date[date_str],
                "event_count": len(events_by_date[date_str])
            })

        return timeline

    def get_entity_timeline(self, entity_name: str) -> List[Dict]:
        entity_timeline = []
        for timestamp, event in self.events:
            args = event.get("args", {})
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
        entity_timeline.sort(key=lambda x: x["date"])
        return entity_timeline

    def get_topic_timeline(self, topic_id: int, topic_events: Dict = None) -> List[Dict]:
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
        if not self.events:
            return {"dates": [], "counts": []}

        date_counts = defaultdict(int)
        for timestamp, _ in self.events:
            date_str = timestamp.strftime("%Y-%m-%d")
            date_counts[date_str] += 1

        dates = sorted(date_counts.keys())
        counts = [date_counts[d] for d in dates]
        return {"dates": dates, "counts": counts}

    def get_summary(self) -> Dict:
        if not self.events:
            return {"has_data": False, "total_events": 0, "date_range": None}

        timestamps = [t for t, _ in self.events]
        min_date = min(timestamps).strftime("%Y-%m-%d") if timestamps else None
        max_date = max(timestamps).strftime("%Y-%m-%d") if timestamps else None

        type_counts = defaultdict(int)
        for _, event in self.events:
            event_type = event.get("event_type", "未知")
            type_counts[event_type] += 1

        return {
            "has_data": True,
            "total_events": len(self.events),
            "date_range": {"start": min_date, "end": max_date},
            "event_types": dict(type_counts)
        }

    def build_from_events(self, events: List[Dict]):
        self.events = []
        for event in events:
            timestamp = event.get("timestamp")
            if timestamp:
                self.add_event(timestamp, event)


def build_timeline_from_analyzer(analyzer) -> TimelineBuilder:
    builder = TimelineBuilder()
    events = getattr(analyzer, 'events', [])
    builder.build_from_events(events)
    return builder