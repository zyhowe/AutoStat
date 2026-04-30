"""
实体档案模块 - 只保留 PER/ORG/LOC 类型
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
import math


class EntityProfile:
    """实体档案"""

    def __init__(self, entity_id: str, name: str, entity_type: str):
        self.entity_id = entity_id
        self.name = name
        self.type = entity_type
        self.mentions = []
        self.events = []
        self.related_entities = []
        self.topics = []
        self.sentiment_scores = []
        self.mention_count = 0

    def add_mention(self, text_index: int, sentence: str = ""):
        self.mentions.append({"index": text_index, "sentence": sentence[:200]})
        self.mention_count += 1

    def add_event(self, event: Dict, role: str = "参与"):
        self.events.append({
            "event_type": event.get("event_type", "未知"),
            "trigger": event.get("trigger", ""),
            "role": role,
            "timestamp": event.get("timestamp", ""),
            "args": event.get("args", {})
        })

    def add_related_entity(self, entity_name: str, weight: float, pmi: float = 0):
        for existing in self.related_entities:
            if existing["name"] == entity_name:
                return
        self.related_entities.append({
            "name": entity_name,
            "weight": round(weight, 3),
            "pmi": round(pmi, 2)
        })

    def add_topic(self, topic_id: int, topic_name: str, probability: float):
        self.topics.append({
            "topic_id": topic_id,
            "topic_name": topic_name,
            "probability": round(probability, 3)
        })

    def to_dict(self) -> Dict:
        sorted_topics = sorted(self.topics, key=lambda x: x["probability"], reverse=True)
        sorted_entities = sorted(self.related_entities, key=lambda x: x["weight"], reverse=True)
        sorted_events = sorted(self.events, key=lambda x: x.get("timestamp", ""), reverse=True)

        return {
            "name": self.name,
            "type": self.type,
            "mention_count": self.mention_count,
            "events": sorted_events[:10],
            "related_entities": sorted_entities[:10],
            "topics": sorted_topics[:5],
            "sample_mentions": [m["sentence"] for m in self.mentions[:2]]
        }


class EntityProfileBuilder:
    """实体档案构建器 - 只保留 PER/ORG/LOC"""

    def __init__(self):
        self.profiles = {}
        self.VALID_TYPES = {"PER", "ORG", "LOC"}

    def build_from_analyzer(self, analyzer) -> List[EntityProfile]:
        self.profiles = {}

        # 1. 初始化实体档案（只保留 PER/ORG/LOC）
        self._init_from_entity_results(analyzer)

        # 2. 添加事件关联
        self._add_events_from_analyzer(analyzer)

        # 3. 添加主题关联
        self._add_topics_from_analyzer(analyzer)

        # 4. 添加实体共现关系
        self._add_cooccurrence_from_relation(analyzer)

        # 过滤掉常见的无意义实体
        filtered_profiles = {}
        for entity_id, profile in self.profiles.items():
            name = profile.name
            # 过滤条件
            if len(name) < 2:
                continue
            if name in ["公司", "企业", "产品", "项目", "市场", "行业", "机构", "业务"]:
                continue
            if name.isdigit():
                continue
            filtered_profiles[entity_id] = profile

        profiles_list = list(filtered_profiles.values())
        profiles_list.sort(key=lambda x: x.mention_count, reverse=True)
        return profiles_list[:20]

    def _init_from_entity_results(self, analyzer):
        entity_results = getattr(analyzer, 'entity_results', [])
        entity_stats = getattr(analyzer, 'entity_stats', {})

        for entity_type, stats in entity_stats.items():
            # 只保留 PER/ORG/LOC
            if entity_type.upper() not in self.VALID_TYPES:
                continue

            for entity_name, count in stats.get("top", []):
                if len(entity_name) < 2:
                    continue
                entity_id = f"{entity_type}:{entity_name}"
                profile = EntityProfile(entity_id, entity_name, entity_type.upper())
                profile.mention_count = count
                self.profiles[entity_id] = profile

        for idx, result in enumerate(entity_results):
            for entity_type, entities in result.items():
                if entity_type.upper() not in self.VALID_TYPES:
                    continue
                for entity_text, start, end in entities:
                    if len(entity_text) < 2:
                        continue
                    entity_id = f"{entity_type}:{entity_text}"
                    if entity_id in self.profiles:
                        self.profiles[entity_id].add_mention(idx)

    def _add_events_from_analyzer(self, analyzer):
        events = getattr(analyzer, 'events', [])
        for event in events:
            args = event.get("args", {})
            for arg_name, arg_value in args.items():
                if arg_value and isinstance(arg_value, str) and len(arg_value) >= 2:
                    for entity_id, profile in self.profiles.items():
                        if profile.name == arg_value or arg_value in profile.name:
                            profile.add_event(event, role=arg_name)
                            break

    def _add_topics_from_analyzer(self, analyzer):
        topics = getattr(analyzer, 'topics', [])
        topic_labels = getattr(analyzer.topic_modeler, 'topic_labels', []) if hasattr(analyzer, 'topic_modeler') else []

        if not topics or not topic_labels:
            return

        topic_names = {}
        for topic in topics:
            topic_id = topic.get("topic_id", 0)
            keywords = topic.get("keywords", [])[:3]
            topic_names[topic_id] = "、".join(keywords) if keywords else f"主题{topic_id}"

        entity_topic_counts = defaultdict(lambda: defaultdict(int))
        entity_results = getattr(analyzer, 'entity_results', [])

        for idx, result in enumerate(entity_results):
            if idx < len(topic_labels):
                topic_id = topic_labels[idx]
                for entity_type, entities in result.items():
                    if entity_type.upper() not in self.VALID_TYPES:
                        continue
                    for entity_text, _, _ in entities:
                        if len(entity_text) < 2:
                            continue
                        entity_id = f"{entity_type}:{entity_text}"
                        if entity_id in self.profiles:
                            entity_topic_counts[entity_id][topic_id] += 1

        for entity_id, topic_counts in entity_topic_counts.items():
            total = sum(topic_counts.values())
            for topic_id, count in topic_counts.items():
                probability = count / total if total > 0 else 0
                topic_name = topic_names.get(topic_id, f"主题{topic_id}")
                self.profiles[entity_id].add_topic(topic_id, topic_name, probability)

    def _add_cooccurrence_from_relation(self, analyzer):
        relation_result = getattr(analyzer, 'relation_result', {})
        pairs = relation_result.get('cooccurrence_pairs', [])

        for pair in pairs:
            e1 = pair.get("entity1", "")
            e2 = pair.get("entity2", "")
            pmi = pair.get("pmi", 0)

            e1_name = e1.split(":", 1)[-1] if ":" in e1 else e1
            e2_name = e2.split(":", 1)[-1] if ":" in e2 else e2

            for entity_id, profile in self.profiles.items():
                if profile.name == e1_name:
                    profile.add_related_entity(e2_name, pmi, pmi)
                elif profile.name == e2_name:
                    profile.add_related_entity(e1_name, pmi, pmi)

    def export_for_report(self, top_n: int = 10) -> List[Dict]:
        profiles_list = list(self.profiles.values())
        profiles_list.sort(key=lambda x: x.mention_count, reverse=True)
        return [p.to_dict() for p in profiles_list[:top_n]]