"""
实体档案模块 - 为每个实体生成完整的档案信息
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
import math


class EntityProfile:
    """实体档案 - 单个实体的完整信息"""

    def __init__(self, entity_id: str, name: str, entity_type: str):
        self.entity_id = entity_id
        self.name = name
        self.type = entity_type
        self.mentions = []  # 提及位置和次数
        self.events = []  # 相关事件 [(event, role, timestamp)]
        self.related_entities = []  # 关联实体 [(entity, weight, pmi)]
        self.topics = []  # 归属主题 [(topic_id, topic_name, probability)]
        self.sentiment_scores = []  # 情感得分 [(timestamp, score)]
        self.mention_count = 0

    def add_mention(self, text_index: int, sentence: str = ""):
        """添加提及记录"""
        self.mentions.append({"index": text_index, "sentence": sentence[:200]})
        self.mention_count += 1

    def add_event(self, event: Dict, role: str = "参与"):
        """添加相关事件"""
        self.events.append({
            "event_type": event.get("event_type", "未知"),
            "trigger": event.get("trigger", ""),
            "role": role,
            "timestamp": event.get("timestamp", ""),
            "args": event.get("args", {})
        })

    def add_related_entity(self, entity_name: str, weight: float, pmi: float = 0):
        """添加关联实体"""
        # 避免重复
        for existing in self.related_entities:
            if existing["name"] == entity_name:
                return
        self.related_entities.append({
            "name": entity_name,
            "weight": round(weight, 3),
            "pmi": round(pmi, 2)
        })

    def add_topic(self, topic_id: int, topic_name: str, probability: float):
        """添加归属主题"""
        self.topics.append({
            "topic_id": topic_id,
            "topic_name": topic_name,
            "probability": round(probability, 3)
        })

    def add_sentiment(self, timestamp: str, score: float):
        """添加情感得分"""
        self.sentiment_scores.append({
            "timestamp": timestamp,
            "score": round(score, 2)
        })

    def get_sentiment_trend(self) -> Dict:
        """获取情感趋势"""
        if not self.sentiment_scores:
            return {"trend": "neutral", "avg_score": 0, "points": []}

        scores = [s["score"] for s in self.sentiment_scores]
        avg_score = sum(scores) / len(scores)

        if avg_score > 0.2:
            trend = "positive"
        elif avg_score < -0.2:
            trend = "negative"
        else:
            trend = "neutral"

        return {
            "trend": trend,
            "avg_score": round(avg_score, 2),
            "points": self.sentiment_scores[:20]
        }

    def to_dict(self) -> Dict:
        """输出实体档案字典"""
        # 按概率排序主题
        sorted_topics = sorted(self.topics, key=lambda x: x["probability"], reverse=True)

        # 按权重排序关联实体
        sorted_entities = sorted(self.related_entities, key=lambda x: x["weight"], reverse=True)

        # 按时间排序事件
        sorted_events = sorted(self.events, key=lambda x: x.get("timestamp", ""), reverse=True)

        return {
            "name": self.name,
            "type": self.type,
            "mention_count": self.mention_count,
            "events": sorted_events[:20],
            "related_entities": sorted_entities[:15],
            "topics": sorted_topics[:5],
            "sentiment_trend": self.get_sentiment_trend(),
            "sample_mentions": [m["sentence"] for m in self.mentions[:3]]
        }


class EntityProfileBuilder:
    """实体档案构建器 - 从分析结果构建所有实体的档案"""

    def __init__(self):
        self.profiles = {}  # entity_id -> EntityProfile

    def build_from_analyzer(self, analyzer) -> List[EntityProfile]:
        """
        从分析器结果构建实体档案

        参数:
        - analyzer: TextAnalyzer 实例

        返回: EntityProfile 列表
        """
        self.profiles = {}

        # 1. 从实体识别结果初始化档案
        self._init_from_entity_results(analyzer)

        # 2. 添加事件关联
        self._add_events_from_analyzer(analyzer)

        # 3. 添加主题关联
        self._add_topics_from_analyzer(analyzer)

        # 4. 添加实体共现关系
        self._add_cooccurrence_from_relation(analyzer)

        # 返回按提及次数排序的档案列表
        profiles_list = list(self.profiles.values())
        profiles_list.sort(key=lambda x: x.mention_count, reverse=True)

        return profiles_list

    def _init_from_entity_results(self, analyzer):
        """从实体识别结果初始化档案"""
        entity_results = getattr(analyzer, 'entity_results', [])
        entity_stats = getattr(analyzer, 'entity_stats', {})

        for entity_type, stats in entity_stats.items():
            for entity_name, count in stats.get("top", []):
                if len(entity_name) < 2:
                    continue
                entity_id = f"{entity_type}:{entity_name}"
                profile = EntityProfile(entity_id, entity_name, entity_type.upper())
                profile.mention_count = count
                self.profiles[entity_id] = profile

        # 添加提及位置
        for idx, result in enumerate(entity_results):
            for entity_type, entities in result.items():
                for entity_text, start, end in entities:
                    entity_id = f"{entity_type}:{entity_text}"
                    if entity_id in self.profiles:
                        self.profiles[entity_id].add_mention(idx)

    def _add_events_from_analyzer(self, analyzer):
        """从事件抽取结果添加事件关联"""
        events = getattr(analyzer, 'events', [])

        for event in events:
            event_type = event.get("event_type", "")
            args = event.get("args", {})
            timestamp = event.get("timestamp", "")

            # 提取事件中的实体
            for arg_name, arg_value in args.items():
                if arg_value and isinstance(arg_value, str):
                    # 尝试匹配已知实体
                    for entity_id, profile in self.profiles.items():
                        if arg_value in profile.name or profile.name in arg_value:
                            profile.add_event(event, role=arg_name)
                            break

    def _add_topics_from_analyzer(self, analyzer):
        """从主题建模结果添加主题关联"""
        topics = getattr(analyzer, 'topics', [])
        topic_labels = getattr(analyzer.topic_modeler, 'topic_labels', []) if hasattr(analyzer, 'topic_modeler') else []

        if not topics or not topic_labels:
            return

        # 构建主题名称映射
        topic_names = {}
        for topic in topics:
            topic_id = topic.get("topic_id", 0)
            keywords = topic.get("keywords", [])[:3]
            topic_names[topic_id] = "、".join(keywords) if keywords else f"主题{topic_id}"

        # 为每个实体分配主题
        entity_topic_counts = defaultdict(lambda: defaultdict(int))
        entity_results = getattr(analyzer, 'entity_results', [])

        for idx, result in enumerate(entity_results):
            if idx < len(topic_labels):
                topic_id = topic_labels[idx]
                for entity_type, entities in result.items():
                    for entity_text, _, _ in entities:
                        entity_id = f"{entity_type}:{entity_text}"
                        if entity_id in self.profiles:
                            entity_topic_counts[entity_id][topic_id] += 1

        # 计算概率并添加到档案
        for entity_id, topic_counts in entity_topic_counts.items():
            total = sum(topic_counts.values())
            for topic_id, count in topic_counts.items():
                probability = count / total if total > 0 else 0
                topic_name = topic_names.get(topic_id, f"主题{topic_id}")
                self.profiles[entity_id].add_topic(topic_id, topic_name, probability)

    def _add_cooccurrence_from_relation(self, analyzer):
        """从关系发现结果添加实体共现关系"""
        relation_result = getattr(analyzer, 'relation_result', {})
        pairs = relation_result.get('cooccurrence_pairs', [])

        for pair in pairs:
            e1 = pair.get("entity1", "")
            e2 = pair.get("entity2", "")
            pmi = pair.get("pmi", 0)

            # 提取实体名称（去掉类型前缀）
            e1_name = e1.split(":", 1)[-1] if ":" in e1 else e1
            e2_name = e2.split(":", 1)[-1] if ":" in e2 else e2

            # 查找对应的档案
            for entity_id, profile in self.profiles.items():
                if profile.name == e1_name:
                    profile.add_related_entity(e2_name, pmi, pmi)
                elif profile.name == e2_name:
                    profile.add_related_entity(e1_name, pmi, pmi)

    def get_profile_by_name(self, name: str) -> Optional[EntityProfile]:
        """根据名称获取实体档案"""
        for profile in self.profiles.values():
            if profile.name == name:
                return profile
        return None

    def get_profiles_by_type(self, entity_type: str) -> List[EntityProfile]:
        """根据类型获取实体档案"""
        return [p for p in self.profiles.values() if p.type == entity_type.upper()]

    def export_for_report(self, top_n: int = 10) -> List[Dict]:
        """导出用于报告的实体档案列表"""
        profiles_list = list(self.profiles.values())
        profiles_list.sort(key=lambda x: x.mention_count, reverse=True)

        return [p.to_dict() for p in profiles_list[:top_n]]