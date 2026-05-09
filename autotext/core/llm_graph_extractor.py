"""
大模型知识图谱抽取器 - 基于大模型的信息抽取、合并、对齐
"""

import json
import re
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# ============================================================
# 第一步：单文档抽取的 Function Schema
# ============================================================

STEP1_FUNCTION = {
    "type": "function",
    "function": {
        "name": "extract_local_graph",
        "description": """
从单篇文本中抽取局部知识图谱。输出节点（实体/事件）、边、属性。
每个节点有临时ID（如 e1, e2, ev1），边通过ID引用节点。
""".strip(),
        "parameters": {
            "type": "object",
            "properties": {
                "doc_info": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "doc_time": {"type": "string"},
                        "title": {"type": "string"}
                    }
                },
                "nodes": {
                    "type": "array",
                    "description": "节点列表：实体节点、事件节点",
                    "items": {
                        "type": "object",
                        "properties": {
                            "node_id": {"type": "string", "description": "临时ID，如 e1, ev1"},
                            "node_type": {"type": "string", "enum": ["entity", "event"]},
                            "name": {"type": "string", "description": "实体名或事件摘要"},
                            "entity_type": {"type": "string", "enum": ["PER", "ORG", "LOC", "PRODUCT", "TIME", "NUMBER", "OTHER"]},
                            "event_summary": {"type": "string", "description": "事件一句话描述"},
                            "trigger": {"type": "string"},
                            "time": {"type": "string"},
                            "location": {"type": "string"},
                            "source_sentence": {"type": "string"}
                        },
                        "required": ["node_id", "node_type", "name"]
                    }
                },
                "edges": {
                    "type": "array",
                    "description": "边列表：实体-实体关系、实体-事件参与",
                    "items": {
                        "type": "object",
                        "properties": {
                            "edge_id": {"type": "string"},
                            "from_node": {"type": "string"},
                            "to_node": {"type": "string"},
                            "edge_type": {"type": "string", "enum": ["relation", "participates_in"]},
                            "predicate": {"type": "string", "description": "关系谓语，如'任职于'"},
                            "role": {"type": "string", "description": "参与角色：subject/object/location/time"},
                            "confidence": {"type": "string", "enum": ["explicit", "inferred"]},
                            "source_sentence": {"type": "string"}
                        }
                    }
                },
                "attributes": {
                    "type": "array",
                    "description": "实体属性（营收、市值、职位等）",
                    "items": {
                        "type": "object",
                        "properties": {
                            "node_id": {"type": "string"},
                            "attr_key": {"type": "string"},
                            "attr_value": {"type": "string"},
                            "time": {"type": "string"},
                            "confidence": {"type": "string"},
                            "source_sentence": {"type": "string"}
                        }
                    }
                }
            },
            "required": ["doc_info", "nodes", "edges", "attributes"]
        }
    }
}


STEP2_FUNCTION = {
    "type": "function",
    "function": {
        "name": "generate_graph_operations",
        "description": """
分析多篇文档的局部子图和原文，生成图谱操作指令。
指令包括：实体对齐、事件共指、属性合并、关系推导。
""".strip(),
        "parameters": {
            "type": "object",
            "properties": {
                "entity_alignments": {
                    "type": "array",
                    "description": "实体对齐指令：哪些局部实体指向同一全局实体",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target_global_entity": {"type": "string", "description": "全局实体名"},
                            "entity_type": {"type": "string", "enum": ["PER", "ORG", "LOC", "PRODUCT", "TIME", "NUMBER", "OTHER"]},
                            "source_nodes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "doc_id": {"type": "string"},
                                        "node_id": {"type": "string"},
                                        "name": {"type": "string"}
                                    }
                                }
                            },
                            "confidence": {"type": "string", "enum": ["explicit", "inferred"]},
                            "reason": {"type": "string"}
                        }
                    }
                },
                "event_coreferences": {
                    "type": "array",
                    "description": "事件共指/关系指令：事件之间的关系",
                    "items": {
                        "type": "object",
                        "properties": {
                            "relation_type": {"type": "string", "enum": ["same", "causal", "temporal"]},
                            "direction": {"type": "string", "enum": ["forward", "bidirectional"]},
                            "source_events": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "doc_id": {"type": "string"},
                                        "node_id": {"type": "string"},
                                        "summary": {"type": "string"},
                                        "time": {"type": "string"}
                                    }
                                }
                            },
                            "confidence": {"type": "string"},
                            "description": {"type": "string"}
                        }
                    }
                },
                "attribute_operations": {
                    "type": "array",
                    "description": "属性操作指令：追加、更新或标记冲突",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target_entity": {"type": "string"},
                            "attr_key": {"type": "string"},
                            "attr_value": {"type": "string"},
                            "time": {"type": "string"},
                            "operation": {"type": "string", "enum": ["add", "conflict"]},
                            "confidence": {"type": "string"},
                            "source_doc_id": {"type": "string"},
                            "source_sentence": {"type": "string"}
                        }
                    }
                },
                "relation_inferences": {
                    "type": "array",
                    "description": "关系推导指令：从已有图谱推导新关系",
                    "items": {
                        "type": "object",
                        "properties": {
                            "from_entity": {"type": "string"},
                            "to_entity": {"type": "string"},
                            "predicate": {"type": "string"},
                            "reason": {"type": "string"},
                            "confidence": {"type": "string"}
                        }
                    }
                }
            }
        }
    }
}


# ============================================================
# 第一步：单文档抽取的 Prompt（完整版）
# ============================================================

STEP1_SYSTEM_PROMPT = """你是信息抽取专家。从文本中抽取局部知识图谱，严格按照JSON格式输出。

【重要】输出JSON时，字符串内部的双引号必须转义为 \\"，例如："title": "他说\\"你好\\""

【节点类型】
- entity：实体（PER人物/ORG组织/LOC地点/PRODUCT产品/TIME时间/NUMBER数值/OTHER其他）
- event：事件（动态发生的事）

【边类型】
- relation：实体间静态关系（任职于、位于、属于、收购等）
- participates_in：实体参与事件（角色：subject/object/location/time）

【属性】
实体自身的特征（营收、市值、职位、年龄等），不单独作为节点

【输出格式要求】
必须输出一个完整的JSON对象，格式如下：
{
  "doc_info": {
    "doc_id": "文档ID",
    "doc_time": "文档时间",
    "title": "文档标题"
  },
  "nodes": [
    {
      "node_id": "e1",
      "node_type": "entity",
      "name": "实体名称",
      "entity_type": "PER",
      "source_sentence": "原文句子"
    },
    {
      "node_id": "ev1",
      "node_type": "event",
      "event_summary": "事件描述",
      "trigger": "触发词",
      "time": "2024-01-01",
      "location": "地点",
      "source_sentence": "原文句子"
    }
  ],
  "edges": [
    {
      "edge_id": "edge1",
      "from_node": "e1",
      "to_node": "e2",
      "edge_type": "relation",
      "predicate": "任职于",
      "source_sentence": "原文句子"
    },
    {
      "edge_id": "edge2",
      "from_node": "e1",
      "to_node": "ev1",
      "edge_type": "participates_in",
      "role": "subject",
      "source_sentence": "原文句子"
    }
  ],
  "attributes": [
    {
      "node_id": "e1",
      "attr_key": "营收",
      "attr_value": "847亿元",
      "time": "2024年",
      "source_sentence": "原文句子"
    }
  ]
}

【重要规则】
- 实体节点：name字段是实体名称，entity_type是类型（PER/ORG/LOC/PRODUCT/TIME/NUMBER/OTHER）
- 事件节点：event_summary是一句话描述，trigger是触发动词
- 边：relation类型的边需要predicate谓语，participates_in类型的边需要role角色
- 如果没有相关内容，对应字段使用空数组 []
- 只输出JSON，不要其他文字说明
"""

STEP1_USER_PROMPT_TEMPLATE = """文档信息：
- doc_id: {doc_id}
- doc_time: {doc_time}
- title: {title}

文本内容：
{text}

请输出局部知识图谱JSON："""


# ============================================================
# 第二步：多文档对齐指令的 Prompt（完整版）
# ============================================================

STEP2_SYSTEM_PROMPT = """你是知识图谱分析专家。分析多篇文档的局部子图，生成图谱操作指令。

【操作指令类型】

1. entity_alignments：实体对齐
   判断不同文档的实体是否指向同一真实实体
   格式：[{"target_global_entity": "规范实体名", "entity_type": "PER", "source_nodes": [{"doc_id": "doc1", "node_id": "e1", "name": "原始名"}], "confidence": "explicit", "reason": "判断理由"}]

2. event_coreferences：事件共指/关系
   判断不同事件是否同一事件，或存在因果/时序关系
   格式：[{"relation_type": "same/causal/temporal", "direction": "forward/bidirectional", "source_events": [{"doc_id": "doc1", "node_id": "ev1", "summary": "事件描述", "time": "时间"}], "confidence": "explicit", "description": "关系描述"}]

3. attribute_operations：属性操作
   实体属性如何合并
   格式：[{"target_entity": "实体名", "attr_key": "属性名", "attr_value": "属性值", "time": "时间", "operation": "add/conflict", "confidence": "explicit", "source_doc_id": "doc1", "source_sentence": "原文"}]

4. relation_inferences：关系推导
   从已有关系推导新关系
   格式：[{"from_entity": "实体A", "to_entity": "实体B", "predicate": "关系谓语", "reason": "推导理由", "confidence": "inferred"}]

【输出要求】
- 只输出操作指令JSON，不要其他文字
- 如果没有对应操作，使用空数组 []
- 严格按格式输出
"""


# ============================================================
# 全局知识图谱存储（完整版）
# ============================================================

class GlobalKnowledgeGraph:
    """全局知识图谱 - 确定性操作执行器"""

    def __init__(self):
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.events: Dict[str, Dict[str, Any]] = {}
        self.relations: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.participations: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.event_relations: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.topics: List[Dict[str, Any]] = []
        self.next_event_id = 1
        self.next_topic_id = 1
        self._lock = threading.Lock()  # 添加线程锁

        self.type_map = {
            "人物": "PER",
            "组织": "ORG",
            "地点": "LOC",
            "产品": "PRODUCT",
            "时间": "TIME",
            "数值": "NUMBER",
            "其他": "OTHER",
            "PER": "PER",
            "ORG": "ORG",
            "LOC": "LOC",
            "PRODUCT": "PRODUCT",
            "TIME": "TIME",
            "NUMBER": "NUMBER",
            "OTHER": "OTHER"
        }

    def _normalize_entity_name(self, name: str) -> str:
        if not name:
            return name
        name = name.strip()
        name = re.sub(r'[（(].*?[）)]', '', name)
        return name

    def _get_or_create_entity(self, name: str, entity_type: str, source: Dict = None) -> str:
        canonical_name = self._normalize_entity_name(name)
        mapped_type = self.type_map.get(entity_type, "OTHER")

        with self._lock:
            if canonical_name not in self.entities:
                self.entities[canonical_name] = {
                    "type": mapped_type,
                    "attributes": {},
                    "sources": [],
                    "mention_count": 0,
                    "aliases": []
                }
            else:
                if self.entities[canonical_name]["type"] == "OTHER" and mapped_type != "OTHER":
                    self.entities[canonical_name]["type"] = mapped_type

            if source:
                self.entities[canonical_name]["sources"].append(source)
                self.entities[canonical_name]["mention_count"] += 1

            if canonical_name != name and name not in self.entities[canonical_name]["aliases"]:
                self.entities[canonical_name]["aliases"].append(name)

        return canonical_name

    def apply_entity_alignment(self, instruction: Dict[str, Any]):
        target = instruction["target_global_entity"]
        target_type = instruction.get("entity_type", "OTHER")
        self._get_or_create_entity(target, target_type, None)

        for source in instruction.get("source_nodes", []):
            source_name = source["name"]
            if source_name != target and source_name in self.entities:
                with self._lock:
                    for key, values in self.entities[source_name].get("attributes", {}).items():
                        if key not in self.entities[target]["attributes"]:
                            self.entities[target]["attributes"][key] = []
                        self.entities[target]["attributes"][key].extend(values)

                    self.entities[target]["sources"].extend(self.entities[source_name].get("sources", []))
                    self.entities[target]["mention_count"] += self.entities[source_name].get("mention_count", 0)

    def apply_attribute_operation(self, instruction: Dict[str, Any]):
        entity_name = instruction["target_entity"]
        attr_key = instruction["attr_key"]
        attr_value = instruction["attr_value"]
        time_value = instruction.get("time")
        confidence = instruction.get("confidence", "explicit")
        source = {
            "doc_id": instruction.get("source_doc_id"),
            "sentence": instruction.get("source_sentence"),
            "confidence": confidence
        }

        canonical_name = self._normalize_entity_name(entity_name)
        if canonical_name not in self.entities:
            self._get_or_create_entity(entity_name, "OTHER", source)

        with self._lock:
            if attr_key not in self.entities[canonical_name]["attributes"]:
                self.entities[canonical_name]["attributes"][attr_key] = []

            exists = False
            for existing in self.entities[canonical_name]["attributes"][attr_key]:
                if existing.get("value") == attr_value and existing.get("time") == time_value:
                    exists = True
                    break
            if not exists:
                self.entities[canonical_name]["attributes"][attr_key].append({
                    "value": attr_value,
                    "time": time_value,
                    "confidence": confidence,
                    "source": source
                })

    def apply_event_coreference(self, instruction: Dict[str, Any]):
        rel_type = instruction["relation_type"]
        source_events = instruction.get("source_events", [])
        confidence = instruction.get("confidence", "inferred")
        description = instruction.get("description", "")

        if rel_type == "same" and source_events:
            canonical_summary = source_events[0].get("summary", "") if source_events else ""
            canonical_time = source_events[0].get("time") if source_events else None

            if source_events:
                for se in source_events:
                    if len(se.get("summary", "")) > len(canonical_summary):
                        canonical_summary = se.get("summary", "")
                    if se.get("time") and not canonical_time:
                        canonical_time = se.get("time")

            with self._lock:
                event_id = f"EV_{self.next_event_id}"
                self.next_event_id += 1
                self.events[event_id] = {
                    "summary": canonical_summary,
                    "time": canonical_time,
                    "mentions": source_events,
                    "confidence": confidence,
                    "description": description
                }

    def apply_relation_inference(self, instruction: Dict[str, Any]):
        from_entity = instruction["from_entity"]
        to_entity = instruction["to_entity"]
        predicate = instruction["predicate"]
        confidence = instruction.get("confidence", "inferred")
        reason = instruction.get("reason", "")

        canonical_from = self._normalize_entity_name(from_entity)
        canonical_to = self._normalize_entity_name(to_entity)

        if canonical_from not in self.entities:
            self._get_or_create_entity(from_entity, "OTHER", None)
        if canonical_to not in self.entities:
            self._get_or_create_entity(to_entity, "OTHER", None)

        with self._lock:
            key = (canonical_from, canonical_to, predicate)
            if key not in self.relations:
                self.relations[key] = {
                    "confidence": confidence,
                    "reason": reason,
                    "sources": []
                }
            else:
                if confidence == "explicit" and self.relations[key]["confidence"] == "inferred":
                    self.relations[key]["confidence"] = "explicit"

    def apply_local_graph(self, local_graph: Dict[str, Any]):
        doc_info = local_graph.get("doc_info", {})
        doc_id = doc_info.get("doc_id", "unknown")
        doc_time = doc_info.get("doc_time")

        # 先收集所有要添加的数据
        entities_to_add = []
        events_to_add = []
        attributes_to_add = []
        relations_to_add = []
        participations_to_add = []

        for node in local_graph.get("nodes", []):
            node_id = node["node_id"]
            node_type = node["node_type"]
            name = node.get("name", "")
            source = {
                "doc_id": doc_id,
                "node_id": node_id,
                "sentence": node.get("source_sentence", "")
            }

            if node_type == "entity" and name:
                entity_type = node.get("entity_type", "OTHER")
                entities_to_add.append((name, entity_type, source))

            elif node_type == "event":
                event_summary = node.get("event_summary", name)
                events_to_add.append({
                    "summary": event_summary,
                    "trigger": node.get("trigger"),
                    "time": node.get("time", doc_time),
                    "location": node.get("location"),
                    "source_doc_id": doc_id,
                    "source_node_id": node_id,
                    "source_sentence": node.get("source_sentence", "")
                })

        for attr in local_graph.get("attributes", []):
            node_id = attr["node_id"]
            entity_name = None
            for node in local_graph.get("nodes", []):
                if node["node_id"] == node_id and node["node_type"] == "entity":
                    entity_name = node.get("name")
                    break
            if entity_name:
                attributes_to_add.append({
                    "target_entity": entity_name,
                    "attr_key": attr["attr_key"],
                    "attr_value": attr["attr_value"],
                    "time": attr.get("time", doc_time),
                    "confidence": attr.get("confidence", "explicit"),
                    "source_doc_id": doc_id,
                    "source_sentence": attr.get("source_sentence", "")
                })

        for edge in local_graph.get("edges", []):
            from_id = edge["from_node"]
            to_id = edge["to_node"]
            edge_type = edge["edge_type"]

            from_name = None
            to_name = None
            for node in local_graph.get("nodes", []):
                if node["node_id"] == from_id and node["node_type"] == "entity":
                    from_name = node.get("name")
                if node["node_id"] == to_id and node["node_type"] == "entity":
                    to_name = node.get("name")

            if from_name and to_name and edge_type == "relation":
                relations_to_add.append({
                    "from_entity": from_name,
                    "to_entity": to_name,
                    "predicate": edge.get("predicate", ""),
                    "confidence": edge.get("confidence", "explicit"),
                    "source": {"doc_id": doc_id, "sentence": edge.get("source_sentence", "")}
                })

        # 批量添加（持锁）
        with self._lock:
            for name, entity_type, source in entities_to_add:
                self._get_or_create_entity(name, entity_type, source)

            for event_data in events_to_add:
                event_id = f"EV_{self.next_event_id}"
                self.next_event_id += 1
                self.events[event_id] = event_data

            for attr_data in attributes_to_add:
                self.apply_attribute_operation(attr_data)

            for rel_data in relations_to_add:
                key = (rel_data["from_entity"], rel_data["to_entity"], rel_data["predicate"])
                if key not in self.relations:
                    self.relations[key] = {"confidence": rel_data["confidence"], "sources": []}
                self.relations[key]["sources"].append(rel_data["source"])

    def infer_topics_from_events_and_entities(self, n_topics: int = 10) -> List[Dict[str, Any]]:
        if not self.events:
            return []

        event_entities = defaultdict(list)
        for (entity, event_id, role), _ in self.participations.items():
            event_entities[event_id].append(entity)

        cooccurrence = defaultdict(int)
        entities_set = set()
        for event_id, entities in event_entities.items():
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    e1, e2 = sorted([entities[i], entities[j]])
                    cooccurrence[(e1, e2)] += 1
                    entities_set.add(e1)
                    entities_set.add(e2)

        topics = []
        used_entities = set()

        for (e1, e2), count in sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)[:n_topics]:
            if e1 in used_entities and e2 in used_entities:
                continue

            related_events = []
            for event_id, entities in event_entities.items():
                if e1 in entities or e2 in entities:
                    event = self.events.get(event_id, {})
                    if event:
                        related_events.append(event.get("summary", ""))

            topic = {
                "topic_id": self.next_topic_id,
                "texts_count": count,
                "keywords": [e1, e2],
                "weights": [1.0, 1.0],
                "representative_texts": related_events[:3],
                "textrank_sentences": [],
                "llm_title": f"{e1} & {e2}",
                "llm_summary": f"关于 {e1} 和 {e2} 的相关事件分析"
            }
            topics.append(topic)
            self.next_topic_id += 1
            used_entities.add(e1)
            used_entities.add(e2)

            if len(topics) >= n_topics:
                break

        if len(topics) < 3 and entities_set:
            type_groups = defaultdict(list)
            for entity in entities_set:
                entity_type = self.entities.get(entity, {}).get("type", "OTHER")
                type_groups[entity_type].append(entity)

            for entity_type, entities in type_groups.items():
                if len(entities) >= 2 and len(topics) < n_topics:
                    topic = {
                        "topic_id": self.next_topic_id,
                        "texts_count": 1,
                        "keywords": entities[:5],
                        "weights": [1.0] * min(5, len(entities)),
                        "representative_texts": [],
                        "textrank_sentences": [],
                        "llm_title": f"{entity_type}相关",
                        "llm_summary": f"关于{entity_type}的实体讨论"
                    }
                    topics.append(topic)
                    self.next_topic_id += 1

        self.topics = topics
        return topics

    def get_entity_stats(self) -> Dict[str, Dict]:
        stats = {
            "per": {"total": 0, "unique": 0, "top": []},
            "org": {"total": 0, "unique": 0, "top": []},
            "loc": {"total": 0, "unique": 0, "top": []},
            "product": {"total": 0, "unique": 0, "top": []},
            "time": {"total": 0, "unique": 0, "top": []},
            "number": {"total": 0, "unique": 0, "top": []},
            "other": {"total": 0, "unique": 0, "top": []}
        }

        type_to_key = {
            "PER": "per",
            "ORG": "org",
            "LOC": "loc",
            "PRODUCT": "product",
            "TIME": "time",
            "NUMBER": "number",
            "OTHER": "other"
        }

        for entity_name, entity_info in self.entities.items():
            entity_type = entity_info.get("type", "OTHER")
            key = type_to_key.get(entity_type, "other")
            mention_count = entity_info.get("mention_count", 1)

            stats[key]["total"] += mention_count
            stats[key]["unique"] += 1
            stats[key]["top"].append((entity_name, mention_count))

        for key in stats:
            stats[key]["top"].sort(key=lambda x: x[1], reverse=True)
            stats[key]["top"] = stats[key]["top"][:20]

        return stats

    def get_events_list(self) -> List[Dict]:
        events = []
        for event_id, event_info in self.events.items():
            event = {
                "event_id": event_id,
                "event_type": event_info.get("trigger", "未知事件"),
                "trigger": event_info.get("trigger", ""),
                "description": event_info.get("summary", ""),
                "timestamp": event_info.get("time", ""),
                "location": event_info.get("location", ""),
                "args": {},
                "text_index": 0
            }
            events.append(event)
        return events

    def get_relations_list(self) -> List[Dict]:
        relations = []
        for (from_entity, to_entity, predicate), rel_info in self.relations.items():
            relations.append({
                "entity1": from_entity,
                "entity2": to_entity,
                "predicate": predicate,
                "count": len(rel_info.get("sources", [])),
                "confidence": rel_info.get("confidence", "inferred")
            })
        return relations

    def to_json(self) -> Dict[str, Any]:
        clean_events = {}
        for ev_id, ev_data in self.events.items():
            clean_events[ev_id] = {
                k: v for k, v in ev_data.items()
                if isinstance(v, (str, int, float, bool, dict, list)) or v is None
            }

        return {
            "entities": self.entities,
            "events": clean_events,
            "relations": {f"{k[0]}->{k[2]}->{k[1]}": v for k, v in self.relations.items()},
            "participations": {f"{k[0]}-{k[2]}-{k[1]}": v for k, v in self.participations.items()},
            "event_relations": {f"{k[0]}-{k[2]}-{k[1]}": v for k, v in self.event_relations.items()},
            "topics": self.topics,
            "statistics": {
                "entity_count": len(self.entities),
                "event_count": len(self.events),
                "relation_count": len(self.relations),
                "participation_count": len(self.participations),
                "topic_count": len(self.topics)
            }
        }

    def print_summary(self):
        print(f"\n{'='*60}")
        print("全局知识图谱摘要")
        print("="*60)
        print(f"实体数量: {len(self.entities)}")
        print(f"事件数量: {len(self.events)}")
        print(f"实体间关系: {len(self.relations)}")
        print(f"实体-事件参与: {len(self.participations)}")
        print(f"事件间关系: {len(self.event_relations)}")
        print(f"主题数量: {len(self.topics)}")


# ============================================================
# 大模型抽取器主类（支持多线程）
# ============================================================

class LLMGraphExtractor:
    """大模型知识图谱抽取器"""

    def __init__(self, llm_client, delay_seconds: float = 0.05, max_workers: int = 5):
        """
        初始化抽取器

        参数:
        - llm_client: 大模型客户端（需实现 chat_complete 方法）
        - delay_seconds: API调用延迟（秒），多线程时建议 0.05
        - max_workers: 最大并发线程数
        """
        self.llm_client = llm_client
        self.delay_seconds = delay_seconds
        self.max_workers = max_workers
        self._progress_lock = threading.Lock()
        self._completed_count = 0

    def is_available(self) -> bool:
        return self.llm_client is not None

    def _extract_local_graph(self, text: str, doc_id: str, doc_time: str = None,
                              title: str = None, max_retries: int = 2) -> Dict[str, Any]:
        """单文档抽取局部子图"""
        doc_info = {
            "doc_id": doc_id,
            "doc_time": doc_time or None,
            "title": title or text[:50]
        }

        safe_title = (title or text[:50]).replace('"', '\\"')
        safe_text = text[:2000].replace('"', '\\"')  # 减少到 2000 字符加快速度

        user_prompt = STEP1_USER_PROMPT_TEMPLATE.format(
            doc_id=doc_id,
            doc_time=doc_time or "未知",
            title=safe_title,
            text=safe_text
        )

        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat_complete(
                    messages=[
                        {"role": "system", "content": STEP1_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0,
                    tools=[STEP1_FUNCTION],
                    tool_choice={"type": "function", "function": {"name": "extract_local_graph"}}
                )

                if "error" in response:
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                        continue
                    return self._empty_local_graph(doc_id)

                if 'choices' in response and len(response['choices']) > 0:
                    message = response['choices'][0].get('message', {})
                    if 'tool_calls' in message and message['tool_calls']:
                        arguments = message['tool_calls'][0]['function']['arguments']
                        result = json.loads(arguments)
                        if "doc_info" not in result:
                            result["doc_info"] = doc_info
                        return result

                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                return self._empty_local_graph(doc_id)

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                return self._empty_local_graph(doc_id)

        return self._empty_local_graph(doc_id)

    def _empty_local_graph(self, doc_id: str) -> Dict[str, Any]:
        return {
            "doc_info": {"doc_id": doc_id, "doc_time": None, "title": ""},
            "nodes": [],
            "edges": [],
            "attributes": []
        }

    def _extract_one(self, doc: Dict) -> Dict[str, Any]:
        """抽取单篇文档（用于多线程）"""
        result = self._extract_local_graph(
            doc["text"],
            doc["doc_id"],
            doc["doc_time"],
            doc["title"]
        )

        with self._progress_lock:
            self._completed_count += 1
            if self._completed_count % 10 == 0:
                print(f"    进度: {self._completed_count}/{self._total_count}")

        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)

        return result

    def _generate_operations(self, local_graphs: List[Dict[str, Any]],
                              max_retries: int = 2) -> Dict[str, Any]:
        """生成操作指令"""
        docs_input = []
        for doc in local_graphs:
            docs_input.append({
                "doc_id": doc["doc_info"]["doc_id"],
                "doc_time": doc["doc_info"]["doc_time"],
                "local_graph": {
                    "nodes": doc.get("nodes", [])[:30],
                    "edges": doc.get("edges", [])[:30],
                    "attributes": doc.get("attributes", [])[:20]
                }
            })

        user_prompt = f"""
以下是多篇文档的局部知识图谱，请分析并生成图谱操作指令：

{json.dumps(docs_input, ensure_ascii=False, indent=2)[:6000]}

请输出操作指令JSON：
"""

        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat_complete(
                    messages=[
                        {"role": "system", "content": STEP2_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0,
                    tools=[STEP2_FUNCTION],
                    tool_choice={"type": "function", "function": {"name": "generate_graph_operations"}}
                )

                if "error" in response:
                    if attempt < max_retries - 1:
                        continue
                    return self._empty_operations()

                if 'choices' in response and len(response['choices']) > 0:
                    message = response['choices'][0].get('message', {})
                    if 'tool_calls' in message and message['tool_calls']:
                        return json.loads(message['tool_calls'][0]['function']['arguments'])

                if attempt < max_retries - 1:
                    continue
                return self._empty_operations()

            except Exception as e:
                print(f"生成指令失败 (尝试 {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return self._empty_operations()

        return self._empty_operations()

    def _empty_operations(self) -> Dict[str, Any]:
        return {
            "entity_alignments": [],
            "event_coreferences": [],
            "attribute_operations": [],
            "relation_inferences": []
        }

    def build_global_graph(self, texts: List[str], doc_ids: List[str] = None,
                           doc_times: List[str] = None, titles: List[str] = None,
                           show_progress: bool = True) -> GlobalKnowledgeGraph:
        """
        从文本列表构建全局知识图谱（支持多线程并发）
        """
        if not self.is_available():
            raise ValueError("大模型客户端未设置，无法进行图谱抽取")

        documents = []
        for idx, text in enumerate(texts):
            if not text or len(text) < 20:
                continue
            doc_id = doc_ids[idx] if doc_ids and idx < len(doc_ids) else f"doc_{idx}"
            doc_time = doc_times[idx] if doc_times and idx < len(doc_times) else None
            title = titles[idx] if titles and idx < len(titles) else text[:30]
            documents.append({
                "text": text,
                "doc_id": doc_id,
                "doc_time": doc_time,
                "title": title
            })

        if not documents:
            return GlobalKnowledgeGraph()

        self._total_count = len(documents)
        self._completed_count = 0

        # 第一步：多线程并发抽取局部子图
        if show_progress:
            print(f"  开始并发抽取 {len(documents)} 篇文档 (并发数: {self.max_workers})...")

        local_graphs = [None] * len(documents)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._extract_one, doc): idx
                for idx, doc in enumerate(documents)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    local_graphs[idx] = future.result()
                except Exception as e:
                    print(f"    文档 {documents[idx]['doc_id']} 抽取失败: {e}")
                    local_graphs[idx] = self._empty_local_graph(documents[idx]['doc_id'])

        if show_progress:
            print(f"  并发抽取完成")

        # 创建全局图谱
        kg = GlobalKnowledgeGraph()

        # 应用局部图
        if show_progress:
            print("  应用局部图到全局图谱...")
        for lg in local_graphs:
            if lg:
                kg.apply_local_graph(lg)

        # 第二步：生成操作指令（只用前 20 篇做对齐，减少调用）
        if show_progress and len(local_graphs) > 1:
            print("  生成图谱操作指令...")
            # 只用前 20 篇做对齐（数量少也能发现主要对齐关系）
            sample_size = min(20, len(local_graphs))
            sample_graphs = local_graphs[:sample_size]
            operations = self._generate_operations(sample_graphs)

            if show_progress:
                print("  执行操作指令...")

            for alignment in operations.get("entity_alignments", []):
                kg.apply_entity_alignment(alignment)

            for attr_op in operations.get("attribute_operations", []):
                kg.apply_attribute_operation(attr_op)

            for event_rel in operations.get("event_coreferences", []):
                kg.apply_event_coreference(event_rel)

            for inference in operations.get("relation_inferences", []):
                kg.apply_relation_inference(inference)

        # 推断主题
        if show_progress:
            print("  推断主题...")
        kg.infer_topics_from_events_and_entities()

        return kg