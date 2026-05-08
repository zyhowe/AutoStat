"""
DeepSeek 信息抽取 - 最终完善版
架构：大模型输出操作指令，确定性代码执行图谱操作

核心流程：
1. 第一步：单文档抽取，输出局部子图（节点+边+属性）
2. 第二步：多文档输入+局部子图，大模型输出对齐/合并/推导指令
3. 后处理：确定性代码执行指令，构建全局知识图谱
"""
import os
import json
import re
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key="sk-c0e1f1ad1a3b41429a92f29251775ecf",
    base_url="https://api.deepseek.com"
)


# ============================================================
# 第一步：单文档抽取的 Function Schema
# 输出：局部子图（节点+边+属性，带临时ID和原文引用）
# ============================================================

STEP1_FUNCTION = {
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
                        "entity_type": {"type": "string", "enum": ["人物", "组织", "地点", "产品", "时间", "其他"]},
                        "event_summary": {"type": "string", "description": "事件一句话描述"},
                        "trigger": {"type": "string"},
                        "time": {"type": "object"},
                        "location": {"type": "string"}
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


STEP1_SYSTEM_PROMPT = """
你是信息抽取专家。从文本中抽取局部知识图谱。

【节点类型】
- entity：实体（人物/组织/地点/产品/时间）
- event：事件（动态发生的事）

【边类型】
- relation：实体间静态关系（任职于、位于、属于）
- participates_in：实体参与事件（角色：subject/object/location/time）

【属性】
实体自身的特征（营收、市值、职位、年龄等），不单独作为节点

【输出要求】
- 每个节点有唯一临时ID：实体用 e1,e2...，事件用 ev1,ev2...
- 边通过ID引用节点
- 保留原文来源句子
- temperature=0，严格按JSON输出
""".strip()


# ============================================================
# 第二步：多文档对齐指令的 Function Schema
# 输出：操作指令（实体对齐、事件共指、属性合并、关系推导）
# ============================================================

STEP2_FUNCTION = {
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
                                    "summary": {"type": "string"}
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


STEP2_SYSTEM_PROMPT = """
你是知识图谱分析专家。分析多篇文档，生成图谱操作指令。

【任务说明】
1. 实体对齐：判断不同文档的实体是否指向同一真实实体
2. 事件共指：判断不同事件是否同一事件，或存在因果/时序关系
3. 属性合并：实体属性如何合并（追加或冲突标记）
4. 关系推导：从已有关系推导新关系（如传递性）

【输入格式】
你会收到多篇文档，每篇包含：原文 + 局部子图（节点、边、属性）

【输出要求】
- 只输出操作指令，不输出最终图谱
- 操作指令要具体、明确，可被代码直接执行
- 每个指令标注置信度
- temperature=0，严格按JSON输出
""".strip()


# ============================================================
# 全局图谱存储器（确定性操作执行器）
# ============================================================

class GlobalKnowledgeGraph:
    """全局知识图谱 - 确定性操作执行器"""

    def __init__(self):
        # 实体存储 {实体名: {type, attributes, source_details}}
        self.entities: Dict[str, Dict[str, Any]] = {}
        # 事件存储 {事件ID: {summary, time, location, source_details}}
        self.events: Dict[str, Dict[str, Any]] = {}
        # 关系存储 {(from, to, predicate): {confidence, sources}}
        self.relations: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        # 实体-事件参与 {(entity, event, role): {confidence, sources}}
        self.participations: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        # 事件间关系 {(event1, event2, rel_type): {confidence, sources}}
        self.event_relations: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        self.next_event_id = 1

    def _get_or_create_entity(self, name: str, entity_type: str) -> str:
        """获取或创建实体节点"""
        if name not in self.entities:
            self.entities[name] = {
                "type": entity_type,
                "attributes": {},  # {key: [(value, time, confidence, source)]}
                "sources": []
            }
        else:
            # 如果类型冲突，保留更具体的
            if self.entities[name]["type"] == "其他" and entity_type != "其他":
                self.entities[name]["type"] = entity_type
        return name

    def apply_entity_alignment(self, instruction: Dict[str, Any]):
        """执行实体对齐指令"""
        target = instruction["target_global_entity"]
        target_type = instruction.get("entity_type", "其他")

        # 创建或获取目标实体
        self._get_or_create_entity(target, target_type)

        for source in instruction["source_nodes"]:
            source_name = source["name"]
            if source_name != target and source_name in self.entities:
                # 合并属性
                source_attrs = self.entities[source_name].get("attributes", {})
                for key, values in source_attrs.items():
                    if key not in self.entities[target]["attributes"]:
                        self.entities[target]["attributes"][key] = []
                    self.entities[target]["attributes"][key].extend(values)

                # 合并来源
                self.entities[target]["sources"].extend(
                    self.entities[source_name].get("sources", [])
                )

                # 删除源实体（可选，保留作为别名）
                # 这里选择保留但标记为别名
                self.entities[source_name]["alias_of"] = target

    def apply_attribute_operation(self, instruction: Dict[str, Any]):
        """执行属性操作指令"""
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

        if entity_name not in self.entities:
            self._get_or_create_entity(entity_name, "其他")

        if attr_key not in self.entities[entity_name]["attributes"]:
            self.entities[entity_name]["attributes"][attr_key] = []

        operation = instruction.get("operation", "add")

        if operation == "add":
            self.entities[entity_name]["attributes"][attr_key].append({
                "value": attr_value,
                "time": time_value,
                "confidence": confidence,
                "source": source
            })
        elif operation == "conflict":
            # 标记冲突
            self.entities[entity_name]["attributes"][attr_key].append({
                "value": attr_value,
                "time": time_value,
                "confidence": confidence,
                "source": source,
                "conflict": True
            })

    def apply_event_coreference(self, instruction: Dict[str, Any]):
        """执行事件共指/关系指令"""
        rel_type = instruction["relation_type"]
        source_events = instruction["source_events"]
        confidence = instruction.get("confidence", "inferred")
        description = instruction.get("description", "")

        if rel_type == "same":
            # 合并为同一事件
            canonical = source_events[0]
            canonical_summary = canonical.get("summary", "")
            canonical_time = canonical.get("time", {})

            event_id = f"ev_{self.next_event_id}"
            self.next_event_id += 1

            self.events[event_id] = {
                "summary": canonical_summary,
                "time": canonical_time,
                "mentions": source_events,
                "confidence": confidence
            }

            # 记录每个源事件到规范事件的映射
            for se in source_events:
                key = (se["doc_id"], se["node_id"])
                if "_event_mapping" not in self.__dict__:
                    self._event_mapping = {}
                self._event_mapping[key] = event_id

        elif rel_type == "causal":
            # 建立因果关系
            if len(source_events) >= 2:
                cause = source_events[0]
                effect = source_events[1]
                direction = instruction.get("direction", "forward")

                if direction == "forward":
                    key = (cause["summary"], effect["summary"], "causes")
                else:
                    key = (effect["summary"], cause["summary"], "caused_by")

                self.event_relations[key] = {
                    "type": "causal",
                    "description": description,
                    "confidence": confidence,
                    "sources": source_events
                }

        elif rel_type == "temporal":
            # 建立时序关系
            if len(source_events) >= 2:
                earlier = source_events[0]
                later = source_events[1]
                key = (earlier["summary"], later["summary"], "before")
                self.event_relations[key] = {
                    "type": "temporal",
                    "description": description,
                    "confidence": confidence,
                    "sources": source_events
                }

    def apply_relation_inference(self, instruction: Dict[str, Any]):
        """执行关系推导指令"""
        from_entity = instruction["from_entity"]
        to_entity = instruction["to_entity"]
        predicate = instruction["predicate"]
        confidence = instruction.get("confidence", "inferred")
        reason = instruction.get("reason", "")

        # 检查实体是否存在
        if from_entity not in self.entities:
            self._get_or_create_entity(from_entity, "其他")
        if to_entity not in self.entities:
            self._get_or_create_entity(to_entity, "其他")

        key = (from_entity, to_entity, predicate)
        if key not in self.relations:
            self.relations[key] = {
                "confidence": confidence,
                "reason": reason,
                "sources": []
            }
        else:
            # 提升置信度
            if confidence == "explicit" and self.relations[key]["confidence"] == "inferred":
                self.relations[key]["confidence"] = "explicit"
            self.relations[key]["reason"] = reason

    def apply_local_graph(self, local_graph: Dict[str, Any]):
        """直接应用局部图到全局"""
        doc_id = local_graph.get("doc_info", {}).get("doc_id", "unknown")

        # 添加节点
        for node in local_graph.get("nodes", []):
            node_id = node["node_id"]
            node_type = node["node_type"]
            name = node.get("name", "")

            if node_type == "entity" and name:
                entity_type = node.get("entity_type", "其他")
                self._get_or_create_entity(name, entity_type)

                # 记录来源
                source = {
                    "doc_id": doc_id,
                    "node_id": node_id,
                    "sentence": node.get("source_sentence", "")
                }
                if "sources" not in self.entities[name]:
                    self.entities[name]["sources"] = []
                self.entities[name]["sources"].append(source)

            elif node_type == "event":
                event_id = f"ev_{self.next_event_id}"
                self.next_event_id += 1

                self.events[event_id] = {
                    "summary": node.get("event_summary", name),
                    "trigger": node.get("trigger"),
                    "time": node.get("time"),
                    "location": node.get("location"),
                    "source_doc_id": doc_id,
                    "source_node_id": node_id,
                    "source_sentence": node.get("source_sentence", "")
                }

                # 记录映射
                if not hasattr(self, "_event_id_mapping"):
                    self._event_id_mapping = {}
                self._event_id_mapping[(doc_id, node_id)] = event_id

        # 添加属性
        for attr in local_graph.get("attributes", []):
            node_id = attr["node_id"]
            # 查找对应的实体名
            entity_name = None
            for node in local_graph.get("nodes", []):
                if node["node_id"] == node_id and node["node_type"] == "entity":
                    entity_name = node.get("name")
                    break

            if entity_name:
                self.apply_attribute_operation({
                    "target_entity": entity_name,
                    "attr_key": attr["attr_key"],
                    "attr_value": attr["attr_value"],
                    "time": attr.get("time"),
                    "operation": "add",
                    "confidence": attr.get("confidence", "explicit"),
                    "source_doc_id": doc_id,
                    "source_sentence": attr.get("source_sentence", "")
                })

        # 添加边
        for edge in local_graph.get("edges", []):
            from_id = edge["from_node"]
            to_id = edge["to_node"]
            edge_type = edge["edge_type"]

            # 查找实体名
            from_name = None
            to_name = None
            for node in local_graph.get("nodes", []):
                if node["node_id"] == from_id:
                    from_name = node.get("name")
                if node["node_id"] == to_id:
                    to_name = node.get("name")

            if from_name and to_name:
                if edge_type == "relation":
                    key = (from_name, to_name, edge.get("predicate", ""))
                    if key not in self.relations:
                        self.relations[key] = {
                            "confidence": edge.get("confidence", "explicit"),
                            "sources": []
                        }
                    self.relations[key]["sources"].append({
                        "doc_id": doc_id,
                        "sentence": edge.get("source_sentence", "")
                    })

                elif edge_type == "participates_in":
                    # 查找事件ID
                    ev_id = None
                    if hasattr(self, "_event_id_mapping"):
                        ev_id = self._event_id_mapping.get((doc_id, to_id))

                    if ev_id:
                        key = (from_name, ev_id, edge.get("role", "participant"))
                        if key not in self.participations:
                            self.participations[key] = {
                                "confidence": edge.get("confidence", "explicit"),
                                "sources": []
                            }
                        self.participations[key]["sources"].append({
                            "doc_id": doc_id,
                            "sentence": edge.get("source_sentence", "")
                        })

    def to_json(self) -> Dict[str, Any]:
        """导出为JSON"""
        # 清理事件中的不可序列化内容
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
            "statistics": {
                "entity_count": len(self.entities),
                "event_count": len(self.events),
                "relation_count": len(self.relations),
                "participation_count": len(self.participations)
            }
        }

    def print_summary(self):
        """打印摘要"""
        print(f"\n{'='*60}")
        print("全局知识图谱摘要")
        print("="*60)
        print(f"实体数量: {len(self.entities)}")
        print(f"事件数量: {len(self.events)}")
        print(f"实体间关系: {len(self.relations)}")
        print(f"实体-事件参与: {len(self.participations)}")
        print(f"事件间关系: {len(self.event_relations)}")

        print("\n【实体示例】")
        for name, data in list(self.entities.items())[:5]:
            attrs = list(data.get("attributes", {}).keys())
            print(f"  - {name} ({data.get('type', '未知')}) 属性: {attrs}")

        print("\n【事件示例】")
        for ev_id, data in list(self.events.items())[:5]:
            print(f"  - {data.get('summary', '')[:50]}")


# ============================================================
# 第一步：单文档抽取
# ============================================================

def extract_local_graph(
    text: str,
    doc_id: str,
    doc_time: Optional[str] = None,
    title: Optional[str] = None,
    max_retries: int = 2
) -> Dict[str, Any]:
    """第一步：单文档抽取局部子图"""

    doc_info = {
        "doc_id": doc_id,
        "doc_time": doc_time or None,
        "title": title or text[:30]
    }

    user_prompt = f"""
文档信息：
- doc_id: {doc_id}
- doc_time: {doc_time or "未知"}
- title: {title or text[:30]}

文本内容：
{text}

请输出局部知识图谱。
"""

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": STEP1_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                tools=[{"type": "function", "function": STEP1_FUNCTION}],
                tool_choice={"type": "function", "function": {"name": "extract_local_graph"}},
                temperature=0,
                top_p=0.5,
                seed=42
            )

            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                result = json.loads(tool_calls[0].function.arguments)
                if "doc_info" not in result:
                    result["doc_info"] = doc_info
                return result

            if attempt < max_retries:
                continue
            return _empty_local_graph(doc_id)

        except Exception as e:
            print(f"  抽取失败 (尝试 {attempt + 1}): {e}")
            if attempt < max_retries:
                continue
            return _empty_local_graph(doc_id)

    return _empty_local_graph(doc_id)


def _empty_local_graph(doc_id: str) -> Dict[str, Any]:
    """空局部图"""
    return {
        "doc_info": {"doc_id": doc_id, "doc_time": None, "title": ""},
        "nodes": [],
        "edges": [],
        "attributes": []
    }


# ============================================================
# 第二步：生成操作指令
# ============================================================

def generate_operations(
    documents: List[Dict[str, Any]],
    max_retries: int = 2
) -> Dict[str, Any]:
    """第二步：多文档分析，生成操作指令"""

    # 构建输入：每篇文档的原文 + 局部子图
    docs_input = []
    for doc in documents:
        docs_input.append({
            "doc_id": doc["doc_info"]["doc_id"],
            "doc_time": doc["doc_info"]["doc_time"],
            "original_text": doc.get("_original_text", ""),
            "local_graph": {
                "nodes": doc.get("nodes", []),
                "edges": doc.get("edges", []),
                "attributes": doc.get("attributes", [])
            }
        })

    user_prompt = f"""
以下是多篇文档的原文和局部知识图谱，请分析并生成图谱操作指令：

{json.dumps(docs_input, ensure_ascii=False, indent=2)}

请输出操作指令 JSON。
"""

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": STEP2_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                tools=[{"type": "function", "function": STEP2_FUNCTION}],
                tool_choice={"type": "function", "function": {"name": "generate_graph_operations"}},
                temperature=0,
                top_p=0.5,
                seed=42
            )

            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                return json.loads(tool_calls[0].function.arguments)

            if attempt < max_retries:
                continue
            return _empty_operations()

        except Exception as e:
            print(f"生成指令失败 (尝试 {attempt + 1}): {e}")
            if attempt < max_retries:
                continue
            return _empty_operations()

    return _empty_operations()


def _empty_operations() -> Dict[str, Any]:
    """空操作指令"""
    return {
        "entity_alignments": [],
        "event_coreferences": [],
        "attribute_operations": [],
        "relation_inferences": []
    }


# ============================================================
# 完整流程：批量处理
# ============================================================

def build_knowledge_graph_from_documents(
    texts: List[Dict[str, str]],
    delay_seconds: float = 0.5,
    show_progress: bool = True
) -> GlobalKnowledgeGraph:
    """
    从文档列表构建全局知识图谱

    参数:
        texts: [{"text": "...", "doc_id": "...", "doc_time": "...", "title": "..."}]
        delay_seconds: API调用延迟
        show_progress: 显示进度

    返回:
        GlobalKnowledgeGraph 实例
    """

    # 第一步：抽取每篇文档的局部子图
    local_graphs = []

    for idx, doc in enumerate(texts):
        if show_progress:
            print(f"第一步：处理第 {idx + 1}/{len(texts)} 篇文档 - {doc.get('doc_id', 'unknown')}")

        result = extract_local_graph(
            doc["text"],
            doc.get("doc_id", f"doc_{idx}"),
            doc.get("doc_time"),
            doc.get("title")
        )
        # 保存原文用于第二步
        result["_original_text"] = doc["text"]
        local_graphs.append(result)

        if delay_seconds > 0 and idx < len(texts) - 1:
            time.sleep(delay_seconds)

    # 创建全局图谱
    kg = GlobalKnowledgeGraph()

    # 先直接应用局部图（建立基础）
    if show_progress:
        print("\n应用局部图到全局图谱...")
    for lg in local_graphs:
        kg.apply_local_graph(lg)

    # 第二步：生成操作指令
    if show_progress:
        print("\n第二步：生成图谱操作指令...")

    operations = generate_operations(local_graphs)

    # 执行操作指令
    if show_progress:
        print("\n执行操作指令...")

    for alignment in operations.get("entity_alignments", []):
        kg.apply_entity_alignment(alignment)

    for attr_op in operations.get("attribute_operations", []):
        kg.apply_attribute_operation(attr_op)

    for event_rel in operations.get("event_coreferences", []):
        kg.apply_event_coreference(event_rel)

    for inference in operations.get("relation_inferences", []):
        kg.apply_relation_inference(inference)

    if show_progress:
        kg.print_summary()

    return kg


# ============================================================
# 格式化输出
# ============================================================

def format_local_graph(local_graph: Dict[str, Any]) -> str:
    """格式化局部子图"""
    lines = []
    doc_info = local_graph.get("doc_info", {})

    lines.append(f"【文档】{doc_info.get('doc_id', 'unknown')}")
    lines.append(f"【时间】{doc_info.get('doc_time', '未知')}")

    nodes = local_graph.get("nodes", [])
    lines.append(f"\n【节点】（{len(nodes)}个）")
    for node in nodes:
        if node.get("node_type") == "entity":
            lines.append(f"  - {node.get('node_id')}: 实体 [{node.get('entity_type')}] {node.get('name')}")
        else:
            time_info = node.get("time", {})
            time_str = f" @{time_info.get('value', '')}" if time_info else ""
            lines.append(f"  - {node.get('node_id')}: 事件 {node.get('event_summary')}{time_str}")

    edges = local_graph.get("edges", [])
    if edges:
        lines.append(f"\n【边】（{len(edges)}个）")
        for edge in edges:
            lines.append(f"  - {edge.get('from_node')} -> {edge.get('to_node')} [{edge.get('edge_type')}]")

    attrs = local_graph.get("attributes", [])
    if attrs:
        lines.append(f"\n【属性】（{len(attrs)}个）")
        for attr in attrs[:5]:
            lines.append(f"  - {attr.get('node_id')}.{attr.get('attr_key')} = {attr.get('attr_value')}")

    return "\n".join(lines)


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":

    # 测试数据
    documents = [
        {
            "doc_id": "news_20240315_001",
            "doc_time": "2024-03-15",
            "title": "宁德时代业绩说明会",
            "text": """
            2024年3月15日，宁德时代在福建宁德召开年度业绩说明会。
            会上，董事长曾毓群宣布，公司2024年全年营收847亿元，同比增长15%，净利润120亿元。
            同时，宁德时代与特斯拉签署长期供货协议，将从2025年起向特斯拉供应储能电池，合同金额约300亿元。
            曾毓群表示，公司计划在匈牙利投资50亿欧元建设第二座欧洲工厂，预计2026年投产。
            """
        },
        {
            "doc_id": "news_20241201_002",
            "doc_time": "2024-12-01",
            "title": "宁德时代匈牙利工厂动工",
            "text": """
            2024年12月1日，宁德时代宣布匈牙利工厂正式开工建设。
            该工厂投资50亿欧元，预计2026年投产，年产能100GWh。
            宁德时代表示，这是公司欧洲战略的重要一步。
            """
        },
        {
            "doc_id": "news_20241215_003",
            "doc_time": "2024-12-15",
            "title": "宁德时代股价突破千元",
            "text": """
            2024年12月15日，宁德时代股价突破1000元大关，市值达到1.2万亿元。
            分析认为，匈牙利工厂动工和与特斯拉的供货协议是推动股价上涨的主要原因。
            今年以来，宁德时代股价累计上涨超过80%。
            """
        },
        {
            "doc_id": "news_20250320_004",
            "doc_time": "2025-03-20",
            "title": "宁德时代年报发布",
            "text": """
            2025年3月20日，宁德时代正式发布2024年年度报告。
            报告显示，公司全年营收847亿元，同比增长15%，净利润120亿元。
            公司同时披露，已开始向特斯拉供应储能电池，合同履行顺利。
            """
        }
    ]

    print("=" * 80)
    print("构建全局知识图谱")
    print("=" * 80)

    kg = build_knowledge_graph_from_documents(documents, show_progress=True)

    # 保存结果
    result_json = kg.to_json()
    with open("knowledge_graph.json", "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    print("\n全局知识图谱已保存到 knowledge_graph.json")

    # 打印详细实体信息
    print("\n" + "=" * 60)
    print("实体详情示例")
    print("=" * 60)
    for name, data in list(kg.entities.items())[:3]:
        print(f"\n【{name}】({data.get('type', '未知')})")
        attrs = data.get("attributes", {})
        for key, values in attrs.items():
            for v in values:
                print(f"  {key}: {v.get('value')} @{v.get('time', '未知')} [{v.get('confidence')}]")