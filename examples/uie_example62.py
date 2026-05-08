"""
知识图谱整理模块 - 确定性清洗，不依赖大模型
功能：去重、属性合并、格式统一、建立追溯映射
输入：knowledge_graph_raw.json
输出：knowledge_graph_cleaned.json + cleaning_report.json
"""

import json
import re
from typing import Dict, Any, List, Set, Tuple, Optional
from datetime import datetime
from collections import defaultdict
from copy import deepcopy


class KnowledgeGraphCleaner:
    """知识图谱整理器 - 确定性清洗，保留完整追溯"""

    def __init__(self, raw_data: Dict[str, Any]):
        """
        初始化整理器

        参数:
            raw_data: 原始知识图谱数据（来自 GlobalKnowledgeGraph.to_json()）
        """
        self.raw = raw_data
        self.cleaning_log = []

        # 清洗后的数据结构
        self.cleaned = {
            "version": "cleaned_v1",
            "cleaned_at": datetime.now().isoformat(),
            "source_file": "knowledge_graph_raw.json",
            "entities": {},
            "events": {},
            "relations": {},
            "participations": {},
            "event_relations": {},
            "mappings": {
                "entity": {},
                "event": {},
                "relation": {},
                "participation": {},
                "event_relation": {}
            },
            "statistics": {
                "entities_raw": 0,
                "entities_cleaned": 0,
                "events_raw": 0,
                "events_cleaned": 0,
                "relations_raw": 0,
                "relations_cleaned": 0
            }
        }

        # 中间映射
        self.entity_canonical_map = {}  # 原始实体名 -> 规范化实体名
        self.event_canonical_map = {}  # 原始事件ID -> 规范化事件ID
        self.relation_canonical_map = {}  # 原始关系key -> 规范化关系key

        # 属性名映射表
        self.attribute_name_map = {
            "2024年全年营收": "营收_2024",
            "全年营收": "营收_2024",
            "2024年全年净利润": "净利润_2024",
            "净利润": "净利润_2024",
            "营收同比增长率": "营收同比增长_2024",
            "营收同比增长": "营收同比增长_2024",
            "与特斯拉合同金额": "合同金额_特斯拉",
            "匈牙利投资额": "匈牙利投资额",
            "投资额": "匈牙利投资额",
            "预计投产时间": "预计投产时间",
            "年产能": "年产能_GWh",
            "股价": "股价_元",
            "市值": "市值_亿元",
            "年内涨幅": "年内涨幅_百分比",
            "战略": "战略",
            "职位": "职位"
        }

        # 关系谓语映射表
        self.predicate_map = {
            "与...有供货关系": "供货",
            "供应储能电池": "供货",
            "向...供应": "供货",
            "股价达到": "股价",
            "市值达到": "市值",
            "拥有": "拥有"
        }

    def _add_to_log(self, action: str, target: str, details: Dict[str, Any]):
        """添加清洗日志"""
        self.cleaning_log.append({
            "action": action,
            "target": target,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })

    def _normalize_time(self, time_value) -> Optional[str]:
        """归一化时间格式为 YYYY-MM-DD"""
        if not time_value:
            return None

        if isinstance(time_value, str):
            # 已经是标准格式
            if re.match(r'\d{4}-\d{2}-\d{2}', time_value):
                return time_value
            # 年月日
            match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', time_value)
            if match:
                return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
            # 年月
            match = re.search(r'(\d{4})年(\d{1,2})月', time_value)
            if match:
                return f"{match.group(1)}-{int(match.group(2)):02d}-01"
            # 年份
            match = re.search(r'(\d{4})年', time_value)
            if match:
                return f"{match.group(1)}-01-01"
            # 已经是 YYYY-MM-DD 格式的其他变体
            match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', time_value)
            if match:
                return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"

        if isinstance(time_value, dict):
            if "date" in time_value:
                return self._normalize_time(time_value["date"])
            if "year" in time_value:
                return f"{time_value['year']}-01-01"

        return str(time_value) if time_value else None

    def _normalize_entity_name(self, name: str) -> str:
        """规范化实体名称（用于合并）"""
        # 去除首尾空格
        name = name.strip()
        # 可选：添加更多规范化规则
        # 如 "CATL" -> "宁德时代" 这类需要外部知识，暂不处理
        return name

    def _get_canonical_entity(self, raw_name: str) -> str:
        """获取规范化实体名"""
        norm_name = self._normalize_entity_name(raw_name)
        if norm_name in self.entity_canonical_map:
            return self.entity_canonical_map[norm_name]
        return raw_name

    def _merge_entity_attributes(self, entity_name: str) -> Dict[str, Any]:
        """合并实体的属性（去重、归一化key）"""
        if entity_name not in self.raw.get("entities", {}):
            return {}

        raw_entity = self.raw["entities"][entity_name]
        raw_attributes = raw_entity.get("attributes", {})
        merged_attributes = {}

        for attr_key, attr_values in raw_attributes.items():
            # 归一化属性名
            clean_key = self.attribute_name_map.get(attr_key, attr_key)

            if clean_key not in merged_attributes:
                merged_attributes[clean_key] = []

            # 去重：同一值+同一时间只保留一条
            seen = set()
            for attr in attr_values:
                if isinstance(attr, dict):
                    value = attr.get("value", "")
                    time = attr.get("time", "")
                    dedup_key = f"{value}|{time}"
                    if dedup_key not in seen:
                        seen.add(dedup_key)
                        # 添加来源引用
                        clean_attr = {
                            "value": value,
                            "time": self._normalize_time(time),
                            "confidence": attr.get("confidence", "explicit"),
                            "_source": f"entities/{entity_name}/attributes/{attr_key}"
                        }
                        merged_attributes[clean_key].append(clean_attr)

        return merged_attributes

    def _normalize_event_time(self, event_data: Dict[str, Any]) -> Optional[str]:
        """归一化事件时间"""
        time_data = event_data.get("time", {})
        if isinstance(time_data, dict):
            if "date" in time_data:
                return self._normalize_time(time_data["date"])
            if "year" in time_data:
                return self._normalize_time(str(time_data["year"]))
            if "value" in time_data:
                return self._normalize_time(time_data["value"])
        elif isinstance(time_data, str):
            return self._normalize_time(time_data)

        # 兼容直接在 event 中的 time_str
        time_str = event_data.get("time_str", "")
        if time_str:
            return self._normalize_time(time_str)

        return None

    def _get_event_key(self, event_data: Dict[str, Any]) -> str:
        """生成事件去重的key（基于摘要规范化）"""
        summary = event_data.get("summary", "")
        # 规范化摘要：去除多余空格，统一标点
        summary = re.sub(r'\s+', ' ', summary).strip()
        summary = summary.replace("，", ",").replace("。", ".")

        # 提取核心部分（去除可能的变化部分）
        # 例如去掉具体金额数字
        core = re.sub(r'\d+\.?\d*亿[元]?', '[金额]', summary)
        core = re.sub(r'\d+%', '[百分比]', core)
        core = re.sub(r'\d+年', '[年份]', core)

        return core

    def _merge_events(self) -> Dict[str, Any]:
        """合并重复事件"""
        merged_events = {}
        event_groups = defaultdict(list)

        # 按事件key分组
        for event_id, event_data in self.raw.get("events", {}).items():
            # 跳过已经是合并结果的事件
            if event_id.startswith("ev_merged_"):
                continue
            key = self._get_event_key(event_data)
            event_groups[key].append((event_id, event_data))

        # 合并每组内的事件
        for key, events in event_groups.items():
            if len(events) == 1:
                event_id, event_data = events[0]
                merged_events[event_id] = deepcopy(event_data)
                self.event_canonical_map[event_id] = event_id
            else:
                # 合并多个事件
                canonical_id = f"ev_merged_{len(merged_events)}"
                canonical_event = {
                    "canonical_summary": events[0][1].get("summary", ""),
                    "summary": events[0][1].get("summary", ""),
                    "trigger": events[0][1].get("trigger", ""),
                    "time": self._normalize_event_time(events[0][1]),
                    "location": events[0][1].get("location"),
                    "confidence": "explicit",
                    "_sources": []
                }

                for event_id, event_data in events:
                    canonical_event["_sources"].append({
                        "raw_id": event_id,
                        "path": f"events/{event_id}"
                    })
                    self.event_canonical_map[event_id] = canonical_id

                merged_events[canonical_id] = canonical_event

        return merged_events

    def _merge_relations(self) -> Dict[str, Any]:
        """合并重复关系"""
        merged_relations = {}
        relation_groups = defaultdict(list)

        for rel_key, rel_data in self.raw.get("relations", {}).items():
            # 解析关系key
            parts = rel_key.split("->")
            if len(parts) >= 3:
                subject = parts[0]
                predicate = parts[1]
                obj = parts[2]

                # 规范化谓语
                predicate = self.predicate_map.get(predicate, predicate)

                # 获取规范化实体名
                canonical_subject = self._get_canonical_entity(subject)
                canonical_obj = self._get_canonical_entity(obj)

                group_key = (canonical_subject, predicate, canonical_obj)
                relation_groups[group_key].append((rel_key, rel_data))

        for (subject, predicate, obj), relations in relation_groups.items():
            canonical_key = f"{subject}->{predicate}->{obj}"

            if len(relations) == 1:
                rel_key, rel_data = relations[0]
                merged_relations[canonical_key] = deepcopy(rel_data)
                merged_relations[canonical_key]["_source"] = f"relations/{rel_key}"
                self.relation_canonical_map[rel_key] = canonical_key
            else:
                # 合并多个关系
                merged_rel = {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "confidence": "explicit",
                    "_sources": []
                }

                for rel_key, rel_data in relations:
                    merged_rel["_sources"].append({
                        "raw_key": rel_key,
                        "path": f"relations/{rel_key}"
                    })
                    self.relation_canonical_map[rel_key] = canonical_key

                    # 取最高的置信度
                    if rel_data.get("confidence") == "explicit":
                        merged_rel["confidence"] = "explicit"

                merged_relations[canonical_key] = merged_rel

        return merged_relations

    def _merge_participations(self) -> Dict[str, Any]:
        """合并参与关系"""
        merged_participations = {}
        part_groups = defaultdict(list)

        for part_key, part_data in self.raw.get("participations", {}).items():
            parts = part_key.split("-")
            if len(parts) >= 3:
                entity = parts[0]
                role = parts[1]
                event = parts[2]

                # 获取规范化名称
                canonical_entity = self._get_canonical_entity(entity)
                canonical_event = self.event_canonical_map.get(event, event)

                group_key = (canonical_entity, role, canonical_event)
                part_groups[group_key].append((part_key, part_data))

        for (entity, role, event), participations in part_groups.items():
            canonical_key = f"{entity}-{role}-{event}"

            if len(participations) == 1:
                part_key, part_data = participations[0]
                merged_participations[canonical_key] = deepcopy(part_data)
                merged_participations[canonical_key]["_source"] = f"participations/{part_key}"
            else:
                merged_part = {
                    "entity": entity,
                    "role": role,
                    "event": event,
                    "confidence": "explicit",
                    "_sources": []
                }

                for part_key, part_data in participations:
                    merged_part["_sources"].append({
                        "raw_key": part_key,
                        "path": f"participations/{part_key}"
                    })

                merged_participations[canonical_key] = merged_part

        return merged_participations

    def _merge_event_relations(self) -> Dict[str, Any]:
        """合并事件间关系"""
        merged_event_rels = {}
        rel_groups = defaultdict(list)

        for rel_key, rel_data in self.raw.get("event_relations", {}).items():
            parts = rel_key.split("-")
            if len(parts) >= 3:
                # 格式：事件A-关系类型-事件B
                event_a = parts[0]
                rel_type = parts[1]
                event_b = parts[2]

                # 获取规范化事件ID
                canonical_a = self.event_canonical_map.get(event_a, event_a)
                canonical_b = self.event_canonical_map.get(event_b, event_b)

                group_key = (canonical_a, rel_type, canonical_b)
                rel_groups[group_key].append((rel_key, rel_data))

        for (event_a, rel_type, event_b), relations in rel_groups.items():
            canonical_key = f"{event_a}-{rel_type}-{event_b}"

            if len(relations) == 1:
                rel_key, rel_data = relations[0]
                merged_event_rels[canonical_key] = deepcopy(rel_data)
                merged_event_rels[canonical_key]["_source"] = f"event_relations/{rel_key}"
            else:
                merged_rel = {
                    "type": rel_type,
                    "description": relations[0][1].get("description", ""),
                    "confidence": "explicit",
                    "_sources": []
                }

                for rel_key, rel_data in relations:
                    merged_rel["_sources"].append({
                        "raw_key": rel_key,
                        "path": f"event_relations/{rel_key}"
                    })

                merged_event_rels[canonical_key] = merged_rel

        return merged_event_rels

    def _build_entity_canonical(self):
        """构建实体规范化映射"""
        for entity_name in self.raw.get("entities", {}).keys():
            canonical = self._normalize_entity_name(entity_name)
            if canonical not in self.entity_canonical_map:
                self.entity_canonical_map[canonical] = canonical
            # 记录映射关系
            if entity_name != canonical:
                if canonical not in self.entity_canonical_map:
                    self.entity_canonical_map[entity_name] = canonical
                else:
                    self.entity_canonical_map[entity_name] = canonical

    def _build_cleaned_entities(self) -> Dict[str, Any]:
        """构建清洗后的实体"""
        cleaned_entities = {}

        # 按规范化实体名分组
        entity_groups = defaultdict(list)
        for entity_name in self.raw.get("entities", {}).keys():
            canonical = self._get_canonical_entity(entity_name)
            entity_groups[canonical].append(entity_name)

        for canonical_name, raw_names in entity_groups.items():
            raw_entity = self.raw["entities"].get(raw_names[0], {})

            # 构建清洗后的实体
            cleaned_entity = {
                "canonical_name": canonical_name,
                "type": raw_entity.get("type", "其他"),
                "attributes": {},
                "_sources": [],
                "_aliases": []
            }

            # 合并所有原始实体的属性
            for raw_name in raw_names:
                cleaned_entity["_sources"].append({
                    "raw_name": raw_name,
                    "path": f"entities/{raw_name}"
                })
                if raw_name != canonical_name:
                    cleaned_entity["_aliases"].append(raw_name)

                # 合并属性
                attrs = self._merge_entity_attributes(raw_name)
                for attr_key, attr_values in attrs.items():
                    if attr_key not in cleaned_entity["attributes"]:
                        cleaned_entity["attributes"][attr_key] = []
                    cleaned_entity["attributes"][attr_key].extend(attr_values)

            # 去重属性值
            for attr_key in cleaned_entity["attributes"]:
                seen_values = set()
                unique_values = []
                for attr in cleaned_entity["attributes"][attr_key]:
                    value_key = f"{attr.get('value')}|{attr.get('time')}"
                    if value_key not in seen_values:
                        seen_values.add(value_key)
                        unique_values.append(attr)
                cleaned_entity["attributes"][attr_key] = unique_values

            cleaned_entities[canonical_name] = cleaned_entity

        return cleaned_entities

    def _build_mappings(self):
        """构建映射区块"""
        # 实体映射
        for raw_name, canonical in self.entity_canonical_map.items():
            if raw_name != canonical:
                if canonical not in self.cleaned["mappings"]["entity"]:
                    self.cleaned["mappings"]["entity"][canonical] = []
                if raw_name not in self.cleaned["mappings"]["entity"][canonical]:
                    self.cleaned["mappings"]["entity"][canonical].append(raw_name)

        # 事件映射
        for raw_id, canonical in self.event_canonical_map.items():
            if raw_id != canonical:
                if canonical not in self.cleaned["mappings"]["event"]:
                    self.cleaned["mappings"]["event"][canonical] = []
                if raw_id not in self.cleaned["mappings"]["event"][canonical]:
                    self.cleaned["mappings"]["event"][canonical].append(raw_id)

        # 关系映射
        for raw_key, canonical in self.relation_canonical_map.items():
            if raw_key != canonical:
                if canonical not in self.cleaned["mappings"]["relation"]:
                    self.cleaned["mappings"]["relation"][canonical] = []
                if raw_key not in self.cleaned["mappings"]["relation"][canonical]:
                    self.cleaned["mappings"]["relation"][canonical].append(raw_key)

    def _update_statistics(self):
        """更新统计信息"""
        self.cleaned["statistics"]["entities_raw"] = len(self.raw.get("entities", {}))
        self.cleaned["statistics"]["entities_cleaned"] = len(self.cleaned["entities"])
        self.cleaned["statistics"]["events_raw"] = len(self.raw.get("events", {}))
        self.cleaned["statistics"]["events_cleaned"] = len(self.cleaned["events"])
        self.cleaned["statistics"]["relations_raw"] = len(self.raw.get("relations", {}))
        self.cleaned["statistics"]["relations_cleaned"] = len(self.cleaned["relations"])

        # 添加日志统计
        self.cleaned["statistics"]["cleaning_actions"] = len(self.cleaning_log)

    def clean(self) -> Dict[str, Any]:
        """执行完整清洗流程"""
        print("\n开始清洗知识图谱...")

        # 1. 构建实体规范化映射
        print("  [1/6] 构建实体规范化映射...")
        self._build_entity_canonical()
        self._add_to_log("build_mapping", "entity", {"count": len(self.entity_canonical_map)})

        # 2. 清洗实体
        print("  [2/6] 清洗实体（去重、合并属性）...")
        self.cleaned["entities"] = self._build_cleaned_entities()
        self._add_to_log("clean_entities", "all", {"raw": len(self.raw.get("entities", {})),
                                                   "cleaned": len(self.cleaned["entities"])})

        # 3. 清洗事件
        print("  [3/6] 清洗事件（共指合并）...")
        self.cleaned["events"] = self._merge_events()
        self._add_to_log("clean_events", "all", {"raw": len(self.raw.get("events", {})),
                                                 "cleaned": len(self.cleaned["events"])})

        # 4. 清洗关系
        print("  [4/6] 清洗关系...")
        self.cleaned["relations"] = self._merge_relations()
        self._add_to_log("clean_relations", "all", {"raw": len(self.raw.get("relations", {})),
                                                    "cleaned": len(self.cleaned["relations"])})

        # 5. 清洗参与关系
        print("  [5/6] 清洗参与关系...")
        self.cleaned["participations"] = self._merge_participations()
        self._add_to_log("clean_participations", "all",
                         {"raw": len(self.raw.get("participations", {})),
                          "cleaned": len(self.cleaned["participations"])})

        # 6. 清洗事件间关系
        print("  [6/6] 清洗事件间关系...")
        self.cleaned["event_relations"] = self._merge_event_relations()
        self._add_to_log("clean_event_relations", "all",
                         {"raw": len(self.raw.get("event_relations", {})),
                          "cleaned": len(self.cleaned["event_relations"])})

        # 7. 构建映射
        print("  构建映射表...")
        self._build_mappings()

        # 8. 更新统计
        self._update_statistics()

        print(f"\n清洗完成！")
        print(
            f"  实体: {self.cleaned['statistics']['entities_raw']} -> {self.cleaned['statistics']['entities_cleaned']}")
        print(f"  事件: {self.cleaned['statistics']['events_raw']} -> {self.cleaned['statistics']['events_cleaned']}")
        print(
            f"  关系: {self.cleaned['statistics']['relations_raw']} -> {self.cleaned['statistics']['relations_cleaned']}")

        return self.cleaned

    def get_cleaning_report(self) -> Dict[str, Any]:
        """获取清洗报告"""
        return {
            "cleaned_at": self.cleaned["cleaned_at"],
            "source_file": self.cleaned["source_file"],
            "statistics": self.cleaned["statistics"],
            "logs": self.cleaning_log,
            "mappings_summary": {
                "entity_mappings": len(self.cleaned["mappings"]["entity"]),
                "event_mappings": len(self.cleaned["mappings"]["event"]),
                "relation_mappings": len(self.cleaned["mappings"]["relation"])
            }
        }


def clean_knowledge_graph(
        input_path: str = "knowledge_graph_raw.json",
        output_path: str = "knowledge_graph_cleaned.json",
        report_path: str = "cleaning_report.json"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    清洗知识图谱并保存结果

    参数:
        input_path: 原始JSON文件路径
        output_path: 清洗后JSON输出路径
        report_path: 清洗报告输出路径

    返回:
        (清洗后数据, 清洗报告)
    """
    # 读取原始数据
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 创建清洗器并执行清洗
    cleaner = KnowledgeGraphCleaner(raw_data)
    cleaned_data = cleaner.clean()

    # 保存清洗后的数据
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    print(f"\n清洗后数据已保存到 {output_path}")

    # 保存清洗报告
    report = cleaner.get_cleaning_report()
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"清洗报告已保存到 {report_path}")

    return cleaned_data, report


def compare_graphs(raw_path: str = "knowledge_graph_raw.json",
                   cleaned_path: str = "knowledge_graph_cleaned.json"):
    """
    对比原始图谱和清洗后图谱的差异
    """
    with open(raw_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    with open(cleaned_path, "r", encoding="utf-8") as f:
        cleaned = json.load(f)

    print("\n" + "=" * 60)
    print("图谱对比报告")
    print("=" * 60)

    raw_stats = raw.get("statistics", {})
    cleaned_stats = cleaned.get("statistics", {})

    print(f"\n实体数量: {raw_stats.get('entity_count', 0)} -> {cleaned_stats.get('entities_cleaned', 0)}")
    print(f"事件数量: {raw_stats.get('event_count', 0)} -> {cleaned_stats.get('events_cleaned', 0)}")
    print(f"关系数量: {raw_stats.get('relation_count', 0)} -> {cleaned_stats.get('relations_cleaned', 0)}")

    # 显示实体合并示例
    mappings = cleaned.get("mappings", {})
    entity_mappings = mappings.get("entity", {})
    if entity_mappings:
        print("\n实体合并示例:")
        for canonical, aliases in list(entity_mappings.items())[:3]:
            print(f"  {canonical} ← {', '.join(aliases)}")

    event_mappings = mappings.get("event", {})
    if event_mappings:
        print("\n事件合并示例:")
        for canonical, aliases in list(event_mappings.items())[:3]:
            print(f"  {canonical} ← {', '.join(aliases)}")


if __name__ == "__main__":
    import os

    # 检查原始文件是否存在
    if os.path.exists("knowledge_graph_raw.json"):
        # 执行清洗
        cleaned_data, report = clean_knowledge_graph(
            input_path="knowledge_graph_raw.json",
            output_path="knowledge_graph_cleaned.json",
            report_path="cleaning_report.json"
        )

        # 对比图谱
        compare_graphs()

        print("\n清洗完成！")
        print("  - knowledge_graph_raw.json (原始数据，保留不变)")
        print("  - knowledge_graph_cleaned.json (清洗后数据)")
        print("  - cleaning_report.json (清洗报告)")
    else:
        print("请先运行信息抽取脚本，生成 knowledge_graph_raw.json 文件")