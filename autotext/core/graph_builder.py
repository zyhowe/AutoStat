"""
关系图谱构建模块
"""

import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import math


class GraphBuilder:
    """关系图谱构建器"""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self._node_counter = 0
        self._edge_counter = 0

        self.node_colors = {
            "ENTITY": "#4facfe",
            "EVENT": "#ff9f43",
            "TOPIC": "#2ed573"
        }

        self.node_categories = {
            "ENTITY": "实体",
            "EVENT": "事件",
            "TOPIC": "主题"
        }

    def _add_node(self, node_id: str, name: str, node_type: str, metadata: Dict = None) -> str:
        if metadata is None:
            metadata = {}
        self.graph.add_node(node_id,
                           name=name,
                           type=node_type,
                           category=self.node_categories.get(node_type, node_type),
                           color=self.node_colors.get(node_type, "#a4b0be"),
                           **metadata)
        return node_id

    def _add_edge(self, source: str, target: str, edge_type: str, weight: float = 1.0, **kwargs) -> str:
        edge_id = f"e_{self._edge_counter}"
        self._edge_counter += 1
        self.graph.add_edge(source, target,
                           edge_id=edge_id,
                           type=edge_type,
                           weight=weight,
                           **kwargs)
        return edge_id

    def add_entity_node(self, entity_id: str, name: str, entity_type: str, count: int = 1) -> str:
        # 去掉前缀，直接使用名称作为显示名
        clean_name = name.split(":", 1)[-1] if ":" in name else name
        return self._add_node(
            clean_name,
            clean_name,
            "ENTITY",
            {"entity_type": entity_type, "count": count, "original_id": entity_id}
        )

    def add_event_node(self, event_id: str, event_type: str, trigger: str, timestamp: str = None, text: str = None) -> str:
        display_name = f"{event_type}" if len(event_type) <= 15 else event_type[:12] + "..."
        return self._add_node(
            event_id,
            display_name,
            "EVENT",
            {
                "event_type": event_type,
                "trigger": trigger,
                "timestamp": timestamp,
                "full_name": f"{event_type}({trigger})" if trigger else event_type
            }
        )

    def add_topic_node(self, topic_id: int, topic_name: str, keywords: List[str] = None) -> str:
        display_name = topic_name if len(topic_name) <= 15 else topic_name[:12] + "..."
        return self._add_node(
            f"T{topic_id}",
            display_name,
            "TOPIC",
            {"topic_id": topic_id, "keywords": keywords or [], "full_name": topic_name}
        )

    def add_entity_event_edge(self, entity_id: str, event_id: str, role: str = "参与") -> str:
        return self._add_edge(entity_id, event_id, "PARTICIPATE", role=role)

    def add_entity_entity_edge(self, entity_a: str, entity_b: str, weight: float = 1.0, pmi: float = 0) -> str:
        clean_a = entity_a.split(":", 1)[-1] if ":" in entity_a else entity_a
        clean_b = entity_b.split(":", 1)[-1] if ":" in entity_b else entity_b
        return self._add_edge(clean_a, clean_b, "CO_OCCUR", weight=weight, pmi=pmi)

    def add_entity_topic_edge(self, entity_id: str, topic_id: int, probability: float = 0.5) -> str:
        return self._add_edge(entity_id, f"T{topic_id}", "BELONG_TO", weight=probability)

    def add_event_topic_edge(self, event_id: str, topic_id: int, probability: float = 0.5) -> str:
        return self._add_edge(event_id, f"T{topic_id}", "BELONG_TO", weight=probability)

    def get_graph(self) -> nx.MultiDiGraph:
        return self.graph

    def export_for_visualization(self) -> Dict:
        nodes = []
        links = []

        for node_id, attrs in self.graph.nodes(data=True):
            node_type = attrs.get("type", "UNKNOWN")
            node_color = self.node_colors.get(node_type, "#a4b0be")
            node_category = self.node_categories.get(node_type, node_type)

            nodes.append({
                "id": node_id,
                "name": attrs.get("name", node_id),
                "category": node_category,
                "value": attrs.get("count", 1),
                "itemStyle": {"color": node_color},
                "symbolSize": min(40, 15 + attrs.get("count", 1) // 5)
            })

        for u, v, attrs in self.graph.edges(data=True):
            links.append({
                "source": u,
                "target": v,
                "value": attrs.get("weight", 1),
                "type": attrs.get("type", "UNKNOWN")
            })

        return {"nodes": nodes, "links": links}

    def get_statistics(self) -> Dict:
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "entity_count": sum(1 for n, d in self.graph.nodes(data=True) if d.get("type") == "ENTITY"),
            "event_count": sum(1 for n, d in self.graph.nodes(data=True) if d.get("type") == "EVENT"),
            "topic_count": sum(1 for n, d in self.graph.nodes(data=True) if d.get("type") == "TOPIC")
        }