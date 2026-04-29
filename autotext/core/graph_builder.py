"""
关系图谱构建模块 - 构建实体-事件-主题知识图谱
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

        # 节点类型颜色映射（用于前端）
        self.node_colors = {
            "ENTITY": "#4facfe",  # 蓝色
            "EVENT": "#ff9f43",  # 橙色
            "TOPIC": "#2ed573"  # 绿色
        }

        # 边类型样式映射
        self.edge_styles = {
            "PARTICIPATE": {"color": "#4facfe", "width": 1, "style": "solid"},
            "CO_OCCUR": {"color": "#a4b0be", "width": 1, "style": "dashed"},
            "BELONG_TO": {"color": "#2ed573", "width": 1, "style": "solid"},
            "TEMPORAL": {"color": "#ff9f43", "width": 1, "style": "dotted"}
        }

    def _add_node(self, node_id: str, name: str, node_type: str,
                  metadata: Dict = None) -> str:
        """添加节点"""
        if metadata is None:
            metadata = {}

        self.graph.add_node(node_id,
                            name=name,
                            type=node_type,
                            color=self.node_colors.get(node_type, "#a4b0be"),
                            **metadata)
        return node_id

    def _add_edge(self, source: str, target: str, edge_type: str,
                  weight: float = 1.0, **kwargs) -> str:
        """添加边"""
        edge_id = f"e_{self._edge_counter}"
        self._edge_counter += 1

        style = self.edge_styles.get(edge_type, {"color": "#a4b0be", "width": 1, "style": "solid"})

        self.graph.add_edge(source, target,
                            edge_id=edge_id,
                            type=edge_type,
                            weight=weight,
                            color=style["color"],
                            width=style["width"],
                            style=style["style"],
                            **kwargs)
        return edge_id

    def add_entity_node(self, entity_id: str, name: str, entity_type: str,
                        count: int = 1) -> str:
        """添加实体节点"""
        return self._add_node(
            f"ENTITY:{entity_id}",
            name,
            "ENTITY",
            {"entity_type": entity_type, "count": count, "original_id": entity_id}
        )

    def add_event_node(self, event_id: str, event_type: str, trigger: str,
                       timestamp: str = None, text: str = None) -> str:
        """添加事件节点"""
        return self._add_node(
            f"EVENT:{event_id}",
            f"{event_type}",
            "EVENT",
            {
                "event_type": event_type,
                "trigger": trigger,
                "timestamp": timestamp,
                "text": text[:100] if text else "",
                "original_id": event_id
            }
        )

    def add_topic_node(self, topic_id: int, topic_name: str,
                       keywords: List[str] = None) -> str:
        """添加主题节点"""
        return self._add_node(
            f"TOPIC:{topic_id}",
            topic_name,
            "TOPIC",
            {"topic_id": topic_id, "keywords": keywords or []}
        )

    def add_entity_event_edge(self, entity_id: str, event_id: str, role: str = "参与") -> str:
        """添加实体-事件边（实体参与事件）"""
        return self._add_edge(
            f"ENTITY:{entity_id}",
            f"EVENT:{event_id}",
            "PARTICIPATE",
            role=role
        )

    def add_entity_entity_edge(self, entity_a: str, entity_b: str,
                               weight: float = 1.0, pmi: float = 0) -> str:
        """添加实体-实体边（共现关系）"""
        return self._add_edge(
            f"ENTITY:{entity_a}",
            f"ENTITY:{entity_b}",
            "CO_OCCUR",
            weight=weight,
            pmi=pmi
        )

    def add_entity_topic_edge(self, entity_id: str, topic_id: int,
                              probability: float = 0.5) -> str:
        """添加实体-主题边（实体属于主题）"""
        return self._add_edge(
            f"ENTITY:{entity_id}",
            f"TOPIC:{topic_id}",
            "BELONG_TO",
            weight=probability,
            probability=probability
        )

    def add_event_topic_edge(self, event_id: str, topic_id: int,
                             probability: float = 0.5) -> str:
        """添加事件-主题边（事件属于主题）"""
        return self._add_edge(
            f"EVENT:{event_id}",
            f"TOPIC:{topic_id}",
            "BELONG_TO",
            weight=probability,
            probability=probability
        )

    def add_event_event_edge(self, event_a: str, event_b: str,
                             relation_type: str = "PRECEDES",
                             time_gap: int = 0) -> str:
        """添加事件-事件边（时序关系）"""
        return self._add_edge(
            f"EVENT:{event_a}",
            f"EVENT:{event_b}",
            "TEMPORAL",
            relation=relation_type,
            time_gap=time_gap
        )

    def get_graph(self) -> nx.MultiDiGraph:
        """获取 NetworkX 图对象"""
        return self.graph

    def export_for_visualization(self) -> Dict:
        """导出 ECharts 可用的格式"""
        nodes = []
        links = []

        # 导出节点
        for node_id, attrs in self.graph.nodes(data=True):
            nodes.append({
                "id": node_id,
                "name": attrs.get("name", node_id),
                "type": attrs.get("type", "UNKNOWN"),
                "value": attrs.get("count", 1),
                "itemStyle": {"color": attrs.get("color", "#a4b0be")},
                "symbolSize": min(50, 10 + attrs.get("count", 1) * 2),
                "category": attrs.get("type", "UNKNOWN")
            })

        # 导出边
        for u, v, attrs in self.graph.edges(data=True):
            links.append({
                "source": u,
                "target": v,
                "value": attrs.get("weight", 1),
                "type": attrs.get("type", "UNKNOWN"),
                "lineStyle": {
                    "color": attrs.get("color", "#a4b0be"),
                    "width": attrs.get("width", 1),
                    "type": attrs.get("style", "solid")
                }
            })

        return {"nodes": nodes, "links": links}

    def get_statistics(self) -> Dict:
        """获取图谱统计"""
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "entity_count": sum(1 for n, d in self.graph.nodes(data=True) if d.get("type") == "ENTITY"),
            "event_count": sum(1 for n, d in self.graph.nodes(data=True) if d.get("type") == "EVENT"),
            "topic_count": sum(1 for n, d in self.graph.nodes(data=True) if d.get("type") == "TOPIC")
        }