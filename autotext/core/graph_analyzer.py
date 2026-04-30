"""
图算法分析模块 - 中心节点、桥梁节点、社区发现、事件链
"""

import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
import math
from collections import defaultdict


class GraphAnalyzer:
    """图算法分析器"""

    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        # 转换为无向图用于社区发现
        self.undirected = graph.to_undirected()

    def get_center_nodes(self, top_n: int = 10) -> List[Dict]:
        """
        PageRank 中心节点检测

        返回: [{"node_id": "...", "name": "...", "type": "...", "score": 0.087}, ...]
        """
        try:
            # 计算 PageRank
            pagerank = nx.pagerank(self.undirected, weight='weight')

            # 获取节点信息
            results = []
            for node_id, score in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_n]:
                attrs = self.graph.nodes[node_id]
                results.append({
                    "node_id": node_id,
                    "name": attrs.get("name", node_id),
                    "type": attrs.get("type", "UNKNOWN"),
                    "score": round(score, 4)
                })
            return results
        except Exception as e:
            print(f"⚠️ PageRank 计算失败: {e}")
            return []

    def get_bridge_nodes(self, top_n: int = 10) -> List[Dict]:
        """
        Betweenness Centrality 桥梁节点检测

        返回: [{"node_id": "...", "name": "...", "type": "...", "score": 0.087}, ...]
        """
        try:
            # 计算介数中心性（采样提高性能）
            if self.undirected.number_of_nodes() > 500:
                # 大图采样计算
                betweenness = nx.betweenness_centrality(self.undirected, k=500)
            else:
                betweenness = nx.betweenness_centrality(self.undirected)

            results = []
            for node_id, score in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_n]:
                attrs = self.graph.nodes[node_id]
                results.append({
                    "node_id": node_id,
                    "name": attrs.get("name", node_id),
                    "type": attrs.get("type", "UNKNOWN"),
                    "score": round(score, 4)
                })
            return results
        except Exception as e:
            print(f"⚠️ Betweenness 计算失败: {e}")
            return []

    def get_communities(self) -> List[List[Dict]]:
        """
        Louvain 社区发现

        返回: [[{"node_id": "...", "name": "..."}, ...], ...]
        """
        try:
            import community.community_louvain as community_louvain

            # 计算社区
            partition = community_louvain.best_partition(self.undirected)

            # 按社区分组
            communities = defaultdict(list)
            for node_id, community_id in partition.items():
                attrs = self.graph.nodes[node_id]
                communities[community_id].append({
                    "node_id": node_id,
                    "name": attrs.get("name", node_id),
                    "type": attrs.get("type", "UNKNOWN")
                })

            # 转换为列表并按大小排序
            result = [nodes for nodes in communities.values()]
            result.sort(key=len, reverse=True)
            return result[:20]  # 只返回前20个社区
        except ImportError:
            print("⚠️ python-louvain 未安装，跳过社区发现")
            return []
        except Exception as e:
            print(f"⚠️ 社区发现失败: {e}")
            return []

    def find_event_chains(self, start_node_id: str = None,
                          max_depth: int = 5) -> List[List[str]]:
        """事件链发现 - 简化版（不使用时间比较）"""
        try:
            # 只考虑事件节点
            event_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "EVENT"]

            if len(event_nodes) < 2:
                return []

            # 按名称排序（简单方案）
            event_nodes.sort()

            # 分成多个链
            chains = []
            current_chain = [event_nodes[0]]

            for node in event_nodes[1:]:
                if len(current_chain) < max_depth:
                    current_chain.append(node)
                else:
                    if len(current_chain) >= 2:
                        chains.append(current_chain)
                    current_chain = [node]

            if len(current_chain) >= 2:
                chains.append(current_chain)

            return chains[:10]

        except Exception as e:
            print(f"⚠️ 事件链发现失败: {e}")
            return []

    def get_node_connections(self, node_id: str) -> Dict:
        """
        获取节点的关联信息

        返回: {
            "events": [...],
            "entities": [...],
            "topics": [...],
            "neighbors": [...]
        }
        """
        if node_id not in self.graph:
            return {}

        attrs = self.graph.nodes[node_id]
        node_type = attrs.get("type", "UNKNOWN")

        connections = {
            "events": [],
            "entities": [],
            "topics": [],
            "neighbors": []
        }

        for neighbor in self.graph.neighbors(node_id):
            neighbor_attrs = self.graph.nodes[neighbor]
            neighbor_type = neighbor_attrs.get("type", "UNKNOWN")

            connections["neighbors"].append({
                "node_id": neighbor,
                "name": neighbor_attrs.get("name", neighbor),
                "type": neighbor_type
            })

            if neighbor_type == "EVENT":
                connections["events"].append(neighbor_attrs.get("name", neighbor))
            elif neighbor_type == "ENTITY":
                connections["entities"].append(neighbor_attrs.get("name", neighbor))
            elif neighbor_type == "TOPIC":
                connections["topics"].append(neighbor_attrs.get("name", neighbor))

        return connections

    def get_shortest_path(self, source: str, target: str) -> List[str]:
        """获取两个节点之间的最短路径"""
        try:
            path = nx.shortest_path(self.undirected, source=source, target=target)
            return path
        except Exception:
            return []

    def get_summary_insights(self) -> Dict:
        """获取综合分析洞察"""
        insights = {
            "center_nodes": self.get_center_nodes(5),
            "bridge_nodes": self.get_bridge_nodes(5),
            "event_chains": self.find_event_chains(),
            "statistics": {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "entity_count": sum(1 for n, d in self.graph.nodes(data=True) if d.get("type") == "ENTITY"),
                "event_count": sum(1 for n, d in self.graph.nodes(data=True) if d.get("type") == "EVENT"),
                "topic_count": sum(1 for n, d in self.graph.nodes(data=True) if d.get("type") == "TOPIC")
            }
        }

        return insights