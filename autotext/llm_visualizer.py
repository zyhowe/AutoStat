"""
信息抽取结果可视化 - 基于抽取结果绘制图表
"""

import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Any, List
from collections import defaultdict, Counter
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
class InfoVisualizer:
    """信息抽取结果可视化器"""

    def __init__(self, extraction_result: Dict[str, Any]):
        """
        参数:
        - extraction_result: 大模型抽取的 JSON 结果
        """
        self.entities = extraction_result.get("entities", [])
        self.relationships = extraction_result.get("relationships", [])
        self.events = extraction_result.get("events", [])
        self.themes = extraction_result.get("themes", [])
        self.categorization = extraction_result.get("categorization", {})

        # 构建 ID 到名称的映射
        self.entity_map = {e["entity_id"]: e["entity_name"] for e in self.entities}
        self.event_map = {e["event_id"]: e for e in self.events}
        self.theme_map = {t["theme_id"]: t for t in self.themes}

    # ==================== 1. 关系图谱 ====================

    def plot_relationship_graph(self, figsize=(12, 10), title="实体关系图谱"):
        """
        绘制实体关系图谱
        - 节点: 实体（大小代表属性数量）
        - 边: 关系
        """
        if not self.entities:
            print("无实体数据，无法绘制关系图谱")
            return

        G = nx.Graph()

        # 添加节点
        entity_sizes = {}
        for entity in self.entities:
            entity_id = entity["entity_id"]
            entity_name = entity["entity_name"]
            # 节点大小：属性数量越多越大
            size = 20 + len(entity.get("attributes", [])) * 5
            entity_sizes[entity_id] = size
            G.add_node(entity_id, name=entity_name, type=entity.get("entity_type", "Other"))

        # 添加边
        for rel in self.relationships:
            subj = rel.get("subject_entity_id")
            obj = rel.get("object_entity_id")
            predicate = rel.get("predicate", "关联")
            if subj in G and obj in G:
                G.add_edge(subj, obj, label=predicate)

        if len(G.nodes) == 0:
            print("图谱无节点")
            return

        # 布局
        pos = nx.spring_layout(G, k=2, iterations=50)

        plt.figure(figsize=figsize)

        # 按实体类型分组着色
        type_colors = {
            "Person": "#ff6b6b",
            "Organization": "#4facfe",
            "Location": "#2ed573",
            "Product": "#ff9f43",
            "EventName": "#a29bfe",
            "Other": "#dfe6e9"
        }

        node_colors = []
        for node in G.nodes:
            node_type = G.nodes[node].get("type", "Other")
            node_colors.append(type_colors.get(node_type, "#dfe6e9"))

        nx.draw_networkx_nodes(G, pos, node_size=[entity_sizes.get(n, 20) for n in G.nodes],
                               node_color=node_colors, alpha=0.9)
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5, edge_color="#a4b0be")

        # 标签
        labels = {n: G.nodes[n].get("name", n) for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight="bold")

        # 边标签
        edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

        plt.title(title, fontsize=14, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # ==================== 2. 事件时间线 ====================

    def plot_event_timeline(self, figsize=(14, 6), title="事件时间线"):
        """
        绘制事件时间线
        - 横轴: 时间
        - 纵轴: 事件类型
        """
        if not self.events:
            print("无事件数据，无法绘制时间线")
            return

        # 解析时间
        events_with_time = []
        for event in self.events:
            time_str = event.get("time", "")
            if not time_str:
                continue
            events_with_time.append(event)

        if not events_with_time:
            print("无带时间的事件")
            return

        # 按时间排序
        events_with_time.sort(key=lambda x: x.get("time", ""))

        plt.figure(figsize=figsize)

        y_pos = {}
        y_offset = 0
        event_types = set()

        for i, event in enumerate(events_with_time):
            event_type = event.get("event_type", "事件")
            event_types.add(event_type)
            time_str = event.get("time", "")
            summary = event.get("summary", event_type)[:40]
            y_pos[event["event_id"]] = y_offset
            y_offset += 1

            # 绘制点
            plt.plot(i, y_offset - 1, 'o', markersize=10, color='#ff9f43')

            # 绘制时间线连接
            if i > 0:
                plt.plot([i-1, i], [y_offset-2, y_offset-1], '--', color='#a4b0be', alpha=0.5)

            # 添加标签
            plt.annotate(f"{time_str}: {summary}", xy=(i, y_offset - 1),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, rotation=45, ha='left')

        plt.xlabel("事件序号", fontsize=12)
        plt.ylabel("事件", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xticks(range(len(events_with_time)), [e.get("time", "") for e in events_with_time], rotation=45)
        plt.yticks([])
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ==================== 3. 主题层级树 ====================

    def plot_theme_tree(self, figsize=(10, 8), title="主题层级结构"):
        """
        绘制主题层级树
        """
        if not self.themes:
            print("无主题数据")
            return

        # 构建树结构
        theme_dict = {t["theme_id"]: t for t in self.themes}
        root_themes = [t for t in self.themes if t.get("parent_theme_id") is None]
        children_map = defaultdict(list)
        for t in self.themes:
            parent = t.get("parent_theme_id")
            if parent:
                children_map[parent].append(t)

        # 使用 networkx 绘制树
        G = nx.DiGraph()

        def add_nodes(theme_id, parent_id=None):
            theme = theme_dict.get(theme_id)
            if not theme:
                return
            G.add_node(theme_id, name=theme["theme_name"], size=theme.get("summary", "")[:50])
            if parent_id:
                G.add_edge(parent_id, theme_id)
            for child in children_map.get(theme_id, []):
                add_nodes(child["theme_id"], theme_id)

        for root in root_themes:
            add_nodes(root["theme_id"])

        if len(G.nodes) == 0:
            print("无主题节点")
            return

        plt.figure(figsize=figsize)

        pos = nx.spring_layout(G, k=1, iterations=30)
        # 或者使用层次布局
        # pos = nx.nx_agraph.graphviz_layout(G, prog='dot')  # 需要 pygraphviz

        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color="#a29bfe", alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5, edge_color="#a4b0be", arrows=True)

        labels = {n: G.nodes[n].get("name", n) for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold")

        plt.title(title, fontsize=14, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # ==================== 4. 事件-实体关联图 ====================

    def plot_event_entity_network(self, figsize=(14, 12), title="事件-实体关联网络"):
        """
        绘制事件与实体的关联网络
        - 橙色: 事件节点
        - 蓝色: 实体节点
        """
        if not self.events or not self.entities:
            print("无事件或实体数据")
            return

        G = nx.Graph()

        # 添加事件节点
        for event in self.events:
            event_id = event["event_id"]
            event_summary = event.get("summary", event["event_type"])[:30]
            G.add_node(event_id, name=event_summary, type="event", event_type=event["event_type"])

        # 添加实体节点
        for entity in self.entities:
            entity_id = entity["entity_id"]
            entity_name = entity["entity_name"]
            G.add_node(entity_id, name=entity_name, type="entity", entity_type=entity.get("entity_type", "Other"))

        # 添加事件-实体边
        for event in self.events:
            event_id = event["event_id"]
            for participant in event.get("participants", []):
                entity_id = participant.get("entity_id")
                role = participant.get("role", "参与")
                if entity_id and entity_id in G:
                    G.add_edge(event_id, entity_id, role=role)

        if len(G.nodes) == 0:
            print("图谱无节点")
            return

        plt.figure(figsize=figsize)

        # 节点着色
        node_colors = []
        node_sizes = []
        for node in G.nodes:
            if G.nodes[node].get("type") == "event":
                node_colors.append("#ff9f43")  # 橙色
                node_sizes.append(1500)
            else:
                node_colors.append("#4facfe")  # 蓝色
                node_sizes.append(800)

        pos = nx.spring_layout(G, k=1.5, iterations=50)

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_edges(G, pos, width=1, alpha=0.4, edge_color="#a4b0be")

        labels = {n: G.nodes[n].get("name", n)[:20] for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)

        plt.title(title, fontsize=14, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # ==================== 5. 统计图表 ====================

    def plot_statistics(self, figsize=(12, 8)):
        """绘制统计图表"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. 实体类型分布
        entity_types = Counter([e.get("entity_type", "Other") for e in self.entities])
        if entity_types:
            axes[0, 0].bar(entity_types.keys(), entity_types.values(), color="#4facfe")
            axes[0, 0].set_title("实体类型分布", fontsize=12)
            axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. 事件类型分布
        event_types = Counter([e.get("event_type", "其他") for e in self.events])
        if event_types:
            axes[0, 1].bar(event_types.keys(), event_types.values(), color="#ff9f43")
            axes[0, 1].set_title("事件类型分布", fontsize=12)
            axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. 实体-事件关联数
        entity_event_count = defaultdict(int)
        for event in self.events:
            for p in event.get("participants", []):
                entity_id = p.get("entity_id")
                if entity_id:
                    entity_event_count[entity_id] += 1

        if entity_event_count:
            top_entities = sorted(entity_event_count.items(), key=lambda x: x[1], reverse=True)[:10]
            top_names = [self.entity_map.get(eid, eid)[:10] for eid, _ in top_entities]
            top_counts = [c for _, c in top_entities]
            axes[1, 0].barh(top_names, top_counts, color="#2ed573")
            axes[1, 0].set_title("实体参与事件次数 TOP 10", fontsize=12)

        # 4. 主题关键词词云（简化）
        all_keywords = []
        for theme in self.themes:
            all_keywords.extend(theme.get("keywords", []))
        if all_keywords:
            keyword_counts = Counter(all_keywords).most_common(15)
            words = [w for w, _ in keyword_counts]
            counts = [c for _, c in keyword_counts]
            axes[1, 1].bar(words, counts, color="#a29bfe")
            axes[1, 1].set_title("主题关键词频次", fontsize=12)
            axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    # ==================== 综合展示 ====================

    def show_all(self):
        """展示所有图表"""
        self.plot_statistics()
        self.plot_relationship_graph()
        self.plot_event_timeline()
        self.plot_event_entity_network()
        self.plot_theme_tree()