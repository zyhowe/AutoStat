"""
知识图谱可视化模块 - 适配当前图谱结构
功能：实体关系图、事件时间线、实体演进图、完整图谱
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import re
import os
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class KnowledgeGraphVisualizer:
    """知识图谱可视化器 - 适配当前图谱结构"""

    # 节点颜色映射
    NODE_COLORS = {
        "人物": "#FF6B6B",      # 红色
        "组织": "#4ECDC4",      # 青色
        "地点": "#45B7D1",      # 蓝色
        "产品": "#96CEB4",      # 绿色
        "时间": "#FFEAA7",      # 黄色
        "事件": "#A29BFE",      # 紫色
        "数值": "#FDCB6E",      # 橙色
        "项目": "#FF9F43",      # 橙色
        "公司": "#4ECDC4",      # 青色
        "其他": "#B2BEC3"       # 灰色
    }

    # 边颜色映射
    EDGE_COLORS = {
        "relation": "#74B9FF",        # 实体关系
        "participates_in": "#00CEC9", # 参与关系
        "causal": "#FF7675",          # 因果关系
        "temporal": "#FDCB6E",        # 时序关系
        "same": "#A29BFE"             # 共指关系
    }

    def __init__(self, kg_data: Dict[str, Any]):
        """
        初始化可视化器

        参数:
            kg_data: 知识图谱数据，格式来自 GlobalKnowledgeGraph.to_json()
        """
        self.entities = kg_data.get("entities", {})
        self.events = kg_data.get("events", {})
        self.relations = kg_data.get("relations", {})
        self.participations = kg_data.get("participations", {})
        self.event_relations = kg_data.get("event_relations", {})
        self.statistics = kg_data.get("statistics", {})

    def _parse_event_relation_key(self, key: str) -> Tuple[str, str, str]:
        """解析事件关系key，格式 '事件A-关系类型-事件B'"""
        parts = key.split('-')
        if len(parts) >= 3:
            return parts[0], parts[-1], '-'.join(parts[1:-1])
        return key, key, "unknown"

    def draw_entity_graph(self,
                          figsize: Tuple[int, int] = (14, 10),
                          max_entities: int = 20,
                          layout: str = "spring",
                          save_path: Optional[str] = None):
        """
        绘制实体关系图
        """
        G = nx.Graph()

        # 统计实体重要性（按关系数量）
        entity_importance = {}
        for rel_key, rel_data in self.relations.items():
            # 解析key格式 "主体->关系->客体"
            parts = rel_key.split('->')
            if len(parts) >= 3:
                subject = parts[0]
                obj = parts[-1]
                if subject:
                    entity_importance[subject] = entity_importance.get(subject, 0) + 1
                if obj:
                    entity_importance[obj] = entity_importance.get(obj, 0) + 1

        # 按重要性排序，取前max_entities个
        sorted_entities = sorted(entity_importance.items(), key=lambda x: x[1], reverse=True)
        selected_entities = set([e[0] for e in sorted_entities[:max_entities]])

        if not selected_entities:
            # 如果没有关系，取所有实体
            selected_entities = set(list(self.entities.keys())[:max_entities])

        # 添加节点
        node_colors = []
        node_sizes = []

        for entity_name in selected_entities:
            if entity_name in self.entities:
                entity_type = self.entities[entity_name].get("type", "其他")
                G.add_node(entity_name, type=entity_type)
                node_colors.append(self.NODE_COLORS.get(entity_type, "#B2BEC3"))
                size = 2000 + entity_importance.get(entity_name, 0) * 200
                node_sizes.append(min(size, 5000))
            else:
                G.add_node(entity_name)
                node_colors.append("#B2BEC3")
                node_sizes.append(2000)

        # 添加边
        edge_labels = {}
        for rel_key, rel_data in self.relations.items():
            parts = rel_key.split('->')
            if len(parts) >= 3:
                subject = parts[0]
                predicate = parts[1]
                obj = parts[2]

                if subject in selected_entities and obj in selected_entities:
                    G.add_edge(subject, obj)
                    edge_labels[(subject, obj)] = predicate

        if len(G.nodes) == 0:
            print("实体关系图：没有足够的节点")
            return None

        # 布局
        if layout == "spring":
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.random_layout(G, seed=42)

        fig, ax = plt.subplots(figsize=figsize)

        # 绘制边
        nx.draw_networkx_edges(G, pos, width=2, edge_color="#74B9FF", alpha=0.7,
                               connectionstyle="arc3,rad=0.1", ax=ax)

        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                               alpha=0.9, ax=ax)

        # 节点标签
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

        # 边标签
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

        # 图例
        legend_patches = []
        used_types = set()
        for entity_name in selected_entities:
            if entity_name in self.entities:
                etype = self.entities[entity_name].get("type", "其他")
                if etype not in used_types:
                    used_types.add(etype)
                    legend_patches.append(mpatches.Patch(color=self.NODE_COLORS.get(etype, "#B2BEC3"), label=etype))

        if legend_patches:
            ax.legend(handles=legend_patches, loc='upper right', fontsize=10)

        ax.set_title("实体关系图", fontsize=16, fontweight='bold')
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"实体关系图已保存到 {save_path}")

        plt.tight_layout()
        return fig

    def draw_event_timeline(self,
                            figsize: Tuple[int, int] = (14, 8),
                            save_path: Optional[str] = None):
        """
        绘制事件时间线图
        """
        # 收集所有带时间的事件
        timeline_events = []

        for event_id, event_data in self.events.items():
            summary = event_data.get("summary", "")
            time_data = event_data.get("time", {})

            # 兼容两种格式
            if isinstance(time_data, dict):
                time_str = time_data.get("value", "")
                is_estimated = time_data.get("is_estimated", False)
            else:
                time_str = str(time_data) if time_data else ""
                is_estimated = False

            location = event_data.get("location", "")

            if time_str:
                timeline_events.append({
                    "event_id": event_id,
                    "summary": summary,
                    "time": time_str,
                    "location": location,
                    "is_estimated": is_estimated
                })

        if not timeline_events:
            print("时间线：没有带时间的事件")
            return None

        # 按时间排序
        def parse_time(t):
            if not t:
                return datetime.min
            if re.match(r'\d{4}-\d{2}-\d{2}', t):
                try:
                    return datetime.strptime(t, "%Y-%m-%d")
                except:
                    pass
            if re.match(r'\d{4}-\d{2}', t):
                try:
                    return datetime.strptime(t, "%Y-%m")
                except:
                    pass
            if re.match(r'\d{4}', t):
                try:
                    return datetime.strptime(t[:4], "%Y")
                except:
                    pass
            return datetime.min

        timeline_events.sort(key=lambda x: parse_time(x["time"]))

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#F8F9FA')

        for i, event in enumerate(timeline_events):
            y_pos = i * 0.8

            # 绘制时间点
            color = "#A29BFE"
            if event.get("is_estimated"):
                color = "#FDCB6E"  # 橙色表示预计时间

            ax.plot(0, y_pos, 'o', markersize=12, color=color, zorder=3)

            # 绘制时间线
            ax.plot([-0.2, 0.2], [y_pos, y_pos], '-', color=color, linewidth=2, alpha=0.5)

            # 时间标签
            time_label = event["time"]
            if event.get("is_estimated"):
                time_label += " (预计)"
            ax.text(-0.8, y_pos, time_label, fontsize=10, ha='right', va='center',
                   fontweight='bold', color='#2C3E50')

            # 事件描述
            summary = event["summary"]
            if len(summary) > 50:
                summary = summary[:47] + "..."
            ax.text(0.3, y_pos, summary, fontsize=11, ha='left', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            # 地点标签
            if event.get("location"):
                ax.text(0.3, y_pos - 0.25, f"📍 {event['location']}", fontsize=8,
                       ha='left', va='center', color='#7F8C8D')

        ax.set_xlim(-1, 8)
        ax.set_ylim(-1, len(timeline_events) * 0.8)
        ax.axis('off')

        ax.set_title("事件时间线", fontsize=16, fontweight='bold', pad=20)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"事件时间线已保存到 {save_path}")

        plt.tight_layout()
        return fig

    def draw_entity_evolution(self,
                              entity_name: str,
                              figsize: Tuple[int, int] = (12, 6),
                              save_path: Optional[str] = None):
        """
        绘制单个实体的属性演进图
        """
        if entity_name not in self.entities:
            print(f"实体 '{entity_name}' 不存在")
            return None

        entity_data = self.entities[entity_name]
        attributes = entity_data.get("attributes", {})

        if not attributes:
            print(f"实体 '{entity_name}' 没有属性")
            return None

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#F8F9FA')

        y_pos = 0
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FDCB6E', '#A29BFE']
        color_idx = 0

        for attr_key, attr_values in attributes.items():
            if not attr_values:
                continue

            color = colors[color_idx % len(colors)]

            # 属性名称
            ax.text(-0.5, y_pos, attr_key, fontsize=11, ha='right', va='center',
                   fontweight='bold', color='#2C3E50')

            # 处理属性值（可能是列表）
            if isinstance(attr_values, list):
                value_str = " | ".join([v.get("value", str(v)) for v in attr_values])
            else:
                value_str = str(attr_values)

            ax.text(0, y_pos, f"= {value_str}", fontsize=11, ha='left', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))

            y_pos += 0.8
            color_idx += 1

        ax.set_xlim(-1, 2)
        ax.set_ylim(-0.5, y_pos + 0.5)
        ax.axis('off')

        ax.set_title(f"实体属性: {entity_name}", fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"实体属性图已保存到 {save_path}")

        plt.tight_layout()
        return fig

    def draw_participation_graph(self,
                                 figsize: Tuple[int, int] = (16, 12),
                                 max_nodes: int = 30,
                                 save_path: Optional[str] = None):
        """
        绘制实体-事件参与关系图
        """
        G = nx.Graph()

        # 解析参与关系
        entities_in_participation = set()
        events_in_participation = set()
        participation_edges = []

        for part_key, part_data in self.participations.items():
            # key格式 "实体-角色-事件"
            parts = part_key.split('-')
            if len(parts) >= 3:
                entity = parts[0]
                role = parts[1]
                event_key = parts[2]

                entities_in_participation.add(entity)
                events_in_participation.add(event_key)
                participation_edges.append((entity, event_key, role))

        if not entities_in_participation:
            print("参与关系图：没有参与关系")
            return None

        # 限制节点数
        entity_list = list(entities_in_participation)[:max_nodes//2]
        event_list = list(events_in_participation)[:max_nodes//2]

        # 添加实体节点
        for entity_name in entity_list:
            if entity_name in self.entities:
                entity_type = self.entities[entity_name].get("type", "其他")
                G.add_node(entity_name, node_type="entity", entity_type=entity_type)
            else:
                G.add_node(entity_name, node_type="entity", entity_type="其他")

        # 添加事件节点
        for event_key in event_list:
            if event_key in self.events:
                summary = self.events[event_key].get("summary", event_key)
            else:
                summary = event_key
            if len(summary) > 25:
                summary = summary[:22] + "..."
            G.add_node(summary, node_type="event", event_id=event_key)

        # 添加参与关系边
        edge_labels = {}
        for entity, event_key, role in participation_edges:
            if entity in G:
                if event_key in self.events:
                    summary = self.events[event_key].get("summary", event_key)
                else:
                    summary = event_key
                if len(summary) > 25:
                    summary = summary[:22] + "..."
                if summary in G:
                    G.add_edge(entity, summary)
                    edge_labels[(entity, summary)] = role

        if len(G.nodes) == 0:
            print("参与关系图：没有足够的节点")
            return None

        # 布局
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#F8F9FA')

        # 分组节点
        entity_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "entity"]
        event_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "event"]

        # 绘制边
        nx.draw_networkx_edges(G, pos, width=2, edge_color="#00CEC9", alpha=0.6,
                               connectionstyle="arc3,rad=0.1", ax=ax)

        # 绘制实体节点
        if entity_nodes:
            node_colors = [self.NODE_COLORS.get(G.nodes[n].get("entity_type", "其他"), "#B2BEC3")
                          for n in entity_nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=entity_nodes,
                                  node_color=node_colors, node_size=3000, alpha=0.9, ax=ax)

        # 绘制事件节点
        if event_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=event_nodes,
                                  node_color=self.NODE_COLORS["事件"], node_size=2500, alpha=0.9, ax=ax)

        # 节点标签
        labels = {n: n for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold', ax=ax)

        # 边标签
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

        # 图例
        used_types = set()
        legend_patches = []
        for node in entity_nodes:
            etype = G.nodes[node].get("entity_type", "其他")
            if etype not in used_types:
                used_types.add(etype)
                legend_patches.append(mpatches.Patch(color=self.NODE_COLORS.get(etype, "#B2BEC3"), label=etype))

        legend_patches.append(mpatches.Patch(color=self.NODE_COLORS["事件"], label="事件"))
        legend_patches.append(mpatches.Patch(color="#00CEC9", label="参与关系"))

        ax.legend(handles=legend_patches, loc='upper right', fontsize=9)

        ax.set_title("实体-事件参与关系图", fontsize=16, fontweight='bold')
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"参与关系图已保存到 {save_path}")

        plt.tight_layout()
        return fig

    def draw_full_graph(self,
                        figsize: Tuple[int, int] = (18, 14),
                        max_nodes: int = 35,
                        save_path: Optional[str] = None):
        """
        绘制完整知识图谱（实体+事件+所有关系）
        """
        G = nx.Graph()

        # 统计实体重要性
        entity_importance = {}
        for rel_key in self.relations.keys():
            parts = rel_key.split('->')
            if len(parts) >= 3:
                subject = parts[0]
                obj = parts[-1]
                if subject:
                    entity_importance[subject] = entity_importance.get(subject, 0) + 1
                if obj:
                    entity_importance[obj] = entity_importance.get(obj, 0) + 1

        # 选择重要实体
        sorted_entities = sorted(entity_importance.items(), key=lambda x: x[1], reverse=True)
        selected_entities = set([e[0] for e in sorted_entities[:max_nodes//3]])

        # 添加实体节点
        for entity_name in selected_entities:
            if entity_name in self.entities:
                entity_type = self.entities[entity_name].get("type", "其他")
                G.add_node(entity_name, node_type="entity", entity_type=entity_type)

        # 添加事件节点
        events_in_participation = set()
        for part_key in self.participations.keys():
            parts = part_key.split('-')
            if len(parts) >= 3:
                events_in_participation.add(parts[-1])

        for event_key in list(events_in_participation)[:max_nodes//3]:
            if event_key in self.events:
                summary = self.events[event_key].get("summary", event_key)
            else:
                summary = event_key
            if len(summary) > 20:
                summary = summary[:17] + "..."
            G.add_node(summary, node_type="event", event_id=event_key)

        # 添加实体间关系边
        for rel_key, rel_data in self.relations.items():
            parts = rel_key.split('->')
            if len(parts) >= 3:
                subject = parts[0]
                predicate = parts[1]
                obj = parts[2]

                if subject in G and obj in G:
                    G.add_edge(subject, obj, label=predicate, edge_type="relation")

        # 添加参与关系边
        for part_key, part_data in self.participations.items():
            parts = part_key.split('-')
            if len(parts) >= 3:
                entity = parts[0]
                role = parts[1]
                event_key = parts[2]

                if entity in G:
                    if event_key in self.events:
                        summary = self.events[event_key].get("summary", event_key)
                    else:
                        summary = event_key
                    if len(summary) > 20:
                        summary = summary[:17] + "..."
                    if summary in G:
                        G.add_edge(entity, summary, label=role, edge_type="participates_in")

        if len(G.nodes) == 0:
            print("完整图谱：没有足够的节点")
            return None

        # 布局
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#F8F9FA')

        # 分组节点
        entity_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "entity"]
        event_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "event"]

        # 绘制边
        for edge in G.edges(data=True):
            source, target, data = edge
            edge_type = data.get("edge_type", "relation")
            color = self.EDGE_COLORS.get(edge_type, "#74B9FF")

            nx.draw_networkx_edges(G, pos, edgelist=[(source, target)],
                                   width=2, edge_color=color, alpha=0.5,
                                   connectionstyle="arc3,rad=0.1", ax=ax)

        # 绘制实体节点
        if entity_nodes:
            node_colors = [self.NODE_COLORS.get(G.nodes[n].get("entity_type", "其他"), "#B2BEC3")
                          for n in entity_nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=entity_nodes,
                                  node_color=node_colors, node_size=3000, alpha=0.9, ax=ax)

        # 绘制事件节点
        if event_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=event_nodes,
                                  node_color=self.NODE_COLORS["事件"], node_size=2500, alpha=0.9, ax=ax)

        # 节点标签
        labels = {n: n for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold', ax=ax)

        # 边标签（只显示部分）
        edge_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True) if d.get("label")}
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax)

        # 图例
        used_types = set()
        legend_patches = []
        for node in entity_nodes:
            etype = G.nodes[node].get("entity_type", "其他")
            if etype not in used_types:
                used_types.add(etype)
                legend_patches.append(mpatches.Patch(color=self.NODE_COLORS.get(etype, "#B2BEC3"), label=etype))

        legend_patches.append(mpatches.Patch(color=self.NODE_COLORS["事件"], label="事件"))
        legend_patches.append(mpatches.Patch(color=self.EDGE_COLORS["relation"], label="实体关系"))
        legend_patches.append(mpatches.Patch(color=self.EDGE_COLORS["participates_in"], label="参与关系"))

        ax.legend(handles=legend_patches, loc='upper right', fontsize=9)

        ax.set_title("完整知识图谱", fontsize=16, fontweight='bold')
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"完整知识图谱已保存到 {save_path}")

        plt.tight_layout()
        return fig

    def generate_all_views(self, output_dir: str = ".", entity_for_evolution: str = None):
        """
        生成所有可视化视图
        """
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print("生成知识图谱可视化视图")
        print("=" * 60)

        # 1. 实体关系图
        print("\n[1/5] 生成实体关系图...")
        self.draw_entity_graph(save_path=os.path.join(output_dir, "01_entity_graph.png"))

        # 2. 事件时间线
        print("\n[2/5] 生成事件时间线...")
        self.draw_event_timeline(save_path=os.path.join(output_dir, "02_event_timeline.png"))

        # 3. 实体属性图
        if entity_for_evolution and entity_for_evolution in self.entities:
            print(f"\n[3/5] 生成实体属性图: {entity_for_evolution}...")
            self.draw_entity_evolution(entity_for_evolution,
                                       save_path=os.path.join(output_dir, f"03_entity_{entity_for_evolution}.png"))
        else:
            # 找第一个有属性的实体
            for name, data in self.entities.items():
                if data.get("attributes"):
                    print(f"\n[3/5] 生成实体属性图: {name}...")
                    self.draw_entity_evolution(name,
                                               save_path=os.path.join(output_dir, f"03_entity_{name}.png"))
                    break
            else:
                print("\n[3/5] 跳过：没有实体属性")

        # 4. 参与关系图
        print("\n[4/5] 生成实体-事件参与关系图...")
        self.draw_participation_graph(save_path=os.path.join(output_dir, "04_participation_graph.png"))

        # 5. 完整图谱
        print("\n[5/5] 生成完整知识图谱...")
        self.draw_full_graph(save_path=os.path.join(output_dir, "05_full_graph.png"))

        print(f"\n所有图表已保存到 {output_dir}/")


def visualize_from_json(json_path: str, output_dir: str = ".", entity_for_evolution: str = None):
    """
    从JSON文件加载知识图谱并生成可视化
    """
    with open(json_path, "r", encoding="utf-8") as f:
        kg_data = json.load(f)

    visualizer = KnowledgeGraphVisualizer(kg_data)
    visualizer.generate_all_views(output_dir, entity_for_evolution)


if __name__ == "__main__":
    if os.path.exists("knowledge_graph_cleaned.json"):
        visualize_from_json("knowledge_graph_cleaned.json", output_dir=".", entity_for_evolution="宁德时代")
    else:
        print("请先运行信息抽取脚本，生成 knowledge_graph.json 文件")