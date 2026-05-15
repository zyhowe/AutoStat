# ==================== autotext/reporter.py ====================
"""
文本分析报告生成器 - 生成 HTML/JSON/Markdown 报告
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from jinja2 import Template


class TextReporter:
    """文本分析报告生成器"""

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def to_html(self, output_file: Optional[str] = None, title: str = "文本分析报告") -> str:
        try:
            html = self._build_html(title)
            if output_file:
                os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(html)
                print(f"✅ HTML报告已保存到 {output_file}")
            return html
        except Exception as e:
            error_html = self._get_error_html(title, f"报告生成失败: {str(e)}")
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(error_html)
            return error_html

    def to_json(self, output_file: Optional[str] = None, indent: int = 2) -> str:
        report_data = self._build_json_data()
        report_data["analysis_time"] = datetime.now().isoformat()
        report_data["source"] = getattr(self.analyzer, "source_name", "未知")

        json_str = json.dumps(report_data, ensure_ascii=False, indent=indent, default=str)

        if output_file:
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_str)
            print(f"✅ JSON报告已保存到 {output_file}")

        return json_str

    def to_markdown(self, output_file: Optional[str] = None) -> str:
        data = self._build_json_data()

        md = f"""# 文本分析报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 数据概览

| 指标 | 数值 |
|------|------|
| 总文本数 | {data['stats']['total_count']} |
| 空文本数 | {data['stats']['empty_count']} |
| 空文本率 | {data['stats']['empty_rate']:.1%} |
| 平均长度 | {data['stats']['char_length']['mean']:.1f} 字符 |
| 重复文本对 | {data['quality'].get('duplicates', {}).get('count', 0)} |

## 📈 情感分布

- 积极: {data['sentiment']['distribution']['positive_rate']:.1%}
- 消极: {data['sentiment']['distribution']['negative_rate']:.1%}
- 中性: {data['sentiment']['distribution']['neutral_rate']:.1%}

## 🔑 高频关键词 TOP 30

"""
        for word, count in data['keywords']['frequency'][:30]:
            md += f"- {word}: {count}\n"

        # 大模型实体统计
        entity_stats_by_type = data.get('entity_stats_by_type', {})
        if entity_stats_by_type:
            md += "\n## 🤖 大模型实体识别\n"
            for type_name, stats in entity_stats_by_type.items():
                md += f"\n### {type_name} ({stats.get('unique', 0)}个)\n"
                for item in stats.get('top', [])[:10]:
                    md += f"- {item.get('name')}: {item.get('count')}次\n"

        # 实体列表表格
        llm_extraction = data.get('llm_extraction', {})
        entities = llm_extraction.get('entities', [])
        if entities:
            md += "\n## 📋 实体与属性总览\n\n"
            md += "| 实体ID | 实体名称 | 实体类型 | 关键属性 | 关联实体 | 参与事件 | 归属主题 | 原文出处 |\n"
            md += "|--------|----------|----------|----------|----------|----------|----------|----------|\n"
            for entity in entities[:30]:
                entity_id = entity.get('entity_id', '')
                entity_name = entity.get('entity_name', '')
                entity_type = entity.get('entity_type', '')
                attrs = []
                for attr in entity.get('attributes', [])[:3]:
                    key = attr.get('attr_name') or attr.get('attr_key', '')
                    value = attr.get('attr_value', '')
                    attrs.append(f"{key}:{value}")
                attr_str = '; '.join(attrs) if attrs else '—'
                evidence = entity.get('evidence', '')[:50]
                md += f"| {entity_id} | {entity_name} | {entity_type} | {attr_str} | — | — | — | {evidence} |\n"

        # 事件一览表格
        events = llm_extraction.get('events', [])
        if events:
            md += "\n## 📰 事件一览\n\n"
            md += "| 事件ID | 事件类型 | 触发词 | 时间 | 参与者 | 事件摘要 |\n"
            md += "|--------|----------|--------|------|--------|----------|\n"
            for event in events[:30]:
                event_id = event.get('event_id', '')
                event_type = event.get('event_type', '')
                trigger = event.get('trigger_word', '—')
                time = event.get('time', '—')
                participants = []
                for p in event.get('participants', [])[:2]:
                    participants.append(p.get('entity_id', ''))
                participant_str = ', '.join(participants) if participants else '—'
                summary = event.get('summary', '')[:60]
                md += f"| {event_id} | {event_type} | {trigger} | {time} | {participant_str} | {summary} |\n"

        # 主题分析
        topics = data.get('topics', [])
        if topics:
            md += "\n## 📚 主题分析\n"
            for topic in topics:
                title = topic.get('llm_title', f'主题{topic.get("topic_id", 0)}')
                md += f"\n### {title}\n"
                if topic.get('llm_summary'):
                    md += f"{topic['llm_summary']}\n\n"
                kw_list = ', '.join(topic.get('keywords', [])[:8])
                md += f"**关键词:** {kw_list}\n"
                md += f"**文本数量:** {topic.get('texts_count', 0)} 条\n"

        # 清洗建议
        if data.get('cleaning_suggestions'):
            md += "\n## 🧹 数据清洗建议\n"
            for s in data['cleaning_suggestions'][:5]:
                md += f"- {s}\n"

        if output_file:
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(md)
            print(f"✅ Markdown报告已保存到 {output_file}")

        return md

    def _build_tree_from_global_graph(self, static_graph: Dict) -> Dict:
        """
        从静态图谱构建树形数据结构
        只使用静态关系（包含、属于、组成），从根节点向下展开
        """
        if not static_graph or not static_graph.get('nodes'):
            return {}

        # 获取实体节点
        entity_nodes = [n for n in static_graph['nodes'] if n.get('category') == 'entity']
        if not entity_nodes:
            return {}

        # 创建节点映射
        node_map = {n['id']: n for n in entity_nodes}

        # 使用所有实体-实体关系边
        all_edges = static_graph.get('links', [])
        relation_edges = [e for e in all_edges if e.get('type') == 'relation']

        # 构建邻接表
        children_map = {}
        in_degree = {}

        for edge in relation_edges:
            source = edge.get('source')
            target = edge.get('target')
            label = edge.get('label', '关联')

            if source not in node_map or target not in node_map:
                continue

            if source not in children_map:
                children_map[source] = []
            children_map[source].append((target, label))

            in_degree[target] = in_degree.get(target, 0) + 1
            if source not in in_degree:
                in_degree[source] = in_degree.get(source, 0)

        # 找出根节点（入度为0的节点）
        root_candidates = [node_id for node_id in node_map.keys() if in_degree.get(node_id, 0) == 0]

        if not root_candidates:
            return {}

        visited = set()

        def build_tree_node(node_id: str, depth: int = 0, max_depth: int = 8, max_children: int = 50):
            """递归构建树节点"""
            if depth > max_depth:
                return None
            if node_id in visited:
                return None
            if node_id not in node_map:
                return None

            visited.add(node_id)

            node = node_map[node_id]
            node_name = node.get('name', node_id)
            if len(node_name) > 30:
                node_name = node_name[:27] + '...'

            tree_node = {
                'id': node_id,
                'name': node_name,
                'value': node.get('value', 1),
                'fact': node.get('fact', ''),
                'type': node.get('type', 'OTHER'),
                'children': []
            }

            # 获取子节点
            children = children_map.get(node_id, [])
            children = children[:max_children]

            for child_id, relation_label in children:
                if child_id in visited:
                    continue
                child_node = build_tree_node(child_id, depth + 1, max_depth, max_children)
                if child_node:
                    child_node['relation'] = relation_label[:15] + '...' if len(relation_label) > 15 else relation_label
                    tree_node['children'].append(child_node)

            return tree_node

        # 构建森林
        forest = []
        for root_id in root_candidates:
            visited.clear()
            tree = build_tree_node(root_id, max_depth=6, max_children=50)
            if tree and tree.get('children'):
                forest.append(tree)

        if not forest:
            return {}

        # 创建虚拟根节点
        virtual_root = {
            'id': 'ROOT',
            'name': '实体关系图',
            'value': 100,
            'fact': f'共 {len(entity_nodes)} 个实体',
            'children': forest
        }

        return virtual_root

    def _build_highlight_data(self, full_text: str, entities: List[Dict]) -> List[Dict]:
        """
        构建高亮数据：从 entity_name 中提取原文中的实际词汇，用 evidence 定位
        """
        highlight_items = []

        for entity in entities:
            entity_name = entity.get('entity_name', '')
            evidence = entity.get('evidence', '')
            entity_type = entity.get('entity_type', 'Other')

            if not entity_name or not evidence:
                continue

            # 1. 从 entity_name 中提取实际出现在 evidence 中的文本
            matched_text = ''
            if entity_name in evidence:
                matched_text = entity_name
            elif len(entity_name) > 1 and entity_name[:-1] in evidence:
                matched_text = entity_name[:-1]
            elif len(entity_name) > 2 and entity_name[:-2] in evidence:
                matched_text = entity_name[:-2]
            elif len(entity_name) > 3 and entity_name[:-3] in evidence:
                matched_text = entity_name[:-3]

            if not matched_text:
                continue

            # 2. 用 evidence 在 full_text 中定位
            search_evidence = evidence
            pos = full_text.find(search_evidence)

            # 如果找不到，去掉省略号及后面内容
            if pos == -1 and '...' in search_evidence:
                search_evidence = search_evidence.split('...')[0]
                pos = full_text.find(search_evidence)

            # 如果还找不到，逐次去掉末尾字符
            if pos == -1:
                for remove_len in range(1, 10):
                    if len(search_evidence) > remove_len:
                        test_text = search_evidence[:-remove_len]
                        pos = full_text.find(test_text)
                        if pos >= 0:
                            search_evidence = test_text
                            break

            if pos >= 0:
                highlight_items.append({
                    'text': matched_text,
                    'evidence': search_evidence,
                    'position': pos,
                    'entity_id': entity.get('entity_id'),
                    'entity_name': entity_name,
                    'entity_type': entity_type
                })

        # 去重并按位置排序
        seen = set()
        unique_items = []
        for item in highlight_items:
            key = f"{item['text']}_{item['position']}"
            if key not in seen:
                seen.add(key)
                unique_items.append(item)

        unique_items.sort(key=lambda x: x['position'])
        return unique_items

    def _generate_entity_table_data(self, entities: List[Dict], entity_profiles: List[Dict]) -> List[Dict]:
        """生成实体表格数据（合并实体档案详情）"""
        profile_map = {}
        for profile in entity_profiles:
            profile_map[profile.get('name')] = profile

        table_data = []
        for entity in entities:
            entity_name = entity.get('entity_name', '')
            profile = profile_map.get(entity_name, {})

            attrs = []
            for attr in entity.get('attributes', [])[:3]:
                key = attr.get('attr_name') or attr.get('attr_key', '')
                value = attr.get('attr_value', '')
                attrs.append(f"{key}:{value}")
            attr_str = '; '.join(attrs) if attrs else '—'

            related_entities = []
            for rel in profile.get('related_entities', [])[:3]:
                related_entities.append(f"{rel.get('name')}({rel.get('relation', '关联')})")
            related_str = ', '.join(related_entities) if related_entities else '—'

            participated_events = []
            for evt in profile.get('participated_events', [])[:2]:
                participated_events.append(evt.get('summary', '')[:30])
            event_str = '; '.join(participated_events) if participated_events else '—'

            themes = []
            for theme in profile.get('belong_to_themes', [])[:2]:
                themes.append(theme.get('name', ''))
            theme_str = ', '.join(themes) if themes else '—'

            table_data.append({
                'entity_id': entity.get('entity_id', ''),
                'entity_name': entity_name,
                'entity_type': entity.get('entity_type', ''),
                'key_attributes': attr_str,
                'related_entities': related_str,
                'participated_events': event_str,
                'belong_to_themes': theme_str,
                'evidence': entity.get('evidence', '')[:60]
            })

        return table_data

    def _build_json_data(self) -> Dict[str, Any]:
        """构建完整的 JSON 数据（优先使用大模型数据）"""

        keywords_frequency = []
        for item in self.analyzer.keywords.get("frequency", [])[:50]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                keywords_frequency.append([item[0], item[1]])

        has_llm = hasattr(self.analyzer, '_extracted_entities') and self.analyzer._extracted_entities

        if has_llm and hasattr(self.analyzer, 'entity_stats') and self.analyzer.entity_stats:
            entity_stats = {
                "org": {"unique": 0, "top": []},
                "per": {"unique": 0, "top": []},
                "loc": {"unique": 0, "top": []}
            }
            for key in ["org", "per", "loc"]:
                if key in self.analyzer.entity_stats:
                    stats = self.analyzer.entity_stats[key]
                    entity_stats[key] = {
                        "unique": stats.get("unique", 0),
                        "top": stats.get("top", [])
                    }
        else:
            entity_stats = {
                "org": {"unique": 0, "top": []},
                "per": {"unique": 0, "top": []},
                "loc": {"unique": 0, "top": []}
            }

        if has_llm and hasattr(self.analyzer, 'events_by_type') and self.analyzer.events_by_type:
            events_by_type = self.analyzer.events_by_type
        else:
            events_by_type = {}

        if has_llm and hasattr(self.analyzer, 'topics') and self.analyzer.topics:
            topics = []
            for topic in self.analyzer.topics:
                topics.append({
                    "topic_id": topic.get("topic_id", 0),
                    "texts_count": topic.get("texts_count", 0),
                    "keywords": topic.get("keywords", [])[:10],
                    "llm_title": topic.get("llm_title", f"主题{topic.get('topic_id', 0)}"),
                    "llm_summary": topic.get("llm_summary", ""),
                    "textrank_sentences": topic.get("textrank_sentences", [])[:2]
                })
        else:
            topics = []

        quality_data = self.analyzer.quality_report
        cleaning_suggestions = self.analyzer.cleaning_suggestions
        duplicates_detail = quality_data.get('duplicates', {})
        similarity_detail = quality_data.get('similarity', {})
        length_detail = quality_data.get('length_anomalies', {})

        llm_extraction = {}
        entity_stats_by_type = {}
        relation_network = {"nodes": [], "links": []}
        event_timeline = {"events": [], "timeline_data": []}
        theme_hierarchy = {"roots": [], "total_themes": 0}
        entity_profiles = []
        event_chains = []
        global_graph = {"nodes": [], "links": [], "statistics": {}}
        static_graph = {"nodes": [], "links": [], "statistics": {}}
        llm_statistics = {"entity_count": 0, "relation_count": 0, "event_count": 0, "theme_count": 0}
        relation_tree_data = {}
        highlight_data = []

        if has_llm:
            type_names = {
                'per': '人物', 'org': '组织', 'loc': '地点',
                'product': '产品', 'event': '事件', 'metric': '指标',
                'industry': '工业', 'tech': '技术', 'time': '时间',
                'number': '数值', 'other': '其他'
            }

            if hasattr(self.analyzer, 'entity_stats'):
                for etype, stats in self.analyzer.entity_stats.items():
                    display_name = type_names.get(etype, etype)
                    entity_stats_by_type[display_name] = {
                        "total": stats.get("total", 0),
                        "unique": stats.get("unique", 0),
                        "top": [{"name": name, "count": count} for name, count in stats.get("top", [])[:20]]
                    }

            if hasattr(self.analyzer, 'relation_result'):
                relation_pairs = self.analyzer.relation_result.get("cooccurrence_pairs", [])
                nodes_set = set()
                for pair in relation_pairs:
                    e1 = pair.get("entity1", "")
                    e2 = pair.get("entity2", "")
                    if e1:
                        nodes_set.add(e1)
                    if e2:
                        nodes_set.add(e2)

                for node_name in nodes_set:
                    relation_network["nodes"].append({
                        "id": node_name,
                        "name": node_name,
                        "category": "entity",
                        "value": 1
                    })

                for pair in relation_pairs:
                    relation_network["links"].append({
                        "source": pair.get("entity1", ""),
                        "target": pair.get("entity2", ""),
                        "label": pair.get("predicate", "关联"),
                        "value": pair.get("count", 1)
                    })

            events = getattr(self.analyzer, 'events', [])
            for event in events:
                timestamp = event.get("timestamp")
                if timestamp:
                    event_timeline["events"].append({
                        "id": event.get("event_id"),
                        "name": event.get("event_type"),
                        "description": event.get("description", ""),
                        "time": timestamp
                    })

            event_timeline["events"].sort(key=lambda x: x.get("time", ""))
            timeline_dict = {}
            for event in event_timeline["events"]:
                date = event.get("time", "")[:10]
                if date:
                    timeline_dict[date] = timeline_dict.get(date, 0) + 1
            event_timeline["timeline_data"] = [
                {"date": date, "count": count}
                for date, count in sorted(timeline_dict.items())
            ]

            if hasattr(self.analyzer, 'theme_hierarchy'):
                theme_hierarchy = self.analyzer.theme_hierarchy

            if hasattr(self.analyzer, 'entity_profiles'):
                entity_profiles = self.analyzer.entity_profiles

            if hasattr(self.analyzer, 'event_chains'):
                event_chains = self.analyzer.event_chains

            if hasattr(self.analyzer, 'global_graph'):
                global_graph = self.analyzer.global_graph

            if hasattr(self.analyzer, 'static_graph'):
                static_graph = self.analyzer.static_graph

            if hasattr(self.analyzer, 'llm_statistics'):
                llm_statistics = self.analyzer.llm_statistics

            llm_extraction = {
                "entities": getattr(self.analyzer, '_extracted_entities', []),
                "relationships": getattr(self.analyzer, '_extracted_relationships', []),
                "events": getattr(self.analyzer, '_extracted_events', []),
                "themes": getattr(self.analyzer, '_extracted_themes', []),
                "categorization": getattr(self.analyzer, '_categorization', {})
            }

            # 构建树形关系图（使用静态图谱）
            if static_graph and static_graph.get('nodes'):
                relation_tree_data = self._build_tree_from_global_graph(static_graph)

            # 构建高亮数据
            sample_texts = self.analyzer.raw_texts if hasattr(self.analyzer,
                                                              'raw_texts') else self.analyzer.content_texts
            if sample_texts:
                full_text = sample_texts[0] if sample_texts else ''
                highlight_data = self._build_highlight_data(full_text, llm_extraction.get('entities', []))

        entity_table_data = []
        if has_llm:
            entity_table_data = self._generate_entity_table_data(
                llm_extraction.get('entities', []),
                entity_profiles
            )

        data = {
            'stats': self.analyzer.stats_result,
            'language_distribution': getattr(self.analyzer, 'language_distribution', {}),
            'quality': quality_data,
            'cleaning_suggestions': cleaning_suggestions[:10],
            'keywords': {'frequency': keywords_frequency, 'tfidf': []},
            'sentiment': {
                'distribution': self.analyzer.sentiment_distribution,
                'is_polarized': self.analyzer.sentiment_distribution.get('positive_rate', 0) > 0.4 or
                                self.analyzer.sentiment_distribution.get('negative_rate', 0) > 0.2
            },
            'templates': {
                'start': list(self.analyzer.start_templates)[:20],
                'end': list(self.analyzer.end_templates)[:20]
            },
            'insights': getattr(self.analyzer, 'insights', [])[:15],
            'sample_texts': getattr(self.analyzer, 'raw_texts', self.analyzer.content_texts)[:10],
            'quality_details': {
                'duplicates': {
                    'count': duplicates_detail.get('count', 0),
                    'rate': duplicates_detail.get('rate', 0),
                    'duplicate_summary': duplicates_detail.get('duplicate_summary', [])
                },
                'similarity': {
                    'count': len(similarity_detail.get('high_similarity_pairs', [])),
                    'similarity_summary': similarity_detail.get('similarity_summary', [])
                },
                'length': {
                    'short_count': len(length_detail.get('short', [])),
                    'long_count': len(length_detail.get('long', [])),
                    'short_summary': length_detail.get('short_summary', []),
                    'long_summary': length_detail.get('long_summary', [])
                }
            },
            'llm_extraction': llm_extraction,
            'entity_stats_by_type': entity_stats_by_type,
            'relation_network': relation_network,
            'event_timeline': event_timeline,
            'theme_hierarchy': theme_hierarchy,
            'entity_profiles': entity_profiles,
            'event_chains': event_chains,
            'global_graph': global_graph,
            'static_graph': static_graph,
            'llm_statistics': llm_statistics,
            'has_llm_data': has_llm,
            'relation_tree_data': relation_tree_data,
            'entity_table_data': entity_table_data,
            'highlight_data': highlight_data
        }

        return data

    def _get_error_html(self, title: str, error_msg: str) -> str:
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #d32f2f; border-bottom: 3px solid #d32f2f; }}
        .error {{ background: #ffebee; padding: 15px; border-radius: 8px; margin: 20px 0; }}
    </style>
</head>
<body>
<div class="container">
    <h1>❌ 报告生成失败</h1>
    <div class="error">
        <p><strong>错误信息：</strong></p>
        <p>{error_msg}</p>
    </div>
</div>
</body>
</html>"""

    def _build_html(self, title: str) -> str:
        data = self._build_json_data()

        template_path = os.path.join(os.path.dirname(__file__), '..', 'templates', 'report_text.html')
        if not os.path.exists(template_path):
            template_path = os.path.join(os.path.dirname(__file__), 'templates', 'report_text.html')
        if not os.path.exists(template_path):
            return self._get_error_html(title, f"模板文件不存在: {template_path}")

        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            template = Template(template_content)

            # 只提取模板实际使用的变量
            keywords = data['keywords']['frequency']
            sample_texts = data.get('sample_texts', [])
            has_llm_data = data.get('has_llm_data', False)

            # 大模型相关变量
            llm_statistics = data.get('llm_statistics', {})
            entity_stats_by_type = data.get('entity_stats_by_type', {})
            relation_tree_data = data.get('relation_tree_data', {})
            entity_table_data = data.get('entity_table_data', [])
            llm_extraction = data.get('llm_extraction', {})
            global_graph = data.get('global_graph', {})
            event_timeline = data.get('event_timeline', {})
            event_chains = data.get('event_chains', [])
            theme_hierarchy = data.get('theme_hierarchy', {})
            topics = data.get('topics', [])
            highlight_data = data.get('highlight_data', [])

            return template.render(
                # 基础变量
                title=title,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                keywords=keywords,
                sample_texts=sample_texts,

                # 大模型开关
                has_llm_data=has_llm_data,

                # 大模型统计
                llm_statistics=llm_statistics,

                # 实体相关
                entity_stats_by_type=entity_stats_by_type,
                entity_table_data=entity_table_data,

                # 关系图
                relation_tree_data=relation_tree_data,

                # 事件相关
                llm_extraction=llm_extraction,
                event_timeline=event_timeline,
                event_chains=event_chains,

                # 主题相关
                theme_hierarchy=theme_hierarchy,
                topics=topics,

                # 知识图谱
                global_graph=global_graph,

                # 高亮数据
                highlight_data=highlight_data
            )
        except Exception as e:
            import traceback
            print(f"模板渲染失败: {traceback.format_exc()}")
            return self._get_error_html(title, f"模板渲染失败: {str(e)}")