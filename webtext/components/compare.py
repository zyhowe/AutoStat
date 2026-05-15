# webtext/components/compare.py
"""项目对比组件 - 基于大模型抽取的5项核心数据结构"""

import streamlit as st
import re
from datetime import datetime
from typing import Dict, Any, List, Set

from webtext.services.session_service import TextSessionService


def extract_llm_data(json_data: dict) -> Dict[str, Any]:
    """从 JSON 中提取大模型抽取的5项核心数据"""
    llm_extraction = json_data.get('llm_extraction', {})
    return {
        'entities': llm_extraction.get('entities', []),
        'relationships': llm_extraction.get('relationships', []),
        'events': llm_extraction.get('events', []),
        'themes': llm_extraction.get('themes', []),
        'categorization': llm_extraction.get('categorization', {})
    }


def extract_entity_names(entities: List[Dict]) -> Set[str]:
    """提取实体名称集合"""
    names = set()
    for e in entities:
        name = e.get('entity_name', '')
        if name:
            names.add(name)
    return names


def extract_relationship_items(relationships: List[Dict]) -> Set[str]:
    """提取关系内容（主语→谓语→宾语）"""
    items = set()
    for r in relationships:
        subj = r.get('subject_entity_name', '') or r.get('subject_entity_id', '')
        pred = r.get('predicate', '')
        obj = r.get('object_entity_name', '') or r.get('object_entity_id', '')
        if subj and obj:
            items.add(f"{subj} → {pred} → {obj}")
    return items


def extract_event_items(events: List[Dict]) -> Set[str]:
    """提取事件内容（类型: 摘要）"""
    items = set()
    for e in events:
        etype = e.get('event_type', '')
        summary = e.get('summary', '')[:50]
        if etype or summary:
            items.add(f"{etype}: {summary}")
    return items


def extract_theme_items(themes: List[Dict]) -> Set[str]:
    """提取主题内容（名称 - 关键词）"""
    items = set()
    for t in themes:
        name = t.get('theme_name', '')
        keywords = t.get('keywords', [])
        kw_str = ', '.join(keywords[:3])
        if name:
            items.add(f"{name} [{kw_str}]")
    return items


def extract_category_items(categorization: Dict) -> Set[str]:
    """提取分类内容（类别名 - 成员）"""
    items = set()
    categories = categorization.get('categories', [])
    for cat in categories:
        name = cat.get('category_name', '')
        members = cat.get('member_entity_names', []) or cat.get('member_entity_ids', [])
        member_str = ', '.join(members[:3])
        if name:
            items.add(f"{name}: {member_str}")
    return items


def get_entity_highlight_data(entities: List[Dict]) -> List[Dict]:
    """获取实体高亮数据（用于文本高亮）"""
    highlight_data = []
    for e in entities:
        name = e.get('entity_name', '')
        if name and len(name) >= 2:
            etype = e.get('entity_type', '')
            highlight_data.append({
                'text': name,
                'entity_name': name,
                'entity_type': etype,
                'evidence': e.get('evidence', '')
            })
    return highlight_data


def get_original_text(json_data: Dict) -> str:
    """获取原始文本"""
    sample_texts = json_data.get('sample_texts', [])
    if sample_texts:
        return sample_texts[0]

    # 从 llm_extraction 中获取 evidence 组合
    entities = json_data.get('llm_extraction', {}).get('entities', [])
    if entities:
        evidences = [e.get('evidence', '') for e in entities if e.get('evidence')]
        if evidences:
            return '\n'.join(evidences[:5])

    return "无文本内容"


def render_highlighted_text(text: str, highlight_data: List[Dict], title: str):
    """渲染高亮文本（与 HTML 报告逻辑一致）"""

    if not text or text == "无文本内容":
        st.caption("无文本内容")
        return

    highlighted = text

    # 按长度排序，从长到短
    sorted_highlights = sorted(highlight_data, key=lambda x: len(x.get('text', '')), reverse=True)

    for item in sorted_highlights:
        target = item.get('text', '')
        if target and len(target) >= 2 and target in highlighted:
            entity_type = item.get('entity_type', '')
            color_map = {
                'Metric': '#ffeb3b',
                'Product': '#c8e6c9',
                'Industry': '#ffccbc',
                'EventName': '#b3e5fc',
                'Organization': '#e1bee7',
                'Location': '#ffe0b2',
                'Person': '#f8bbd0'
            }
            border_map = {
                'Metric': '#f9a825',
                'Product': '#388e3c',
                'Industry': '#e65100',
                'EventName': '#0288d1',
                'Organization': '#7b1fa2',
                'Location': '#ef6c00',
                'Person': '#c2185b'
            }
            color = color_map.get(entity_type, '#e0e0e0')
            border = border_map.get(entity_type, '#757575')

            # 替换所有出现
            highlighted = highlighted.replace(
                target,
                f'<mark style="background: {color}; border-left: 3px solid {border}; padding: 0 4px; border-radius: 4px; color: #333;">{target}</mark>'
            )

    st.markdown(f"**{title}**")
    st.markdown(
        f'<div style="background: #fafafa; padding: 12px; border-radius: 8px; max-height: 300px; overflow-y: auto; font-size: 13px; line-height: 1.6;">{highlighted}</div>',
        unsafe_allow_html=True
    )


def render_diff_row(
        category: str,
        icon: str,
        items_a: Set,
        items_b: Set,
        name_a: str,
        name_b: str,
        max_display: int = 10
):
    """渲染差异对比行（带高亮和展开）"""

    only_a = items_a - items_b
    only_b = items_b - items_a
    common = items_a & items_b

    col1, col2, col3 = st.columns([1.5, 3, 3])

    with col1:
        st.markdown(f"{icon} **{category}**")
        # 差异徽章
        if only_a or only_b:
            st.markdown(
                f'<span style="background: #ff9800; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px;">⚠️ 有差异</span>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<span style="background: #4caf50; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px;">✅ 一致</span>',
                unsafe_allow_html=True)
        st.caption(f"共 {len(items_a)} vs {len(items_b)}")

    with col2:
        st.markdown(f"**{name_a}**")
        if only_a:
            st.markdown(
                f'<span style="background: #ffeb3b; padding: 2px 6px; border-radius: 4px; font-size: 11px;">✨ 独有 {len(only_a)} 个</span>',
                unsafe_allow_html=True)

        # 先显示共同项（普通样式）
        if common:
            st.markdown("**共同项:**")
            for item in list(common)[:max_display]:
                st.caption(f"✓ {item}")
            if len(common) > max_display:
                st.caption(f"... 还有 {len(common) - max_display} 个")

        # 再显示独有项（高亮背景）
        if only_a:
            st.markdown("**仅此项目有:**")
            for item in list(only_a)[:max_display]:
                st.markdown(
                    f'<span style="background: #fff3e0; padding: 2px 6px; border-radius: 4px; display: inline-block; margin: 2px 0;">⚠️ {item}</span>',
                    unsafe_allow_html=True)
            if len(only_a) > max_display:
                st.caption(f"... 还有 {len(only_a) - max_display} 个")

    with col3:
        st.markdown(f"**{name_b}**")
        if only_b:
            st.markdown(
                f'<span style="background: #ffeb3b; padding: 2px 6px; border-radius: 4px; font-size: 11px;">✨ 独有 {len(only_b)} 个</span>',
                unsafe_allow_html=True)

        # 先显示共同项（普通样式）
        if common:
            st.markdown("**共同项:**")
            for item in list(common)[:max_display]:
                st.caption(f"✓ {item}")
            if len(common) > max_display:
                st.caption(f"... 还有 {len(common) - max_display} 个")

        # 再显示独有项（高亮背景）
        if only_b:
            st.markdown("**仅此项目有:**")
            for item in list(only_b)[:max_display]:
                st.markdown(
                    f'<span style="background: #fff3e0; padding: 2px 6px; border-radius: 4px; display: inline-block; margin: 2px 0;">⚠️ {item}</span>',
                    unsafe_allow_html=True)
            if len(only_b) > max_display:
                st.caption(f"... 还有 {len(only_b) - max_display} 个")

    st.divider()


def render_compare_tab():
    """渲染项目对比标签页"""
    st.markdown("### 🔍 项目对比")
    st.caption("基于大模型抽取结果，对比两个项目的实体、关系、事件、主题、分类信息")

    projects = TextSessionService.list_projects()

    if len(projects) < 2:
        st.info("需要至少两个项目才能进行对比，请先创建更多分析项目。")
        return

    # 构建选项（只显示 session_id）
    project_options = {}
    for p in projects:
        session_id = p["session_id"]
        if p.get("has_analysis", False):
            project_options[session_id] = session_id

    if len(project_options) < 2:
        st.info("需要至少两个已完成分析的项目才能进行对比。")
        return

    col_left, col_right = st.columns(2)

    with col_left:
        selected_left = st.selectbox(
            "选择项目 A",
            options=list(project_options.keys()),
            format_func=lambda x: x,
            key="text_compare_left"
        )

    with col_right:
        selected_right = st.selectbox(
            "选择项目 B",
            options=list(project_options.keys()),
            format_func=lambda x: x,
            key="text_compare_right"
        )

    if st.button("📊 开始对比", type="primary", use_container_width=True):
        if selected_left == selected_right:
            st.warning("请选择两个不同的项目")
            return

        # 加载 JSON 数据
        json_data_a = TextSessionService.load_json(selected_left)
        json_data_b = TextSessionService.load_json(selected_right)

        if json_data_a is None or json_data_b is None:
            st.error("无法加载分析结果，请确保两个项目都已完成分析")
            return

        # 项目名称
        name_a = selected_left
        name_b = selected_right

        # 提取数据
        data_a = extract_llm_data(json_data_a)
        data_b = extract_llm_data(json_data_b)

        # 提取各类内容
        entities_a = extract_entity_names(data_a['entities'])
        entities_b = extract_entity_names(data_b['entities'])

        relationships_a = extract_relationship_items(data_a['relationships'])
        relationships_b = extract_relationship_items(data_b['relationships'])

        events_a = extract_event_items(data_a['events'])
        events_b = extract_event_items(data_b['events'])

        themes_a = extract_theme_items(data_a['themes'])
        themes_b = extract_theme_items(data_b['themes'])

        categories_a = extract_category_items(data_a['categorization'])
        categories_b = extract_category_items(data_b['categorization'])

        # 获取高亮数据
        highlight_a = get_entity_highlight_data(data_a['entities'])
        highlight_b = get_entity_highlight_data(data_b['entities'])

        # 获取原始文本
        text_a = get_original_text(json_data_a)
        text_b = get_original_text(json_data_b)

        # ========== 对比表格 ==========
        st.markdown("---")
        st.markdown("### 📊 对比详情")

        # 渲染各行
        render_diff_row("实体", "📦", entities_a, entities_b, name_a, name_b)
        render_diff_row("关系", "🔗", relationships_a, relationships_b, name_a, name_b)
        render_diff_row("事件", "📰", events_a, events_b, name_a, name_b)
        render_diff_row("主题", "📚", themes_a, themes_b, name_a, name_b)
        render_diff_row("分类", "🏷️", categories_a, categories_b, name_a, name_b)

        # ========== 文本高亮区域 ==========
        st.markdown("---")
        st.markdown("### 📝 原始文本（实体高亮）")

        col1, col2 = st.columns(2)
        with col1:
            render_highlighted_text(text_a, highlight_a, name_a)
        with col2:
            render_highlighted_text(text_b, highlight_b, name_b)

        # ========== 导出按钮 ==========
        st.markdown("---")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        export_html = generate_compare_html(
            entities_a, entities_b,
            relationships_a, relationships_b,
            events_a, events_b,
            themes_a, themes_b,
            categories_a, categories_b,
            text_a, text_b,
            highlight_a, highlight_b,
            name_a, name_b, timestamp
        )

        st.download_button(
            "📥 导出对比报告 (HTML)",
            export_html,
            f"autotext_compare_{timestamp}.html",
            "text/html",
            use_container_width=True,
            key="text_download_compare"
        )


def generate_compare_html(
        entities_a, entities_b,
        relationships_a, relationships_b,
        events_a, events_b,
        themes_a, themes_b,
        categories_a, categories_b,
        text_a, text_b,
        highlight_a, highlight_b,
        name_a, name_b, timestamp
) -> str:
    """生成对比报告的 HTML 导出"""

    def format_items_with_diff(items_a: Set, items_b: Set, max_items: int = 20):
        """格式化项目，高亮差异"""
        only_a = items_a - items_b
        only_b = items_b - items_a
        common = items_a & items_b

        html = ""

        if common:
            html += "<div style='margin-bottom: 10px;'><strong>共同项:</strong></div>"
            html += "<ul style='margin: 0 0 15px 20px;'>"
            for item in list(common)[:max_items]:
                html += f"<li>✓ {item}</li>"
            if len(common) > max_items:
                html += f"<li>... 还有 {len(common) - max_items} 个</li>"
            html += "</ul>"

        if only_a:
            html += "<div style='margin-bottom: 5px;'><strong>✨ 仅此项目有:</strong></div>"
            html += "<ul style='margin: 0 0 15px 20px; background: #fff3e0; padding: 10px 10px 10px 30px; border-radius: 8px;'>"
            for item in list(only_a)[:max_items]:
                html += f"<li style='color: #e65100;'>⚠️ {item}</li>"
            if len(only_a) > max_items:
                html += f"<li style='color: #e65100;'>... 还有 {len(only_a) - max_items} 个</li>"
            html += "</ul>"

        return html if (common or only_a) else "<span style='color: #999;'>无</span>"

    def format_items_simple(items: Set, max_items: int = 20):
        """简单格式化（用于 B 列）"""
        if not items:
            return "<span style='color: #999;'>无</span>"
        result = "<ul style='margin: 0; padding-left: 20px;'>"
        for item in list(items)[:max_items]:
            result += f"<li>{item}</li>"
        if len(items) > max_items:
            result += f"<li>... 还有 {len(items) - max_items} 个</li>"
        result += "</ul>"
        return result

    def highlight_text_html(text: str, highlights: List[Dict]) -> str:
        if not text or text == "无文本内容":
            return "<span style='color: #999;'>无文本内容</span>"

        highlighted = text
        sorted_highlights = sorted(highlights, key=lambda x: len(x.get('text', '')), reverse=True)

        for item in sorted_highlights:
            target = item.get('text', '')
            if target and len(target) >= 2 and target in highlighted:
                etype = item.get('entity_type', '')
                color_map = {
                    'Metric': '#ffeb3b', 'Product': '#c8e6c9', 'Industry': '#ffccbc',
                    'EventName': '#b3e5fc', 'Organization': '#e1bee7', 'Location': '#ffe0b2',
                    'Person': '#f8bbd0'
                }
                border_map = {
                    'Metric': '#f9a825', 'Product': '#388e3c', 'Industry': '#e65100',
                    'EventName': '#0288d1', 'Organization': '#7b1fa2', 'Location': '#ef6c00',
                    'Person': '#c2185b'
                }
                color = color_map.get(etype, '#e0e0e0')
                border = border_map.get(etype, '#757575')
                replacement = f'<mark style="background: {color}; border-left: 3px solid {border}; padding: 0 4px; border-radius: 4px;">{target}</mark>'
                highlighted = highlighted.replace(target, replacement)

        return f'<div style="background: #fafafa; padding: 15px; border-radius: 8px; max-height: 300px; overflow-y: auto; font-size: 13px; line-height: 1.6;">{highlighted}</div>'

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AutoText 对比报告</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #1f77b4; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 25px; font-size: 18px; border-left: 4px solid #1f77b4; padding-left: 12px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #e0e0e0; padding: 12px; vertical-align: top; }}
        th {{ background: #f0f2f5; font-weight: 600; }}
        .footer {{ text-align: center; padding: 20px; color: #999; font-size: 12px; border-top: 1px solid #e9ecef; margin-top: 30px; }}
        .diff-badge {{ background: #ff9800; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; display: inline-block; }}
        .same-badge {{ background: #4caf50; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; display: inline-block; }}
        .highlight-col {{ background: #fff3e0; }}
    </style>
</head>
<body>
<div class="container">
    <h1>📊 AutoText 文本对比报告</h1>
    <p>生成时间: {timestamp}</p>
    <hr>

    <h2>📋 对比详情</h2>
    <table>
        <thead>
            <tr>
                <th style="width: 15%;">类别</th>
                <th style="width: 40%;">{name_a}</th>
                <th style="width: 40%;">{name_b}</th>
                <th style="width: 5%;">状态</th>
             </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>📦 实体</strong><br><small>共 {len(entities_a)} vs {len(entities_b)}</small></td>
                <td class="{'highlight-col' if entities_a - entities_b else ''}">{format_items_with_diff(entities_a, entities_b)}</td>
                <td class="{'highlight-col' if entities_b - entities_a else ''}">{format_items_simple(entities_b)}</td>
                <td style="text-align: center;">{'<span class="diff-badge">差异</span>' if entities_a != entities_b else '<span class="same-badge">一致</span>'}</td>
            </tr>
            <tr>
                <td><strong>🔗 关系</strong><br><small>共 {len(relationships_a)} vs {len(relationships_b)}</small></td>
                <td class="{'highlight-col' if relationships_a - relationships_b else ''}">{format_items_with_diff(relationships_a, relationships_b)}</td>
                <td class="{'highlight-col' if relationships_b - relationships_a else ''}">{format_items_simple(relationships_b)}</td>
                <td style="text-align: center;">{'<span class="diff-badge">差异</span>' if relationships_a != relationships_b else '<span class="same-badge">一致</span>'}</td>
            </tr>
            <tr>
                <td><strong>📰 事件</strong><br><small>共 {len(events_a)} vs {len(events_b)}</small></td>
                <td class="{'highlight-col' if events_a - events_b else ''}">{format_items_with_diff(events_a, events_b)}</td>
                <td class="{'highlight-col' if events_b - events_a else ''}">{format_items_simple(events_b)}</td>
                <td style="text-align: center;">{'<span class="diff-badge">差异</span>' if events_a != events_b else '<span class="same-badge">一致</span>'}</td>
            </tr>
            <tr>
                <td><strong>📚 主题</strong><br><small>共 {len(themes_a)} vs {len(themes_b)}</small></td>
                <td class="{'highlight-col' if themes_a - themes_b else ''}">{format_items_with_diff(themes_a, themes_b)}</td>
                <td class="{'highlight-col' if themes_b - themes_a else ''}">{format_items_simple(themes_b)}</td>
                <td style="text-align: center;">{'<span class="diff-badge">差异</span>' if themes_a != themes_b else '<span class="same-badge">一致</span>'}</td>
            </tr>
            <tr>
                <td><strong>🏷️ 分类</strong><br><small>共 {len(categories_a)} vs {len(categories_b)}</small></td>
                <td class="{'highlight-col' if categories_a - categories_b else ''}">{format_items_with_diff(categories_a, categories_b)}</td>
                <td class="{'highlight-col' if categories_b - categories_a else ''}">{format_items_simple(categories_b)}</td>
                <td style="text-align: center;">{'<span class="diff-badge">差异</span>' if categories_a != categories_b else '<span class="same-badge">一致</span>'}</td>
            </tr>
        </tbody>
    </table>

    <h2>📝 原始文本（实体高亮）</h2>
    <div style="display: flex; gap: 20px;">
        <div style="flex: 1;">
            <h3>{name_a}</h3>
            {highlight_text_html(text_a, highlight_a)}
        </div>
        <div style="flex: 1;">
            <h3>{name_b}</h3>
            {highlight_text_html(text_b, highlight_b)}
        </div>
    </div>

    <div class="footer">
        <p>🤖 AutoText 智能文本分析工具 | 对比报告自动生成</p>
        <p>✨ 高亮部分表示该项目独有的内容</p>
    </div>
</div>
</body>
</html>"""