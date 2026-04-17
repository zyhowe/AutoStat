"""AI智能解读组件 - 消息历史、输入框、推荐问题"""

import streamlit as st
import pandas as pd
import re
from autostat.prompts import get_recommended_questions


def get_initial_question(analysis_type: str) -> str:
    """根据分析类型获取初始提问内容"""
    if analysis_type == "single":
        return "请根据这份数据分析报告，帮我解读一下数据的主要特征、质量问题和建模建议。"
    elif analysis_type == "multi":
        return "请根据这份多表关联分析报告，帮我解读一下表间关系、数据特征和跨表分析建议。"
    elif analysis_type == "database":
        return "请根据这份数据库分析报告，帮我解读一下数据模型、性能优化和BI建设建议。"
    else:
        return "请根据这份数据分析报告，帮我解读一下主要内容。"


def extract_json_context(json_data: dict, analysis_type: str) -> str:
    """从 JSON 数据提取上下文信息（纯数据提取，不含提示词前缀）"""

    data_shape = json_data.get('data_shape', {})
    variable_types = json_data.get('variable_types', {})
    quality_report = json_data.get('quality_report', {})
    cleaning_suggestions = json_data.get('cleaning_suggestions', [])
    correlations = json_data.get('correlations', {})
    model_recommendations = json_data.get('model_recommendations', [])
    variable_summaries = json_data.get('variable_summaries', {})

    type_counts = {}
    type_display = {
        'continuous': '连续变量',
        'categorical': '分类变量',
        'categorical_numeric': '数值型分类',
        'ordinal': '有序分类',
        'datetime': '日期时间',
        'identifier': '标识符',
        'text': '文本'
    }
    for col, info in variable_types.items():
        typ = info.get('type', 'unknown')
        type_counts[typ] = type_counts.get(typ, 0) + 1

    type_summary = ", ".join([f"{type_display.get(t, t)}: {c}" for t, c in type_counts.items()])

    missing_list = quality_report.get('missing', [])
    missing_summary = "\n".join([f"  - {m['column']}: {m['percent']:.1f}%" for m in missing_list[:10]])
    if len(missing_list) > 10:
        missing_summary += f"\n  - ... 还有 {len(missing_list) - 10} 个字段"

    outliers = quality_report.get('outliers', {})
    outlier_summary = "\n".join([f"  - {col}: {info.get('count', 0)}个 ({info.get('percent', 0):.1f}%)"
                                 for col, info in list(outliers.items())[:5]])

    high_corrs = correlations.get('high_correlations', [])
    corr_summary = "\n".join([f"  - {c['var1']} ↔ {c['var2']}: r={c['value']}" for c in high_corrs[:5]])

    model_summary = ""
    for rec in model_recommendations[:3]:
        model_summary += f"  - {rec.get('task_type', '')}: {rec.get('ml', '')}\n"

    date_range_info = ""
    for col, summary in variable_summaries.items():
        if summary.get('type') == 'datetime':
            min_date = summary.get('min_date')
            max_date = summary.get('max_date')
            if min_date and max_date:
                date_range_info += f"  - {col}: {min_date} 至 {max_date}\n"

    numeric_summary = ""
    for col, summary in list(variable_summaries.items())[:10]:
        if summary.get('type') == 'continuous':
            min_val = summary.get('min', 'N/A')
            max_val = summary.get('max', 'N/A')
            mean_val = summary.get('mean', 'N/A')
            median_val = summary.get('median', 'N/A')

            if isinstance(min_val, (int, float)):
                min_str = f"{min_val:.2f}"
            else:
                min_str = str(min_val)
            if isinstance(max_val, (int, float)):
                max_str = f"{max_val:.2f}"
            else:
                max_str = str(max_val)
            if isinstance(mean_val, (int, float)):
                mean_str = f"{mean_val:.2f}"
            else:
                mean_str = str(mean_val)
            if isinstance(median_val, (int, float)):
                median_str = f"{median_val:.2f}"
            else:
                median_str = str(median_val)

            numeric_summary += f"  - {col}: 均值={mean_str}, 中位数={median_str}, 范围=[{min_str}, {max_str}]\n"

    multi_info = ""
    if analysis_type in ["multi", "database"]:
        multi_tables = json_data.get('multi_table_info', {}).get('tables', {})
        relationships = json_data.get('multi_table_info', {}).get('relationships', [])
        if multi_tables:
            multi_info = "\n### 多表信息\n"
            multi_info += "**包含的表:**\n"
            for name, info in multi_tables.items():
                shape = info.get('shape', {})
                multi_info += f"  - {name}: {shape.get('rows', 0)}行 x {shape.get('columns', 0)}列\n"
            if relationships:
                multi_info += "\n**表间关系:**\n"
                for rel in relationships[:5]:
                    multi_info += f"  - {rel.get('from_table')}.{rel.get('from_col')} → {rel.get('to_table')}.{rel.get('to_col')}\n"

    return f"""
### JSON 分析结果数据

**数据概览**
- 总行数: {data_shape.get('rows', 0):,}
- 总列数: {data_shape.get('columns', 0)}
- 变量类型分布: {type_summary}

**日期范围**
{date_range_info if date_range_info else '  无日期列'}

**数值变量分布**
{numeric_summary if numeric_summary else '  无数值变量'}

**数据质量**
缺失值:
{missing_summary if missing_summary else '  无缺失值'}

异常值:
{outlier_summary if outlier_summary else '  无异常值'}

重复记录: {quality_report.get('duplicates', {}).get('count', 0)}条

**清洗建议**
{chr(10).join([f"  - {s}" for s in cleaning_suggestions[:5]]) if cleaning_suggestions else '  无清洗建议'}

**相关性分析**
{corr_summary if corr_summary else '  无强相关对'}

**建模建议**
{model_summary if model_summary else '  无建模建议'}

{multi_info}
"""


def extract_html_context(html_content: str) -> str:
    """从 HTML 报告提取关键信息（纯数据提取，不含提示词前缀）"""

    context_parts = []

    title_match = re.search(r'<h1>(.*?)</h1>', html_content)
    if title_match:
        context_parts.append(f"报告标题: {title_match.group(1)}")

    stat_cards = re.findall(r'<div class="stat-card"><div class="value">(.*?)</div><div class="label">(.*?)</div>', html_content)
    if stat_cards:
        stats_summary = "**关键指标:**\n"
        for value, label in stat_cards:
            stats_summary += f"  - {label}: {value}\n"
        context_parts.append(stats_summary)

    insight_match = re.search(r'<h2>💡 核心洞察</h2>(.*?)</div>', html_content, re.DOTALL)
    if insight_match:
        insights = re.findall(r'<li>(.*?)</li>', insight_match.group(1))
        if insights:
            context_parts.append("**核心洞察:**\n" + "\n".join([f"  - {i}" for i in insights[:3]]))

    type_match = re.search(r'变量类型分布</strong><br>(.*?)</div>', html_content, re.DOTALL)
    if type_match:
        type_lines = [line.strip() for line in type_match.group(1).split('<br>') if line.strip()]
        if type_lines:
            context_parts.append("**变量类型:** " + ", ".join(type_lines[:5]))

    cleaning_match = re.search(r'<h2>🧹 数据清洗建议</h2>(.*?)</div>', html_content, re.DOTALL)
    if cleaning_match:
        cleaning_items = re.findall(r'<tr>(.*?)</td>', cleaning_match.group(1))
        if cleaning_items:
            context_parts.append("**清洗建议:** " + ", ".join(cleaning_items[:3]))

    return "\n".join(context_parts) if context_parts else "无法提取HTML摘要信息"


def extract_data_context(raw_data_preview: dict) -> str:
    """从原始数据提取上下文信息（纯数据提取，不含提示词前缀）"""
    if raw_data_preview is None:
        return "无源数据信息"

    shape = raw_data_preview.get('shape', (0, 0))
    dtypes = raw_data_preview.get('dtypes', {})
    summary_stats = raw_data_preview.get('summary_stats', {})
    preview = raw_data_preview.get('preview', pd.DataFrame())
    numeric_cols = raw_data_preview.get('numeric_cols', [])
    cat_cols = raw_data_preview.get('cat_cols', [])
    date_cols = raw_data_preview.get('date_cols', [])

    context_parts = []
    context_parts.append(f"- 数据形状: {shape[0]} 行 × {shape[1]} 列")

    type_counts = {}
    for col, dtype in dtypes.items():
        dtype_str = str(dtype)
        if 'int' in dtype_str or 'float' in dtype_str:
            type_counts['数值型'] = type_counts.get('数值型', 0) + 1
        elif 'datetime' in dtype_str:
            type_counts['日期型'] = type_counts.get('日期型', 0) + 1
        elif 'object' in dtype_str:
            type_counts['文本型'] = type_counts.get('文本型', 0) + 1
        else:
            type_counts['其他'] = type_counts.get('其他', 0) + 1

    type_summary = ", ".join([f"{t}: {c}" for t, c in type_counts.items()])
    context_parts.append(f"- 字段类型分布: {type_summary}")

    columns = list(dtypes.keys())[:30]
    context_parts.append(f"- 字段列表: {', '.join(columns)}")
    if len(dtypes) > 30:
        context_parts.append(f"  ... 还有 {len(dtypes) - 30} 个字段")

    if date_cols:
        context_parts.append("\n### 日期范围")
        for col in date_cols:
            stats = summary_stats.get(col, {})
            if stats.get('type') == 'datetime':
                context_parts.append(f"- {col}: {stats.get('min', 'N/A')} 至 {stats.get('max', 'N/A')} "
                                   f"(共{stats.get('unique_dates', 0)}个日期)")

    if numeric_cols:
        context_parts.append("\n### 数值列统计")
        for col in numeric_cols:
            stats = summary_stats.get(col, {})
            if stats.get('type') == 'numeric':
                context_parts.append(f"- {col}: 范围=[{stats.get('min', 'N/A'):.2f}, {stats.get('max', 'N/A'):.2f}], "
                                   f"均值={stats.get('mean', 'N/A'):.2f}, "
                                   f"中位数={stats.get('median', 'N/A'):.2f}")

    if cat_cols:
        context_parts.append("\n### 分类列统计")
        for col in cat_cols:
            stats = summary_stats.get(col, {})
            if stats.get('type') == 'categorical':
                top_vals = stats.get('top_values', {})
                top_str = ", ".join([f"{k}:{v}" for k, v in list(top_vals.items())[:3]])
                context_parts.append(f"- {col}: {stats.get('unique_count', 0)}个唯一值, "
                                   f"最高频占比={stats.get('top_percent', 0):.1f}%, "
                                   f"前3: {top_str}")

    if len(preview) > 0:
        preview_str = preview.head(50).to_string()
        context_parts.append(f"\n### 数据预览 (前50行)\n```\n{preview_str}\n```")

    return "\n".join(context_parts)


def build_context_prompt(selected_contexts, analysis_type, json_data, html_content, raw_data_preview, source_name):
    """根据选中的上下文构建完整提示词"""

    context_parts = []

    analysis_type_map = {
        "single": "单文件分析",
        "multi": "多文件分析",
        "database": "数据库分析"
    }
    context_parts.append(f"**分析类型:** {analysis_type_map.get(analysis_type, '未知')}")
    context_parts.append(f"**数据源:** {source_name}")
    context_parts.append("")

    if "json_result" in selected_contexts and json_data:
        context_parts.append(extract_json_context(json_data, analysis_type))

    if "html_report" in selected_contexts and html_content:
        html_summary = extract_html_context(html_content)
        if html_summary:
            context_parts.append(f"### HTML 报告摘要\n{html_summary}")

    if "raw_data" in selected_contexts and raw_data_preview:
        context_parts.append(f"### 源数据信息\n{extract_data_context(raw_data_preview)}")

    full_context = "\n\n".join(context_parts)

    full_prompt = f"""
你是一位专业的数据分析师。以下是根据用户选择的上下文提供的分析数据。

{full_context}

---

## 注意事项
- 请基于上述提供的数据上下文回答问题
- 源数据包含完整的数据预览（前50行）和统计信息
- 你可以根据日期范围、数值范围等条件查询具体数据
- 如果用户询问具体数据（如某年某月的数据），请根据数据预览和统计信息回答
- 用中文回答，结构清晰
"""
    return full_prompt


def render_chat_interface():
    """
    渲染聊天界面（历史 + 临时 + 输入框）

    历史区块：显示 session_state.chat_messages 中的消息
    临时区块：显示当前正在生成的 AI 回答（流式）
    """
    # ==================== 历史区块 ====================
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ==================== 临时区块 ====================
    # 检查是否有正在处理的请求
    if st.session_state.get("pending_question"):
        # 显示用户消息（临时区块）
        with st.chat_message("user"):
            st.markdown(st.session_state.pending_question)

        # 流式显示 AI 回答
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            # 构建消息
            system_prompt = build_context_prompt(
                st.session_state.selected_contexts,
                st.session_state.current_analysis_type,
                st.session_state.current_json_data,
                st.session_state.current_html,
                st.session_state.raw_data_preview,
                st.session_state.current_source_name
            )

            messages = [{"role": "system", "content": system_prompt}]
            for msg in st.session_state.chat_messages:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": st.session_state.pending_question})

            # 流式输出
            for chunk in st.session_state.llm_client.chat_stream(messages):
                if chunk:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        # 将完整对话追加到历史
        st.session_state.chat_messages.append({"role": "user", "content": st.session_state.pending_question})
        st.session_state.chat_messages.append({"role": "assistant", "content": full_response})

        # 清除临时状态并刷新
        del st.session_state.pending_question
        st.rerun()

    # ==================== 输入框 ====================
    prompt = st.chat_input("输入您的问题...", key="chat_input")
    if prompt and prompt.strip():
        # 设置临时状态，触发临时区块显示
        st.session_state.pending_question = prompt.strip()
        st.rerun()


def render_recommended_questions():
    """渲染推荐问题（对话标签页专用）"""
    st.markdown("#### 💡 推荐问题")
    st.caption("点击下方问题快速提问")

    if st.session_state.current_json_data is None:
        st.info("请先完成数据分析")
        return

    has_datetime = False
    if st.session_state.current_json_data.get('variable_types'):
        for info in st.session_state.current_json_data.get('variable_types', {}).values():
            if info.get('type') == 'datetime':
                has_datetime = True
                break

    recommended_qs = get_recommended_questions(
        st.session_state.current_analysis_type,
        has_datetime
    )

    cols = st.columns(min(len(recommended_qs), 3))
    for i, q in enumerate(recommended_qs):
        col_idx = i % 3
        if cols[col_idx].button(q, key=f"rec_q_{i}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()