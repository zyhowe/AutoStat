# web/components/chat_interface.py

"""AI智能解读组件 - 消息历史、输入框、推荐问题（Agent模式）"""

import streamlit as st
import pandas as pd
import re
from autostat.prompts import get_recommended_questions
from web.services.session_service import SessionService
from web.components.agent_inference import parse_and_execute_tool, ModelInferenceTool


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
    """从 JSON 数据提取上下文信息"""
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
            numeric_summary += f"  - {col}: 均值={mean_val}, 中位数={median_val}, 范围=[{min_val}, {max_val}]\n"

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
    """从 HTML 报告提取关键信息"""
    context_parts = []

    title_match = re.search(r'<h1>(.*?)</h1>', html_content)
    if title_match:
        context_parts.append(f"报告标题: {title_match.group(1)}")

    stat_cards = re.findall(r'<div class="stat-card"><div class="value">(.*?)</div><div class="label">(.*?)</div>',
                            html_content)
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
        cleaning_items = re.findall(r'<tr>(.*?)</tr>', cleaning_match.group(1))
        if cleaning_items:
            context_parts.append("**清洗建议:** " + ", ".join(cleaning_items[:3]))

    return "\n".join(context_parts) if context_parts else "无法提取HTML摘要信息"


def extract_data_context(raw_data_preview: dict) -> str:
    """从原始数据提取上下文信息"""
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


def build_analysis_prompt(selected_contexts, analysis_type, json_data, html_content, raw_data_preview, source_name):
    """构建普通分析提示词"""
    data_context = ""
    if "json_result" in selected_contexts and json_data:
        data_context = extract_json_context(json_data, analysis_type)
    if "html_report" in selected_contexts and html_content:
        data_context += "\n" + extract_html_context(html_content)
    if "raw_data" in selected_contexts and raw_data_preview:
        data_context += "\n" + extract_data_context(raw_data_preview)

    return f"""你是专业的数据分析师，正在回答用户关于数据的问题。

## 分析信息
- 分析类型: {analysis_type}
- 数据源: {source_name}

{data_context}

## 重要说明
1. 用中文回答，结构清晰，友好专业
2. 基于提供的数据上下文回答问题
3. 如果用户询问预测相关问题，请告知用户切换到「推理预测」标签页
"""


def build_prediction_prompt(selected_contexts, analysis_type, json_data, html_content, raw_data_preview, source_name):
    """构建预测专用提示词 - 强制输出JSON"""
    session_id = SessionService.get_current_session()

    tool = ModelInferenceTool(session_id)
    available_models = tool.get_available_models()

    if not available_models:
        return build_analysis_prompt(selected_contexts, analysis_type, json_data, html_content, raw_data_preview,
                                     source_name)

    models_desc = ""
    for model in available_models:
        model_name = model.get('user_model_name', model.get('model_key'))
        model_key = model.get('model_key')
        task_type = model.get('task_type', 'unknown')
        target = model.get('target_column', '未知')
        features = model.get('features', [])
        feature_str = ', '.join(features[:5])
        if len(features) > 5:
            feature_str += '...'
        models_desc += f"- **{model_name}**\n"
        models_desc += f"  模型ID: {model_key}\n"
        models_desc += f"  类型: {task_type}, 预测目标: {target}\n"
        models_desc += f"  特征: {feature_str}\n\n"

    return f"""你是预测助手，负责调用模型进行预测。

## 可用模型列表

{models_desc}

## 规则
1. 当用户需要进行预测时，只输出JSON，不要输出任何其他文字
2. JSON格式：{{"tool": "predict", "model_key": "模型ID", "input_values": {{"特征名1": 值1}}}}
3. 特征名必须与模型定义完全一致
4. 如果用户没有指定模型，选择第一个模型

示例：
{{"tool": "predict", "model_key": "classification_random_forest_1776665800", "input_values": {{"销售额": 8000}}}}

请直接输出JSON。
"""


def render_chat_interface():
    """渲染聊天界面"""
    session_id = SessionService.get_current_session()
    chat_mode = st.session_state.get("chat_mode", "analysis")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.get("pending_question"):
        question = st.session_state.pending_question

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            if chat_mode == "prediction":
                system_prompt = build_prediction_prompt(
                    st.session_state.selected_contexts,
                    st.session_state.current_analysis_type,
                    st.session_state.current_json_data,
                    st.session_state.current_html,
                    st.session_state.raw_data_preview,
                    st.session_state.current_source_name
                )
            else:
                system_prompt = build_analysis_prompt(
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
            messages.append({"role": "user", "content": question})

            for chunk in st.session_state.llm_client.chat_stream(messages):
                if chunk:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

            if chat_mode == "prediction":
                tool_result = parse_and_execute_tool(full_response, session_id)
                if tool_result:
                    full_response = tool_result
                    response_placeholder.markdown(full_response)

            response_placeholder.markdown(full_response)

        st.session_state.chat_messages.append({"role": "user", "content": question})
        st.session_state.chat_messages.append({"role": "assistant", "content": full_response})

        del st.session_state.pending_question
        st.rerun()

    prompt = st.chat_input("输入您的问题...", key="chat_input")
    if prompt and prompt.strip():
        st.session_state.pending_question = prompt.strip()
        st.rerun()


def render_recommended_questions():
    """渲染推荐问题"""
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