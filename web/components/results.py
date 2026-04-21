# web/components/results.py

"""分析结果展示组件 - 预览报告、AI解读标签页"""

import streamlit as st
import json
from web.components.chat_interface import render_chat_interface, render_recommended_questions
from web.components.scenario_recommendation import render_scenario_recommendation
from web.components.natural_query import render_natural_query
from web.components.sql_generator import render_sql_generator
from web.components.agent_inference import render_agent_inference
from web.services.session_service import SessionService
from web.services.storage_service import StorageService


def render_preview_tab():
    """渲染预览报告标签页"""
    session_id = SessionService.get_current_session()
    if session_id is None:
        st.info("暂无报告预览")
        return

    # 每次都从存储重新加载，不使用 session_state 缓存
    html_content = StorageService.load_text("analysis_report", session_id)
    json_data = StorageService.load_json("analysis_result", session_id)
    log_content = StorageService.load_text("analysis_log", session_id)

    if html_content and json_data:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "📥 下载 HTML 报告",
                html_content,
                "autostat_report.html",
                "text/html",
                use_container_width=True
            )
        with col2:
            json_str = json.dumps(json_data, ensure_ascii=False, indent=2) if json_data else ""
            st.download_button(
                "📥 下载 JSON 结果",
                json_str,
                "autostat_result.json",
                "application/json",
                use_container_width=True
            )
        with col3:
            if log_content:
                st.download_button(
                    "📥 下载分析日志",
                    log_content,
                    "autostat_log.txt",
                    "text/plain",
                    use_container_width=True
                )
            else:
                st.download_button(
                    "📥 下载分析日志",
                    "暂无分析日志",
                    "autostat_log.txt",
                    "text/plain",
                    disabled=True,
                    use_container_width=True
                )

        st.divider()
        st.html(html_content)
    else:
        st.info("暂无报告预览")


def render_ai_tab():
    """渲染AI解读标签页"""
    is_database = False
    session_id = SessionService.get_current_session()
    if session_id:
        metadata = SessionService.load_metadata(session_id)
        is_database = metadata.get("analysis_type") == "database"

    render_context_selector()

    st.divider()

    render_chat_interface()

    st.divider()

    if is_database:
        sub_tab_labels = ["💬 智能解读", "🎯 场景推荐", "🔍 自然查询", "📝 SQL生成", "🔮 推理预测"]
    else:
        sub_tab_labels = ["💬 智能解读", "🎯 场景推荐", "🔍 自然查询", "🔮 推理预测"]

    sub_tabs = st.tabs(sub_tab_labels)

    with sub_tabs[0]:
        if st.session_state.llm_client is None:
            st.warning("请先在侧边栏配置大模型")
        else:
            st.session_state.chat_mode = "analysis"
            render_recommended_questions()

    with sub_tabs[1]:
        if st.session_state.llm_client is None:
            st.warning("请先在侧边栏配置大模型")
        else:
            st.session_state.chat_mode = "scenario"
            render_scenario_recommendation()

    with sub_tabs[2]:
        if st.session_state.llm_client is None:
            st.warning("请先在侧边栏配置大模型")
        else:
            st.session_state.chat_mode = "natural_query"
            render_natural_query()

    if is_database:
        with sub_tabs[3]:
            if st.session_state.llm_client is None:
                st.warning("请先在侧边栏配置大模型")
            else:
                st.session_state.chat_mode = "sql"
                render_sql_generator()
        with sub_tabs[4]:
            if st.session_state.llm_client is None:
                st.warning("请先在侧边栏配置大模型")
            else:
                st.session_state.chat_mode = "prediction"
                render_agent_inference()
    else:
        with sub_tabs[3]:
            if st.session_state.llm_client is None:
                st.warning("请先在侧边栏配置大模型")
            else:
                st.session_state.chat_mode = "prediction"
                render_agent_inference()


def render_context_selector():
    """渲染上下文选择器"""
    st.markdown("#### 📚 选择分析上下文")
    st.caption("勾选要提供给AI的上下文信息，多选可获得更全面的分析")

    col_ctx1, col_ctx2, col_ctx3 = st.columns(3)

    with col_ctx1:
        ctx_json = st.checkbox(
            "📊 JSON 结果",
            value="json_result" in st.session_state.selected_contexts,
            key="ctx_json",
            help="包含完整的统计分析数据（变量类型、相关性、质量报告等）"
        )

    with col_ctx2:
        ctx_html = st.checkbox(
            "📄 HTML 报告",
            value="html_report" in st.session_state.selected_contexts,
            key="ctx_html",
            help="包含可视化的分析报告（图表、统计摘要等）"
        )

    with col_ctx3:
        ctx_data = st.checkbox(
            "🗃️ 源数据",
            value="raw_data" in st.session_state.selected_contexts,
            key="ctx_data",
            help="包含原始数据预览（前50行）及完整统计摘要"
        )

    selected = []
    if ctx_json:
        selected.append("json_result")
    if ctx_html:
        selected.append("html_report")
    if ctx_data:
        selected.append("raw_data")

    if set(selected) != set(st.session_state.selected_contexts):
        st.session_state.selected_contexts = selected

    if not st.session_state.selected_contexts:
        st.warning("⚠️ 请至少选择一种上下文信息")

    with st.expander(f"📋 已选上下文 ({len(st.session_state.selected_contexts)}项)", expanded=False):
        ctx_names = {
            "json_result": "📊 JSON 结果 - 完整的统计分析数据",
            "html_report": "📄 HTML 报告 - 可视化分析报告",
            "raw_data": "🗃️ 源数据 - 原始数据预览（前50行）及统计摘要"
        }
        for ctx in st.session_state.selected_contexts:
            st.markdown(f"- {ctx_names.get(ctx, ctx)}")