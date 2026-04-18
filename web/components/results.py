"""分析结果展示组件 - 预览报告、分析日志、AI解读标签页"""

import streamlit as st
from web.components.chat_interface import render_chat_interface, render_recommended_questions
from web.components.scenario_recommendation import render_scenario_recommendation
from web.components.natural_query import render_natural_query
from web.components.sql_generator import render_sql_generator


def render_preview_tab():
    """渲染预览报告标签页"""
    if st.session_state.current_html and st.session_state.current_json_data:
        # 下载按钮行
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 下载 HTML 报告",
                st.session_state.current_html,
                "autostat_report.html",
                "text/html",
                use_container_width=True
            )
        with col2:
            # 获取当前分析器的 JSON 输出
            if st.session_state.current_analysis_type == "single":
                json_output = st.session_state.single_analyzer.to_json() if st.session_state.single_analyzer else ""
            elif st.session_state.current_analysis_type == "multi":
                json_output = st.session_state.multi_analyzer.to_json() if st.session_state.multi_analyzer else ""
            elif st.session_state.current_analysis_type == "database":
                json_output = st.session_state.db_analyzer.to_json() if st.session_state.db_analyzer else ""
            else:
                json_output = ""

            st.download_button(
                "📥 下载 JSON 结果",
                json_output,
                "autostat_result.json",
                "application/json",
                use_container_width=True
            )

        st.divider()
        st.html(st.session_state.current_html)
    else:
        st.info("暂无报告预览")


def render_log_tab():
    """渲染分析日志标签页"""
    output = None
    if st.session_state.current_analysis_type == "single":
        output = st.session_state.single_output
    elif st.session_state.current_analysis_type == "multi":
        output = st.session_state.multi_output
    elif st.session_state.current_analysis_type == "database":
        output = st.session_state.db_output

    if output:
        st.code(output, language='text')
    else:
        st.info("暂无分析日志")


def render_ai_tab():
    """渲染AI解读标签页"""
    # 判断是否为数据库模式
    is_database = st.session_state.current_analysis_type == "database"

    # ==================== 顶部：选择分析上下文 ====================
    render_context_selector()

    st.divider()

    # ==================== 中间：聊天界面（历史消息 + 输入框） ====================
    render_chat_interface()

    st.divider()

    # ==================== 底部：二级标签页 ====================
    if is_database:
        sub_tab_labels = ["💬 自由提问", "🎯 场景推荐", "🔍 自然查询", "📝 SQL生成", "🔮 推理预测"]
    else:
        sub_tab_labels = ["💬 自由提问", "🎯 场景推荐", "🔍 自然查询", "🔮 推理预测"]

    sub_tabs = st.tabs(sub_tab_labels)

    with sub_tabs[0]:
        if st.session_state.llm_client is None:
            st.warning("请先在侧边栏配置大模型")
        else:
            render_recommended_questions()

    with sub_tabs[1]:
        if st.session_state.llm_client is None:
            st.warning("请先在侧边栏配置大模型")
        else:
            render_scenario_recommendation()

    with sub_tabs[2]:
        if st.session_state.llm_client is None:
            st.warning("请先在侧边栏配置大模型")
        else:
            render_natural_query()

    with sub_tabs[3]:
        if is_database:
            if st.session_state.llm_client is None:
                st.warning("请先在侧边栏配置大模型")
            else:
                render_sql_generator()
        else:
            # 推理预测标签页
            from web.components.inference import render_inference_interface
            render_inference_interface()

    if is_database:
        with sub_tabs[4]:
            from web.components.inference import render_inference_interface
            render_inference_interface()


def render_context_selector():
    """渲染上下文选择器（公共组件）"""
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