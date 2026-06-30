"""
自助探索面板组件
"""

import streamlit as st
import pandas as pd

from autostat.core.explore import NL2SQL, ChartRecommender, StoryGenerator
from web.services.session_service import SessionService
from web.services.storage_service import StorageService


def render_explore_panel():
    """渲染自助探索面板"""
    st.markdown("### 🔎 自助探索")
    st.caption("用自然语言查询数据，自动推荐图表，生成分析故事")

    session_id = SessionService.get_current_session()
    if session_id is None:
        st.info("请先完成数据分析")
        return

    data = StorageService.load_dataframe("processed_data", session_id)
    analysis_result = StorageService.load_json("analysis_result", session_id)

    if data is None or analysis_result is None:
        st.info("请先完成数据分析")
        return

    # ==================== Tab页 ====================
    tab1, tab2, tab3 = st.tabs(["💬 自然语言查询", "📊 智能图表", "📝 故事生成"])

    with tab1:
        render_nl2sql_tab(data, analysis_result)

    with tab2:
        render_chart_tab(data)

    with tab3:
        render_story_tab(data, analysis_result)


def render_nl2sql_tab(data: pd.DataFrame, analysis_result: dict):
    """自然语言查询标签页"""
    st.markdown("用自然语言描述你想查询的数据")

    # 构建Schema
    variable_types = analysis_result.get("variable_types", {})
    schema = {
        "table_name": analysis_result.get("source_table", "data"),
        "fields": [
            {"name": col, "type": info.get("type", "unknown")}
            for col, info in variable_types.items()
        ]
    }

    # 示例查询
    examples = [
        "总共有多少行数据",
        "列出前10条记录",
        "统计各分类的数量",
        "按时间查看趋势",
    ]

    st.markdown("**💡 示例查询：**")
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i % len(cols)].button(ex, key=f"explore_ex_{i}"):
            st.session_state.explore_query = ex

    # 输入框
    query = st.text_input(
        "输入查询",
        value=st.session_state.get("explore_query", ""),
        placeholder="例如：统计各产品的销售额",
        key="explore_input"
    )

    if query and st.button("🔍 执行查询", type="primary"):
        with st.spinner("正在处理查询..."):
            # 使用大模型（如果有）
            llm_client = st.session_state.get("llm_client")

            nl2sql = NL2SQL(llm_client)
            result = nl2sql.convert(query, schema)

            st.markdown("**生成的SQL:**")
            st.code(result["sql"], language="sql")

            # 执行查询
            try:
                df_result = nl2sql.execute(result["sql"], data)
                if df_result is not None and len(df_result) > 0:
                    st.markdown("**查询结果:**")
                    st.dataframe(df_result, use_container_width=True)
                else:
                    st.info("查询无结果")
            except Exception as e:
                st.error(f"执行失败: {e}")


def render_chart_tab(data: pd.DataFrame):
    """智能图表标签页"""
    st.markdown("选择字段，自动推荐最佳图表")

    # 字段选择
    all_fields = data.columns.tolist()
    selected_fields = st.multiselect(
        "选择要展示的字段",
        options=all_fields,
        default=all_fields[:3] if len(all_fields) >= 3 else all_fields
    )

    # 意图选择
    intent = st.selectbox(
        "分析意图",
        options=["自动", "趋势", "对比", "构成", "分布", "关系"],
        index=0
    )

    intent_map = {
        "自动": None,
        "趋势": "trend",
        "对比": "compare",
        "构成": "composition",
        "分布": "distribution",
        "关系": "relationship"
    }

    if selected_fields and st.button("📊 推荐图表", type="primary"):
        with st.spinner("正在推荐图表..."):
            # 检测时间字段
            time_fields = data.select_dtypes(include=['datetime64']).columns
            time_field = time_fields[0] if len(time_fields) > 0 else None

            recommender = ChartRecommender()
            recommendations = recommender.recommend(
                data,
                selected_fields,
                intent=intent_map.get(intent),
                time_field=time_field
            )

            if recommendations:
                for rec in recommendations[:3]:
                    st.markdown(f"**{rec.title}**")
                    st.caption(rec.description)

                    # 显示数据预览
                    if rec.data is not None and len(rec.data) > 0:
                        st.dataframe(rec.data.head(10), use_container_width=True)

                    # 显示配置
                    config = recommender.get_chart_config(rec)
                    st.caption(f"图表类型: {rec.chart_type.value}")
                    st.json(config)

                    st.divider()
            else:
                st.info("无法推荐图表，请尝试其他字段组合")


def render_story_tab(data: pd.DataFrame, analysis_result: dict):
    """故事生成标签页"""
    st.markdown("自动生成数据叙事报告")

    if st.button("📝 生成故事", type="primary"):
        with st.spinner("正在生成故事..."):
            llm_client = st.session_state.get("llm_client")

            generator = StoryGenerator(llm_client)
            story = generator.generate(data, analysis_result)

            st.markdown(f"## {story.title}")
            st.markdown(f"> {story.summary}")

            for section in story.sections:
                st.markdown(f"### {section.title}")
                st.markdown(section.content)

            if story.key_findings:
                st.markdown("### 🔍 关键发现")
                for finding in story.key_findings:
                    st.markdown(f"- {finding}")

            if story.recommendations:
                st.markdown("### 💡 建议")
                for rec in story.recommendations:
                    st.markdown(f"- {rec}")

            st.caption(f"生成时间: {story.generated_at[:19]}")