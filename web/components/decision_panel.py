"""
决策建议面板组件
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from autostat.core.decision import AnomalyDetector, RootCauseAnalyzer, ActionRecommender
from autostat.shared.storage import SharedStorage
from web.services.session_service import SessionService
from web.services.storage_service import StorageService


def render_decision_panel():
    """渲染决策建议面板"""
    st.markdown("### 💡 智能决策建议")
    st.caption("自动发现数据异常，定位根因，生成行动建议")

    session_id = SessionService.get_current_session()
    if session_id is None:
        st.info("请先完成数据分析")
        return

    # 加载数据
    data = StorageService.load_dataframe("processed_data", session_id)
    analysis_result = StorageService.load_json("analysis_result", session_id)

    if data is None or analysis_result is None:
        st.info("请先完成数据分析")
        return

    # ==================== 异常检测 ====================
    st.markdown("#### 🔍 异常检测")

    if st.button("🔄 检测异常", use_container_width=True):
        with st.spinner("正在检测异常..."):
            detector = AnomalyDetector()
            anomalies = detector.detect_from_analysis_result(analysis_result)

            if anomalies:
                SharedStorage.save_anomalies(session_id, anomalies)
                st.success(f"发现 {len(anomalies)} 个异常")
            else:
                st.info("未发现明显异常")

    # 加载已有异常
    anomalies = SharedStorage.load_anomalies(session_id)

    if anomalies:
        # 按严重程度分组显示
        critical = [a for a in anomalies if a.severity == "critical"]
        high = [a for a in anomalies if a.severity == "high"]
        medium = [a for a in anomalies if a.severity == "medium"]
        low = [a for a in anomalies if a.severity == "low"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔴 严重", len(critical))
        with col2:
            st.metric("🟠 高", len(high))
        with col3:
            st.metric("🟡 中", len(medium))
        with col4:
            st.metric("🟢 低", len(low))

        # 显示异常列表
        for anomaly in anomalies[:5]:
            severity_icon = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢"
            }.get(anomaly.severity, "⚪")

            with st.expander(f"{severity_icon} {anomaly.message} (严重: {anomaly.severity})", expanded=False):
                st.caption(f"目标: {anomaly.target}")
                st.caption(f"当前值: {anomaly.value}")
                st.caption(f"预期值: {anomaly.expected}")
                if anomaly.evidence:
                    st.caption("证据:")
                    for ev in anomaly.evidence[:3]:
                        st.caption(f"  - {ev}")

    st.divider()

    # ==================== 根因分析 ====================
    st.markdown("#### 🔎 根因分析")

    if anomalies and st.button("🔬 分析根因", use_container_width=True):
        with st.spinner("正在分析根因..."):
            # 选择最严重的异常
            selected = next((a for a in anomalies if a.severity in ["critical", "high"]), anomalies[0])

            # 获取维度和指标列
            variable_types = analysis_result.get("variable_types", {})
            dimension_cols = [
                col for col, info in variable_types.items()
                if info.get("type") in ["categorical", "categorical_numeric", "ordinal"]
            ]
            metric_cols = [
                col for col, info in variable_types.items()
                if info.get("type") == "continuous"
            ]

            if dimension_cols and metric_cols:
                analyzer = RootCauseAnalyzer()
                result = analyzer.analyze(
                    selected.__dict__,
                    data,
                    dimension_cols[:5],
                    metric_cols[:5]
                )

                st.markdown("**根因结论:**")
                st.info(result.summary)

                if result.root_causes:
                    for rc in result.root_causes:
                        st.markdown(f"- {rc.description} (置信度: {rc.confidence:.0%})")
            else:
                st.warning("缺少维度或指标列，无法进行根因分析")

    st.divider()

    # ==================== 行动建议 ====================
    st.markdown("#### 💡 行动建议")

    if anomalies and st.button("📋 生成建议", use_container_width=True):
        with st.spinner("正在生成行动建议..."):
            # 使用大模型（如果有）
            llm_client = st.session_state.get("llm_client")

            recommender = ActionRecommender(llm_client)

            # 使用最严重的异常
            selected = next((a for a in anomalies if a.severity in ["critical", "high"]), anomalies[0])

            suggestions = recommender.recommend(
                selected.__dict__,
                [],
                {"session_id": session_id, "table": analysis_result.get("source_table", "未知")}
            )

            if suggestions:
                SharedStorage.save_suggestions(session_id, suggestions)

                for s in suggestions:
                    priority_color = "🔴" if s.priority == "高" else "🟠" if s.priority == "中" else "🟢"
                    with st.container():
                        st.markdown(f"**{priority_color} {s.title}**")
                        st.caption(f"描述: {s.description}")
                        st.caption(f"预期效果: {s.expected_effect}")
                        st.caption(f"置信度: {s.confidence:.0%}")
                        if s.steps:
                            st.caption("执行步骤:")
                            for step in s.steps[:3]:
                                st.caption(f"  - {step}")
                        st.divider()
            else:
                st.info("暂无建议")

    # 加载已有建议
    suggestions = SharedStorage.load_suggestions(session_id)
    if suggestions:
        st.markdown("#### 📋 已有建议")
        for s in suggestions[:3]:
            st.caption(f"- {s.title} ({s.priority}优先级)")