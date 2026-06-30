"""
质量看板组件 - 用于预览报告内嵌
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from autostat.core.quality import QualityScorer, QualityMonitor
from autostat.shared.storage import SharedStorage
from web.services.session_service import SessionService
from web.services.storage_service import StorageService


def render_quality_dashboard():
    """独立质量看板页面（原质量监控标签页）"""
    st.markdown("### 📊 数据质量看板")
    st.caption("实时监控数据质量五维评分和趋势")
    _render_quality_content()


def render_quality_dashboard_inline():
    """内嵌质量看板（用于预览报告折叠）"""
    _render_quality_content()


def _render_quality_content():
    """质量看板核心内容"""
    session_id = SessionService.get_current_session()
    if session_id is None:
        st.info("请先完成数据分析")
        return

    # 加载质量数据
    quality_score = SharedStorage.load_quality_score(session_id)
    quality_history = SharedStorage.load_quality_history(session_id)

    if quality_score is None:
        st.info("暂无质量评分数据，请先执行质量检查")
        if st.button("🔍 执行质量检查", key="quality_check_btn"):
            _run_quality_check(session_id)
        return

    # ==================== 概览卡片 ====================
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "综合评分",
            f"{quality_score.overall_score:.1f}",
            delta=f"{quality_score.grade} {quality_score.grade_icon}"
        )

    with col2:
        st.metric("完整性", f"{quality_score.dimensions.get('completeness', 0):.1f}%")

    with col3:
        st.metric("准确性", f"{quality_score.dimensions.get('accuracy', 0):.1f}%")

    with col4:
        st.metric("一致性", f"{quality_score.dimensions.get('consistency', 0):.1f}%")

    with col5:
        st.metric("唯一性", f"{quality_score.dimensions.get('uniqueness', 0):.1f}%")

    st.divider()

    # ==================== 趋势图 ====================
    if quality_history:
        st.markdown("#### 📈 质量趋势")

        df_history = pd.DataFrame(quality_history)
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_history['timestamp'],
            y=df_history['overall_score'],
            mode='lines+markers',
            name='综合评分',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))

        colors = ['#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
        dims = ['completeness', 'accuracy', 'consistency', 'uniqueness']
        dim_labels = ['完整性', '准确性', '一致性', '唯一性']

        for dim, label, color in zip(dims, dim_labels, colors):
            values = [h.get('dimensions', {}).get(dim, 0) * 100 for h in quality_history]
            fig.add_trace(go.Scatter(
                x=df_history['timestamp'],
                y=values,
                mode='lines',
                name=label,
                line=dict(color=color, width=1.5, dash='dash')
            ))

        fig.update_layout(
            height=350,
            title="质量评分趋势",
            xaxis_title="时间",
            yaxis_title="评分",
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

    # ==================== 告警列表 ====================
    st.markdown("#### 🚨 告警列表")

    alerts = quality_score.alerts if quality_score else []

    if alerts:
        for alert in alerts[:10]:
            level = alert.get('level', 'warning')
            level_icon = "🔴" if level == "error" else "🟡" if level == "warning" else "ℹ️"
            st.markdown(f"{level_icon} {alert.get('message', '')}")

        if len(alerts) > 10:
            st.caption(f"... 还有 {len(alerts) - 10} 条告警")
    else:
        st.success("✅ 暂无告警，数据质量良好")

    # ==================== 字段评分 ====================
    with st.expander("📋 字段评分详情", expanded=False):
        field_scores = quality_score.field_scores if quality_score else {}
        if field_scores:
            df_fields = pd.DataFrame([
                {
                    "字段": col,
                    "完整性": scores.get('completeness', 0),
                    "准确性": scores.get('accuracy', 0)
                }
                for col, scores in field_scores.items()
            ]).sort_values("完整性")

            st.dataframe(df_fields, use_container_width=True)
        else:
            st.caption("暂无字段评分数据")


def _run_quality_check(session_id: str):
    """执行质量检查"""
    data = StorageService.load_dataframe("processed_data", session_id)
    if data is None:
        st.error("未找到数据")
        return

    with st.spinner("正在执行质量检查..."):
        scorer = QualityScorer()
        result = scorer.score(data, table_name=session_id)

        SharedStorage.save_quality_score(session_id, result)

        st.success("质量检查完成")
        st.rerun()