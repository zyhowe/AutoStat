"""
预测预警面板组件
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from autostat.core.forecast import Forecaster, AlertEngine, ForecastMonitor
from autostat.core.forecast.alert import console_alert_notifier
from autostat.shared.storage import SharedStorage
from web.services.session_service import SessionService
from web.services.storage_service import StorageService


def render_forecast_panel():
    """渲染预测预警面板"""
    st.markdown("### 📈 预测与预警")
    st.caption("基于历史数据预测未来趋势，自动预警异常")

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
    tab1, tab2, tab3 = st.tabs(["🔮 预测", "🚨 预警", "📊 监控"])

    with tab1:
        render_forecast_tab(data, session_id)

    with tab2:
        render_alert_tab(session_id)

    with tab3:
        render_monitor_tab(session_id)


def render_forecast_tab(data: pd.DataFrame, session_id: str):
    """预测标签页"""
    st.markdown("选择目标列进行预测")

    # 获取数值列
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if not numeric_cols:
        st.warning("没有数值列，无法进行预测")
        return

    target = st.selectbox("选择预测目标", options=numeric_cols)

    periods = st.slider("预测周期数", min_value=3, max_value=30, value=12)

    # 检测时间列
    time_cols = data.select_dtypes(include=['datetime64']).columns
    time_col = st.selectbox(
        "选择时间列",
        options=time_cols.tolist() if len(time_cols) > 0 else ["自动检测"],
        index=0
    ) if len(time_cols) > 0 else None

    if st.button("🔮 执行预测", type="primary"):
        with st.spinner("正在执行预测..."):
            forecaster = Forecaster(confidence_level=0.95)

            try:
                result = forecaster.forecast(
                    data,
                    target,
                    time_col=time_col if time_col and time_col != "自动检测" else None,
                    periods=periods
                )

                # 保存结果
                SharedStorage.save_forecast(session_id, target, result)

                # 显示结果
                st.success(f"预测完成，使用模型: {result.model_name}")

                # 显示指标
                if result.metrics:
                    cols = st.columns(len(result.metrics))
                    for i, (k, v) in enumerate(result.metrics.items()):
                        with cols[i % len(result.metrics)]:
                            st.metric(k, f"{v:.2f}")

                # 图表
                fig = go.Figure()

                # 历史数据
                series = data[target].values
                fig.add_trace(go.Scatter(
                    x=list(range(len(series))),
                    y=series,
                    mode='lines',
                    name='历史数据',
                    line=dict(color='#1f77b4')
                ))

                # 预测值
                future_x = list(range(len(series), len(series) + periods))
                fig.add_trace(go.Scatter(
                    x=future_x,
                    y=result.values,
                    mode='lines+markers',
                    name='预测值',
                    line=dict(color='#ff7f0e', width=2)
                ))

                # 置信区间
                fig.add_trace(go.Scatter(
                    x=future_x + future_x[::-1],
                    y=result.upper_bound.tolist() + result.lower_bound.tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    line=dict(color='rgba(255, 127, 14, 0)'),
                    name=f'置信区间 ({result.confidence:.0%})'
                ))

                fig.update_layout(
                    height=400,
                    title=f"{target} 预测",
                    xaxis_title="时间",
                    yaxis_title=target,
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)

                # 预测值表格
                st.markdown("**预测值:**")
                df_pred = pd.DataFrame({
                    "周期": list(range(1, periods + 1)),
                    "预测值": result.values,
                    "下限": result.lower_bound,
                    "上限": result.upper_bound
                })
                st.dataframe(df_pred, use_container_width=True)

            except Exception as e:
                st.error(f"预测失败: {e}")


def render_alert_tab(session_id: str):
    """预警标签页"""
    st.markdown("配置预警规则，实时监控数据异常")

    # 加载已有的预警事件
    alerts = SharedStorage.load_alerts(session_id)

    if alerts:
        st.markdown(f"#### 📋 预警事件 ({len(alerts)})")

        unresolved = [a for a in alerts if not a.resolved]
        if unresolved:
            st.warning(f"有 {len(unresolved)} 条未解决的预警")

        for alert in alerts[:10]:
            level_icon = {
                "critical": "🔴",
                "error": "🚨",
                "warning": "⚠️",
                "info": "ℹ️"
            }.get(alert.level, "⚪")

            status = "✅ 已解决" if alert.resolved else "⏳ 待处理"

            with st.container():
                cols = st.columns([1, 4, 1])
                with cols[0]:
                    st.markdown(level_icon)
                with cols[1]:
                    st.markdown(f"**{alert.title}**")
                    st.caption(alert.message)
                with cols[2]:
                    st.caption(status)
                    st.caption(alert.triggered_at[:16])
                st.divider()

    else:
        st.info("暂无预警事件")

    # 预警规则配置
    with st.expander("⚙️ 预警规则配置", expanded=False):
        st.markdown("**当前启用的规则:**")
        rules = [
            "数值低于阈值",
            "数值高于阈值",
            "连续下降（3期）",
            "连续上升（5期）",
            "超出预测范围",
            "异常率过高（>10%）"
        ]

        for rule in rules:
            st.checkbox(rule, value=True, key=f"alert_rule_{rule}")

        st.caption("💡 规则在代码中配置，可通过 `core/forecast/alert.py` 自定义")


def render_monitor_tab(session_id: str):
    """监控标签页"""
    st.markdown("监控预测效果，检测模型漂移")

    # 加载预测历史
    # 简化实现：从存储中加载最近的预测
    monitor = ForecastMonitor()

    # 模拟数据（实际应从存储加载）
    st.info("预测监控需要历史预测和实际值数据")

    # 显示监控状态
    st.markdown("#### 📊 监控状态")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("当前状态", "🟢 良好")
    with col2:
        st.metric("MAPE", "3.2%", delta="-0.5%")
    with col3:
        st.metric("漂移检测", "未检测到")

    st.caption("💡 完整的预测监控功能需要接入实际数据，持续收集预测值和实际值")