"""
模型中心组件 - 模型训练/预测/预警
"""

import streamlit as st
import pandas as pd

from web.services.session_service import SessionService
from web.services.storage_service import StorageService
from web.components.model_training import render_model_training


def render_model_center():
    """渲染模型中心 - 3个子Tab"""
    st.markdown("### 🤖 模型中心")
    st.caption("模型训练、预测、预警监控一体化管理")

    session_id = SessionService.get_current_session()
    if session_id is None:
        st.info("请先完成数据分析")
        return

    # 加载数据
    data = StorageService.load_dataframe("processed_data", session_id)

    # 子Tab
    tab1, tab2, tab3 = st.tabs(["🏋️ 模型训练", "🔮 预测", "🚨 预警监控"])

    with tab1:
        render_model_training()

    with tab2:
        if data is not None and not data.empty:
            render_forecast_tab(data, session_id)
        else:
            st.warning("请先完成数据分析，加载数据")

    with tab3:
        render_alert_tab(session_id)


def render_forecast_tab(data: pd.DataFrame, session_id: str):
    """预测子Tab"""
    from web.components.forecast_panel import render_forecast_tab as _render_forecast
    _render_forecast(data, session_id)


def render_alert_tab(session_id: str):
    """预警监控子Tab"""
    from web.components.forecast_panel import render_alert_tab as _render_alert
    _render_alert(session_id)


def render_monitor_tab(session_id: str):
    """监控子Tab"""
    from web.components.forecast_panel import render_monitor_tab as _render_monitor
    _render_monitor(session_id)