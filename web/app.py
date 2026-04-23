# web/app.py

"""
Streamlit Web界面 - 主入口
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.components.sidebar import render_sidebar
from web.components.tabs import render_tabs, scroll_to_top
from web.components.data_preparation import render_data_preparation
from web.components.results import render_preview_tab, render_ai_tab
from web.components.model_training import render_model_training
from web.services.cache_service import CacheService
from web.services.session_service import SessionService
from web.services.feature_flags import FeatureFlags  # 保留功能开关

st.set_page_config(
    page_title="AutoStat 智能数据分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 Session State
CacheService.init_session_state()

# 初始化滚动标志
if 'scroll_to_top' not in st.session_state:
    st.session_state.scroll_to_top = False

# 样式
st.markdown("""
<style>
    hr { margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# 侧边栏
render_sidebar()

# 检查是否需要滚动到顶部
if st.session_state.scroll_to_top:
    scroll_to_top()
    st.session_state.scroll_to_top = False

# 渲染功能开关弹窗（只渲染，不自动打开）
#FeatureFlags.render_settings_dialog()

# web/app.py (在 render_tabs() 之前添加)

# 处理核心结论和推理预测中的操作按钮
if st.session_state.get("pending_action"):
    action = st.session_state.pending_action
    st.session_state.pending_action = None

    if action == "cluster":
        # 跳转到模型训练页，自动填充聚类配置
        st.session_state.current_tab = 2
        st.session_state.auto_fill_task_type = "clustering"
        st.session_state.auto_fill_features = None  # 让系统自动选择所有数值列
        st.session_state.auto_fill_model_key = "kmeans"
        st.session_state.auto_fill_model_name = "自动_聚类分析_KMeans"
        st.rerun()
    elif action == "association":
        # 跳转到AI解读页，发送关联规则分析请求
        st.session_state.current_tab = 3
        st.session_state.pending_question = "请对分类变量进行关联规则挖掘，找出频繁项集和强关联规则，并解释业务含义。"
        st.rerun()
    elif action == "outlier":
        # 跳转到AI解读页，发送异常值分析请求
        st.session_state.current_tab = 3
        st.session_state.pending_question = "请分析数据中的异常值，判断哪些是数据错误，哪些是有意义的异常，并给出处理建议。"
        st.rerun()


# 渲染标签页
current_tab = render_tabs()

# 根据当前标签页渲染内容
if current_tab == 0:
    render_data_preparation()
elif current_tab == 1:
    session_id = SessionService.get_current_session()
    if session_id is not None:
        render_preview_tab()
    else:
        st.info("📌 请先在「数据准备」中上传数据并点击「开始分析」，或从侧边栏选择历史项目")
        st.caption("分析完成后，报告将显示在此处")
elif current_tab == 2:
    session_id = SessionService.get_current_session()
    if session_id is not None:
        render_model_training()
    else:
        st.info("📌 请先在「数据准备」中上传数据并点击「开始分析」，或从侧边栏选择历史项目")
        st.caption("分析完成后，可基于标准化数据进行模型训练")
elif current_tab == 3:
    session_id = SessionService.get_current_session()
    if session_id is not None:
        render_ai_tab()
    else:
        st.info("📌 请先在「数据准备」中上传数据并点击「开始分析」，或从侧边栏选择历史项目")
        st.caption("分析完成后，可使用大模型进行智能解读")