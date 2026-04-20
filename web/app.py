# web/app.py

"""
Streamlit Web界面 - 主入口
"""

import streamlit as st
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.components.sidebar import render_sidebar
from web.components.tabs import render_tabs, scroll_to_top
from web.components.data_preparation import render_data_preparation
from web.components.results import render_preview_tab, render_ai_tab
from web.components.model_training import render_model_training
from web.services.cache_service import CacheService
from web.services.session_service import SessionService

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
analysis_mode = render_sidebar()

# 检查分析模式是否变化
if analysis_mode != st.session_state.get('analysis_mode'):
    st.session_state.analysis_mode = analysis_mode
    CacheService.reset_analysis_state()

# 检查是否需要滚动到顶部
if st.session_state.scroll_to_top:
    scroll_to_top()
    st.session_state.scroll_to_top = False

# 渲染标签页
current_tab = render_tabs()

# 根据当前标签页渲染内容
if current_tab == 0:
    render_data_preparation(analysis_mode)
elif current_tab == 1:
    if st.session_state.analysis_completed and st.session_state.current_html:
        render_preview_tab()
    else:
        st.info("📌 请先在「数据准备」中上传数据并点击「开始分析」")
        st.caption("分析完成后，报告将显示在此处")
elif current_tab == 2:
    # 小模型训练标签页
    render_model_training()
elif current_tab == 3:
    # 大模型智能解读标签页
    if st.session_state.analysis_completed and st.session_state.current_json_data:
        render_ai_tab()
    else:
        st.info("📌 请先在「数据准备」中上传数据并点击「开始分析」")
        st.caption("分析完成后，可使用大模型进行智能解读")