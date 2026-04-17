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
from web.components.results import render_preview_tab, render_log_tab, render_ai_tab
from web.services.cache_service import CacheService

st.set_page_config(
    page_title="AutoStat 智能数据分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 Session State
CacheService.init_session_state()

# 样式
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 1rem; }
    hr { margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 AutoStat 智能数据分析助手</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">自动识别数据类型、检测数据质量、选择统计方法、生成分析报告</div>', unsafe_allow_html=True)

# 侧边栏
analysis_mode = render_sidebar()

# 检查分析模式是否变化
if analysis_mode != st.session_state.get('analysis_mode'):
    st.session_state.analysis_mode = analysis_mode
    CacheService.reset_analysis_state()

# 渲染标签页
current_tab = render_tabs()

# 滚动到顶部（分析完成时）
if st.session_state.get('scroll_to_top', False):
    scroll_to_top()
    st.session_state.scroll_to_top = False

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
    if st.session_state.analysis_completed:
        render_log_tab()
    else:
        st.info("📌 请先在「数据准备」中上传数据并点击「开始分析」")
        st.caption("分析完成后，日志将显示在此处")
elif current_tab == 3:
    if st.session_state.analysis_completed and st.session_state.current_json_data:
        render_ai_tab()
    else:
        st.info("📌 请先在「数据准备」中上传数据并点击「开始分析」")
        st.caption("分析完成后，可使用 AI 进行智能解读")