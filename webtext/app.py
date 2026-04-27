"""
Streamlit Web界面 - 文本分析主入口
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webtext.components.sidebar import render_sidebar
from webtext.components.tabs import render_tabs, scroll_to_top
from webtext.components.data_preparation import render_data_preparation
from webtext.components.results import render_results_tab
from webtext.components.model_training import render_model_training
from webtext.components.chat_interface import render_chat_tab

st.set_page_config(
    page_title="AutoText 智能文本分析",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 Session State
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 0
if 'analysis_completed' not in st.session_state:
    st.session_state.analysis_completed = False
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'report_html' not in st.session_state:
    st.session_state.report_html = None
if 'report_json' not in st.session_state:
    st.session_state.report_json = None
if 'scroll_to_top' not in st.session_state:
    st.session_state.scroll_to_top = False

# 样式
st.markdown("""
<style>
    hr { margin: 10px 0; }
    .stButton button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# 侧边栏
render_sidebar()

# 检查是否需要滚动到顶部
if st.session_state.scroll_to_top:
    scroll_to_top()
    st.session_state.scroll_to_top = False

# 渲染标签页
current_tab = render_tabs()

if current_tab == 0:
    render_data_preparation()
elif current_tab == 1:
    render_results_tab()
elif current_tab == 2:
    render_model_training()
elif current_tab == 3:
    render_chat_tab()