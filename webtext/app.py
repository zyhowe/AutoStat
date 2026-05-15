# webtext/app.py
"""AutoText 智能文本分析 - Web 界面主入口"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webtext.components.sidebar import render_sidebar, init_session_state
from webtext.components.tabs import render_tabs, scroll_to_top
from webtext.components.data_preparation import render_data_preparation
from webtext.components.results import render_results_tab
from webtext.components.chat_interface import render_chat_tab
from webtext.components.compare import render_compare_tab  # 新增

st.set_page_config(
    page_title="AutoText智能文本分析",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_session_state()

st.markdown("""
<style>
    hr { margin: 10px 0; }
    .stButton button { width: 100%; }
</style>
""", unsafe_allow_html=True)

render_sidebar()

if st.session_state.get("text_scroll_to_top", False):
    scroll_to_top()
    st.session_state.text_scroll_to_top = False

current_tab = render_tabs()

if current_tab == 0:
    render_data_preparation()
elif current_tab == 1:
    render_results_tab()
elif current_tab == 2:
    render_chat_tab()
elif current_tab == 3:
    render_compare_tab()  # 新增