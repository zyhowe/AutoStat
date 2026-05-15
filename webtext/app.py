# webtext/app.py
"""文本分析 Web 界面 - 可导入模块"""
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
from webtext.components.compare import render_compare_tab


def render_text_analysis():
    """文本分析主渲染函数"""

    # 初始化 session state
    init_session_state()

    # 侧边栏
    render_sidebar()

    # 检查是否需要滚动到顶部
    if st.session_state.get("text_scroll_to_top", False):
        scroll_to_top()
        st.session_state.text_scroll_to_top = False

    # 渲染标签页
    current_tab = render_tabs()

    # 根据当前标签页渲染内容
    if current_tab == 0:
        render_data_preparation()
    elif current_tab == 1:
        render_results_tab()
    elif current_tab == 2:
        render_chat_tab()
    elif current_tab == 3:
        render_compare_tab()


# 直接运行时入口（独立运行或调试时使用）
if __name__ == "__main__":
    # 仅在独立运行时设置页面配置
    st.set_page_config(
        page_title="AutoText 智能文本分析",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    render_text_analysis()