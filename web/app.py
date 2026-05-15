# web/app.py
"""数据分析 Web 界面 - 可导入模块"""
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.components.sidebar import render_sidebar
from web.components.tabs import render_tabs, scroll_to_top
from web.components.data_preparation import render_data_preparation
from web.components.results import render_preview_tab, render_ai_tab, render_compare_tab
from web.components.model_training import render_model_training
from web.services.cache_service import CacheService
from web.services.session_service import SessionService
from web.services.feature_flags import FeatureFlags


def render_analysis():
    """数据分析主渲染函数"""

    # 初始化 Session State
    CacheService.init_session_state()

    # 初始化滚动标志
    if 'scroll_to_top' not in st.session_state:
        st.session_state.scroll_to_top = False

    # 侧边栏
    render_sidebar()

    # 检查是否需要滚动到顶部
    if st.session_state.scroll_to_top:
        scroll_to_top()
        st.session_state.scroll_to_top = False

    # 渲染标签页
    current_tab = render_tabs()

    # 根据当前标签页渲染内容
    if current_tab == 0:
        render_data_preparation()
    elif current_tab == 1:
        render_preview_tab()
    elif current_tab == 2:
        render_model_training()
    elif current_tab == 3:
        render_ai_tab()
    elif current_tab == 4:
        render_compare_tab()


# 直接运行时入口（独立运行或调试时使用）
if __name__ == "__main__":
    # 仅在独立运行时设置页面配置
    st.set_page_config(
        page_title="AutoStat 智能数据分析",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    render_analysis()