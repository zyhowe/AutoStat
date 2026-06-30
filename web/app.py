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
from web.components.data_preparation import render_data_preparation
from web.components.preview_report import render_preview_report
from web.components.model_center import render_model_center
from web.components.ai_assistant import render_ai_assistant
from web.components.results import render_compare_tab
from web.services.cache_service import CacheService


def render_analysis():
    """数据分析主渲染函数"""

    # 初始化 Session State
    CacheService.init_session_state()

    # 初始化滚动标志
    if 'scroll_to_top' not in st.session_state:
        st.session_state.scroll_to_top = False

    # ==================== 修复标签页被遮挡 ====================
    st.markdown("""
    <style>
        /* 减少页面顶部空白 */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 0rem !important;
        }
        /* 调整 header 高度 */
        header {
            height: 1.5rem !important;
        }
        header .stDecoration {
            display: none !important;
        }
        /* 调整 main 区域 */
        .main > div {
            padding-top: 0rem !important;
        }
        /* 标签页样式 - 更紧凑 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0px !important;
            background-color: #f0f2f6;
            border-radius: 8px;
            padding: 2px 4px !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 6px 16px !important;
            font-size: 14px !important;
            height: 36px !important;
            min-height: 36px !important;
            border-radius: 6px !important;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: white !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
        }
        /* 侧边栏样式 */
        section[data-testid="stSidebar"] {
            padding-top: 0.5rem !important;
        }
        /* 减少标题区域空白 */
        .stApp > header {
            background-color: transparent !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # 侧边栏
    render_sidebar()

    # 检查是否需要滚动到顶部
    if st.session_state.scroll_to_top:
        st.session_state.scroll_to_top = False

    # ==================== 5标签页 ====================
    tab_names = [
        "📁 数据准备",
        "📄 预览报告",
        "🤖 模型中心",
        "🧠 AI助手",
        "🔍 项目对比"
    ]

    # 使用 st.tabs，标签页会更紧凑
    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_data_preparation()

    with tabs[1]:
        render_preview_report()

    with tabs[2]:
        render_model_center()

    with tabs[3]:
        render_ai_assistant()

    with tabs[4]:
        render_compare_tab()


# 直接运行时入口
if __name__ == "__main__":
    st.set_page_config(
        page_title="AutoStat 智能数据分析",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    render_analysis()