# pages/2_📝_文本分析.py
"""文本分析页面 - Streamlit 多页模式"""

import streamlit as st
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webtext.app import render_text_analysis

# 设置页面配置
st.set_page_config(
    page_title="AutoText 智能文本分析",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 调整 CSS 以减少页面顶部空白
st.markdown("""
<style>
    /* 减少主容器顶部空白 */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0rem !important;
    }
    /* 调整 header 高度 */
    header {
        height: 2rem !important;
    }
    header .stDecoration {
        display: none !important;
    }
    /* 调整 main 区域 */
    .main > div {
        padding-top: 0rem !important;
    }
    /* 标签页样式优化 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
    }
    /* 侧边栏样式 */
    section[data-testid="stSidebar"] {
        padding-top: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# 渲染文本分析界面
render_text_analysis()