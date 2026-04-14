"""
Streamlit Web界面 - 主入口
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.config.storage import load_db_configs, load_llm_configs
from web.components.sidebar import render_sidebar
from web.components.single_analysis import single_file_mode
from web.components.multi_analysis import multi_file_mode
from web.components.db_analysis import database_mode

st.set_page_config(
    page_title="AutoStat 智能数据分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 初始化 Session State ====================

# 分析器
if 'single_analyzer' not in st.session_state:
    st.session_state.single_analyzer = None
if 'single_output' not in st.session_state:
    st.session_state.single_output = None
if 'multi_analyzer' not in st.session_state:
    st.session_state.multi_analyzer = None
if 'multi_output' not in st.session_state:
    st.session_state.multi_output = None
if 'db_analyzer' not in st.session_state:
    st.session_state.db_analyzer = None
if 'db_output' not in st.session_state:
    st.session_state.db_output = None

# 当前分析结果
if 'current_html' not in st.session_state:
    st.session_state.current_html = None
if 'current_json_data' not in st.session_state:
    st.session_state.current_json_data = None
if 'current_analysis_type' not in st.session_state:
    st.session_state.current_analysis_type = None
if 'current_source_name' not in st.session_state:
    st.session_state.current_source_name = None

# 大模型相关
if 'llm_client' not in st.session_state:
    st.session_state.llm_client = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'selected_contexts' not in st.session_state:
    st.session_state.selected_contexts = ["json_result"]
if 'raw_data_preview' not in st.session_state:
    st.session_state.raw_data_preview = None

# UI 状态
if 'selected_db_config' not in st.session_state:
    st.session_state.selected_db_config = None
if 'selected_llm_config' not in st.session_state:
    st.session_state.selected_llm_config = None

# ==================== 样式 ====================

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
    hr { margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 AutoStat 智能数据分析助手</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">自动识别数据类型、检测数据质量、选择统计方法、生成分析报告</div>', unsafe_allow_html=True)

# ==================== 侧边栏 ====================

analysis_mode = render_sidebar()

# ==================== 主入口 ====================

if analysis_mode == "📁 单文件分析":
    single_file_mode()
elif analysis_mode == "📚 多文件分析":
    multi_file_mode()
elif analysis_mode == "🗄️ 数据库分析":
    database_mode()