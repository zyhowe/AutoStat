# Home.py
"""AutoStat/AutoText 统一入口 - 欢迎页"""

import streamlit as st

st.set_page_config(
    page_title="AutoStat 智能分析平台",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 全局样式
st.markdown("""
<style>
    /* 减少主容器顶部空白 */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0rem !important;
    }
    /* 调整 header 高度 */
    header {
        height: 2rem !important;
    }
    header .stDecoration {
        display: none !important;
    }
    .main-header {
        text-align: center;
        padding: 30px 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        color: white;
        margin-bottom: 30px;
    }
    .main-header h1 {
        font-size: 42px;
        margin-bottom: 10px;
    }
    .main-header p {
        font-size: 16px;
        opacity: 0.9;
    }
    .feature-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .feature-icon {
        font-size: 48px;
        margin-bottom: 12px;
    }
    .feature-title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #333;
    }
    .feature-desc {
        font-size: 13px;
        color: #666;
        line-height: 1.5;
    }
    .quick-start {
        background: #f0f2f5;
        border-radius: 16px;
        padding: 20px;
        margin-top: 30px;
        text-align: center;
    }
    .footer {
        text-align: center;
        padding: 30px 20px 20px;
        color: #999;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# 头部
st.markdown("""
<div class="main-header">
    <h1>🎯 AutoStat 智能分析平台</h1>
    <p>数据分析 · 文本分析 · AI 驱动</p>
</div>
""", unsafe_allow_html=True)

# 功能卡片
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">数据分析</div>
        <div class="feature-desc">
            智能统计分析工具，自动识别数据类型、检测数据质量、分析变量关系、生成专业报告。<br>
            支持 CSV、Excel、JSON、TXT 及 SQL Server 数据库。
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("进入数据分析 →", key="goto_analysis", use_container_width=True):
        st.switch_page("pages/autostat.py")

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📝</div>
        <div class="feature-title">文本分析</div>
        <div class="feature-desc">
            智能文本分析工具，支持关键词提取、情感分析、实体识别、文本聚类、主题建模。<br>
            大模型增强，自动抽取实体关系、事件、主题，构建知识图谱。
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("进入文本分析 →", key="goto_text", use_container_width=True):
        st.switch_page("pages/autotext.py")

# 快速开始区域
st.markdown("""
<div class="quick-start">
    <h3>🚀 快速开始</h3>
    <p>选择上方任意功能，上传数据即可开始分析</p>
    <p style="font-size: 12px; color: #888; margin-top: 12px;">💡 提示：首次使用请在侧边栏配置大模型 API 获取 AI 智能解读</p>
</div>
""", unsafe_allow_html=True)

# 页脚
st.markdown("""
<div class="footer">
    <p>🤖 AutoStat/AutoText 智能分析平台 | 版本 0.2.0</p>
</div>
""", unsafe_allow_html=True)