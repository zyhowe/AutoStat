"""
文本分类模型训练组件 - 占位实现
"""

import streamlit as st


def render_model_training():
    """渲染模型训练标签页"""
    st.markdown("### 🤖 文本分类")
    st.caption("基于文本内容训练分类模型（开发中）")

    st.info("📌 文本分类功能正在开发中，敬请期待")

    st.markdown("#### 计划支持的功能")
    st.markdown("""
    - 情感分类（积极/消极/中性）
    - 主题分类
    - 自定义标签分类
    - 模型：TextCNN、LSTM、BERT
    """)