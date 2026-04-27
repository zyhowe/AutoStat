"""
数据准备组件 - 文本上传、选择、分析
"""

import streamlit as st
import pandas as pd
import tempfile
import os
from typing import Dict, Any

from autotext.analyzer import TextAnalyzer
from webtext.services.analysis_service import TextAnalysisService


def render_data_preparation():
    """渲染数据准备标签页"""
    st.markdown("### 📁 数据准备")
    st.caption("上传文本文件，或选择已有项目中的文本列")

    # 数据源选择
    source_type = st.radio(
        "选择数据源",
        options=["📄 上传文本文件", "📊 从数据分析项目选择"],
        horizontal=True
    )

    if source_type == "📄 上传文本文件":
        render_file_upload()
    else:
        render_dataframe_selection()


def render_file_upload():
    """渲染文件上传界面"""
    st.markdown("#### 上传文本文件")

    uploaded_files = st.file_uploader(
        "选择文本文件（支持 .txt，可多选）",
        type=['txt'],
        accept_multiple_files=True,
        help="支持 TXT 格式，可同时选择多个文件"
    )

    if not uploaded_files:
        return

    st.success(f"已选择 {len(uploaded_files)} 个文件")

    if st.button("▶️ 开始分析", type="primary", use_container_width=True):
        with st.spinner("正在分析文本..."):
            # 保存临时文件
            texts = []
            for f in uploaded_files:
                content = f.read().decode('utf-8', errors='ignore')
                texts.append(content)

            if not texts:
                st.error("没有有效的文本数据")
                return

            # 创建分析器并分析
            analyzer = TextAnalyzer(texts, source_name=f"{len(uploaded_files)}个文件", quiet=False)
            analyzer.generate_full_report()

            # 保存到 session
            st.session_state.analyzer = analyzer
            st.session_state.analysis_completed = True
            st.session_state.current_tab = 1
            st.session_state.scroll_to_top = True

            st.success("分析完成！")
            st.rerun()


def render_dataframe_selection():
    """渲染从 DataFrame 选择文本列"""
    st.markdown("#### 从已有项目选择文本列")

    # 检查是否有数据分析的会话
    from web.services.session_service import SessionService
    from web.services.storage_service import StorageService

    session_id = SessionService.get_current_session()
    if session_id is None:
        st.info("暂无数据分析项目，请先在数据分析中创建项目")
        st.caption("提示：在数据分析页面完成分析后，可在此处选择文本列进行文本分析")
        return

    # 加载处理后的数据
    processed_data = StorageService.load_dataframe("processed_data", session_id)
    if processed_data is None:
        st.warning("项目中没有数据")
        return

    st.info(f"当前项目: {session_id}")
    st.dataframe(processed_data.head(100))

    # 选择文本列
    text_cols = [col for col in processed_data.columns
                 if processed_data[col].dtype == 'object' or processed_data[col].dtype == 'string']

    if not text_cols:
        st.warning("未检测到文本类型的列")
        return

    text_col = st.selectbox("选择文本列", options=text_cols)

    # 可选：标题列
    title_col = st.selectbox("选择标题列（可选）", options=["无"] + text_cols)
    title_col = None if title_col == "无" else title_col

    # 可选：时间列
    time_cols = [col for col in processed_data.columns if 'date' in col.lower() or 'time' in col.lower()]
    time_col = st.selectbox("选择时间列（可选）", options=["无"] + time_cols)
    time_col = None if time_col == "无" else time_col

    if st.button("▶️ 开始分析", type="primary", use_container_width=True):
        with st.spinner("正在分析文本..."):
            # 提取文本
            texts = processed_data[text_col].fillna("").astype(str).tolist()
            texts = [t for t in texts if t.strip()]

            if not texts:
                st.error("没有有效的文本数据")
                return

            # 创建分析器
            analyzer = TextAnalyzer(texts, source_name=text_col, quiet=False)
            if title_col:
                analyzer.titles = processed_data[title_col].fillna("").astype(str).tolist()
            if time_col:
                analyzer.dates = processed_data[time_col].tolist()

            analyzer.generate_full_report()

            st.session_state.analyzer = analyzer
            st.session_state.analysis_completed = True
            st.session_state.current_tab = 1
            st.session_state.scroll_to_top = True

            st.success("分析完成！")
            st.rerun()