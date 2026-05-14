# webtext/components/results.py
"""报告预览组件 - HTML展示和下载"""

import streamlit as st
import json
from datetime import datetime


def render_results_tab():
    """渲染报告预览标签页"""
    st.markdown("### 📄 报告预览")

    if not st.session_state.get("text_analysis_completed"):
        st.info("请先在「数据准备」中输入文本并点击「开始分析」")
        return

    html_content = st.session_state.text_html_content
    json_data = st.session_state.text_json_data
    analyzer = st.session_state.text_analyzer

    if not html_content or not json_data:
        st.warning("暂无分析结果，请重新分析")
        return

    # 导出按钮区域
    st.markdown("#### 📥 导出报告")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📄 下载 HTML 报告",
            html_content,
            f"text_report_{timestamp}.html",
            "text/html",
            use_container_width=True,
            key="text_download_html"
        )
    with col2:
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
        st.download_button(
            "📋 下载 JSON 结果",
            json_str,
            f"text_result_{timestamp}.json",
            "application/json",
            use_container_width=True,
            key="text_download_json"
        )

    st.markdown("---")

    # 显示完整 HTML 报告
    st.markdown("#### 📊 完整分析报告")
    # 使用 st.components.v1.html 替代 st.html，更好地支持 JS 图表
    try:
        # 设置一个较高的高度以容纳图表
        st.components.v1.html(html_content, height=800, scrolling=True)
    except Exception as e:
        # 如果失败，回退到 st.html
        st.warning(f"图表渲染失败，使用简化视图: {str(e)}")
        st.html(html_content)