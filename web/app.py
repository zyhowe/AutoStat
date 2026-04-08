"""
Streamlit Web界面
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autostat.analyzer import AutoStatisticalAnalyzer
from autostat.reporter import Reporter

st.set_page_config(page_title="AutoStat 智能数据分析", layout="wide")

st.title("📊 AutoStat 智能数据分析助手")
st.markdown("上传数据文件，自动分析并生成报告")

uploaded_file = st.file_uploader(
    "选择数据文件",
    type=['csv', 'xlsx', 'xls', 'json', 'txt'],
    help="支持 CSV、Excel、JSON、TXT 格式"
)

if uploaded_file is not None:
    # 保存临时文件
    temp_path = f"/tmp/{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 显示数据预览
    st.subheader("📄 数据预览")
    df_preview = pd.read_csv(temp_path) if uploaded_file.name.endswith('.csv') else pd.read_excel(temp_path)
    st.dataframe(df_preview.head(100))
    st.write(f"数据形状: {df_preview.shape[0]} 行 × {df_preview.shape[1]} 列")

    # 分析按钮
    if st.button("🚀 开始分析", type="primary"):
        with st.spinner("分析中，请稍候..."):
            try:
                analyzer = AutoStatisticalAnalyzer(temp_path, auto_clean=False, quiet=True)
                reporter = Reporter(analyzer)

                # 生成报告
                report_path = "/tmp/report.html"
                reporter.to_html(report_path)

                # 显示结果摘要
                st.success("分析完成！")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总行数", f"{len(analyzer.data):,}")
                with col2:
                    st.metric("总列数", len(analyzer.data.columns))
                with col3:
                    missing_count = len([m for m in analyzer.quality_report.get('missing', []) if m['percent'] > 20])
                    st.metric("高缺失字段", missing_count)

                # 变量类型分布
                st.subheader("📋 变量类型分布")
                type_counts = {}
                for typ in analyzer.variable_types.values():
                    type_counts[typ] = type_counts.get(typ, 0) + 1
                st.json(type_counts)

                # 下载报告
                with open(report_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.download_button(
                    label="📥 下载完整报告 (HTML)",
                    data=html_content,
                    file_name="analysis_report.html",
                    mime="text/html"
                )

            except Exception as e:
                st.error(f"分析失败: {str(e)}")

    # 清理临时文件
    if os.path.exists(temp_path):
        os.remove(temp_path)