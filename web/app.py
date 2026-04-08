"""
Streamlit Web界面
"""

import streamlit as st
import pandas as pd
import sys
import os
import tempfile

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
    # 显示数据预览
    st.subheader("📄 数据预览")

    # 根据文件类型读取预览
    file_extension = uploaded_file.name.split('.')[-1].lower()

    try:
        if file_extension == 'csv':
            df_preview = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df_preview = pd.read_excel(uploaded_file)
        elif file_extension == 'json':
            df_preview = pd.read_json(uploaded_file)
        else:
            df_preview = pd.read_csv(uploaded_file, delimiter='\t')

        st.dataframe(df_preview.head(100))
        st.write(f"数据形状: {df_preview.shape[0]} 行 × {df_preview.shape[1]} 列")

        # 分析按钮
        if st.button("🚀 开始分析", type="primary"):
            with st.spinner("分析中，请稍候..."):
                try:
                    # 保存临时文件（使用系统临时目录）
                    with tempfile.NamedTemporaryFile(
                        mode='w',
                        suffix=f'.{file_extension}',
                        encoding='utf-8',
                        delete=False
                    ) as tmp_file:
                        if file_extension == 'csv':
                            df_preview.to_csv(tmp_file.name, index=False)
                        elif file_extension in ['xlsx', 'xls']:
                            df_preview.to_excel(tmp_file.name, index=False)
                        elif file_extension == 'json':
                            df_preview.to_json(tmp_file.name, orient='records', force_ascii=False)
                        else:
                            df_preview.to_csv(tmp_file.name, sep='\t', index=False)
                        temp_path = tmp_file.name

                    # 创建分析器
                    analyzer = AutoStatisticalAnalyzer(temp_path, auto_clean=False, quiet=True)
                    reporter = Reporter(analyzer)

                    # 生成报告
                    report_path = os.path.join(tempfile.gettempdir(), 'autostat_report.html')
                    reporter.to_html(report_path)

                    # 显示结果摘要
                    st.success("分析完成！")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("总行数", f"{len(analyzer.data):,}")
                    with col2:
                        st.metric("总列数", len(analyzer.data.columns))
                    with col3:
                        missing_count = len([m for m in analyzer.quality_report.get('missing', []) if m.get('percent', 0) > 20])
                        st.metric("高缺失字段", missing_count)

                    # 变量类型分布
                    st.subheader("📋 变量类型分布")
                    type_counts = {}
                    for typ in analyzer.variable_types.values():
                        type_counts[typ] = type_counts.get(typ, 0) + 1

                    type_desc = {
                        'continuous': '连续变量',
                        'categorical': '分类变量',
                        'categorical_numeric': '数值型分类',
                        'ordinal': '有序分类',
                        'datetime': '日期时间',
                        'identifier': '标识符',
                        'text': '文本',
                        'other': '其他'
                    }

                    for typ, count in type_counts.items():
                        st.write(f"- {type_desc.get(typ, typ)}: {count} 列")

                    # 下载报告
                    with open(report_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()

                    st.download_button(
                        label="📥 下载完整报告 (HTML)",
                        data=html_content,
                        file_name="autostat_report.html",
                        mime="text/html"
                    )

                    # 清理临时文件
                    try:
                        os.unlink(temp_path)
                        os.unlink(report_path)
                    except:
                        pass

                except Exception as e:
                    st.error(f"分析失败: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    except Exception as e:
        st.error(f"读取文件失败: {str(e)}")