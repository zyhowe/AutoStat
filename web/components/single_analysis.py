"""单文件分析组件"""

import streamlit as st
import pandas as pd
import tempfile
import os
import io
import json
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autostat import AutoStatisticalAnalyzer
from autostat.reporter import Reporter
from web.components.chat_interface import render_chat_interface
from web.utils.helpers import capture_and_run, get_raw_data_preview


def single_file_mode():
    """单文件分析模式"""
    st.markdown("### 📁 单文件分析")

    uploaded = st.file_uploader(
        "选择数据文件", type=['csv', 'xlsx', 'xls', 'json', 'txt'],
        help="支持 CSV、Excel、JSON、TXT 格式",
        key="single_uploader"
    )

    if uploaded:
        ext = uploaded.name.split('.')[-1].lower()
        date_level = st.selectbox("日期特征级别", ['basic', 'none'], index=0, key="single_date")

        try:
            df = load_file(uploaded, ext)

            if df is None:
                st.error("文件加载失败")
                return

            # 清理列名
            df.columns = [str(col).strip().replace('\n', '_').replace('\r', '_') for col in df.columns]

            # 处理空字符串为NaN
            df = df.replace(r'^\s*$', pd.NA, regex=True)

            # 尝试解析日期列
            for col in df.columns:
                if 'date' in col.lower() or '时间' in col or '日期' in col:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        pass

            st.subheader("📄 数据预览")
            st.dataframe(df.head(100))
            st.write(f"形状: {df.shape[0]} 行 × {df.shape[1]} 列")

            if st.button("🚀 开始分析", type="primary", key="single_start"):
                with st.spinner("分析中..."):
                    with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{ext}', delete=False) as tmp:
                        save_file(df, tmp, ext)
                        path = tmp.name

                    def run():
                        a = AutoStatisticalAnalyzer(path, auto_clean=False, quiet=False, date_features_level=date_level)
                        a.generate_full_report()
                        return a

                    analyzer, output = capture_and_run(run)

                    try:
                        os.unlink(path)
                    except:
                        pass

                    st.session_state.single_analyzer = analyzer
                    st.session_state.single_output = output
                    st.session_state.current_analysis_type = "single"
                    st.session_state.current_source_name = uploaded.name

                    # 生成 HTML 和 JSON
                    reporter = Reporter(analyzer)
                    st.session_state.current_html = reporter.to_html()
                    st.session_state.current_json_data = json.loads(analyzer.to_json())

                    # 保存原始数据预览
                    st.session_state.raw_data_preview = get_raw_data_preview(df)

                    # 重置聊天历史
                    st.session_state.chat_messages = []

                    st.rerun()

        except Exception as e:
            st.error(f"读取失败: {str(e)}")

    # 显示结果
    display_results()


def load_file(uploaded, ext):
    """加载文件"""
    if ext == 'csv':
        return load_csv(uploaded)
    elif ext in ['xlsx', 'xls']:
        return load_excel(uploaded, ext)
    elif ext == 'json':
        return load_json(uploaded)
    else:
        return pd.read_csv(uploaded, delimiter='\t', engine='python', on_bad_lines='skip')


def load_csv(uploaded):
    """加载 CSV 文件"""
    try:
        content = uploaded.read()
        content = content.replace(b'\x00', b'')
        content_str = content.decode('utf-8', errors='ignore')
        return pd.read_csv(io.StringIO(content_str))
    except:
        uploaded.seek(0)
        return pd.read_csv(uploaded, encoding='utf-8', engine='python', on_bad_lines='skip')


def load_excel(uploaded, ext):
    """加载 Excel 文件"""
    with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as tmp_excel:
        tmp_excel.write(uploaded.getbuffer())
        tmp_excel_path = tmp_excel.name

    try:
        df = pd.read_excel(tmp_excel_path, engine='openpyxl')
    except:
        try:
            df = pd.read_excel(tmp_excel_path, engine='xlrd')
        except:
            df = pd.read_excel(tmp_excel_path)

    os.unlink(tmp_excel_path)
    return df


def load_json(uploaded):
    """加载 JSON 文件"""
    content = uploaded.read()
    content_str = content.decode('utf-8', errors='ignore')
    content_str = re.sub(r'[\x00-\x1f\x7f]', '', content_str)

    try:
        data = json.loads(content_str)
        return pd.DataFrame(data)
    except:
        lines = content_str.strip().split('\n')
        data_list = []
        for line in lines:
            if line.strip():
                data_list.append(json.loads(line))
        return pd.DataFrame(data_list)


def save_file(df, tmp, ext):
    """保存文件到临时文件"""
    if ext == 'csv':
        df.to_csv(tmp.name, index=False, encoding='utf-8')
    elif ext in ['xlsx', 'xls']:
        df.to_excel(tmp.name, index=False)
    elif ext == 'json':
        df.to_json(tmp.name, orient='records', force_ascii=False)
    else:
        df.to_csv(tmp.name, sep='\t', index=False, encoding='utf-8')


def display_results():
    """显示分析结果"""
    if st.session_state.single_analyzer is not None:
        analyzer = st.session_state.single_analyzer
        output = st.session_state.single_output

        st.success("✅ 分析完成！")

        col_success, col_dl1, col_dl2 = st.columns([3, 1, 1])
        with col_dl1:
            st.download_button("📥 下载 HTML 报告", st.session_state.current_html,
                               "autostat_report.html", "text/html", use_container_width=True)
        with col_dl2:
            st.download_button("📥 下载 JSON 结果", analyzer.to_json(),
                               "autostat_result.json", "application/json", use_container_width=True)

        with st.expander("📝 分析过程日志", expanded=False):
            st.code(output, language='text')

        with st.expander("📄 预览报告", expanded=False):
            if st.session_state.current_html:
                st.html(st.session_state.current_html)
            else:
                st.info("暂无报告预览")

        with st.expander("🤖 AI 智能解读", expanded=False):
            if st.session_state.llm_client is None:
                st.warning("请先在侧边栏配置大模型")
            elif st.session_state.current_json_data is None:
                st.info("请先完成数据分析")
            else:
                render_chat_interface()