"""多文件分析组件"""

import streamlit as st
import pandas as pd
import tempfile
import os
import json
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autostat import MultiTableStatisticalAnalyzer
from autostat.loader import DataLoader
from autostat.reporter import Reporter
from web.components.chat_interface import render_chat_interface
from web.utils.helpers import capture_and_run, get_raw_data_preview


def multi_file_mode():
    """多文件分析模式"""
    st.markdown("### 📚 多文件分析")

    files = st.file_uploader(
        "选择多个数据文件", type=['csv', 'xlsx', 'xls', 'json', 'txt'],
        accept_multiple_files=True, key="multi_uploader"
    )

    if files and len(files) >= 2:
        st.success(f"已选择 {len(files)} 个文件")

        info = [{"文件名": f.name, "大小": f"{len(f.getvalue()) / 1024:.1f} KB"} for f in files]
        st.dataframe(pd.DataFrame(info))

        with st.expander("🔗 手动定义表间关系（可选）"):
            rel_text = st.text_area("关系定义（每行一个）", placeholder="orders.user_id = users.user_id", key="multi_rel")
            relationships = parse_relationships(rel_text)

        if st.button("🚀 开始分析", type="primary", key="multi_start"):
            with st.spinner("分析中..."):
                tmp_dir, tables = load_tables(files)

                if not tables:
                    st.error("没有成功加载任何表")
                    shutil.rmtree(tmp_dir)
                    return

                def run():
                    if relationships:
                        a = MultiTableStatisticalAnalyzer(tables, relationships={'foreign_keys': relationships})
                    else:
                        a = MultiTableStatisticalAnalyzer(tables)
                    a.analyze_all_tables()
                    return a

                analyzer, output = capture_and_run(run)

                shutil.rmtree(tmp_dir)

                st.session_state.multi_analyzer = analyzer
                st.session_state.multi_output = output
                st.session_state.current_analysis_type = "multi"
                st.session_state.current_source_name = f"{len(files)}个文件"

                merged_analyzer = analyzer.get_merged_analyzer()
                reporter = Reporter(merged_analyzer)
                st.session_state.current_html = reporter.to_html()
                st.session_state.current_json_data = json.loads(merged_analyzer.to_json())

                st.session_state.raw_data_preview = get_raw_data_preview(merged_analyzer.data)

                st.session_state.chat_messages = []

                st.rerun()

    elif files and len(files) == 1:
        st.warning("多文件分析需要至少2个文件")

    # 显示结果
    display_results()


def parse_relationships(rel_text):
    """解析关系定义"""
    relationships = []
    if rel_text.strip():
        for line in rel_text.strip().split('\n'):
            if '=' in line:
                parts = line.split('=')
                if len(parts) == 2:
                    from_part, to_part = parts[0].strip(), parts[1].strip()
                    if '.' in from_part and '.' in to_part:
                        ft, fc = from_part.split('.')
                        tt, tc = to_part.split('.')
                        relationships.append({'from_table': ft, 'from_col': fc, 'to_table': tt, 'to_col': tc})
    return relationships


def load_tables(files):
    """加载所有表"""
    tmp_dir = tempfile.mkdtemp()
    paths = {}
    for f in files:
        p = os.path.join(tmp_dir, f.name)
        with open(p, 'wb') as w:
            w.write(f.getbuffer())
        paths[os.path.splitext(f.name)[0]] = p

    tables = {}
    for name, p in paths.items():
        try:
            tables[name] = DataLoader.load_from_file(p)
        except Exception as e:
            st.warning(f"加载 {name} 失败: {e}")
            tables[name] = None

    # 过滤空表
    tables = {k: v for k, v in tables.items() if v is not None and not v.empty}

    return tmp_dir, tables


def display_results():
    """显示分析结果"""
    if st.session_state.multi_analyzer is not None:
        analyzer = st.session_state.multi_analyzer
        output = st.session_state.multi_output

        st.success("✅ 分析完成！")

        col_success, col_dl1, col_dl2 = st.columns([3, 1, 1])
        with col_dl1:
            st.download_button("📥 下载 HTML 报告", st.session_state.current_html,
                               "autostat_multi_report.html", "text/html", use_container_width=True)
        with col_dl2:
            st.download_button("📥 下载 JSON 结果", analyzer.to_json(),
                               "autostat_multi_result.json", "application/json", use_container_width=True)

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