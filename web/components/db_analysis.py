"""数据库分析组件"""

import streamlit as st
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autostat import MultiTableStatisticalAnalyzer
from autostat.loader import DataLoader
from autostat.reporter import Reporter
from web.components.chat_interface import render_chat_interface
from web.utils.helpers import capture_and_run, get_raw_data_preview


def database_mode():
    """数据库分析模式"""
    st.markdown("### 🗄️ 数据库分析")

    if st.session_state.selected_db_config is None:
        st.warning("请先在侧边栏配置并选择数据库")
        return

    config = st.session_state.selected_db_config

    with st.form("db_form"):
        st.info(f"使用配置: {config.get('name')}")
        st.caption(f"服务器: {config.get('server')} | 数据库: {config.get('database')}")

        tables_input = st.text_area("表名列表（每行一个）", placeholder="users\norders", key="db_tables")
        rel_input = st.text_area("关系定义（可选）", placeholder="orders.user_id = users.user_id", key="db_rels")
        limit = st.number_input("最大加载行数", min_value=100, max_value=100000, value=5000, key="db_limit")

        submitted = st.form_submit_button("🔌 连接并分析", type="primary")

    if submitted:
        if not tables_input.strip():
            st.error("请至少输入一个表名")
            return

        table_names = [t.strip() for t in tables_input.strip().split('\n') if t.strip()]
        relationships = parse_relationships(rel_input)

        with st.spinner("连接并分析..."):
            try:
                tables = DataLoader.load_multiple_tables(
                    server=config.get('server'),
                    database=config.get('database'),
                    table_names=table_names,
                    username=config.get('username') if not config.get('trusted_connection') else None,
                    password=config.get('password') if not config.get('trusted_connection') else None,
                    trusted_connection=config.get('trusted_connection', False),
                    limit=limit,
                    relationships=relationships if relationships else None
                )

                success = {n: df for n, df in tables.items() if df is not None and not df.empty}

                if not success:
                    st.error("没有成功加载任何表")
                    return

                st.success(f"成功加载 {len(success)} 个表")

                def run():
                    if relationships:
                        a = MultiTableStatisticalAnalyzer(success, relationships={'foreign_keys': relationships})
                    else:
                        a = MultiTableStatisticalAnalyzer(success)
                    a.analyze_all_tables()
                    return a

                analyzer, output = capture_and_run(run)

                st.session_state.db_analyzer = analyzer
                st.session_state.db_output = output
                st.session_state.current_analysis_type = "database"
                st.session_state.current_source_name = f"{config.get('name')}/{config.get('database')}"

                merged_analyzer = analyzer.get_merged_analyzer()
                reporter = Reporter(merged_analyzer)
                st.session_state.current_html = reporter.to_html()
                json_data = json.loads(merged_analyzer.to_json())
                json_data['db_server'] = config.get('server')
                json_data['db_database'] = config.get('database')
                st.session_state.current_json_data = json_data

                st.session_state.raw_data_preview = get_raw_data_preview(merged_analyzer.data)

                st.session_state.chat_messages = []

                st.rerun()

            except Exception as e:
                st.error(f"连接失败: {str(e)}")

    # 显示结果
    display_results()


def parse_relationships(rel_input):
    """解析关系定义"""
    relationships = []
    if rel_input.strip():
        for line in rel_input.strip().split('\n'):
            if '=' in line:
                parts = line.split('=')
                if len(parts) == 2:
                    from_part, to_part = parts[0].strip(), parts[1].strip()
                    if '.' in from_part and '.' in to_part:
                        ft, fc = from_part.split('.')
                        tt, tc = to_part.split('.')
                        relationships.append({'from_table': ft, 'from_col': fc, 'to_table': tt, 'to_col': tc})
    return relationships


def display_results():
    """显示分析结果"""
    if st.session_state.db_analyzer is not None:
        analyzer = st.session_state.db_analyzer
        output = st.session_state.db_output

        st.success("✅ 分析完成！")

        col_success, col_dl1, col_dl2 = st.columns([3, 1, 1])
        with col_dl1:
            st.download_button("📥 下载 HTML 报告", st.session_state.current_html,
                               "autostat_db_report.html", "text/html", use_container_width=True)
        with col_dl2:
            st.download_button("📥 下载 JSON 结果", analyzer.to_json(),
                               "autostat_db_result.json", "application/json", use_container_width=True)

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