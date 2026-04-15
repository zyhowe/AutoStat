"""数据库分析组件"""

import streamlit as st
import json
import sys
import os
import traceback
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autostat import MultiTableStatisticalAnalyzer
from autostat.loader import DataLoader
from autostat.reporter import Reporter
from web.components.chat_interface import render_chat_interface
from web.utils.helpers import capture_and_run, get_raw_data_preview
from web.utils.data_preprocessor import render_multi_preprocessing_interface


def database_mode():
    """数据库分析模式"""
    st.markdown("### 🗄️ 数据库分析")

    # 注意事项提示
    with st.expander("ℹ️ 使用说明与注意事项", expanded=False):
        st.markdown("""
        **适用场景：** 需要分析 SQL Server 数据库中的数据，多表关联分析
        
        **前置要求：** 安装 pyodbc 驱动，确保网络可访问数据库服务器
        
        **限制建议：** 表数量 < 10，每个表记录数 < 5万，字段数 < 100
        
        **预处理功能：** 
        - 可以勾选要保留的字段，调整变量类型
        - 系统会自动发现表间关系，你可以在"表间关系管理"中修改或删除
        - 也可以手动添加新的关系
        """)

    if st.session_state.selected_db_config is None:
        st.warning("请先在侧边栏配置并选择数据库")
        return

    config = st.session_state.selected_db_config

    # 数据库连接表单
    with st.form("db_form"):
        st.info(f"📡 使用配置: {config.get('name')}")
        st.caption(f"服务器: {config.get('server')} | 数据库: {config.get('database')}")

        tables_input = st.text_area(
            "表名列表（每行一个）",
            placeholder="users\norders\nproducts",
            key="db_tables",
            help="输入要分析的表名，每行一个，建议不超过10个表"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            limit = st.number_input(
                "每个表最大加载行数",
                min_value=100,
                max_value=100000,
                value=5000,
                key="db_limit",
                help="每个表最多加载的行数，建议不超过50000"
            )
        with col2:
            max_text_length = st.number_input(
                "文本字段最大长度",
                min_value=50,
                max_value=500,
                value=100,
                key="db_text_length",
                help="文本字段保留的最大字符数"
            )
        with col3:
            max_columns = st.number_input(
                "每个表最大字段数",
                min_value=10,
                max_value=200,
                value=100,
                key="db_max_columns",
                help="每个表最多保留的字段数"
            )

        submitted = st.form_submit_button("🔌 连接并分析", type="primary")

    # 处理连接和分析
    if submitted:
        if not tables_input.strip():
            st.error("请至少输入一个表名")
            return

        table_names = [t.strip() for t in tables_input.strip().split('\n') if t.strip()]

        if len(table_names) > 10:
            st.error(f"表数量 {len(table_names)} 超过限制，请减少表数量")
            return

        st.info(f"将分析 {len(table_names)} 个表，每个表最多 {limit} 行")

        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        try:
            status_placeholder.info("🔌 正在连接数据库...")
            progress_bar.progress(10)

            # 测试连接
            test_conn = test_db_connection(config)
            if not test_conn:
                st.error("数据库连接测试失败，请检查配置")
                return
            test_conn.close()

            status_placeholder.success("✅ 数据库连接成功")
            progress_bar.progress(20)
            status_placeholder.info("📊 正在加载表数据...")

            # 加载表
            tables = DataLoader.load_multiple_tables(
                server=config.get('server'),
                database=config.get('database'),
                table_names=table_names,
                username=config.get('username') if not config.get('trusted_connection') else None,
                password=config.get('password') if not config.get('trusted_connection') else None,
                trusted_connection=config.get('trusted_connection', False),
                limit=limit,
                relationships=None,
                max_text_length=max_text_length
            )

            success = {n: df for n, df in tables.items() if df is not None and not df.empty}

            if not success:
                st.error("没有成功加载任何表")
                return

            # 数据加载完成，清除进度条
            progress_bar.empty()
            status_placeholder.empty()

            # 清除数据库相关的缓存
            if "multi_relationships" in st.session_state:
                del st.session_state.multi_relationships
            if "relationship_refresh_ts" in st.session_state:
                del st.session_state.relationship_refresh_ts
            if "multi_table_type_keys" in st.session_state:
                del st.session_state.multi_table_type_keys
            keys_to_delete = [k for k in st.session_state.keys() if k.startswith("saved_variable_types_")]
            for key in keys_to_delete:
                del st.session_state[key]
            if "field_selector_refresh_ts" in st.session_state:
                del st.session_state.field_selector_refresh_ts

            # 将加载的表保存到 session_state，供预处理界面使用
            st.session_state.db_loaded_tables = success

        except Exception as e:
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            progress_bar.empty()
            status_placeholder.empty()

            st.error(f"连接失败: {error_msg}")

            if "Login failed" in error_msg:
                st.info("🔧 请检查用户名和密码是否正确")
            elif "Cannot open database" in error_msg:
                st.info("🔧 请检查数据库名称是否正确")
            elif "Cannot find" in error_msg or "Invalid object" in error_msg:
                st.info("🔧 请检查表名是否正确")
            elif "timeout" in error_msg.lower():
                st.info("🔧 连接超时，请检查网络和防火墙设置")

            with st.expander("📋 详细错误信息", expanded=False):
                st.code(error_traceback, language='python')

            return

    # 预处理界面（放在 submitted 块外面）
    if "db_loaded_tables" in st.session_state and st.session_state.db_loaded_tables:
        tables = st.session_state.db_loaded_tables

        # 显示加载结果统计
        st.subheader("📊 已加载的表")
        for name, df in tables.items():
            st.caption(f"  📋 {name}: {len(df)}行 x {len(df.columns)}列")

        # 显示预处理界面
        confirmed, filtered_tables, variable_types_dict, filtered_relationships = render_multi_preprocessing_interface(
            tables,
            relationships=None,
            initial_types_dict=None
        )

        if confirmed:
            # 用户确认后，显示分析进度条
            status_placeholder = st.empty()
            progress_bar = st.progress(0)

            try:
                status_placeholder.info("📁 正在准备数据...")
                progress_bar.progress(20)

                status_placeholder.info("🔍 正在分析数据...")
                progress_bar.progress(50)

                def run():
                    if filtered_relationships:
                        a = MultiTableStatisticalAnalyzer(
                            filtered_tables,
                            relationships={'foreign_keys': filtered_relationships},
                            predefined_types=variable_types_dict
                        )
                    else:
                        a = MultiTableStatisticalAnalyzer(
                            filtered_tables,
                            predefined_types=variable_types_dict
                        )
                    a.analyze_all_tables()
                    return a

                analyzer, output = capture_and_run(run)

                status_placeholder.info("📝 正在生成报告...")
                progress_bar.progress(80)

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

                progress_bar.progress(100)
                status_placeholder.success("✅ 分析完成！")

                import time
                time.sleep(0.5)
                status_placeholder.empty()
                progress_bar.empty()

                st.rerun()

            except Exception as e:
                progress_bar.empty()
                status_placeholder.empty()
                st.error(f"分析失败: {str(e)}")
        else:
            # 用户还没有确认，继续显示预处理界面
            pass

    # 显示结果
    display_results()


def test_db_connection(config):
    """测试数据库连接"""
    import pyodbc

    server = config.get('server')
    database = config.get('database')
    username = config.get('username')
    password = config.get('password')
    trusted_connection = config.get('trusted_connection', False)

    possible_drivers = [
        'ODBC Driver 17 for SQL Server',
        'ODBC Driver 13 for SQL Server',
        'SQL Server Native Client 11.0',
        'SQL Server'
    ]

    available_drivers = pyodbc.drivers()

    for driver in possible_drivers:
        if driver in available_drivers:
            if trusted_connection or not username:
                conn_str = f'DRIVER={{{driver}}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'
            else:
                conn_str = f'DRIVER={{{driver}}};SERVER={server};DATABASE={database};UID={username};PWD={password};'

            try:
                conn = pyodbc.connect(conn_str, timeout=10)
                return conn
            except:
                continue

    return None


def display_results():
    """显示分析结果"""
    if st.session_state.db_analyzer is not None:
        analyzer = st.session_state.db_analyzer
        output = st.session_state.db_output

        st.success("✅ 分析完成！")

        col_success, col_dl1, col_dl2 = st.columns([3, 1, 1])
        with col_dl1:
            st.download_button("📥 下载 HTML 报告", st.session_state.current_html,
                              "autostat_db_report.html", "text/html", width="stretch")
        with col_dl2:
            st.download_button("📥 下载 JSON 结果", analyzer.to_json(),
                              "autostat_db_result.json", "application/json", width="stretch")

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