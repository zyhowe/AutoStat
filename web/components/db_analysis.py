"""数据库分析组件"""

import streamlit as st
import json
import sys
import os
import traceback

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
            **适用场景：**
            - 需要分析 SQL Server 数据库中的数据
            - 多表关联分析

            **前置要求：**
            - 需要安装 pyodbc 驱动：`pip install pyodbc`
            - 确保网络可以访问数据库服务器
            - 需要有相应的数据库访问权限

            **配置说明：**
            - 在侧边栏"数据库配置"中添加连接信息
            - 支持 Windows 身份认证和 SQL Server 身份认证

            **限制建议：**
            - 表数量：建议 < 10 个
            - 每个表记录数：建议 < 5万行（通过 limit 控制）
            - 每个表字段数：建议 < 100列
            - 文本字段会自动截断（默认保留100字符）

            **分析内容：**
            - 各表独立分析（变量类型、数据质量）
            - 自动发现表间关联关系
            - 智能采样（基于外键关联）
            - 生成多表关联报告

            **预处理功能：**
            - 可以为每个表勾选要保留的字段
            - 可以调整每个字段的变量类型
            - 可以添加/删除/修改表间关系

            **常见错误及解决：**
            - 连接失败：检查服务器地址、端口、防火墙
            - 登录失败：检查用户名/密码，或尝试 Windows 身份认证
            - 找不到表：检查表名是否正确（区分大小写）
            - 超时：增加 limit 值或检查网络

            **注意事项：**
            - 大表建议设置较小的加载行数（limit）
            - 文本字段会被截断（默认保留100字符）
            - 大字段类型（text、ntext、MAX类型）会自动过滤
            - 系统字段（如 tmstamp、entrydt 等）默认会被标记为"排除"
            - 建议在非高峰时段进行大规模数据分析
            """)

    if st.session_state.selected_db_config is None:
        st.warning("请先在侧边栏配置并选择数据库")
        return

    config = st.session_state.selected_db_config

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
                relationships=None,  # 不传关系，让预处理界面自动发现
                max_text_length=max_text_length
            )

            success = {n: df for n, df in tables.items() if df is not None and not df.empty}

            if not success:
                st.error("没有成功加载任何表")
                return

            progress_bar.progress(40)
            status_placeholder.info("🔧 正在准备预处理界面...")

            # 显示预处理界面（包含自动关系发现）
            confirmed, filtered_tables, variable_types_dict, filtered_relationships = render_multi_preprocessing_interface(
                success,
                relationships=None,  # 让预处理界面自动发现关系
                initial_types_dict=None
            )

            if not confirmed:
                progress_bar.empty()
                status_placeholder.empty()
                return

            progress_bar.progress(60)
            status_placeholder.info("🔍 正在分析数据...")

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

            progress_bar.progress(80)
            status_placeholder.info("📝 正在生成报告...")

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
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            progress_bar.empty()
            status_placeholder.empty()

            st.error(f"分析失败: {error_msg}")

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


def parse_relationships(rel_input):
    """解析关系定义"""
    relationships = []
    if rel_input.strip():
        for line in rel_input.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            if '=' in line:
                parts = line.split('=')
                if len(parts) == 2:
                    from_part, to_part = parts[0].strip(), parts[1].strip()
                    if '.' in from_part and '.' in to_part:
                        ft, fc = from_part.split('.')
                        tt, tc = to_part.split('.')
                        relationships.append({
                            'from_table': ft.strip(),
                            'from_col': fc.strip(),
                            'to_table': tt.strip(),
                            'to_col': tc.strip()
                        })
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