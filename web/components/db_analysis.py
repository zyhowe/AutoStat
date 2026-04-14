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

        **常见错误及解决：**
        - 连接失败：检查服务器地址、端口、防火墙
        - 登录失败：检查用户名/密码，或尝试 Windows 身份认证
        - 找不到表：检查表名是否正确（区分大小写）
        - 超时：增加 limit 值或检查网络
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

        # 表数量检查
        if tables_input.strip():
            table_count = len([t.strip() for t in tables_input.strip().split('\n') if t.strip()])
            if table_count > 10:
                st.warning(f"⚠️ 表数量 {table_count} 超过建议限制 10个，分析可能较慢。")
            elif table_count > 5:
                st.info(f"📊 表数量 {table_count}，分析可能需要一些时间。")

        rel_input = st.text_area(
            "关系定义（可选）",
            placeholder="orders.user_id = users.user_id\norders.product_id = products.product_id",
            key="db_rels",
            help="定义表间关系，格式: 表名.列名 = 表名.列名",
            height=100
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
            if limit > 50000:
                st.warning("⚠️ 行数超过50000，分析可能较慢")
            elif limit < 1000:
                st.info("💡 行数较小，分析速度较快")
        with col2:
            max_text_length = st.number_input(
                "文本字段最大长度",
                min_value=50,
                max_value=500,
                value=100,
                key="db_text_length",
                help="文本字段保留的最大字符数，超出部分会被截断"
            )
        with col3:
            max_columns = st.number_input(
                "每个表最大字段数",
                min_value=10,
                max_value=200,
                value=100,
                key="db_max_columns",
                help="每个表最多保留的字段数，超出部分会自动过滤"
            )
            if max_columns > 150:
                st.warning("⚠️ 字段数较多，可能影响性能")

        submitted = st.form_submit_button("🔌 连接并分析", type="primary")

    if submitted:
        if not tables_input.strip():
            st.error("请至少输入一个表名")
            return

        table_names = [t.strip() for t in tables_input.strip().split('\n') if t.strip()]

        # 最终检查
        if len(table_names) > 10:
            st.error(f"表数量 {len(table_names)} 超过限制，请减少表数量")
            return

        if limit > 50000:
            st.warning(f"加载行数 {limit} 较大，请确保数据库性能充足")

        relationships = parse_relationships(rel_input)

        st.info(f"将分析 {len(table_names)} 个表，每个表最多 {limit} 行")

        # 估算总加载量
        estimated_total = len(table_names) * limit
        if estimated_total > 200000:
            st.warning(f"⚠️ 估算总加载量约 {estimated_total:,} 行，请确保内存充足")

        # 使用进度条和状态显示
        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        try:
            status_placeholder.info("🔌 正在连接数据库...")
            progress_bar.progress(10)

            # 先测试连接
            test_conn = None
            try:
                test_conn = test_db_connection(config)
                if not test_conn:
                    st.error("数据库连接测试失败，请检查配置")
                    return
                status_placeholder.success("✅ 数据库连接成功")
            except Exception as e:
                st.error(f"数据库连接失败: {str(e)}")
                st.info("请检查：\n1. 服务器地址和端口是否正确\n2. 用户名和密码是否正确\n3. 防火墙是否允许连接\n4. 是否安装了 pyodbc 驱动")
                return
            finally:
                if test_conn:
                    test_conn.close()

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
                relationships=relationships if relationships else None,
                max_text_length=max_text_length
            )

            progress_bar.progress(50)
            status_placeholder.info("📊 正在处理加载结果...")

            success = {n: df for n, df in tables.items() if df is not None and not df.empty}

            if not success:
                st.error("没有成功加载任何表，请检查：\n1. 表名是否正确\n2. 是否有访问权限\n3. 数据库连接是否正常")
                return

            # 显示加载结果统计
            st.success(f"成功加载 {len(success)} 个表")
            total_rows = 0
            total_cols = 0
            large_tables = []
            for name, df in success.items():
                rows, cols = df.shape
                total_rows += rows
                total_cols += cols
                if rows > 30000:
                    large_tables.append(f"{name}({rows}行)")
                st.caption(f"  📊 {name}: {rows}行 x {cols}列")

            st.caption(f"📈 总计: {total_rows:,}行, {total_cols}列")

            if total_rows > 500000:
                st.warning(f"⚠️ 总行数 {total_rows:,} 超过建议限制，分析可能较慢")
            if large_tables:
                st.info(f"💡 以下表较大，已采样至 {limit} 行: {', '.join(large_tables)}")

            progress_bar.progress(70)
            status_placeholder.info("🔍 正在分析数据（可能需要几分钟）...")

            def run():
                if relationships:
                    a = MultiTableStatisticalAnalyzer(success, relationships={'foreign_keys': relationships})
                else:
                    a = MultiTableStatisticalAnalyzer(success)
                a.analyze_all_tables()
                return a

            analyzer, output = capture_and_run(run)

            progress_bar.progress(90)
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

            # 清除进度条
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

            # 根据错误类型给出具体建议
            if "Login failed" in error_msg or "登录失败" in error_msg:
                st.info("🔧 解决方案：\n1. 检查用户名和密码是否正确\n2. 尝试勾选「Windows身份认证」\n3. 确认账号有数据库访问权限")
            elif "Cannot open database" in error_msg or "无法打开数据库" in error_msg:
                st.info("🔧 解决方案：\n1. 检查数据库名称是否正确\n2. 确认账号有该数据库的访问权限")
            elif "Cannot find" in error_msg or "找不到" in error_msg or "Invalid object" in error_msg:
                st.info("🔧 解决方案：\n1. 检查表名是否正确（注意大小写）\n2. 确认表是否存在于该数据库中\n3. 检查表名是否包含特殊字符或空格")
            elif "timeout" in error_msg.lower() or "超时" in error_msg:
                st.info("🔧 解决方案：\n1. 检查网络连接\n2. 检查防火墙设置\n3. 尝试减小 limit 值\n4. 确认数据库服务器响应正常")
            elif "driver" in error_msg.lower() or "odbc" in error_msg.lower():
                st.info("🔧 解决方案：\n1. 安装 ODBC Driver: `pip install pyodbc`\n2. 安装 SQL Server ODBC 驱动\n3. Windows: 安装「SQL Server Native Client」\n4. Linux: 安装 `unixODBC` 和 `msodbcsql17`")
            elif "memory" in error_msg.lower() or "内存" in error_msg:
                st.info("🔧 解决方案：\n1. 减小 limit 值\n2. 减少表数量\n3. 关闭其他程序释放内存")
            else:
                st.info("🔧 请检查：\n1. 数据库服务器是否可访问\n2. 用户名/密码是否正确\n3. 是否有访问该数据库的权限\n4. 表名是否正确")

            # 显示详细错误信息（可展开）
            with st.expander("📋 详细错误信息（用于调试）", expanded=False):
                st.code(error_traceback, language='python')

    # 结果显示放在表单下方
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
                    else:
                        st.warning(f"关系格式错误，应为 '表名.列名 = 表名.列名': {line}")
            else:
                st.warning(f"关系格式错误，缺少 '=' : {line}")
    return relationships


def display_results():
    """显示分析结果（放在表单下方）"""
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