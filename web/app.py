"""
Streamlit Web界面 - 支持单文件、多文件、数据库连接、大模型解读
"""

import streamlit as st
import pandas as pd
import sys
import os
import tempfile
import contextlib
import io
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autostat import AutoStatisticalAnalyzer, MultiTableStatisticalAnalyzer
from autostat.loader import DataLoader
from autostat.reporter import Reporter
from autostat.core.base import BaseAnalyzer
from autostat.config_manager import (
    load_db_configs, save_db_configs, add_db_config, update_db_config, delete_db_config,
    load_llm_configs, save_llm_configs, add_llm_config, update_llm_config, delete_llm_config,
    test_llm_connection
)
from autostat.llm_client import LLMClient
from autostat.prompts import (
    build_single_table_prompt, build_multi_table_prompt,
    build_database_prompt, build_chat_prompt
)

st.set_page_config(
    page_title="AutoStat 智能数据分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 初始化 Session State ====================

# 分析器
if 'single_analyzer' not in st.session_state:
    st.session_state.single_analyzer = None
if 'single_output' not in st.session_state:
    st.session_state.single_output = None
if 'multi_analyzer' not in st.session_state:
    st.session_state.multi_analyzer = None
if 'multi_output' not in st.session_state:
    st.session_state.multi_output = None
if 'db_analyzer' not in st.session_state:
    st.session_state.db_analyzer = None
if 'db_output' not in st.session_state:
    st.session_state.db_output = None

# 当前分析结果
if 'current_html' not in st.session_state:
    st.session_state.current_html = None
if 'current_json_data' not in st.session_state:
    st.session_state.current_json_data = None
if 'current_analysis_type' not in st.session_state:
    st.session_state.current_analysis_type = None
if 'current_source_name' not in st.session_state:
    st.session_state.current_source_name = None

# 大模型相关
if 'llm_client' not in st.session_state:
    st.session_state.llm_client = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'auto_send_initial' not in st.session_state:
    st.session_state.auto_send_initial = False

# UI 状态
if 'selected_db_config' not in st.session_state:
    st.session_state.selected_db_config = None
if 'selected_llm_config' not in st.session_state:
    st.session_state.selected_llm_config = None

# ==================== 样式 ====================

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
    hr { margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 AutoStat 智能数据分析助手</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">自动识别数据类型、检测数据质量、选择统计方法、生成分析报告</div>', unsafe_allow_html=True)

# ==================== 侧边栏 ====================

st.sidebar.title("⚙️ 分析模式")
analysis_mode = st.sidebar.selectbox(
    "选择分析模式",
    ["📁 单文件分析", "📚 多文件分析", "🗄️ 数据库分析"],
    key="analysis_mode"
)

# ==================== 数据库配置管理 ====================

st.sidebar.markdown("---")
with st.sidebar.expander("🗄️ 数据库配置"):
    db_configs = load_db_configs()

    if db_configs:
        config_names = [c.get('name', '未命名') for c in db_configs]
        selected_idx = st.selectbox(
            "选择数据库配置",
            range(len(config_names)),
            format_func=lambda i: config_names[i],
            key="db_select"
        )
        st.session_state.selected_db_config = db_configs[selected_idx]

        if st.session_state.selected_db_config:
            st.info(f"当前配置: {st.session_state.selected_db_config.get('name')}")
            st.caption(f"服务器: {st.session_state.selected_db_config.get('server')}")
            st.caption(f"数据库: {st.session_state.selected_db_config.get('database')}")
    else:
        st.info("暂无数据库配置，请添加")
        st.session_state.selected_db_config = None

    st.markdown("---")

    with st.form("add_db_form"):
        st.markdown("**添加新配置**")
        db_name = st.text_input("配置名称", key="db_name")
        db_server = st.text_input("服务器地址", key="db_server")
        db_database = st.text_input("数据库名称", key="db_database")
        db_username = st.text_input("用户名", placeholder="可选", key="db_username")
        db_password = st.text_input("密码", type="password", placeholder="可选", key="db_password")
        db_trusted = st.checkbox("Windows身份认证", key="db_trusted")

        if st.form_submit_button("添加配置"):
            if db_name and db_server and db_database:
                new_config = {
                    "name": db_name,
                    "server": db_server,
                    "database": db_database,
                    "username": db_username if db_username else None,
                    "password": db_password if db_password else None,
                    "trusted_connection": db_trusted
                }
                if add_db_config(new_config):
                    st.success(f"配置 {db_name} 添加成功")
                    st.rerun()
                else:
                    st.error(f"配置名称 {db_name} 已存在")
            else:
                st.error("请填写配置名称、服务器和数据库名称")

    if db_configs:
        st.markdown("---")
        delete_name = st.selectbox("删除配置", config_names, key="db_delete")
        if st.button("删除选中配置", key="db_delete_btn"):
            if delete_db_config(delete_name):
                st.success(f"配置 {delete_name} 已删除")
                if st.session_state.selected_db_config and st.session_state.selected_db_config.get('name') == delete_name:
                    st.session_state.selected_db_config = None
                st.rerun()
            else:
                st.error("删除失败")

# ==================== 大模型配置管理 ====================

with st.sidebar.expander("🤖 大模型配置"):
    llm_configs = load_llm_configs()

    if llm_configs:
        config_names = [c.get('name', '未命名') for c in llm_configs]
        selected_idx = st.selectbox(
            "选择大模型配置",
            range(len(config_names)),
            format_func=lambda i: config_names[i],
            key="llm_select"
        )
        st.session_state.selected_llm_config = llm_configs[selected_idx]

        if st.session_state.selected_llm_config:
            st.info(f"当前模型: {st.session_state.selected_llm_config.get('name')}")
            st.caption(f"API: {st.session_state.selected_llm_config.get('api_base')}")
            st.caption(f"模型: {st.session_state.selected_llm_config.get('model')}")

            if st.session_state.llm_client is None or st.session_state.llm_client.model != st.session_state.selected_llm_config.get('model'):
                st.session_state.llm_client = LLMClient(st.session_state.selected_llm_config)

            if st.button("测试连接", key="test_llm"):
                with st.spinner("测试中..."):
                    success, msg = test_llm_connection(st.session_state.selected_llm_config)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
    else:
        st.info("暂无大模型配置，请添加")
        st.session_state.selected_llm_config = None
        st.session_state.llm_client = None

    st.markdown("---")

    with st.form("add_llm_form"):
        st.markdown("**添加新配置**")
        llm_name = st.text_input("配置名称", key="llm_name", placeholder="例如: DeepSeek, 本地Qwen")
        llm_api_base = st.text_input("API地址", key="llm_api_base", placeholder="https://api.deepseek.com/v1")
        llm_api_key = st.text_input("API密钥", type="password", key="llm_api_key", placeholder="sk-xxx")
        llm_model = st.text_input("模型名称", key="llm_model", placeholder="deepseek-chat, qwen-7b")

        if st.form_submit_button("添加配置"):
            if llm_name and llm_api_base and llm_model:
                new_config = {
                    "name": llm_name,
                    "api_base": llm_api_base.rstrip('/'),
                    "api_key": llm_api_key,
                    "model": llm_model,
                    "timeout": 60
                }
                if add_llm_config(new_config):
                    st.success(f"配置 {llm_name} 添加成功")
                    st.rerun()
                else:
                    st.error(f"配置名称 {llm_name} 已存在")
            else:
                st.error("请填写配置名称、API地址和模型名称")

    if llm_configs:
        st.markdown("---")
        delete_llm_name = st.selectbox("删除配置", config_names, key="llm_delete")
        if st.button("删除选中配置", key="llm_delete_btn"):
            if delete_llm_config(delete_llm_name):
                st.success(f"配置 {delete_llm_name} 已删除")
                if st.session_state.selected_llm_config and st.session_state.selected_llm_config.get('name') == delete_llm_name:
                    st.session_state.selected_llm_config = None
                    st.session_state.llm_client = None
                st.rerun()
            else:
                st.error("删除失败")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 使用说明")
st.sidebar.markdown("""
**单文件分析**：上传 CSV、Excel、JSON、TXT 文件
**多文件分析**：上传多个相关文件，自动发现表间关系
**数据库分析**：连接 SQL Server 数据库
**大模型解读**：分析完成后，在下方展开"AI 智能解读"获取解读
""")


# ==================== 辅助函数 ====================

def capture_and_run(func, *args, **kwargs):
    """捕获输出并运行函数"""
    f = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = f
        sys.stderr = f
        result = func(*args, **kwargs)
        output = f.getvalue()
        return result, output
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def get_initial_question(analysis_type: str) -> str:
    """根据分析类型获取初始提问内容"""
    if analysis_type == "single":
        return "请根据这份数据分析报告，帮我解读一下数据的主要特征、质量问题和建模建议。"
    elif analysis_type == "multi":
        return "请根据这份多表关联分析报告，帮我解读一下表间关系、数据特征和跨表分析建议。"
    elif analysis_type == "database":
        return "请根据这份数据库分析报告，帮我解读一下数据模型、性能优化和BI建设建议。"
    else:
        return "请根据这份数据分析报告，帮我解读一下主要内容。"


def render_chat_interface():
    """渲染聊天界面"""
    # 显示历史消息
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    # 输入框
    prompt = st.chat_input("请输入您的问题...", key="llm_chat_input")

    # 处理自动发送初始消息
    if st.session_state.auto_send_initial and len(st.session_state.chat_messages) == 0:
        st.session_state.auto_send_initial = False
        initial_question = get_initial_question(st.session_state.current_analysis_type)
        st.session_state.chat_messages.append({"role": "user", "content": initial_question})
        st.rerun()

    # 处理用户输入
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        st.rerun()

    # 处理最后一条消息（如果是用户消息，需要调用大模型）
    if st.session_state.chat_messages and st.session_state.chat_messages[-1]["role"] == "user":
        user_question = st.session_state.chat_messages[-1]["content"]

        # 构建系统提示词（使用 JSON 数据）
        system_prompt = ""
        if st.session_state.current_json_data:
            if st.session_state.current_analysis_type == "single":
                system_prompt = build_single_table_prompt(st.session_state.current_json_data, "")
            elif st.session_state.current_analysis_type == "multi":
                multi_info = st.session_state.current_json_data.get('multi_table_info', {})
                system_prompt = build_multi_table_prompt(st.session_state.current_json_data, multi_info)
            elif st.session_state.current_analysis_type == "database":
                multi_info = st.session_state.current_json_data.get('multi_table_info', {})
                system_prompt = build_database_prompt(
                    st.session_state.current_json_data, multi_info,
                    st.session_state.current_json_data.get('db_server', ''),
                    st.session_state.current_json_data.get('db_database', '')
                )
            else:
                system_prompt = build_single_table_prompt(st.session_state.current_json_data, "")
        else:
            system_prompt = "请分析数据"

        # 构建消息列表
        messages = [{"role": "system", "content": system_prompt}]
        for msg in st.session_state.chat_messages[:-1]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_question})

        # 显示加载状态并流式输出
        with st.spinner("AI 思考中..."):
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for chunk in st.session_state.llm_client.chat_stream(messages):
                    if chunk:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

        st.session_state.chat_messages.append({"role": "assistant", "content": full_response})
        st.rerun()


# ==================== 单文件分析 ====================

def single_file_mode():
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
            if ext == 'csv':
                df = pd.read_csv(uploaded)
            elif ext in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded)
            elif ext == 'json':
                df = pd.read_json(uploaded)
            else:
                df = pd.read_csv(uploaded, delimiter='\t')

            st.subheader("📄 数据预览")
            st.dataframe(df.head(100))
            st.write(f"形状: {df.shape[0]} 行 × {df.shape[1]} 列")

            if st.button("🚀 开始分析", type="primary", key="single_start"):
                with st.spinner("分析中..."):
                    with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{ext}', delete=False) as tmp:
                        if ext == 'csv':
                            df.to_csv(tmp.name, index=False)
                        elif ext in ['xlsx', 'xls']:
                            df.to_excel(tmp.name, index=False)
                        elif ext == 'json':
                            df.to_json(tmp.name, orient='records', force_ascii=False)
                        else:
                            df.to_csv(tmp.name, sep='\t', index=False)
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

                    st.session_state.chat_messages = []
                    st.session_state.auto_send_initial = True

                    st.rerun()

        except Exception as e:
            st.error(f"读取失败: {str(e)}")

    # 显示结果
    if st.session_state.single_analyzer is not None:
        analyzer = st.session_state.single_analyzer
        output = st.session_state.single_output

        st.success("✅ 分析完成！")

        # 下载按钮放在成功消息旁边
        col_success, col_dl1, col_dl2 = st.columns([3, 1, 1])
        with col_dl1:
            st.download_button("📥 下载 HTML 报告", st.session_state.current_html,
                              "autostat_report.html", "text/html", use_container_width=True)
        with col_dl2:
            st.download_button("📥 下载 JSON 结果", analyzer.to_json(),
                              "autostat_result.json", "application/json", use_container_width=True)

        # 分析过程日志（默认不展开）
        with st.expander("📝 分析过程日志", expanded=False):
            st.code(output, language='text')

        # 预览报告（默认显示，但收缩状态）- 直接显示 HTML 源代码
        with st.expander("📄 预览报告", expanded=False):
            if st.session_state.current_html:
                st.html(st.session_state.current_html)
            else:
                st.info("暂无报告预览")

        # AI 智能解读（默认显示，但收缩状态）
        with st.expander("🤖 AI 智能解读", expanded=False):
            if st.session_state.llm_client is None:
                st.warning("请先在侧边栏配置大模型")
            else:
                render_chat_interface()


# ==================== 多文件分析 ====================

def multi_file_mode():
    st.markdown("### 📚 多文件分析")

    files = st.file_uploader(
        "选择多个数据文件", type=['csv', 'xlsx', 'xls', 'json', 'txt'],
        accept_multiple_files=True, key="multi_uploader"
    )

    if files and len(files) >= 2:
        st.success(f"已选择 {len(files)} 个文件")

        info = [{"文件名": f.name, "大小": f"{len(f.getvalue())/1024:.1f} KB"} for f in files]
        st.dataframe(pd.DataFrame(info))

        with st.expander("🔗 手动定义表间关系（可选）"):
            rel_text = st.text_area("关系定义（每行一个）", placeholder="orders.user_id = users.user_id", key="multi_rel")
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

        if st.button("🚀 开始分析", type="primary", key="multi_start"):
            with st.spinner("分析中..."):
                tmp_dir = tempfile.mkdtemp()
                paths = {}
                for f in files:
                    p = os.path.join(tmp_dir, f.name)
                    with open(p, 'wb') as w:
                        w.write(f.getbuffer())
                    paths[os.path.splitext(f.name)[0]] = p

                tables = {}
                for name, p in paths.items():
                    tables[name] = DataLoader.load_from_file(p)

                def run():
                    if relationships:
                        a = MultiTableStatisticalAnalyzer(tables, relationships={'foreign_keys': relationships})
                    else:
                        a = MultiTableStatisticalAnalyzer(tables)
                    a.analyze_all_tables()
                    return a

                analyzer, output = capture_and_run(run)

                import shutil
                shutil.rmtree(tmp_dir)

                st.session_state.multi_analyzer = analyzer
                st.session_state.multi_output = output
                st.session_state.current_analysis_type = "multi"
                st.session_state.current_source_name = f"{len(files)}个文件"

                # 获取合并后的分析器（用于生成完整报告）
                merged_analyzer = analyzer.get_merged_analyzer()
                reporter = Reporter(merged_analyzer)
                st.session_state.current_html = reporter.to_html()
                st.session_state.current_json_data = json.loads(merged_analyzer.to_json())

                st.session_state.chat_messages = []
                st.session_state.auto_send_initial = True

                st.rerun()

    elif files and len(files) == 1:
        st.warning("多文件分析需要至少2个文件")

    # 显示结果
    if st.session_state.multi_analyzer is not None:
        analyzer = st.session_state.multi_analyzer
        output = st.session_state.multi_output

        st.success("✅ 分析完成！")

        # 下载按钮放在成功消息旁边
        col_success, col_dl1, col_dl2 = st.columns([3, 1, 1])
        with col_dl1:
            st.download_button("📥 下载 HTML 报告", st.session_state.current_html,
                              "autostat_multi_report.html", "text/html", use_container_width=True)
        with col_dl2:
            st.download_button("📥 下载 JSON 结果", analyzer.to_json(),
                              "autostat_multi_result.json", "application/json", use_container_width=True)

        # 分析过程日志（默认不展开）
        with st.expander("📝 分析过程日志", expanded=False):
            st.code(output, language='text')

        # 预览报告（默认显示，但收缩状态）- 直接显示 HTML 源代码
        with st.expander("📄 预览报告", expanded=False):
            if st.session_state.current_html:
                st.html(st.session_state.current_html)
            else:
                st.info("暂无报告预览")

        # AI 智能解读（默认显示，但收缩状态）
        with st.expander("🤖 AI 智能解读", expanded=False):
            if st.session_state.llm_client is None:
                st.warning("请先在侧边栏配置大模型")
            else:
                render_chat_interface()


# ==================== 数据库分析 ====================

def database_mode():
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

                success = {n: df for n, df in tables.items() if df is not None and len(df) > 0}

                if success:
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

                    # 获取合并后的分析器（用于生成完整报告）
                    merged_analyzer = analyzer.get_merged_analyzer()
                    reporter = Reporter(merged_analyzer)
                    st.session_state.current_html = reporter.to_html()
                    json_data = json.loads(merged_analyzer.to_json())
                    json_data['db_server'] = config.get('server')
                    json_data['db_database'] = config.get('database')
                    st.session_state.current_json_data = json_data

                    st.session_state.chat_messages = []
                    st.session_state.auto_send_initial = True

                    st.rerun()
                else:
                    st.error("没有成功加载任何表")

            except Exception as e:
                st.error(f"连接失败: {str(e)}")

    # 显示结果
    if st.session_state.db_analyzer is not None:
        analyzer = st.session_state.db_analyzer
        output = st.session_state.db_output

        st.success("✅ 分析完成！")

        # 下载按钮放在成功消息旁边
        col_success, col_dl1, col_dl2 = st.columns([3, 1, 1])
        with col_dl1:
            st.download_button("📥 下载 HTML 报告", st.session_state.current_html,
                              "autostat_db_report.html", "text/html", use_container_width=True)
        with col_dl2:
            st.download_button("📥 下载 JSON 结果", analyzer.to_json(),
                              "autostat_db_result.json", "application/json", use_container_width=True)

        # 分析过程日志（默认不展开）
        with st.expander("📝 分析过程日志", expanded=False):
            st.code(output, language='text')

        # 预览报告（默认显示，但收缩状态）- 直接显示 HTML 源代码
        with st.expander("📄 预览报告", expanded=False):
            if st.session_state.current_html:
                st.html(st.session_state.current_html)
            else:
                st.info("暂无报告预览")

        # AI 智能解读（默认显示，但收缩状态）
        with st.expander("🤖 AI 智能解读", expanded=False):
            if st.session_state.llm_client is None:
                st.warning("请先在侧边栏配置大模型")
            else:
                render_chat_interface()


# ==================== 关于 ====================

st.sidebar.markdown("---")
if st.sidebar.button("📖 关于"):
    with st.expander("📖 关于 AutoStat", expanded=True):
        st.markdown("""
        **AutoStat** 智能统计分析工具
        
        - 自动识别数据类型
        - 数据质量体检
        - 智能统计方法选择
        - 多表关联分析
        - 时间序列分析
        - 大模型智能解读
        - 多种输出格式 (HTML/JSON/Markdown/Excel)
        
        版本: 0.2.0 | 许可证: MIT
        """)


# ==================== 主入口 ====================

if analysis_mode == "📁 单文件分析":
    single_file_mode()
elif analysis_mode == "📚 多文件分析":
    multi_file_mode()
elif analysis_mode == "🗄️ 数据库分析":
    database_mode()