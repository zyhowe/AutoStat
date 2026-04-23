# web/components/sidebar.py

"""侧边栏组件 - 配置管理、项目最近、技巧推送"""

import streamlit as st
from web.services.session_service import SessionService
from web.config.storage import (
    load_db_configs, add_db_config, delete_db_config,
    load_llm_configs, add_llm_config, delete_llm_config, test_llm_connection
)
from autostat.llm_client import LLMClient
from web.services.feature_flags import FeatureFlags


def _show_delete_dialog(session_id, source_name, is_current):
    """显示删除确认对话框"""

    @st.dialog("确认删除")
    def confirm():
        st.warning(f"确定要删除项目「{source_name}」吗？此操作不可恢复。")
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("确认删除", use_container_width=True):
                if SessionService.delete_project(session_id):
                    st.success(f"已删除项目: {source_name}")
                    if is_current:
                        SessionService.clear_current_session()
                    st.rerun()
                else:
                    st.error("删除失败")
        with col_cancel:
            if st.button("取消", use_container_width=True):
                st.rerun()

    confirm()


def _show_settings_dialog():
    """显示设置对话框"""
    if 'show_settings_dialog' not in st.session_state:
        st.session_state.show_settings_dialog = False

    @st.dialog("⚙️ 设置", width="large")
    def settings_dialog():
        _render_db_config_dialog()
        st.markdown("---")
        _render_llm_config_dialog()
        st.markdown("---")
        if st.button("关闭", use_container_width=True):
            st.session_state.show_settings_dialog = False
            st.rerun()

    settings_dialog()


def _render_db_config_dialog():
    """渲染对话框中的数据库配置"""
    db_configs = load_db_configs()

    if db_configs:
        config_names = [c.get('name', '未命名') for c in db_configs]
        selected_idx = st.selectbox(
            "选择数据库配置",
            range(len(config_names)),
            format_func=lambda i: config_names[i],
            key="db_select_dialog"
        )
        selected_config = db_configs[selected_idx]

        st.info(f"当前配置: {selected_config.get('name')}")
        st.caption(f"服务器: {selected_config.get('server')} | 数据库: {selected_config.get('database')}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("测试连接", key="test_db_dialog", use_container_width=True):
                _test_db_connection(selected_config)
        with col2:
            if st.button("删除选中配置", key="db_delete_dialog", use_container_width=True):
                if delete_db_config(selected_config.get('name')):
                    st.success(f"配置 {selected_config.get('name')} 已删除")
                    if st.session_state.selected_db_config and st.session_state.selected_db_config.get(
                            'name') == selected_config.get('name'):
                        st.session_state.selected_db_config = None
                    st.rerun()
                else:
                    st.error("删除失败")
    else:
        st.info("暂无数据库配置，请添加")

    st.markdown("---")

    with st.expander("添加新配置"):
        with st.form("add_db_form_dialog"):
            db_name = st.text_input("配置名称")
            db_server = st.text_input("服务器地址")
            db_database = st.text_input("数据库名称")
            db_username = st.text_input("用户名", placeholder="可选")
            db_password = st.text_input("密码", type="password", placeholder="可选")
            db_trusted = st.checkbox("Windows身份认证")

            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("添加配置", use_container_width=True)
            with col2:
                test_clicked = st.form_submit_button("测试连接", use_container_width=True)

            if submitted:
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
                        st.session_state.show_settings_dialog = False
                        st.rerun()
                    else:
                        st.error(f"配置名称 {db_name} 已存在")
                else:
                    st.error("请填写配置名称、服务器和数据库名称")

            if test_clicked:
                if db_server and db_database:
                    test_config = {
                        "server": db_server,
                        "database": db_database,
                        "username": db_username if db_username else None,
                        "password": db_password if db_password else None,
                        "trusted_connection": db_trusted
                    }
                    _test_db_connection(test_config)
                else:
                    st.error("请填写服务器和数据库名称")


def _test_db_connection(config):
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
                conn = pyodbc.connect(conn_str, timeout=5)
                conn.close()
                st.success(f"✅ 连接成功！使用驱动: {driver}")
                return
            except Exception:
                continue

    st.error(f"❌ 连接失败: 无法连接到服务器 {server}")


def _render_llm_config_dialog():
    """渲染对话框中的大模型配置"""
    llm_configs = load_llm_configs()

    if llm_configs:
        config_names = [c.get('name', '未命名') for c in llm_configs]
        selected_idx = st.selectbox(
            "选择大模型配置",
            range(len(config_names)),
            format_func=lambda i: config_names[i],
            key="llm_select_dialog"
        )
        selected_config = llm_configs[selected_idx]

        st.info(f"当前模型: {selected_config.get('name')}")
        st.caption(f"API: {selected_config.get('api_base')}")
        st.caption(f"模型: {selected_config.get('model')}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("测试连接", key="test_llm_existing_dialog", use_container_width=True):
                success, msg = test_llm_connection(selected_config)
                if success:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")
        with col2:
            if st.button("删除选中配置", key="llm_delete_dialog", use_container_width=True):
                if delete_llm_config(selected_config.get('name')):
                    st.success(f"配置 {selected_config.get('name')} 已删除")
                    if st.session_state.selected_llm_config and st.session_state.selected_llm_config.get(
                            'name') == selected_config.get('name'):
                        st.session_state.selected_llm_config = None
                        st.session_state.llm_client = None
                    st.rerun()
                else:
                    st.error("删除失败")
    else:
        st.info("暂无大模型配置，请添加")

    st.markdown("---")

    with st.expander("添加新配置"):
        with st.form("add_llm_form_dialog"):
            llm_name = st.text_input("配置名称", placeholder="例如: DeepSeek, 本地Qwen")
            llm_api_base = st.text_input("API地址", placeholder="https://api.deepseek.com/v1")
            llm_api_key = st.text_input("API密钥", type="password", placeholder="sk-xxx")
            llm_model = st.text_input("模型名称", placeholder="deepseek-chat, qwen-7b")

            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("添加配置", use_container_width=True)
            with col2:
                test_clicked = st.form_submit_button("测试连接", use_container_width=True)

            if submitted:
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
                        st.session_state.show_settings_dialog = False
                        st.rerun()
                    else:
                        st.error(f"配置名称 {llm_name} 已存在")
                else:
                    st.error("请填写配置名称、API地址和模型名称")

            if test_clicked:
                if llm_api_base and llm_model:
                    test_config = {
                        "api_base": llm_api_base.rstrip('/'),
                        "api_key": llm_api_key,
                        "model": llm_model
                    }
                    success, msg = test_llm_connection(test_config)
                    if success:
                        st.success(f"✅ {msg}")
                    else:
                        st.error(f"❌ {msg}")
                else:
                    st.error("请填写API地址和模型名称")

    # 显示当前使用的配置
    if st.session_state.get('selected_llm_config'):
        st.markdown("---")
        st.success(f"✅ 当前使用: {st.session_state.selected_llm_config.get('name')}")


def render_db_selector():
    """渲染数据库配置选择器"""
    db_configs = load_db_configs()

    if not db_configs:
        st.sidebar.warning("暂无数据库配置，请点击⚙️设置添加")
        return

    config_names = [c.get('name', '未命名') for c in db_configs]

    current_name = None
    if st.session_state.get('selected_db_config'):
        current_name = st.session_state.selected_db_config.get('name')

    current_index = 0
    if current_name and current_name in config_names:
        current_index = config_names.index(current_name)

    selected_name = st.sidebar.selectbox(
        "🗄️ 数据库",
        options=config_names,
        index=current_index,
        key="db_selector"
    )

    for config in db_configs:
        if config.get('name') == selected_name:
            st.session_state.selected_db_config = config
            break


def render_llm_selector():
    """渲染大模型配置选择器"""
    llm_configs = load_llm_configs()

    if not llm_configs:
        st.sidebar.warning("暂无大模型配置，请点击⚙️设置添加")
        return

    config_names = [c.get('name', '未命名') for c in llm_configs]

    current_name = None
    if st.session_state.get('selected_llm_config'):
        current_name = st.session_state.selected_llm_config.get('name')

    current_index = 0
    if current_name and current_name in config_names:
        current_index = config_names.index(current_name)

    selected_name = st.sidebar.selectbox(
        "🤖 大模型",
        options=config_names,
        index=current_index,
        key="llm_selector"
    )

    for config in llm_configs:
        if config.get('name') == selected_name:
            if st.session_state.get('selected_llm_config') != config:
                st.session_state.selected_llm_config = config
                st.session_state.llm_client = LLMClient(config)
            break


def render_auto_analysis_selector():
    """渲染自动分析选择器"""
    current = FeatureFlags.is_auto_analysis_enabled()
    options = ["开启", "关闭"]
    current_index = 0 if current else 1

    selected = st.sidebar.selectbox(
        "⚡ 自动分析",
        options=options,
        index=current_index,
        key="auto_analysis_selector",
        help="开启后上传文件/加载数据库表后自动开始分析"
    )

    new_value = (selected == "开启")
    if new_value != current:
        FeatureFlags.set_auto_analysis(new_value)
        st.rerun()


def render_auto_training_selector():
    """渲染自动训练选择器"""
    current = FeatureFlags.is_auto_training_enabled()
    options = ["开启", "关闭"]
    current_index = 0 if current else 1

    selected = st.sidebar.selectbox(
        "🤖 自动训练",
        options=options,
        index=current_index,
        key="auto_training_selector",
        help="开启后分析完成自动训练推荐模型（需开启自动分析）"
    )

    new_value = (selected == "开启")
    if new_value != current:
        FeatureFlags.set_auto_training(new_value)
        st.rerun()


def render_project_history():
    """渲染项目最近列表"""
    st.sidebar.markdown("""
        <style>
        section[data-testid="stSidebar"] .stButton button {
            justify-content: flex-start !important;
        }
        section[data-testid="stSidebar"] .stButton button div {
            display: flex !important;
            justify-content: flex-start !important;
            width: 100% !important;
        }
        section[data-testid="stSidebar"] .stButton button div p {
            margin: 0 !important;
            text-align: left !important;
        }
        section[data-testid="stSidebar"] .stButton button[kind="primary"],
        section[data-testid="stSidebar"] .stButton button[kind="tertiary"]{
            justify-content: center !important;
        }
        section[data-testid="stSidebar"] .stButton button[kind="primary"] div,
        section[data-testid="stSidebar"] .stButton button[kind="tertiary"] div{
            display: flex !important;
            justify-content: center !important;
            width: 100% !important;
        }
        section[data-testid="stSidebar"] .stButton button[kind="primary"] div p,
        section[data-testid="stSidebar"] .stButton button[kind="tertiary"] div p{
            margin: 0 !important;
            text-align: center !important;
        }
        </style>
        """, unsafe_allow_html=True)

    st.sidebar.markdown("### 📜 最近项目")

    projects = SessionService.list_user_projects()

    if not projects:
        st.sidebar.caption("暂无最近项目")
        return

    current_session = SessionService.get_current_session()

    for project in projects:
        session_id = project.get("session_id")
        source_name = project.get("source_name", "未知")
        display_name = source_name[:15] + "..." if len(source_name) > 15 else source_name
        created_at = project.get("created_at", "")
        created_short = created_at[:16].replace("T", " ") if created_at else "未知时间"

        is_current = (current_session == session_id)

        col1, col2 = st.sidebar.columns([4, 1])

        with col1:
            if is_current:
                button_text = f"✅ {display_name}"
            else:
                button_text = f"📁 {display_name}"
            if st.button(button_text, key=f"project_{session_id}", use_container_width=True):
                SessionService.set_current_session(session_id)
                st.rerun()

        with col2:
            if st.button("🗑️", key=f"del_project_{session_id}", help="删除项目"):
                _show_delete_dialog(session_id, source_name, is_current)

        st.sidebar.caption(f"   {created_short}")


def render_sidebar():
    """渲染侧边栏"""
    # 标题
    st.sidebar.markdown(
        '<div style="text-align: left; font-size: 1.4rem; font-weight: bold; color: #1f77b4;padding-left:0pm;padding-bottom:20pm">📊AutoStat智能分析助手</div>',
        unsafe_allow_html=True
    )

    # 开启新分析按钮
    if st.sidebar.button("➕ 开启新分析", use_container_width=True, type="primary"):
        SessionService.clear_current_session()
        cache_keys = [
            'current_html', 'current_json_data', 'raw_data_preview',
            'single_output', 'multi_output', 'db_output',
            'single_analyzer', 'multi_analyzer', 'db_analyzer',
            'chat_messages', 'analysis_completed',
            'current_analysis_type', 'current_source_name',
            'single_cached_df', 'single_cached_name', 'single_cached_ext',
            'multi_cached_tables', 'multi_tmp_dir', 'db_cached_tables'
        ]
        for key in cache_keys:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.sidebar.markdown("---")

    render_project_history()

    st.sidebar.markdown("---")

    # 数据库配置下拉框
    render_db_selector()

    # 大模型配置下拉框
    render_llm_selector()

    # 自动分析选择器
    render_auto_analysis_selector()

    # 自动训练选择器
    render_auto_training_selector()

    st.sidebar.markdown("---")

    # 单文件分析固定使用 basic 级别
    st.session_state.date_features_level = "basic"

    # 技巧推送
    with st.sidebar.expander("💡 使用技巧", expanded=False):
        st.markdown("""
        **快速开始**
        - 点击「销售数据示例」立即体验
        - 上传CSV/Excel文件自动分析

        **高级功能**
        - 上传多个文件自动发现表间关系
        - 配置大模型获得AI智能解读
        - 训练预测模型进行数据预测

        **导出报告**
        - 支持HTML/JSON/Excel格式
        - 一键导出全部格式为ZIP包
        """)

        # 动态提示：未配置大模型
        if st.session_state.get("selected_llm_config") is None:
            st.info("💡 提示：配置大模型后可以获得AI智能解读")
            if st.button("去配置大模型", key="tip_go_llm", use_container_width=True):
                st.session_state.show_settings_dialog = True
                st.rerun()

        # 动态提示：分析完成可训练模型
        if st.session_state.get("analysis_completed"):
            st.success("💡 分析完成！试试「小模型训练」标签页")
            if st.button("前往模型训练", key="tip_go_train", use_container_width=True):
                st.session_state.current_tab = 2
                st.rerun()

    st.sidebar.markdown("---")

    # 设置按钮放最后
    if st.sidebar.button("⚙️ 设置", use_container_width=True, type="tertiary"):
        st.session_state.show_settings_dialog = True
        _show_settings_dialog()