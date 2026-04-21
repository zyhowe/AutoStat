# web/components/sidebar.py

"""侧边栏组件 - 配置管理、项目最近"""

import streamlit as st
from web.services.session_service import SessionService
from web.config.storage import (
    load_db_configs, add_db_config, delete_db_config,
    load_llm_configs, add_llm_config, delete_llm_config, test_llm_connection
)
from autostat.llm_client import LLMClient


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

    @st.dialog("⚙️ 设置", width="large")
    def settings_dialog():
        # 数据库配置
        st.markdown("### 🗄️ 数据库配置")
        render_db_config()

        st.markdown("---")

        # 大模型配置
        st.markdown("### 🤖 大模型配置")
        render_llm_config()

        st.markdown("---")
        if st.button("关闭", use_container_width=True):
            st.rerun()

    settings_dialog()


def render_db_config():
    """渲染数据库配置"""
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

        if st.button("删除选中配置", key="db_delete_dialog"):
            if delete_db_config(selected_config.get('name')):
                st.success(f"配置 {selected_config.get('name')} 已删除")
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


def render_llm_config():
    """渲染大模型配置"""
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
            if st.button("测试连接", key="test_llm_dialog", use_container_width=True):
                with st.spinner("测试中..."):
                    success, msg = test_llm_connection(selected_config)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
        with col2:
            if st.button("使用此配置", key="use_llm_dialog", use_container_width=True):
                st.session_state.selected_llm_config = selected_config
                st.session_state.llm_client = LLMClient(selected_config)
                st.success(f"已切换到模型: {selected_config.get('name')}")
                st.rerun()

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

    # 显示当前使用的配置
    if st.session_state.get('selected_llm_config'):
        st.markdown("---")
        st.success(f"✅ 当前使用: {st.session_state.selected_llm_config.get('name')}")


def render_sidebar():
    """渲染侧边栏"""
    # 标题
    st.sidebar.markdown(
        '<div style="text-align: left; font-size: 1.4rem; font-weight: bold; color: #1f77b4;padding-left:0pm;padding-bottom:10pm">📊 AutoStat智能分析助手</div>',
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

    # 单文件分析固定使用 basic 级别
    st.session_state.date_features_level = "basic"

    # 设置按钮放在底部
    if st.sidebar.button("⚙️ 设置", use_container_width=True):
        _show_settings_dialog()


def render_project_history():
    """渲染项目最近列表"""
    st.sidebar.markdown("### 📜 最近项目")

    projects = SessionService.list_user_projects()

    if not projects:
        st.sidebar.caption("暂无最近项目")
        return

    current_session = SessionService.get_current_session()

    for project in projects:
        session_id = project.get("session_id")
        source_name = project.get("source_name", "未知")
        created_at = project.get("created_at", "")
        created_short = created_at[:16].replace("T", " ") if created_at else "未知时间"

        is_current = (current_session == session_id)

        col1, col2 = st.sidebar.columns([4, 1])

        with col1:
            button_text = f"✅ {source_name} (当前)" if is_current else f"📁 {source_name}"
            if st.button(
                    button_text,
                    key=f"project_{session_id}",
                    use_container_width=True
            ):
                SessionService.set_current_session(session_id)
                st.rerun()

        with col2:
            if st.button("🗑️", key=f"del_project_{session_id}", help="删除项目"):
                _show_delete_dialog(session_id, source_name, is_current)

        st.sidebar.caption(f"   {created_short}")