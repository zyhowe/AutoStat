"""侧边栏组件 - 模式选择、配置管理"""

import streamlit as st
from web.config.storage import (
    load_db_configs, add_db_config, delete_db_config,
    load_llm_configs, add_llm_config, delete_llm_config, test_llm_connection
)
from autostat.llm_client import LLMClient


def render_sidebar():
    """渲染侧边栏"""

    # 侧边栏头部
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <div style="font-size: 1.8rem; font-weight: bold; color: #1f77b4;">📊 AutoStat</div>
        <div style="font-size: 0.8rem; color: #666;">智能数据分析助手</div>
        <div style="font-size: 0.7rem; color: #999; margin-top: 5px;">自动识别数据类型、检测数据质量、选择统计方法、生成分析报告</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # 1. 分析模式选择
    analysis_mode = st.sidebar.selectbox(
        "分析模式",
        ["📁 单文件分析", "📚 多文件分析", "🗄️ 数据库分析"],
        key="analysis_mode_select"
    )

    st.sidebar.markdown("---")

    # 2. 日期特征级别（仅在单文件分析模式下显示）
    if analysis_mode == "📁 单文件分析":
        date_features_level = st.sidebar.selectbox(
            "📅 日期特征级别",
            ["基础", "无"],
            index=0,
            key="date_features_level_select",
            help="基础: 提取年/月/季度；无: 不提取日期特征"
        )

        # 转换为内部使用的值
        if date_features_level == "基础":
            st.session_state.date_features_level = "basic"
        else:
            st.session_state.date_features_level = "none"
    else:
        # 多文件/数据库模式默认使用基础
        st.session_state.date_features_level = "basic"

    st.sidebar.markdown("---")

    # 3. 数据库配置管理
    render_db_config_section()

    st.sidebar.markdown("---")

    # 4. 大模型配置管理
    render_llm_config_section()

    st.sidebar.markdown("---")

    # 5. 使用说明
    st.sidebar.markdown("### 📖 使用说明")
    st.sidebar.markdown("""
    **单文件分析**：上传 CSV、Excel、JSON、TXT 文件
    **多文件分析**：上传多个相关文件，自动发现表间关系
    **数据库分析**：连接 SQL Server 数据库
    **大模型解读**：分析完成后，在 AI 解读标签页使用
    """)

    return analysis_mode


def render_db_config_section():
    """渲染数据库配置区域"""
    st.sidebar.markdown("### 🗄️ 数据库配置")
    db_configs = load_db_configs()

    if db_configs:
        config_names = [c.get('name', '未命名') for c in db_configs]
        selected_idx = st.sidebar.selectbox(
            "选择数据库配置",
            range(len(config_names)),
            format_func=lambda i: config_names[i],
            key="db_select"
        )
        st.session_state.selected_db_config = db_configs[selected_idx]

        if st.session_state.selected_db_config:
            st.sidebar.info(f"当前配置: {st.session_state.selected_db_config.get('name')}")
            st.sidebar.caption(f"服务器: {st.session_state.selected_db_config.get('server')}")
            st.sidebar.caption(f"数据库: {st.session_state.selected_db_config.get('database')}")
    else:
        st.sidebar.info("暂无数据库配置，请添加")
        st.session_state.selected_db_config = None

    st.sidebar.markdown("---")

    with st.sidebar.expander("添加新配置"):
        with st.form("add_db_form"):
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
        st.sidebar.markdown("---")
        delete_name = st.sidebar.selectbox("删除配置", config_names, key="db_delete")
        if st.sidebar.button("删除选中配置", key="db_delete_btn"):
            if delete_db_config(delete_name):
                st.success(f"配置 {delete_name} 已删除")
                if st.session_state.selected_db_config and st.session_state.selected_db_config.get(
                        'name') == delete_name:
                    st.session_state.selected_db_config = None
                st.rerun()
            else:
                st.error("删除失败")


def render_llm_config_section():
    """渲染大模型配置区域"""
    st.sidebar.markdown("### 🤖 大模型配置")
    llm_configs = load_llm_configs()

    if llm_configs:
        config_names = [c.get('name', '未命名') for c in llm_configs]
        selected_idx = st.sidebar.selectbox(
            "选择大模型配置",
            range(len(config_names)),
            format_func=lambda i: config_names[i],
            key="llm_select"
        )
        st.session_state.selected_llm_config = llm_configs[selected_idx]

        if st.session_state.selected_llm_config:
            st.sidebar.info(f"当前模型: {st.session_state.selected_llm_config.get('name')}")
            st.sidebar.caption(f"API: {st.session_state.selected_llm_config.get('api_base')}")
            st.sidebar.caption(f"模型: {st.session_state.selected_llm_config.get('model')}")

            if st.session_state.llm_client is None or st.session_state.llm_client.model != st.session_state.selected_llm_config.get(
                    'model'):
                st.session_state.llm_client = LLMClient(st.session_state.selected_llm_config)

            if st.sidebar.button("测试连接", key="test_llm"):
                with st.spinner("测试中..."):
                    success, msg = test_llm_connection(st.session_state.selected_llm_config)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
    else:
        st.sidebar.info("暂无大模型配置，请添加")
        st.session_state.selected_llm_config = None
        st.session_state.llm_client = None

    st.sidebar.markdown("---")

    with st.sidebar.expander("添加新配置"):
        with st.form("add_llm_form"):
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
        st.sidebar.markdown("---")
        delete_llm_name = st.sidebar.selectbox("删除配置", config_names, key="llm_delete")
        if st.sidebar.button("删除选中配置", key="llm_delete_btn"):
            if delete_llm_config(delete_llm_name):
                st.success(f"配置 {delete_llm_name} 已删除")
                if st.session_state.selected_llm_config and st.session_state.selected_llm_config.get(
                        'name') == delete_llm_name:
                    st.session_state.selected_llm_config = None
                    st.session_state.llm_client = None
                st.rerun()
            else:
                st.error("删除失败")