# webtext/components/sidebar.py
"""文本分析侧边栏组件 - 独立于数据分析"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from webtext.services.session_service import TextSessionService
from autostat.config_manager import load_llm_configs
from autostat.llm_client import LLMClient


def init_session_state():
    """初始化 session state"""
    if "text_current_session" not in st.session_state:
        st.session_state.text_current_session = None
    if "text_analysis_completed" not in st.session_state:
        st.session_state.text_analysis_completed = False
    if "text_current_tab" not in st.session_state:
        st.session_state.text_current_tab = 0
    if "text_analyzer" not in st.session_state:
        st.session_state.text_analyzer = None
    if "text_html_content" not in st.session_state:
        st.session_state.text_html_content = None
    if "text_json_data" not in st.session_state:
        st.session_state.text_json_data = None
    if "text_chat_messages" not in st.session_state:
        st.session_state.text_chat_messages = []
    if "text_llm_client" not in st.session_state:
        st.session_state.text_llm_client = None
    if "text_selected_llm_config" not in st.session_state:
        st.session_state.text_selected_llm_config = None


def _show_delete_dialog(session_id: str, text_preview: str, is_current: bool):
    """显示删除确认对话框"""

    @st.dialog("确认删除")
    def confirm():
        st.warning(f"确定要删除项目「{text_preview}」吗？此操作不可恢复。")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("确认删除", use_container_width=True):
                if TextSessionService.delete_session(session_id):
                    st.success(f"已删除项目: {text_preview}")
                    if is_current:
                        st.session_state.text_current_session = None
                        st.session_state.text_analysis_completed = False
                        st.session_state.text_analyzer = None
                        st.session_state.text_html_content = None
                        st.session_state.text_json_data = None
                        st.session_state.text_chat_messages = []
                    st.rerun()
                else:
                    st.error("删除失败")
        with col2:
            if st.button("取消", use_container_width=True):
                st.rerun()

    confirm()


def render_llm_selector():
    """渲染大模型配置选择器（仅在侧边栏调用）"""
    llm_configs = load_llm_configs()

    if not llm_configs:
        st.warning("暂无大模型配置，请点击⚙️设置添加")
        return

    config_names = [c.get('name', '未命名') for c in llm_configs]

    current_name = None
    if st.session_state.get('text_selected_llm_config'):
        current_name = st.session_state.text_selected_llm_config.get('name')

    current_index = 0
    if current_name and current_name in config_names:
        current_index = config_names.index(current_name)

    selected_name = st.selectbox(
        "🤖 大模型",
        options=config_names,
        index=current_index,
        key="text_llm_selector"
    )

    for config in llm_configs:
        if config.get('name') == selected_name:
            if st.session_state.get('text_selected_llm_config') != config:
                st.session_state.text_selected_llm_config = config
                st.session_state.text_llm_client = LLMClient(config)
            break


def render_sidebar():
    """渲染侧边栏"""
    init_session_state()

    with st.sidebar:
        st.markdown("""
        <div style="text-align: left; font-size: 1.4rem; font-weight: bold; color: #1f77b4;">
            📝 AutoText 智能文本分析
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # 开启新分析按钮
        if st.button("➕ 开启新分析", use_container_width=True, type="primary"):
            st.session_state.text_current_session = None
            st.session_state.text_analysis_completed = False
            st.session_state.text_analyzer = None
            st.session_state.text_html_content = None
            st.session_state.text_json_data = None
            st.session_state.text_chat_messages = []
            st.session_state.text_current_tab = 0
            st.rerun()

        st.markdown("---")

        # 最近项目列表
        st.markdown("### 📜 最近项目")

        projects = TextSessionService.list_projects()

        if not projects:
            st.caption("暂无最近项目")
        else:
            current_session = st.session_state.text_current_session

            for project in projects:
                session_id = project.get("session_id")
                text_preview = project.get("text_preview", "未知")
                created_at = project.get("created_at", "")
                created_short = created_at[:16].replace("T", " ") if created_at else "未知时间"
                is_current = (current_session == session_id)

                col1, col2 = st.columns([4, 1])

                with col1:
                    if is_current:
                        button_text = f"✅ {session_id})"
                    else:
                        button_text = f"📁 {session_id})"
                    if st.button(button_text, key=f"text_project_{session_id}", use_container_width=True):
                        TextSessionService.update_last_accessed(session_id)
                        st.session_state.text_current_session = session_id
                        # 加载已有的分析结果
                        html_content = TextSessionService.load_html(session_id)
                        json_data = TextSessionService.load_json(session_id)
                        if html_content and json_data:
                            st.session_state.text_analysis_completed = True
                            st.session_state.text_html_content = html_content
                            st.session_state.text_json_data = json_data
                            st.session_state.text_chat_messages = []
                        st.rerun()

                with col2:
                    if st.button("🗑️", key=f"text_del_{session_id}", help="删除项目"):
                        _show_delete_dialog(session_id, text_preview, is_current)

                st.caption(f"   {created_short}")

        st.markdown("---")


        # 大模型配置
        render_llm_selector()

        st.markdown("---")

        # 使用技巧
        with st.expander("💡 使用技巧", expanded=False):
            st.markdown("""
            **快速开始**
            - 在文本框中输入或粘贴文本
            - 点击「开始分析」自动生成报告

            **支持功能**
            - 关键词提取
            - 情感分析
            - 实体识别
            - 文本聚类
            - 主题建模

            **导出格式**
            - HTML 报告
            - JSON 结果
            """)

            if st.session_state.get("text_llm_client") is None:
                st.info("💡 提示：配置大模型后可以获得AI智能解读")