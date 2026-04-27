"""
侧边栏组件 - 配置管理、项目历史
"""

import streamlit as st
import os
from pathlib import Path
from datetime import datetime


class TextSessionService:
    """文本分析会话管理"""

    BASE_PATH = Path.home() / ".autotext" / "data"

    @classmethod
    def _ensure_base_dir(cls):
        cls.BASE_PATH.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create_session(cls, source_name: str) -> str:
        cls._ensure_base_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in source_name if c.isalnum() or c in "._-")[:30]
        session_id = f"{safe_name}_{timestamp}"
        session_path = cls.BASE_PATH / session_id
        session_path.mkdir(parents=True, exist_ok=True)
        return session_id

    @classmethod
    def list_sessions(cls):
        cls._ensure_base_dir()
        sessions = []
        for path in cls.BASE_PATH.iterdir():
            if path.is_dir():
                sessions.append({
                    "id": path.name,
                    "name": path.name.rsplit("_", 1)[0] if "_" in path.name else path.name,
                    "created": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                })
        return sorted(sessions, key=lambda x: x["created"], reverse=True)

    @classmethod
    def delete_session(cls, session_id: str):
        import shutil
        session_path = cls.BASE_PATH / session_id
        if session_path.exists():
            shutil.rmtree(session_path)


def render_sidebar():
    """渲染侧边栏"""
    st.sidebar.markdown(
        '<div style="text-align: left; font-size: 1.4rem; font-weight: bold; color: #1f77b4;">📝 AutoText</div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown("智能文本分析工具")

    st.sidebar.markdown("---")

    # 开启新分析按钮
    if st.sidebar.button("➕ 开启新分析", use_container_width=True, type="primary"):
        st.session_state.current_tab = 0
        st.session_state.analysis_completed = False
        st.session_state.analyzer = None
        st.session_state.report_html = None
        st.rerun()

    st.sidebar.markdown("---")

    # 历史项目
    st.sidebar.markdown("### 📜 历史项目")
    sessions = TextSessionService.list_sessions()

    if not sessions:
        st.sidebar.caption("暂无历史项目")
    else:
        for session in sessions[:10]:
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                if st.button(f"📁 {session['name']}", key=f"load_{session['id']}", use_container_width=True):
                    st.session_state.current_session = session['id']
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{session['id']}", help="删除项目"):
                    TextSessionService.delete_session(session['id'])
                    st.rerun()
            st.sidebar.caption(f"   {session['created']}")

    st.sidebar.markdown("---")

    # 使用技巧
    with st.sidebar.expander("💡 使用技巧", expanded=False):
        st.markdown("""
        **快速开始**
        - 上传 .txt 文件或选择 CSV 中的文本列
        - 点击「开始分析」自动生成报告

        **支持功能**
        - 关键词提取
        - 情感分析
        - 实体识别
        - 文本聚类
        - 主题建模
        - 时间趋势

        **导出格式**
        - HTML / JSON / Markdown
        """)