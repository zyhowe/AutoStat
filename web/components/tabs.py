"""标签页组件 - 4个主要标签页按钮"""

import streamlit as st


def render_tabs():
    """渲染标签页按钮，返回当前选中的标签页"""

    tab_names = ["📁 数据准备", "📄 预览报告", "📝 分析日志", "🤖 AI 解读"]

    cols = st.columns(4)

    for i, (col, name) in enumerate(zip(cols, tab_names)):
        with col:
            is_active = (st.session_state.current_tab == i)

            if is_active:
                # 激活状态：使用 HTML 模拟高亮按钮
                st.markdown(f"""
                <div style="
                    background-color: #1f77b4;
                    color: white;
                    border-radius: 8px;
                    padding: 8px 0;
                    text-align: center;
                    font-weight: 500;
                    cursor: default;
                ">{name}</div>
                """, unsafe_allow_html=True)
            else:
                # 非激活状态：使用 st.button
                if st.button(name, key=f"tab_{i}", use_container_width=True):
                    st.session_state.current_tab = i
                    st.rerun()

    st.divider()

    return st.session_state.current_tab


def scroll_to_top():
    """滚动到页面顶部"""
    st.markdown("""
    <script>
        window.scrollTo({ top: 0, behavior: 'smooth' });
    </script>
    """, unsafe_allow_html=True)