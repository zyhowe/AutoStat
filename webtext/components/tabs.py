"""
标签页组件 - 主要标签页按钮
"""

import streamlit as st


def render_tabs():
    """渲染标签页按钮，返回当前选中的标签页"""

    st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)
    st.markdown("<style>div.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)

    tab_names = ["📁 数据准备", "📄 分析报告", "🤖 文本分类", "🧠 AI 解读"]

    cols = st.columns(len(tab_names))

    for i, (col, name) in enumerate(zip(cols, tab_names)):
        with col:
            is_active = (st.session_state.current_tab == i)

            if is_active:
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
                if st.button(name, key=f"tab_{i}", use_container_width=True):
                    st.session_state.current_tab = i
                    st.rerun()

    st.divider()

    return st.session_state.current_tab


def scroll_to_top():
    """滚动到页面顶部"""
    st.markdown("""
    <script>
        var element = document.getElementById('top-anchor');
        if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    </script>
    """, unsafe_allow_html=True)