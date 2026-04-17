"""标签页组件 - 4个主要标签页按钮"""

import streamlit as st


def render_tabs():
    """渲染标签页按钮，返回当前选中的标签页"""

    # 锚点用于滚动定位
    st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)

    # 减少顶部空白
    st.markdown("<style>div.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)

    tab_names = ["📁 数据准备", "📄 预览报告", "📝 分析日志", "🧠 模型训练", "🤖 AI 解读"]

    cols = st.columns(5)

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
    # 使用锚点 + JavaScript 滚动
    st.markdown("""
    <script>
        var element = document.getElementById('top-anchor');
        if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    </script>
    """, unsafe_allow_html=True)