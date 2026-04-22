"""
空状态引导组件 - 无数据时显示引导
"""

import streamlit as st


def render_empty_state():
    """渲染空状态引导页面"""

    st.markdown("""
    <style>
    .empty-state-container {
        text-align: center;
        padding: 60px 20px;
        max-width: 800px;
        margin: 0 auto;
    }
    .empty-state-icon {
        font-size: 64px;
        margin-bottom: 20px;
    }
    .empty-state-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 12px;
        color: #333;
    }
    .empty-state-desc {
        font-size: 14px;
        color: #666;
        margin-bottom: 30px;
    }
    .step-container {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin: 40px 0;
        flex-wrap: wrap;
    }
    .step {
        text-align: center;
        width: 120px;
    }
    .step-number {
        width: 40px;
        height: 40px;
        background: #1f77b4;
        color: white;
        border-radius: 20px;
        line-height: 40px;
        margin: 0 auto 12px;
        font-weight: bold;
    }
    .step-title {
        font-weight: 500;
        margin-bottom: 4px;
    }
    .step-desc {
        font-size: 12px;
        color: #999;
    }
    .arrow {
        font-size: 24px;
        color: #ccc;
        line-height: 40px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.html("""
    <div class="empty-state-container">
        <div class="empty-state-icon">📊</div>
        <div class="empty-state-title">开始你的数据分析之旅</div>
        <div class="empty-state-desc">只需三步，获得专业洞察</div>

        <div class="step-container">
            <div class="step">
                <div class="step-number">1</div>
                <div class="step-title">上传数据</div>
                <div class="step-desc">CSV、Excel、JSON</div>
            </div>
            <div class="arrow">→</div>
            <div class="step">
                <div class="step-number">2</div>
                <div class="step-title">自动分析</div>
                <div class="step-desc">AI识别类型和关系</div>
            </div>
            <div class="arrow">→</div>
            <div class="step">
                <div class="step-number">3</div>
                <div class="step-title">获得洞察</div>
                <div class="step-desc">报告+预测+建议</div>
            </div>
        </div>
    </div>
    """)

    # 演示数据按钮
    st.markdown("---")
    st.markdown("##### 或者试试示例数据")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 销售数据示例", use_container_width=True):
            st.session_state.demo_request = "sales"
            st.rerun()

    with col2:
        if st.button("👥 用户数据示例", use_container_width=True):
            st.session_state.demo_request = "user_behavior"
            st.rerun()

    with col3:
        if st.button("🏥 医疗数据示例", use_container_width=True):
            st.session_state.demo_request = "medical"
            st.rerun()


def render_welcome_banner():
    """渲染欢迎横幅（已登录用户）"""
    if st.session_state.get("onboarding_completed", False):
        return

    st.info("""
    👋 **欢迎使用 AutoStat！**

    - 📁 上传数据文件开始分析
    - 🎯 试试下方的示例数据快速体验
    - ⚙️ 在侧边栏配置大模型获得AI智能解读
    """)