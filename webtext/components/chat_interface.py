"""
AI 对话组件 - 文本分析智能解读
"""

import streamlit as st
from autotext.core.llm_enhance import LLMEnhancer


def render_chat_tab():
    """渲染 AI 对话标签页"""
    st.markdown("### 🧠 AI 智能解读")
    st.caption("基于分析结果，与 AI 对话获取深度洞察")

    # 检查分析结果
    if not st.session_state.analysis_completed or st.session_state.analyzer is None:
        st.info("请先完成文本分析")
        return

    # 检查大模型客户端
    if not hasattr(st.session_state, 'llm_client') or st.session_state.llm_client is None:
        st.warning("请先在数据分析页面的侧边栏配置大模型")
        return

    # 初始化对话历史
    if 'text_chat_messages' not in st.session_state:
        st.session_state.text_chat_messages = []

    # 显示历史消息
    for msg in st.session_state.text_chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 推荐问题
    st.markdown("#### 💡 推荐问题")
    questions = [
        "这段文本主要讲了什么？",
        "文本中的情感倾向如何？",
        "有哪些关键词和主题？",
        "文本中提到了哪些实体？",
        "整体分析建议是什么？"
    ]

    cols = st.columns(len(questions))
    for i, q in enumerate(questions):
        if cols[i].button(q, key=f"q_{i}", use_container_width=True):
            st.session_state.text_pending_question = q
            st.rerun()

    # 处理待发送的问题
    if st.session_state.get('text_pending_question'):
        question = st.session_state.text_pending_question
        del st.session_state.text_pending_question

        # 添加用户消息
        st.session_state.text_chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # 生成回答
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            # 构建上下文
            context = _build_context(st.session_state.analyzer)

            system_prompt = f"""你是专业的文本分析助手。以下是文本分析结果：

{context}

请根据分析结果回答用户问题。用中文回答，简洁专业。"""

            messages = [{"role": "system", "content": system_prompt}]
            for msg in st.session_state.text_chat_messages:
                messages.append({"role": msg["role"], "content": msg["content"]})

            for chunk in st.session_state.llm_client.chat_stream(messages):
                if chunk:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        st.session_state.text_chat_messages.append({"role": "assistant", "content": full_response})
        st.rerun()

    # 输入框
    prompt = st.chat_input("输入您的问题...")
    if prompt:
        st.session_state.text_pending_question = prompt
        st.rerun()


def _build_context(analyzer) -> str:
    """构建分析结果上下文"""
    stats = analyzer.stats_result
    dist = analyzer.sentiment_distribution

    context = f"""
## 基础统计
- 总文本数: {stats['total_count']}
- 平均长度: {stats['char_length']['mean']:.0f} 字符
- 空文本率: {stats['empty_rate']:.1%}

## 情感分布
- 积极: {dist['positive_rate']:.1%}
- 消极: {dist['negative_rate']:.1%}
- 中性: {dist['neutral_rate']:.1%}
"""

    if analyzer.keywords:
        keywords = analyzer.keywords.get("frequency", [])[:20]
        context += f"\n## 高频关键词\n{', '.join([w for w, _ in keywords])}"

    if hasattr(analyzer, 'entity_stats') and analyzer.entity_stats:
        persons = [p for p, _ in analyzer.entity_stats.get("person", {}).get("top", [])[:5]]
        locations = [l for l, _ in analyzer.entity_stats.get("location", {}).get("top", [])[:5]]
        if persons:
            context += f"\n## 人名\n{', '.join(persons)}"
        if locations:
            context += f"\n## 地名\n{', '.join(locations)}"

    return context