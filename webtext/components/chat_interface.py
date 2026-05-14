# webtext/components/chat_interface.py
"""大模型对话组件 - 基于 JSON 上下文"""

import streamlit as st
import json
from typing import Dict, Any, List


def generate_dynamic_questions(json_data: Dict[str, Any]) -> List[str]:
    """基于 JSON 结果动态生成推荐问题"""
    questions = []

    # 1. 基于情感分布
    sentiment = json_data.get('sentiment', {}).get('distribution', {})
    pos_rate = sentiment.get('positive_rate', 0)
    neg_rate = sentiment.get('negative_rate', 0)

    if pos_rate > 0.5:
        questions.append(f"📈 文本整体情感偏积极（{pos_rate:.0%}），请分析主要原因")
    elif neg_rate > 0.3:
        questions.append(f"📉 文本整体情感偏消极（{neg_rate:.0%}），主要问题是什么")
    elif pos_rate > 0 or neg_rate > 0:
        questions.append(f"😊 文本情感分布：积极{pos_rate:.0%}、消极{neg_rate:.0%}，请解读")

    # 2. 基于高频关键词
    keywords = json_data.get('keywords', {}).get('frequency', [])
    if len(keywords) >= 3:
        top_keywords = [k for k, _ in keywords[:5]]
        kw_str = "、".join(top_keywords)
        questions.append(f"🔑 核心关键词是「{kw_str}」，请总结文本主题")
    elif len(keywords) >= 1:
        questions.append(f"🔑 高频关键词包括「{keywords[0][0]}」，文本主要讨论什么")

    # 3. 基于实体识别
    entities = json_data.get('entity_stats', {})

    # 人名
    persons = entities.get('per', {}).get('top', [])
    if persons:
        names = "、".join([n for n, _ in persons[:3]])
        questions.append(f"👤 文本提到的人物「{names}」，请介绍相关背景")

    # 组织名
    orgs = entities.get('org', {}).get('top', [])
    if orgs:
        org_names = "、".join([n for n, _ in orgs[:2]])
        questions.append(f"🏢 文本涉及的组织「{org_names}」，请分析相关内容")

    # 地名
    locs = entities.get('loc', {}).get('top', [])
    if locs:
        loc_names = "、".join([n for n, _ in locs[:2]])
        questions.append(f"📍 文本提到的地点「{loc_names}」，有何关联")

    # 4. 基于主题（如果有）
    topics = json_data.get('topics', [])
    if topics:
        first_topic = topics[0]
        topic_keywords = first_topic.get('keywords', [])[:3]
        if topic_keywords:
            questions.append(f"📚 文本主要围绕「{'、'.join(topic_keywords)}」展开，请深入分析")

    # 5. 基于文本长度
    data_shape = json_data.get('data_shape', {})
    rows = data_shape.get('rows', 1)
    if rows == 1:
        char_stats = json_data.get('variable_summaries', {}).get('char_length', {})
        char_len = char_stats.get('mean', 0)
        if char_len > 500:
            questions.append("📄 这是一篇长文本，请提取3-5个核心观点")
        elif char_len < 100:
            questions.append("📝 文本较短，请判断其信息完整度")

    # 6. 默认问题（始终保留，确保至少有3个问题）
    if "总结文本核心内容" not in [q[:10] for q in questions]:
        questions.append("📋 请用3句话总结这段文本的核心内容")

    if "值得关注的信息" not in str(questions):
        questions.append("🔍 文本中有哪些值得关注的数据或事实？")

    if len(questions) < 3:
        questions.append("💡 这段文本适合用于什么场景？")
        questions.append("🎯 请给出阅读建议或后续行动")

    return questions[:5]  # 最多5个


def build_agent_prompt(json_data: Dict[str, Any], source_name: str, include_text: bool = False,
                       text_content: str = "") -> str:
    """构建 Agent 提示词"""

    data_shape = json_data.get('data_shape', {})
    variable_types = json_data.get('variable_types', {})
    quality_report = json_data.get('quality_report', {})
    cleaning_suggestions = json_data.get('cleaning_suggestions', [])
    sentiment = json_data.get('sentiment', {}).get('distribution', {})
    keywords = json_data.get('keywords', {}).get('frequency', [])
    entities = json_data.get('entity_stats', {})

    # 类型分布
    type_counts = {}
    type_display = {
        'continuous': '连续变量',
        'categorical': '分类变量',
        'datetime': '日期时间',
        'identifier': '标识符',
        'text': '文本'
    }
    for col, info in variable_types.items():
        typ = info.get('type', 'unknown')
        type_counts[typ] = type_counts.get(typ, 0) + 1

    type_summary = ", ".join([f"{type_display.get(t, t)}: {c}" for t, c in type_counts.items()])

    # 缺失值
    missing_list = quality_report.get('missing', [])
    missing_summary = "\n".join([f"  - {m['column']}: {m['percent']:.1f}%" for m in missing_list[:5]])

    # 情感分布
    pos_rate = sentiment.get('positive_rate', 0)
    neg_rate = sentiment.get('negative_rate', 0)
    neu_rate = sentiment.get('neutral_rate', 0)

    # 关键词
    keyword_list = [w for w, _ in keywords[:15]]

    # 实体统计
    entity_summary = ""
    entity_names = {
        'per': '人物',
        'org': '组织',
        'loc': '地点',
        'product': '产品',
        'time': '时间',
        'number': '数值'
    }
    for etype, display in entity_names.items():
        if etype in entities:
            stats = entities[etype]
            if stats.get('unique', 0) > 0:
                top = stats.get('top', [])[:3]
                top_names = [name for name, _ in top]
                entity_summary += f"  - {display}: {stats['unique']} 个, 示例: {', '.join(top_names)}\n"

    # 主题（如果有）
    topics = json_data.get('topics', [])
    topic_summary = ""
    if topics:
        for topic in topics[:2]:
            topic_keywords = topic.get('keywords', [])[:3]
            if topic_keywords:
                topic_summary += f"  - 主题: {'、'.join(topic_keywords)}\n"

    prompt = f"""
你是专业的文本分析助手，正在回答用户关于已分析文本的问题。

## 文本分析结果

### 数据概览
- 分析时间: {json_data.get('analysis_time', '未知')}
- 文本长度: 均值 {json_data.get('variable_summaries', {}).get('char_length', {}).get('mean', 0):.0f} 字符

### 情感分析
- 积极: {pos_rate:.1%}
- 消极: {neg_rate:.1%}
- 中性: {neu_rate:.1%}

### 高频关键词
{', '.join(keyword_list) if keyword_list else '无'}

### 实体识别
{entity_summary if entity_summary else '  未检测到明显实体'}

### 主题分析
{topic_summary if topic_summary else '  未进行主题建模'}

### 数据质量
{missing_summary if missing_summary else '  无缺失值'}

### 清洗建议
{chr(10).join([f"  - {s}" for s in cleaning_suggestions[:3]]) if cleaning_suggestions else '  无清洗建议'}
"""

    # 可选：添加源文本
    if include_text and text_content:
        prompt += f"\n\n## 源文本\n\n{text_content[:2000]}"

    prompt += """

## 重要说明
1. 用中文回答，结构清晰，友好专业
2. 基于上述分析结果回答用户问题
3. 如果问题与文本分析无关，请礼貌引导
"""

    return prompt


def render_chat_tab():
    """渲染大模型对话标签页"""
    st.markdown("### 🧠 大模型智能解读")
    st.caption("基于分析结果，与 AI 对话获取深度洞察")

    # 检查分析结果
    if not st.session_state.get("text_analysis_completed"):
        st.info("请先在「数据准备」中输入文本并点击「开始分析」")
        return

    # 检查大模型客户端
    if st.session_state.text_llm_client is None:
        st.warning("请先在侧边栏配置大模型")
        return

    json_data = st.session_state.text_json_data
    source_name = "文本分析"

    # 上下文选择器（简化）
    st.markdown("#### 📚 分析上下文")
    col1, col2 = st.columns(2)
    with col1:
        use_json = st.checkbox(
            "📊 JSON 分析结果",
            value=True,
            key="text_ctx_json",
            help="包含完整的统计分析数据"
        )
    with col2:
        use_text = st.checkbox(
            "📝 源文本",
            value=False,
            key="text_ctx_text",
            help="包含原始文本内容（会增加 token 消耗）"
        )

    # 获取源文本内容
    text_content = ""
    if use_text and st.session_state.text_analyzer:
        texts = getattr(st.session_state.text_analyzer, 'raw_texts', [])
        if texts:
            text_content = texts[0]

    # 对话历史
    for msg in st.session_state.text_chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 动态生成推荐问题
    st.markdown("#### 💡 推荐问题")

    if use_json and json_data:
        questions = generate_dynamic_questions(json_data)
    else:
        # 默认问题
        questions = [
            "📋 请用3句话总结这段文本的核心内容",
            "😊 分析文本的情感倾向",
            "🔑 提取文本的关键词",
            "🔍 文本中有哪些值得关注的信息"
        ]

    # 显示推荐问题按钮
    cols = st.columns(min(len(questions), 3))
    for i, q in enumerate(questions):
        col_idx = i % 3
        with cols[col_idx]:
            # 简化显示，取前30字符
            display_q = q[:30] + "..." if len(q) > 30 else q
            if st.button(display_q, key=f"text_q_{i}", use_container_width=True, help=q):
                st.session_state.text_pending_question = q
                st.rerun()

    st.markdown("---")

    # 处理待发送的问题
    if st.session_state.get("text_pending_question"):
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

            # 构建系统提示词
            system_prompt = build_agent_prompt(
                json_data,
                source_name,
                include_text=use_text,
                text_content=text_content
            )

            # 构建消息列表
            messages = [{"role": "system", "content": system_prompt}]

            # 添加历史消息（最近10条）
            for msg in st.session_state.text_chat_messages[:-1]:
                messages.append({"role": msg["role"], "content": msg["content"]})

            # 添加当前问题
            messages.append({"role": "user", "content": question})

            # 调用大模型
            try:
                for chunk in st.session_state.text_llm_client.chat_stream(messages):
                    if chunk:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)
            except Exception as e:
                response_placeholder.error(f"调用大模型失败: {str(e)}")
                full_response = f"调用失败: {str(e)}"

        st.session_state.text_chat_messages.append({"role": "assistant", "content": full_response})
        st.rerun()

    # 输入框
    prompt = st.chat_input("输入您的问题...", key="text_chat_input")
    if prompt:
        st.session_state.text_pending_question = prompt
        st.rerun()