"""
AI助手组件 - 三层结构：数据特征 / 智能推荐 / 场景模板
"""

import streamlit as st
import json
from typing import Dict, Any, List

from web.services.session_service import SessionService
from web.services.storage_service import StorageService
from web.services.agent_service import AgentService


def render_ai_assistant():
    """渲染AI助手 - 左右分栏"""
    st.markdown("### 🧠 AI助手")
    st.caption("基于数据分析结果，智能问答与场景探索")

    session_id = SessionService.get_current_session()
    if session_id is None:
        st.info("请先完成数据分析")
        return

    if st.session_state.get("llm_client") is None:
        st.warning("请先在侧边栏配置大模型")
        return

    json_data = StorageService.load_json("analysis_result", session_id)
    if json_data is None:
        st.info("请先完成数据分析")
        return

    left_col, right_col = st.columns([1.2, 2], gap="large")

    with left_col:
        render_left_panel(json_data, session_id)

    with right_col:
        render_right_panel(json_data, session_id)


def render_left_panel(json_data: Dict[str, Any], session_id: str):
    """左侧面板 - 三层结构"""
    st.markdown("#### 📌 问题面板")

    # ==================== 第一层：数据特征（只读展示） ====================
    with st.container():
        st.markdown("""
        <div style="
            background: #f0f2f6;
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 12px;
            font-size: 12px;
            color: #555;
        ">
            <div style="font-weight: bold; color: #333; margin-bottom: 4px;">📊 当前数据特征</div>
        """, unsafe_allow_html=True)

        _render_data_features(json_data)

        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ==================== 第二层：智能推荐（点击使用） ====================
    st.markdown("#### 🎯 智能推荐")
    st.caption("基于当前数据特征，推荐最适合的分析问题")

    recommendations = _get_smart_recommendations(json_data)

    if recommendations:
        cols = st.columns(2)
        for i, rec in enumerate(recommendations):
            with cols[i % 2]:
                if st.button(
                    rec["label"],
                    key=f"smart_rec_{i}",
                    use_container_width=True,
                    help=rec.get("description", "")
                ):
                    st.session_state.ai_selected_question = rec["question"]
                    st.rerun()
    else:
        st.caption("暂无智能推荐，请使用下方场景模板")

    st.divider()

    # ==================== 第三层：场景模板（点击使用） ====================
    st.markdown("#### 📌 场景模板")
    st.caption("按需选择，所有预设问题")

    if "ai_selected_question" not in st.session_state:
        st.session_state.ai_selected_question = ""

    _render_scenario_templates()


def _render_data_features(json_data: Dict[str, Any]):
    """渲染数据特征（只读）"""
    data_shape = json_data.get("data_shape", {})
    variable_types = json_data.get("variable_types", {})
    quality = json_data.get("quality_report", {})
    ts_diag = json_data.get("time_series_diagnostics", {})

    rows = data_shape.get("rows", 0)
    cols = data_shape.get("columns", 0)

    type_counts = {}
    for info in variable_types.values():
        typ = info.get("type", "unknown")
        type_counts[typ] = type_counts.get(typ, 0) + 1

    type_display = {
        "continuous": "连续",
        "categorical": "分类",
        "categorical_numeric": "数值分类",
        "ordinal": "有序",
        "datetime": "日期",
        "identifier": "标识符"
    }
    type_str = " / ".join([f"{type_display.get(t, t)}{c}" for t, c in type_counts.items() if t in type_display])

    has_auto = any(v.get("has_autocorrelation") for v in ts_diag.values())
    ts_count = sum(1 for v in ts_diag.values() if v.get("has_autocorrelation"))

    outliers = quality.get("outliers", {})
    outlier_count = len(outliers)

    missing = quality.get("missing", [])
    missing_count = len(missing)

    st.markdown(f"""
    <div style="font-size: 12px; line-height: 1.6;">
        <div>📐 数据量: <strong>{rows:,}</strong> 行 × <strong>{cols}</strong> 列</div>
        <div>🏷️ 变量: {type_str}</div>
        <div>📈 时间序列: {"✅ " + str(ts_count) + "个有自相关" if has_auto else "❌ 无"}</div>
        <div>⚠️ 异常值: {"⚠️ " + str(outlier_count) + "个字段" if outlier_count > 0 else "✅ 无"}</div>
        <div>📊 缺失值: {"⚠️ " + str(missing_count) + "个字段" if missing_count > 0 else "✅ 无"}</div>
    </div>
    """, unsafe_allow_html=True)


def _get_smart_recommendations(json_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """根据数据特征生成智能推荐"""
    recommendations = []

    variable_types = json_data.get("variable_types", {})
    ts_diag = json_data.get("time_series_diagnostics", {})
    quality = json_data.get("quality_report", {})
    correlations = json_data.get("correlations", {})
    high_corrs = correlations.get("high_correlations", [])

    has_continuous = any(info.get("type") == "continuous" for info in variable_types.values())
    has_categorical = any(info.get("type") in ["categorical", "categorical_numeric", "ordinal"] for info in variable_types.values())
    has_datetime = any(info.get("type") == "datetime" for info in variable_types.values())
    has_auto = any(v.get("has_autocorrelation") for v in ts_diag.values())
    has_outliers = len(quality.get("outliers", {})) > 0
    has_high_corr = len(high_corrs) > 0

    numeric_cols = [col for col, info in variable_types.items() if info.get("type") == "continuous"]

    if has_datetime and has_continuous:
        recommendations.append({
            "label": "📈 趋势分析",
            "question": "分析数据的时间变化趋势，找出周期性规律和异常波动",
            "description": "检测到日期和数值字段，适合做趋势分析"
        })

    if has_categorical and has_continuous:
        recommendations.append({
            "label": "📊 对比分析",
            "question": "按分类维度对比数据差异，找出显著特征和异常值",
            "description": "检测到分类和数值字段，适合做对比分析"
        })

    if len(numeric_cols) >= 3:
        recommendations.append({
            "label": "🔗 相关性分析",
            "question": "分析数值变量之间的相关关系，发现强关联对",
            "description": f"检测到 {len(numeric_cols)} 个数值字段，适合做相关性分析"
        })

    if has_auto:
        recommendations.append({
            "label": "🔮 预测分析",
            "question": "基于历史数据预测未来趋势，给出置信区间",
            "description": f"检测到 {sum(1 for v in ts_diag.values() if v.get('has_autocorrelation'))} 个序列有自相关，适合做预测"
        })

    if has_outliers:
        recommendations.append({
            "label": "🚨 异常诊断",
            "question": "检测数据中的异常值和异常模式，分析异常原因",
            "description": f"检测到 {len(quality.get('outliers', {}))} 个字段有异常值，需要诊断"
        })

    if has_high_corr:
        recommendations.append({
            "label": "🔗 关联解读",
            "question": "解读强相关关系的业务含义，分析因果关系",
            "description": f"检测到 {len(high_corrs)} 对强相关关系，值得深入解读"
        })

    missing_count = len(quality.get("missing", []))
    if missing_count > 0:
        recommendations.append({
            "label": "📋 质量分析",
            "question": "分析数据质量问题，给出缺失值和异常值处理建议",
            "description": f"检测到 {missing_count} 个字段有缺失值，需要处理"
        })

    return recommendations[:4]


def _render_scenario_templates():
    """渲染场景模板 - 5大类"""
    scenarios = {
        "📊 探索分析": [
            "数据整体情况如何？有哪些关键特征？",
            "变量之间有什么关系？",
            "数据分布有什么特点？",
            "时间序列有什么规律？",
            "有哪些值得关注的异常？",
        ],
        "🔍 诊断分析": [
            "检测数据中的异常值和异常模式",
            "分析某指标异常的根本原因",
            "诊断数据质量问题并提出改进建议",
            "检查数据中的勾稽关系是否一致",
            "识别数据中的风险点和隐患",
        ],
        "📈 预测分析": [
            "预测未来趋势（时序预测）",
            "预测用户流失风险",
            "预测需求量或销量",
            "用训练好的模型进行预测",
            "评估预测结果的准确性和置信度",
        ],
        "💡 决策支持": [
            "基于数据给出业务优化建议",
            "对比不同业务方案的预期效果",
            "评估当前业务的风险点",
            "发现数据中的增长机会",
            "分析销售/用户/运营的关键驱动因素",
        ],
        "📝 生成输出": [
            "生成数据叙事报告",
            "生成一页纸的执行摘要",
            "生成专题分析报告",
            "生成SQL查询语句",
            "生成仪表板设计建议",
        ],
    }

    for category, questions in scenarios.items():
        with st.expander(category, expanded=False):
            for q in questions:
                key = f"scenario_{hash(category)}_{hash(q)}"
                if st.button(q, key=key, use_container_width=True):
                    st.session_state.ai_selected_question = q
                    st.rerun()


def render_right_panel(json_data: Dict[str, Any], session_id: str):
    """右侧面板 - 对话区域（流式输出在容器内）"""
    st.markdown("#### 💬 对话")

    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = []

    # ==================== 对话容器 - 固定高度，独立滚动 ====================
    chat_container = st.container(height=500)

    with chat_container:
        if not st.session_state.ai_messages:
            st.caption("💡 从左侧选择问题开始对话")

        for msg in st.session_state.ai_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ==================== 处理选中问题 ====================
    if st.session_state.get("ai_selected_question"):
        question = st.session_state.ai_selected_question
        st.session_state.ai_selected_question = ""

        # 添加用户消息到历史
        st.session_state.ai_messages.append({"role": "user", "content": question})

        # 🆕 在对话容器内显示流式输出
        with chat_container:
            # 显示用户消息
            with st.chat_message("user"):
                st.markdown(question)

            # 显示AI回复（流式）
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""

                system_prompt = _build_ai_system_prompt(json_data, session_id)

                messages = [{"role": "system", "content": system_prompt}]
                for msg in st.session_state.ai_messages[:-1]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                messages.append({"role": "user", "content": question})

                try:
                    for chunk in st.session_state.llm_client.chat_stream(messages):
                        if chunk:
                            full_response += chunk
                            response_placeholder.markdown(full_response + "▌")

                    response_placeholder.markdown(full_response)

                    if "[[TOOL_CALL]]" in full_response:
                        agent_service = AgentService(session_id)
                        tool_result = _process_tool_call(full_response, agent_service)
                        if tool_result:
                            full_response = tool_result
                            response_placeholder.markdown(full_response)

                except Exception as e:
                    response_placeholder.error(f"调用大模型失败: {str(e)}")
                    full_response = f"调用失败: {str(e)}"

        # 保存AI回复到历史
        st.session_state.ai_messages.append({"role": "assistant", "content": full_response})
        st.rerun()

    # ==================== 输入框（固定在底部） ====================
    prompt = st.chat_input("输入您的问题...", key="ai_chat_input")
    if prompt and prompt.strip():
        st.session_state.ai_selected_question = prompt.strip()
        st.rerun()


def _build_ai_system_prompt(json_data: Dict[str, Any], session_id: str) -> str:
    """构建AI系统提示词"""
    data_shape = json_data.get("data_shape", {})
    variable_types = json_data.get("variable_types", {})
    quality = json_data.get("quality_report", {})
    correlations = json_data.get("correlations", {})
    ts_diag = json_data.get("time_series_diagnostics", {})
    recommendations = json_data.get("model_recommendations", [])

    type_counts = {}
    type_display = {
        "continuous": "连续变量",
        "categorical": "分类变量",
        "categorical_numeric": "数值型分类",
        "ordinal": "有序分类",
        "datetime": "日期时间",
        "identifier": "标识符"
    }
    for info in variable_types.values():
        typ = info.get("type", "unknown")
        type_counts[typ] = type_counts.get(typ, 0) + 1
    type_summary = "、".join([f"{type_display.get(t, t)} {c}个" for t, c in type_counts.items()])

    high_corrs = correlations.get("high_correlations", [])
    corr_summary = ""
    if high_corrs:
        top = high_corrs[0]
        corr_summary = f"{top.get('var1', '')} ↔ {top.get('var2', '')} (r={top.get('value', 0):.3f})"

    has_auto = any(v.get("has_autocorrelation") for v in ts_diag.values())
    ts_summary = "有" if has_auto else "无"

    model_summary = ""
    if recommendations:
        rec = recommendations[0]
        model_summary = f"{rec.get('task_type', '')} 推荐 {rec.get('ml', '')}"

    dup_count = quality.get("duplicates", {}).get("count", 0)
    try:
        dup_count = int(dup_count) if dup_count else 0
    except (ValueError, TypeError):
        dup_count = 0

    return f"""你是专业的数据分析师，正在回答用户关于数据的问题。

## 数据概况
- 总行数: {data_shape.get('rows', 0):,}
- 总列数: {data_shape.get('columns', 0)}
- 变量类型: {type_summary}

## 数据质量
- 缺失字段: {len(quality.get('missing', []))}个
- 异常字段: {len(quality.get('outliers', {}))}个
- 重复记录: {dup_count}条

## 关键发现
- 强相关: {corr_summary if corr_summary else '无'}
- 时间序列: {ts_summary}
- 模型推荐: {model_summary if model_summary else '无'}

## 重要说明
1. 用中文回答，结构清晰，友好专业
2. 基于数据分析结果回答
3. 回答要具体、可执行
4. 如果问题涉及预测，建议使用模型中心
"""


def _process_tool_call(response: str, agent_service: AgentService) -> str:
    """处理工具调用"""
    import re
    import json

    tool_pattern = r'\[\[TOOL_CALL\]\](.*?)\[\[/TOOL_CALL\]\]'
    match = re.search(tool_pattern, response, re.DOTALL)

    if not match:
        return None

    try:
        tool_call = json.loads(match.group(1))
        tool_name = tool_call.get('tool', '')
        params = tool_call.get('params', {})

        if tool_name == 'predict':
            result = agent_service.predict_with_model(
                params.get('model_key', ''),
                params.get('input_values', {})
            )
            if result.get('success'):
                return f"""
**预测结果**

- 使用模型: {result.get('model_name')}
- 预测值: {result.get('prediction')}
- 置信度: {result.get('confidence', 0):.2%}
"""
            else:
                return f"预测失败: {result.get('error')}"

    except Exception as e:
        return f"工具调用失败: {str(e)}"

    return None