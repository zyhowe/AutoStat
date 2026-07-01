"""
AI助手组件 - 固定高度滚动对话框
所有入口默认折叠，示例动态生成
"""

import streamlit as st
import json
import re
from typing import Dict, Any, List, Optional

from web.services.session_service import SessionService
from web.services.storage_service import StorageService
from web.services.agent_service import AgentService


def render_ai_assistant():
    """渲染AI助手 - 左右分栏，始终渲染界面"""
    st.markdown("### 🧠 AI助手")
    st.caption("基于数据分析结果，智能问答与场景探索")

    # ==================== 获取 session_id ====================
    session_id = SessionService.get_current_session()

    # ==================== 加载 json_data（可能为 None） ====================
    json_data = None
    if session_id:
        json_data = st.session_state.get("current_json_data")
        if json_data is None:
            json_data = StorageService.load_json("analysis_result", session_id)
            if json_data is not None:
                st.session_state.current_json_data = json_data

    # ==================== 初始化状态 ====================
    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = []
    if "ai_pending_question" not in st.session_state:
        st.session_state.ai_pending_question = None
    if "ai_selected_contexts" not in st.session_state:
        st.session_state.ai_selected_contexts = ["json_result"]

    # ==================== 检查大模型配置 ====================
    if st.session_state.get("llm_client") is None:
        st.warning("请先在侧边栏配置大模型")

    # ==================== 渲染左右分栏 ====================
    left_col, right_col = st.columns([1.2, 2], gap="large")

    with left_col:
        render_left_panel(json_data, session_id)

    with right_col:
        render_chat_interface(json_data, session_id)


# ==================== 左侧面板 ====================

def render_left_panel(json_data: Optional[Dict[str, Any]], session_id: str):
    """左侧面板 - 所有入口默认折叠"""
    st.markdown("#### 📌 问题面板")

    # ==================== 1. 数据特征（只读） ====================
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
        _render_data_features(json_data, session_id)
        st.markdown("</div>", unsafe_allow_html=True)

    # ==================== 2. 上下文选择 ====================
    _render_context_selector(json_data)

    st.divider()

    # ==================== 3. 数据解读 ====================
    with st.expander("💬 数据解读", expanded=False):
        st.caption("数据统计分析：数据是什么样的？有什么特征？")
        if json_data is not None:
            _render_data_interpretation()
        else:
            st.info("请先完成数据分析")

    # ==================== 4. 动态分析 ====================
    with st.expander("🎯 动态分析", expanded=False):
        st.caption("探索性分析：数据能做什么？有什么规律？")
        if json_data is not None:
            _render_dynamic_analysis(json_data)
        else:
            st.info("请先完成数据分析")

    # ==================== 5. 场景推荐（动态生成） ====================
    with st.expander("🎯 场景推荐", expanded=False):
        if json_data is not None:
            _render_scenario_recommendation(json_data)
        else:
            st.info("请先完成数据分析")

    # ==================== 6. 自然查询（动态生成） ====================
    with st.expander("🔍 自然查询", expanded=False):
        if json_data is not None:
            _render_natural_query(json_data)
        else:
            st.info("请先完成数据分析")

    # ==================== 7. SQL生成（动态生成，仅数据库模式） ====================
    is_database = False
    if session_id:
        metadata = SessionService.load_metadata(session_id)
        is_database = metadata.get("analysis_type") == "database"
    if is_database:
        with st.expander("📝 SQL生成", expanded=False):
            if json_data is not None:
                _render_sql_generator(json_data)
            else:
                st.info("请先完成数据分析")

    # ==================== 8. 推理预测（有模型时显示） ====================
    has_models = _has_trained_models(session_id) if session_id else False
    if has_models:
        with st.expander("🔮 推理预测", expanded=False):
            if json_data is not None:
                _render_inference_prediction(session_id)
            else:
                st.info("请先完成数据分析")

    # ==================== 9. 勾稽校验 ====================
    with st.expander("🔗 勾稽校验", expanded=False):
        if json_data is not None:
            _render_audit_rule()
        else:
            st.info("请先完成数据分析")


# ==================== 1. 数据特征 ====================

def _render_data_features(json_data: Optional[Dict[str, Any]], session_id: str):
    """渲染数据特征，无数据时显示空状态"""
    if json_data is None:
        st.markdown("""
        <div style="text-align:center;padding:20px 10px;color:#888;">
            <div style="font-size:28px;margin-bottom:8px;">📋</div>
            <div style="font-weight:500;margin-bottom:4px;">未检测到分析结果</div>
            <div style="font-size:11px;color:#aaa;">请先完成数据分析生成JSON报告</div>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 去数据准备", key="goto_data_prep", use_container_width=True):
                st.session_state.current_tab = 0
                st.rerun()
        with col2:
            if st.button("🔄 重新加载", key="reload_json", use_container_width=True):
                if session_id:
                    data = StorageService.load_json("analysis_result", session_id)
                    if data is not None:
                        st.session_state.current_json_data = data
                        st.rerun()
        return

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

    col_names = list(variable_types.keys())
    col_preview = "、".join(col_names[:8])
    if len(col_names) > 8:
        col_preview += f"等{len(col_names)}个"

    st.markdown(f"""
    <div style="font-size:12px;line-height:1.6;">
        <div>📐 数据量: <strong>{rows:,}</strong> 行 × <strong>{cols}</strong> 列</div>
        <div>🏷️ 变量: {type_str}</div>
        <div>📋 字段: {col_preview}</div>
        <div>📈 时间序列: {"✅ " + str(ts_count) + "个有自相关" if has_auto else "❌ 无"}</div>
        <div>⚠️ 异常值: {"⚠️ " + str(outlier_count) + "个字段" if outlier_count > 0 else "✅ 无"}</div>
        <div>📊 缺失值: {"⚠️ " + str(missing_count) + "个字段" if missing_count > 0 else "✅ 无"}</div>
    </div>
    """, unsafe_allow_html=True)


# ==================== 2. 上下文选择 ====================

def _render_context_selector(json_data: Optional[Dict[str, Any]]):
    st.markdown("#### 📚 选择分析上下文")
    st.caption("勾选要提供给AI的上下文信息")

    col_ctx1, col_ctx2, col_ctx3 = st.columns(3)

    with col_ctx1:
        ctx_json = st.checkbox(
            "📊 JSON 结果",
            value="json_result" in st.session_state.ai_selected_contexts,
            key="ai_ctx_json",
            disabled=(json_data is None)
        )
    with col_ctx2:
        ctx_html = st.checkbox(
            "📄 HTML 报告",
            value="html_report" in st.session_state.ai_selected_contexts,
            key="ai_ctx_html",
            disabled=(json_data is None)
        )
    with col_ctx3:
        ctx_data = st.checkbox(
            "🗃️ 源数据",
            value="raw_data" in st.session_state.ai_selected_contexts,
            key="ai_ctx_data",
            disabled=(json_data is None)
        )

    selected = []
    if ctx_json:
        selected.append("json_result")
    if ctx_html:
        selected.append("html_report")
    if ctx_data:
        selected.append("raw_data")

    if set(selected) != set(st.session_state.ai_selected_contexts):
        st.session_state.ai_selected_contexts = selected

    if json_data is None:
        st.caption("⚠️ 无分析结果，部分上下文不可用")


# ==================== 3. 数据解读 ====================

def _render_data_interpretation():
    """数据解读 - 点击发送到对话框"""
    questions = [
        "📊 数据整体概况和核心指标",
        "📋 数据质量怎么样？有什么问题？",
        "📈 数值变量的分布特征",
        "📊 分类变量的分布情况",
        "🔗 变量之间存在什么关系？",
        "📅 数据的时间范围和趋势？",
        "💡 数据的核心洞察有哪些？",
    ]

    cols = st.columns(2)
    for i, q in enumerate(questions):
        with cols[i % 2]:
            if st.button(q, key=f"interpret_{i}", use_container_width=True):
                st.session_state.ai_pending_question = q
                st.rerun()


# ==================== 4. 动态分析 ====================

def _render_dynamic_analysis(json_data: Dict[str, Any]):
    recommendations = _get_smart_recommendations(json_data)
    if not recommendations:
        st.caption("当前数据特征不足以生成动态分析建议")
        return

    cols = st.columns(3)
    for i, rec in enumerate(recommendations):
        with cols[i % 3]:
            if st.button(rec["label"], key=f"dynamic_{i}", use_container_width=True, help=rec.get("description", "")):
                st.session_state.ai_pending_question = rec["question"]
                st.rerun()


def _get_smart_recommendations(json_data: Dict[str, Any]) -> List[Dict[str, str]]:
    recommendations = []
    variable_types = json_data.get("variable_types", {})
    ts_diag = json_data.get("time_series_diagnostics", {})
    quality = json_data.get("quality_report", {})
    correlations = json_data.get("correlations", {})
    high_corrs = correlations.get("high_correlations", [])
    data_shape = json_data.get("data_shape", {})

    has_continuous = any(info.get("type") == "continuous" for info in variable_types.values())
    has_categorical = any(info.get("type") in ["categorical", "categorical_numeric", "ordinal"] for info in variable_types.values())
    has_datetime = any(info.get("type") == "datetime" for info in variable_types.values())
    has_auto = any(v.get("has_autocorrelation") for v in ts_diag.values())
    has_outliers = len(quality.get("outliers", {})) > 0
    has_missing = len(quality.get("missing", [])) > 0
    has_high_corr = len(high_corrs) > 0

    numeric_cols = [col for col, info in variable_types.items() if info.get("type") == "continuous"]
    categorical_cols = [col for col, info in variable_types.items() if info.get("type") in ["categorical", "categorical_numeric", "ordinal"]]
    n_samples = data_shape.get("rows", 0)

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
        auto_count = sum(1 for v in ts_diag.values() if v.get("has_autocorrelation"))
        recommendations.append({
            "label": "🔮 预测分析",
            "question": "基于历史数据预测未来趋势，给出置信区间",
            "description": f"检测到 {auto_count} 个序列有自相关，适合做预测"
        })

    if has_outliers:
        recommendations.append({
            "label": "🚨 异常诊断",
            "question": "检测数据中的异常值和异常模式，分析异常原因",
            "description": f"检测到 {len(quality.get('outliers', {}))} 个字段有异常值，需要诊断"
        })

    if has_missing:
        recommendations.append({
            "label": "📋 质量分析",
            "question": "分析数据质量问题，给出缺失值和异常值处理建议",
            "description": f"检测到 {len(quality.get('missing', []))} 个字段有缺失值，需要处理"
        })

    if has_high_corr:
        recommendations.append({
            "label": "🔗 关联解读",
            "question": "解读强相关关系的业务含义，分析因果关系",
            "description": f"检测到 {len(high_corrs)} 对强相关关系，值得深入解读"
        })

    if len(numeric_cols) >= 2:
        recommendations.append({
            "label": "📊 分布分析",
            "question": "分析各变量的分布特征，判断是否符合正态分布，发现偏态变量",
            "description": f"检测到 {len(numeric_cols)} 个数值字段，适合做分布分析"
        })

    if len(numeric_cols) >= 3 and n_samples >= 100:
        recommendations.append({
            "label": "🔘 聚类分析",
            "question": "对数据进行聚类分析，发现潜在的用户群体或模式",
            "description": f"{len(numeric_cols)}个数值指标，{n_samples}个样本，适合做聚类分析"
        })

    if len(categorical_cols) >= 3:
        recommendations.append({
            "label": "🔗 关联规则",
            "question": "挖掘分类变量之间的关联规则，发现「如果A则B」的模式",
            "description": f"检测到 {len(categorical_cols)} 个分类字段，适合做关联规则挖掘"
        })

    if has_datetime and has_continuous and n_samples >= 30:
        recommendations.append({
            "label": "📈 变化检测",
            "question": "检测数据中的突变点和趋势变化，发现拐点和转折点",
            "description": "有日期和连续字段，适合做变化点检测"
        })

    recommendations.append({
        "label": "📋 数据概览",
        "question": "数据整体情况如何？有哪些关键特征和核心指标？",
        "description": "快速了解数据的整体面貌"
    })

    seen = set()
    unique = []
    for rec in recommendations:
        if rec["label"] not in seen:
            seen.add(rec["label"])
            unique.append(rec)
    return unique[:8]


# ==================== 5. 场景推荐（动态生成） ====================

def _render_scenario_recommendation(json_data: Dict[str, Any]):
    """根据数据特征动态生成场景推荐"""
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
    has_missing = len(quality.get("missing", [])) > 0
    has_high_corr = len(high_corrs) > 0

    numeric_cols = [col for col, info in variable_types.items() if info.get("type") == "continuous"]
    categorical_cols = [col for col, info in variable_types.items() if info.get("type") in ["categorical", "categorical_numeric", "ordinal"]]
    data_shape = json_data.get("data_shape", {})
    n_samples = data_shape.get("rows", 0)

    scenarios = []

    # 根据数据特征生成场景描述
    if has_datetime and has_continuous:
        scenarios.append({
            "label": "📈 趋势分析场景",
            "question": "请从趋势分析角度解读数据，找出时间规律、周期性和异常波动。"
        })

    if has_categorical and has_continuous:
        scenarios.append({
            "label": "📊 对比分析场景",
            "question": "请从对比分析角度解读数据，找出各分类维度的差异和显著特征。"
        })

    if len(numeric_cols) >= 3:
        scenarios.append({
            "label": "🔗 相关性分析场景",
            "question": f"请从相关性分析角度解读数据，找出 {', '.join(numeric_cols[:3])} 等变量之间的关联关系。"
        })

    if has_auto:
        scenarios.append({
            "label": "🔮 预测分析场景",
            "question": "请从预测分析角度解读数据，评估哪些指标适合预测，给出建模建议。"
        })

    if has_outliers:
        scenarios.append({
            "label": "🚨 异常诊断场景",
            "question": "请从异常诊断角度解读数据，识别异常值和异常模式，分析可能的原因。"
        })

    if len(numeric_cols) >= 3 and n_samples >= 100:
        scenarios.append({
            "label": "🔘 聚类分析场景",
            "question": f"请从聚类分析角度解读数据，基于 {', '.join(numeric_cols[:3])} 等指标发现潜在的数据分群。"
        })

    if len(categorical_cols) >= 3:
        scenarios.append({
            "label": "🔗 关联规则场景",
            "question": f"请从关联规则角度解读数据，找出 {', '.join(categorical_cols[:3])} 等分类变量之间的关联模式。"
        })

    if has_missing or has_outliers:
        scenarios.append({
            "label": "📋 数据质量场景",
            "question": f"请从数据质量角度解读数据，{'处理缺失值' if has_missing else ''}{'和异常值' if has_outliers else ''}，给出质量提升建议。"
        })

    # 如果场景太少，补充通用场景
    if len(scenarios) < 3:
        scenarios.append({
            "label": "📋 综合数据概览",
            "question": "请从综合角度解读数据，包括数据特征、质量、关系和潜在价值。"
        })

    # 展示场景
    cols = st.columns(2)
    for i, s in enumerate(scenarios[:6]):  # 最多显示6个
        with cols[i % 2]:
            if st.button(s["label"], key=f"scenario_{i}", use_container_width=True):
                st.session_state.ai_pending_question = s["question"]
                st.rerun()


# ==================== 6. 自然查询（动态生成） ====================

def _render_natural_query(json_data: Dict[str, Any]):
    """根据实际字段生成自然查询示例"""
    variable_types = json_data.get("variable_types", {})
    data_shape = json_data.get("data_shape", {})

    # 提取各类字段
    date_cols = [col for col, info in variable_types.items() if info.get("type") == "datetime"]
    numeric_cols = [col for col, info in variable_types.items() if info.get("type") == "continuous"]
    categorical_cols = [col for col, info in variable_types.items() if info.get("type") in ["categorical", "categorical_numeric", "ordinal"]]
    all_cols = list(variable_types.keys())

    examples = []

    # 1. 如果有日期字段
    if date_cols:
        date_field = date_cols[0]
        examples.append(f"查询最近7天的{date_field}数据")

    # 2. 如果有分类和数值字段
    if categorical_cols and numeric_cols:
        cat_field = categorical_cols[0]
        num_field = numeric_cols[0]
        examples.append(f"统计各{cat_field}的{num_field}")

    # 3. 如果有数值字段
    if numeric_cols:
        num_field = numeric_cols[0]
        examples.append(f"找出{num_field}最大的前10条记录")

    # 4. 如果有日期和数值字段
    if date_cols and numeric_cols:
        date_field = date_cols[0]
        num_field = numeric_cols[0]
        examples.append(f"分析各月{num_field}的变化趋势")

    # 5. 如果有分类字段
    if categorical_cols:
        cat_field = categorical_cols[0]
        examples.append(f"查询{cat_field}为某个值的所有记录")

    # 6. 如果有多个数值字段
    if len(numeric_cols) >= 2:
        examples.append(f"分析{numeric_cols[0]}和{numeric_cols[1]}的关系")

    # 7. 兜底
    if not examples:
        examples = ["查询所有数据", "统计总数", "查看数据分布"]

    st.markdown("**💡 示例查询（点击使用）：**")
    cols = st.columns(2)
    for i, ex in enumerate(examples[:6]):
        with cols[i % 2]:
            if st.button(f"🔍 {ex}", key=f"query_ex_{i}", use_container_width=True):
                st.session_state.ai_pending_question = ex
                st.rerun()

    st.markdown("---")
    st.markdown("**✏️ 自定义查询：**")
    custom_query = st.text_input("输入您的问题", placeholder="例如：查询销售额大于1000的记录", key="ai_custom_query")
    if custom_query and st.button("📤 发送查询", key="send_custom_query", use_container_width=True):
        st.session_state.ai_pending_question = custom_query
        st.rerun()


# ==================== 7. SQL生成（动态生成） ====================

def _render_sql_generator(json_data: Dict[str, Any]):
    """根据实际字段生成SQL示例"""
    variable_types = json_data.get("variable_types", {})
    data_shape = json_data.get("data_shape", {})
    table_name = json_data.get("source_table", "your_table")

    # 提取字段名
    all_cols = list(variable_types.keys())
    date_cols = [col for col, info in variable_types.items() if info.get("type") == "datetime"]
    numeric_cols = [col for col, info in variable_types.items() if info.get("type") == "continuous"]
    categorical_cols = [col for col, info in variable_types.items() if info.get("type") in ["categorical", "categorical_numeric", "ordinal"]]
    identifier_cols = [col for col, info in variable_types.items() if info.get("type") == "identifier"]

    # 选择一个示例字段
    date_field = date_cols[0] if date_cols else None
    num_field = numeric_cols[0] if numeric_cols else None
    cat_field = categorical_cols[0] if categorical_cols else None
    id_field = identifier_cols[0] if identifier_cols else None

    examples = []

    # 1. 基本查询
    cols_str = "、".join(all_cols[:3]) + ("..." if len(all_cols) > 3 else "")
    examples.append(f"查询所有数据（字段：{cols_str}）")

    # 2. 条件查询
    if date_field:
        examples.append(f"查询{date_field}在最近7天的数据")
    elif num_field:
        examples.append(f"查询{num_field}大于平均值的记录")
    elif cat_field:
        examples.append(f"查询{cat_field}为'某值'的记录")

    # 3. 聚合统计
    if cat_field and num_field:
        examples.append(f"按{cat_field}分组统计{num_field}的总和和平均值")
    elif num_field:
        examples.append(f"统计{num_field}的最大值、最小值和平均值")

    # 4. 排序
    if num_field:
        examples.append(f"按{num_field}降序排序，取前10条")

    # 5. 时间聚合
    if date_field and num_field:
        examples.append(f"按{date_field}的月份分组统计{num_field}")

    # 6. 关联查询（如果有多个表，但单表模式只提示单表）
    examples.append(f"按{cat_field if cat_field else '分类'}分组统计记录数")

    # 兜底
    if not examples:
        examples = ["查询所有数据", "统计总数", "按条件筛选"]

    st.markdown("**💡 示例SQL需求（点击使用）：**")
    cols = st.columns(2)
    for i, ex in enumerate(examples[:6]):
        with cols[i % 2]:
            if st.button(f"📝 {ex}", key=f"sql_ex_{i}", use_container_width=True):
                st.session_state.ai_pending_question = f"请生成SQL语句：{ex}，表名是{table_name}"
                st.rerun()

    st.markdown("---")
    st.markdown("**✏️ 自定义SQL需求：**")
    custom_sql = st.text_input("描述您需要的SQL", placeholder="例如：查询上个月的订单数据", key="ai_custom_sql")
    if custom_sql and st.button("📤 生成SQL", key="send_custom_sql", use_container_width=True):
        st.session_state.ai_pending_question = f"请生成SQL语句：{custom_sql}，表名是{table_name}"
        st.rerun()


# ==================== 8. 推理预测 ====================

def _has_trained_models(session_id: str) -> bool:
    from web.services.model_training_service import list_saved_models
    models = list_saved_models(session_id)
    return len(models) > 0


def _render_inference_prediction(session_id: str):
    from web.services.model_training_service import list_saved_models

    models = list_saved_models(session_id)
    if not models:
        st.caption("暂无已训练的模型")
        return

    st.markdown("**📋 已训练模型（点击使用）：**")

    for model in models:
        model_key = model.get("model_key", "")
        model_name = model.get("user_model_name", model_key)
        task_type = model.get("task_type", "unknown")
        target = model.get("target_column", "未知")
        features = model.get("features", [])

        feature_str = "、".join(features[:3])
        if len(features) > 3:
            feature_str += f"等{len(features)}个"

        label = f"🔮 {model_name} ({task_type})"
        help_text = f"目标: {target} | 特征: {feature_str}"

        if st.button(label, key=f"model_{model_key}", use_container_width=True, help=help_text):
            question = f"请使用模型「{model_name}」进行预测，目标列是「{target}」，特征包括：{feature_str}"
            st.session_state.ai_pending_question = question
            st.rerun()


# ==================== 9. 勾稽校验 ====================

def _render_audit_rule():
    options = [
        {"label": "📊 解读已有勾稽规则", "question": "请解读当前数据中的勾稽规则，说明每条规则的含义和违反情况"},
        {"label": "📝 校验表结构", "question": "请校验当前数据的表结构，识别潜在的数据一致性问题和业务逻辑隐患"},
        {"label": "🔍 发现潜在勾稽关系", "question": "请分析数据中可能存在的勾稽关系，推荐可用的数据一致性规则"},
        {"label": "📋 生成勾稽规则报告", "question": "请生成一份勾稽规则报告，汇总所有发现的数据一致性规则和问题"},
    ]

    cols = st.columns(2)
    for i, opt in enumerate(options):
        with cols[i % 2]:
            if st.button(opt["label"], key=f"audit_{i}", use_container_width=True):
                st.session_state.ai_pending_question = opt["question"]
                st.rerun()


# ==================== 右侧 - 固定高度滚动对话框 ====================

def render_chat_interface(json_data: Optional[Dict[str, Any]], session_id: str):
    st.markdown("#### 💬 对话")

    if json_data is None:
        with st.container(height=650):
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;text-align:center;color:#888;">
                <div style="font-size:48px;margin-bottom:16px;">🤖</div>
                <div style="font-size:18px;font-weight:500;color:#555;">AI助手</div>
                <div style="font-size:14px;color:#999;margin-bottom:20px;">请先完成数据分析，生成JSON报告</div>
            </div>
            """, unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 去数据准备", key="goto_data_prep_chat", use_container_width=True):
                    st.session_state.current_tab = 0
                    st.rerun()
            with col2:
                if st.button("🔄 重新加载", key="reload_json_chat", use_container_width=True):
                    if session_id:
                        data = StorageService.load_json("analysis_result", session_id)
                        if data is not None:
                            st.session_state.current_json_data = data
                            st.rerun()
        return

    chat_container = st.container(height=650)

    with chat_container:
        if not st.session_state.ai_messages:
            st.caption("💡 从左侧选择问题开始对话")

        for msg in st.session_state.ai_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if st.session_state.get("ai_pending_question"):
        question = st.session_state.ai_pending_question
        st.session_state.ai_pending_question = ""

        st.session_state.ai_messages.append({"role": "user", "content": question})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(question)

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

        st.session_state.ai_messages.append({"role": "assistant", "content": full_response})

    prompt = st.chat_input("输入您的问题...", key="ai_chat_input")
    if prompt and prompt.strip():
        st.session_state.ai_pending_question = prompt.strip()
        st.rerun()


# ==================== AI系统提示词 ====================

def _build_ai_system_prompt(json_data: Dict[str, Any], session_id: str) -> str:
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