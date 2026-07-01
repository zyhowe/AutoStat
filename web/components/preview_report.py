"""
预览报告组件 - 可折叠设计
"""

import streamlit as st
import json
import re
from datetime import datetime

from web.services.session_service import SessionService
from web.services.storage_service import StorageService
from web.services.insight_service import InsightService
from web.components.quality_dashboard import render_quality_dashboard_inline


def render_preview_report():
    """渲染预览报告 - 可折叠设计"""
    st.markdown("### 📄 预览报告")
    st.caption("点击标题展开查看详情，支持全部展开/折叠")

    session_id = SessionService.get_current_session()
    if session_id is None:
        st.info("暂无报告预览，请先完成数据分析")
        return

    html_content = StorageService.load_text("analysis_report", session_id)
    json_data = StorageService.load_json("analysis_result", session_id)
    log_content = StorageService.load_text("analysis_log", session_id)

    if html_content is None or json_data is None:
        st.info("暂无报告预览，请先完成数据分析")
        return

    # ==================== 导出按钮 + 全部展开/折叠 ====================
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 0.8, 0.8, 0.8])

    with col1:
        st.download_button(
            "📥 导出HTML",
            html_content,
            f"autostat_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            "text/html",
            use_container_width=True
        )

    with col2:
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
        st.download_button(
            "📋 导出JSON",
            json_str,
            f"autostat_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json",
            use_container_width=True
        )

    with col3:
        st.download_button(
            "📦 导出日志",
            log_content or "暂无分析日志",
            f"autostat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain",
            use_container_width=True
        )

    with col4:
        if st.button("📂 全部展开", use_container_width=True):
            st.session_state.preview_expand_all = True
            st.rerun()

    with col5:
        if st.button("📁 全部折叠", use_container_width=True):
            st.session_state.preview_expand_all = False
            st.rerun()

    with col6:
        if st.button("🔄 刷新", use_container_width=True):
            st.rerun()

    st.divider()

    # ==================== 核心结论 - 横向卡片 ====================
    _render_core_conclusions(json_data)

    st.divider()

    # ==================== 折叠区块 ====================
    expand_all = st.session_state.get("preview_expand_all", False)

    with st.expander("📊 数据概览", expanded=expand_all):
        _render_overview_section(json_data)

    with st.expander("📊 数据质量看板", expanded=expand_all):
        render_quality_dashboard_inline()

    with st.expander("📋 变量详情", expanded=expand_all):
        _render_variables_section(json_data)

    with st.expander("🔗 相关性分析", expanded=expand_all):
        _render_correlation_section(json_data)

    with st.expander("📈 时间序列分析", expanded=expand_all):
        _render_timeseries_section(json_data)

    with st.expander("🔗 勾稽规则", expanded=expand_all):
        _render_audit_rules_section(json_data)

    with st.expander("🤖 模型推荐", expanded=expand_all):
        _render_model_recommendations_section(json_data)

    with st.expander("🧹 清洗建议", expanded=expand_all):
        _render_cleaning_suggestions_section(json_data)


# ==================== 核心结论渲染 ====================

def _render_core_conclusions(json_data: dict):
    """核心结论 - 横向卡片，统一高度，每张显示4行"""
    conclusions = InsightService.extract_top_conclusions(json_data)

    if not conclusions:
        return

    top_conclusions = conclusions[:4]

    st.markdown("#### 🎯 核心结论")

    cols = st.columns(len(top_conclusions))

    for i, c in enumerate(top_conclusions):
        with cols[i]:
            icon = c.get("icon", "📌")
            title = c.get("title", "")
            raw_desc = c.get("description", "")

            # 第4个卡片（索引3）特殊处理：字段列表在title里
            if i == 3:
                full_text = title + " " + raw_desc
                desc = _truncate_predict_field_list(full_text)
                title="建立可预测模型"
            else:
                desc = _truncate_field_list(raw_desc)

            # 按换行符分割，取前4行
            lines = desc.split('\n') if desc else []
            if len(lines) > 4:
                lines = lines[:4]
                if lines and not lines[-1].endswith('...'):
                    lines[-1] = lines[-1] + '...'
                desc_display = '\n'.join(lines)
            else:
                desc_display = desc

            if not desc_display.strip():
                desc_display = "—"

            st.markdown(f"""
            <div style="
                background: #f0f2f6;
                border-radius: 10px;
                padding: 12px 16px;
                text-align: center;
                height: 100%;
                min-height: 120px;
                border-left: 3px solid #1f77b4;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
            ">
                <div style="font-size: 20px; margin-bottom: 4px;">{icon}</div>
                <div style="font-weight: bold; font-size: 13px; color: #333; margin-bottom: 6px;">{title}</div>
                <div style="font-size: 11px; color: #666; line-height: 1.4; flex: 1;">{desc_display}</div>
            </div>
            """, unsafe_allow_html=True)


# ==================== 截断函数 ====================

def _truncate_field_list(text: str, max_display: int = 3) -> str:
    """
    通用截断函数 - 用于卡片0和卡片1
    输入: "A、B、C、D、E 后缀文字"
    输出: "A、B、C等5个 后缀文字"
    """
    if not text:
        return text

    # 匹配字段列表（顿号或逗号分隔）
    pattern = r'^((?:[^\s、，,;；]+(?:[、，,;；]\s*[^\s、，,;；]+)*))'
    match = re.match(pattern, text)

    if not match:
        return text

    full_list = match.group(1)
    suffix = text[len(full_list):]

    parts = re.split(r'[、，,;；]\s*', full_list)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) <= max_display:
        return text

    # 检测单位
    unit = "个"
    if "对" in suffix[:30]:
        unit = "对"

    display_parts = parts[:max_display]
    display_str = '、'.join(display_parts)

    return f"{display_str}等{len(parts)}{unit}{suffix}"


def _truncate_predict_field_list(text: str, max_display: int = 3) -> str:
    """
    专门处理第4个卡片 "可预测"
    输入: "可预测 A、B、C、D、E... 基于关联特征可建立预测模型"
    输出: "可预测 A、B、C等42个 基于关联特征可建立预测模型"
    """
    if not text:
        return text

    prefix = "可预测"
    suffix = "基于关联特征可建立预测模型"

    # 检查是否以 "可预测" 开头
    if not text.startswith(prefix):
        return text

    # 去掉前缀
    remaining = text[len(prefix):].strip()

    # 检查是否包含后缀
    if suffix not in remaining:
        return text

    # 分割字段和后缀
    idx = remaining.index(suffix)
    field_part = remaining[:idx].strip()
    suffix_part = remaining[idx:]

    # 提取字段（顿号分隔）
    parts = re.split(r'[、，,]\s*', field_part)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) <= max_display:
        return text

    display_parts = parts[:max_display]
    display_str = '、'.join(display_parts)

    return f"{prefix} {display_str}等{len(parts)}个 {suffix_part}"


# ==================== 折叠区块函数 ====================

def _render_overview_section(json_data: dict):
    """数据概览"""
    data_shape = json_data.get("data_shape", {})
    quality = json_data.get("quality_report", {})
    variable_types = json_data.get("variable_types", {})

    rows = data_shape.get("rows", 0)
    cols = data_shape.get("columns", 0)

    missing_count = len(quality.get("missing", []))
    dup_count = quality.get("duplicates", {}).get("count", 0)
    try:
        dup_count = int(dup_count) if dup_count else 0
    except (ValueError, TypeError):
        dup_count = 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总行数", f"{rows:,}")
    col2.metric("总列数", cols)
    col3.metric("缺失字段数", missing_count)
    col4.metric("重复记录数", dup_count)

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

    if type_counts:
        st.markdown("**变量类型分布：**")
        cols = st.columns(min(len(type_counts), 6))
        for i, (typ, count) in enumerate(type_counts.items()):
            with cols[i % len(cols)]:
                st.metric(type_display.get(typ, typ), count)


def _render_variables_section(json_data: dict):
    """变量详情"""
    variable_summaries = json_data.get("variable_summaries", {})

    if not variable_summaries:
        st.caption("暂无变量详情数据")
        return

    def safe_float(val, default=0):
        try:
            return float(val) if val is not None else default
        except (ValueError, TypeError):
            return default

    def safe_int(val, default=0):
        try:
            return int(val) if val is not None else default
        except (ValueError, TypeError):
            return default

    var_data = []
    for col, info in list(variable_summaries.items())[:20]:
        typ = info.get("type", "")

        if typ == "continuous":
            mean_val = safe_float(info.get("mean"))
            std_val = safe_float(info.get("std"))
            min_val = safe_float(info.get("min"))
            max_val = safe_float(info.get("max"))
            center = f"{mean_val:.2f} ± {std_val:.2f}" if std_val != 0 else f"{mean_val:.2f}"
            spread = f"[{min_val:.2f}, {max_val:.2f}]"

        elif typ in ["categorical", "categorical_numeric", "ordinal"]:
            mode_val = info.get("mode", "-")
            center = str(mode_val) if mode_val and mode_val != "-" else "-"
            n_unique = safe_int(info.get("n_unique"))
            spread = f"{n_unique}个类别"

        else:
            center = "-"
            spread = "-"

        count = safe_int(info.get("count"))
        missing = safe_int(info.get("missing"))
        missing_pct = safe_float(info.get("missing_pct"))

        var_data.append({
            "变量名": col,
            "类型": info.get("type_desc", typ),
            "样本量": count,
            "缺失数": missing,
            "缺失率": f"{missing_pct:.1f}%",
            "中心趋势": center,
            "分布": spread
        })

    st.dataframe(var_data, use_container_width=True)

    if len(variable_summaries) > 20:
        st.caption(f"... 还有 {len(variable_summaries) - 20} 个变量未显示")


def _render_correlation_section(json_data: dict):
    """相关性分析"""
    correlations = json_data.get("correlations", {})
    high_corrs = correlations.get("high_correlations", [])

    if high_corrs:
        st.markdown(f"**发现 {len(high_corrs)} 对强相关关系（|r| > 0.7）：**")
        for corr in high_corrs[:10]:
            direction = "正" if corr.get("value", 0) > 0 else "负"
            st.caption(f"• {corr.get('var1', '')} ↔ {corr.get('var2', '')}：{direction}相关 r={corr.get('value', 0):.3f}")
        if len(high_corrs) > 10:
            st.caption(f"... 还有 {len(high_corrs) - 10} 对")
    else:
        st.caption("未发现强相关关系（|r| > 0.7）")


def _render_timeseries_section(json_data: dict):
    """时间序列分析"""
    ts_diag = json_data.get("time_series_diagnostics", {})

    if not ts_diag:
        st.caption("未检测到时间序列数据")
        return

    ts_data = []
    for key, diag in ts_diag.items():
        n_samples = diag.get("n_samples", 0)
        try:
            n_samples = int(n_samples) if n_samples else 0
        except (ValueError, TypeError):
            n_samples = 0

        ts_data.append({
            "变量/分组": key,
            "样本量": n_samples,
            "平稳性": "✅ 平稳" if diag.get("is_stationary") else "⚠️ 非平稳",
            "自相关性": "✅ 有" if diag.get("has_autocorrelation") else "❌ 无",
            "季节性": "✅ 有" if diag.get("has_seasonality") else "❌ 无"
        })

    st.dataframe(ts_data, use_container_width=True)

    has_auto = any(d.get("has_autocorrelation") for d in ts_diag.values())
    if has_auto:
        st.success("✅ 检测到自相关性，适合进行时间序列预测")


def _render_audit_rules_section(json_data: dict):
    """勾稽规则"""
    audit_rules = json_data.get("quality_report", {}).get("audit_rules", {})

    if not audit_rules:
        st.caption("未发现勾稽规则")
        return

    arithmetic_count = len(audit_rules.get("arithmetic_rules", []))
    fd_count = len(audit_rules.get("functional_dependencies", []))
    temporal_count = len(audit_rules.get("temporal_rules", []))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📐 数值关系", arithmetic_count)
    with col2:
        st.metric("🏷️ 函数依赖", fd_count)
    with col3:
        st.metric("📅 时序约束", temporal_count)

    arithmetic = audit_rules.get("arithmetic_rules", [])
    if arithmetic:
        st.markdown(f"**数值关系（{len(arithmetic)}条）：**")
        for rule in arithmetic[:5]:
            conf = rule.get("confidence", 0)
            try:
                conf = float(conf) if conf else 0
            except (ValueError, TypeError):
                conf = 0
            st.caption(f"• {rule.get('rule', '')}（置信度: {conf:.1%}）")
        if len(arithmetic) > 5:
            st.caption(f"... 还有 {len(arithmetic) - 5} 条")


def _render_model_recommendations_section(json_data: dict):
    """模型推荐 - 仅展示摘要"""
    recommendations = json_data.get("model_recommendations", [])

    if not recommendations:
        st.caption("暂无模型推荐")
        return

    high = [r for r in recommendations if r.get("priority") == "高"]
    medium = [r for r in recommendations if r.get("priority") == "中"]
    low = [r for r in recommendations if r.get("priority") == "低"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 高优先级", len(high))
    with col2:
        st.metric("🟠 中优先级", len(medium))
    with col3:
        st.metric("🟢 低优先级", len(low))

    for rec in recommendations[:3]:
        priority_icon = "🔴" if rec.get("priority") == "高" else "🟠" if rec.get("priority") == "中" else "🟢"
        st.markdown(f"{priority_icon} **{rec.get('task_type', '')}** → {rec.get('ml', '')}")
        if rec.get("target_column"):
            st.caption(f"  目标: {rec.get('target_column', '')}")
        st.divider()

    if len(recommendations) > 3:
        st.caption(f"💡 共 {len(recommendations)} 条推荐，完整列表请查看「模型中心」")


def _render_cleaning_suggestions_section(json_data: dict):
    """清洗建议"""
    suggestions = json_data.get("cleaning_suggestions", [])

    if not suggestions:
        st.success("✅ 数据质量良好，无明显清洗需求")
        return

    for s in suggestions[:10]:
        st.caption(f"• {s}")
    if len(suggestions) > 10:
        st.caption(f"... 还有 {len(suggestions) - 10} 条建议")