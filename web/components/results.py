# web/components/results.py

"""分析结果展示组件 - 预览报告、AI解读、项目对比"""

import streamlit as st
import json
from web.components.chat_interface import render_chat_interface, render_recommended_questions
from web.components.scenario_recommendation import render_scenario_recommendation
from web.components.natural_query import render_natural_query
from web.components.sql_generator import render_sql_generator
from web.components.agent_inference import render_agent_inference
from web.services.session_service import SessionService
from web.services.storage_service import StorageService

from web.services.insight_service import InsightService
from web.components.term_tooltip import apply_term_tooltips_to_html
from web.services.feature_flags import FeatureFlags
import zipfile
import io
from datetime import datetime


def render_export_buttons(html_content: str, json_data: dict, log_content: str):
    """渲染一键导出按钮组"""
    st.markdown("#### 📥 导出报告")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.download_button(
            "📄 HTML",
            html_content,
            f"autostat_report_{timestamp}.html",
            "text/html",
            key=f"download_html_{timestamp}",
            use_container_width=True
        )

    with col2:
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2) if json_data else ""
        st.download_button(
            "📋 JSON",
            json_str,
            f"autostat_result_{timestamp}.json",
            "application/json",
            key=f"download_json_{timestamp}",
            use_container_width=True
        )

    with col3:
        st.download_button(
            "📝 日志",
            log_content or "无日志",
            f"autostat_log_{timestamp}.txt",
            "text/plain",
            key=f"download_log_{timestamp}",
            use_container_width=True
        )

    with col4:
        if st.button("📦 一键导出全部", key=f"export_all_{timestamp}", use_container_width=True):
            _export_all_formats(html_content, json_str, log_content)


def _export_all_formats(html_content: str, json_str: str, log_content: str):
    """导出所有格式为ZIP包"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(f"autostat_report_{timestamp}.html", html_content)
        zip_file.writestr(f"autostat_result_{timestamp}.json", json_str)
        zip_file.writestr(f"autostat_log_{timestamp}.txt", log_content or "无日志")

    zip_buffer.seek(0)

    st.download_button(
        "📦 下载ZIP包",
        zip_buffer,
        f"autostat_export_{timestamp}.zip",
        "application/zip",
        key=f"download_zip_{timestamp}",
        use_container_width=True
    )


def render_conclusions_section(conclusions: list, auto_interpretation: str = None):
    """渲染核心结论区域"""
    st.markdown("""
    <style>
    .conclusions-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        color: white;
    }
    .conclusions-summary {
        background: rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 16px;
    }
    .conclusion-item {
        font-size: 13px;
        line-height: 1.6;
        margin: 4px 0;
    }
    .interpretation-card {
        background: rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 16px;
        border-left: 3px solid #ffd700;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="conclusions-section">', unsafe_allow_html=True)
    st.markdown("## 🎯 核心结论（30秒速览）")

    if conclusions:
        summary_html = '<div class="conclusions-summary">'
        for c in conclusions:
            summary_html += f'<div class="conclusion-item">{c.get("icon", "📌")} {c.get("title", "")}：{c.get("description", "")}</div>'
        summary_html += '</div>'
        st.markdown(summary_html, unsafe_allow_html=True)

    if auto_interpretation:
        st.markdown(f"""
        <div class="interpretation-card">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <span style="font-size: 18px;">🧠</span>
                <span style="font-weight: bold; font-size: 14px;">AI 综合解读</span>
            </div>
            <p style="margin: 0; font-size: 13px; opacity: 0.9; line-height: 1.5;">{auto_interpretation}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_preview_tab():
    """渲染预览报告标签页"""
    session_id = SessionService.get_current_session()
    if session_id is None:
        st.info("暂无报告预览")
        return

    html_content = StorageService.load_text("analysis_report", session_id)
    json_data = StorageService.load_json("analysis_result", session_id)
    log_content = StorageService.load_text("analysis_log", session_id)

    if html_content and json_data:
        if FeatureFlags.is_enabled("one_click_export"):
            render_export_buttons(html_content, json_data, log_content)
            st.divider()
        else:
            col1, col2, col3 = st.columns(3)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            with col1:
                st.download_button(
                    "📥 下载 HTML 报告",
                    html_content,
                    "autostat_report.html",
                    "text/html",
                    key=f"download_html_simple_{timestamp}",
                    use_container_width=True
                )
            with col2:
                json_str = json.dumps(json_data, ensure_ascii=False, indent=2) if json_data else ""
                st.download_button(
                    "📥 下载 JSON 结果",
                    json_str,
                    "autostat_result.json",
                    "application/json",
                    key=f"download_json_simple_{timestamp}",
                    use_container_width=True
                )
            with col3:
                st.download_button(
                    "📥 下载分析日志",
                    log_content or "暂无分析日志",
                    "autostat_log.txt",
                    "text/plain",
                    key=f"download_log_simple_{timestamp}",
                    use_container_width=True
                )
            st.divider()

        if FeatureFlags.is_enabled("conclusion_first") and json_data:
            conclusions = InsightService.extract_top_conclusions(json_data)
            auto_interpretation = st.session_state.get("auto_interpretation")
            render_conclusions_section(conclusions, auto_interpretation)
            st.markdown("---")
            st.markdown("### 📊 详细报告")

        if FeatureFlags.is_enabled("term_tooltip"):
            html_content = apply_term_tooltips_to_html(html_content)

        st.html(html_content)
    else:
        st.info("暂无报告预览")


def render_ai_tab():
    """渲染AI解读标签页"""
    is_database = False
    session_id = SessionService.get_current_session()
    if session_id:
        metadata = SessionService.load_metadata(session_id)
        is_database = metadata.get("analysis_type") == "database"

    render_context_selector()
    st.divider()
    render_chat_interface()
    st.divider()

    if is_database:
        sub_tab_labels = ["💬 智能解读", "🎯 场景推荐", "🔍 自然查询", "📝 SQL生成", "🔮 推理预测"]
    else:
        sub_tab_labels = ["💬 智能解读", "🎯 场景推荐", "🔍 自然查询", "🔮 推理预测"]

    sub_tabs = st.tabs(sub_tab_labels)

    with sub_tabs[0]:
        if st.session_state.llm_client is None:
            st.warning("请先在侧边栏配置大模型")
        else:
            st.session_state.chat_mode = "analysis"
            render_recommended_questions()
    with sub_tabs[1]:
        if st.session_state.llm_client is None:
            st.warning("请先在侧边栏配置大模型")
        else:
            st.session_state.chat_mode = "scenario"
            render_scenario_recommendation()
    with sub_tabs[2]:
        if st.session_state.llm_client is None:
            st.warning("请先在侧边栏配置大模型")
        else:
            st.session_state.chat_mode = "natural_query"
            render_natural_query()
    if is_database:
        with sub_tabs[3]:
            if st.session_state.llm_client is None:
                st.warning("请先在侧边栏配置大模型")
            else:
                st.session_state.chat_mode = "sql"
                render_sql_generator()
        with sub_tabs[4]:
            if st.session_state.llm_client is None:
                st.warning("请先在侧边栏配置大模型")
            else:
                st.session_state.chat_mode = "prediction"
                render_agent_inference()
    else:
        with sub_tabs[3]:
            if st.session_state.llm_client is None:
                st.warning("请先在侧边栏配置大模型")
            else:
                st.session_state.chat_mode = "prediction"
                render_agent_inference()


def render_context_selector():
    """渲染上下文选择器"""
    st.markdown("#### 📚 选择分析上下文")
    st.caption("勾选要提供给AI的上下文信息，多选可获得更全面的分析")

    col_ctx1, col_ctx2, col_ctx3 = st.columns(3)

    with col_ctx1:
        ctx_json = st.checkbox(
            "📊 JSON 结果",
            value="json_result" in st.session_state.selected_contexts,
            key="ctx_json",
            help="包含完整的统计分析数据（变量类型、相关性、质量报告等）"
        )
    with col_ctx2:
        ctx_html = st.checkbox(
            "📄 HTML 报告",
            value="html_report" in st.session_state.selected_contexts,
            key="ctx_html",
            help="包含可视化的分析报告（图表、统计摘要等）"
        )
    with col_ctx3:
        ctx_data = st.checkbox(
            "🗃️ 源数据",
            value="raw_data" in st.session_state.selected_contexts,
            key="ctx_data",
            help="包含原始数据预览（前50行）及完整统计摘要"
        )

    selected = []
    if ctx_json:
        selected.append("json_result")
    if ctx_html:
        selected.append("html_report")
    if ctx_data:
        selected.append("raw_data")

    if set(selected) != set(st.session_state.selected_contexts):
        st.session_state.selected_contexts = selected

    if not st.session_state.selected_contexts:
        st.warning("⚠️ 请至少选择一种上下文信息")

    with st.expander(f"📋 已选上下文 ({len(st.session_state.selected_contexts)}项)", expanded=False):
        ctx_names = {
            "json_result": "📊 JSON 结果 - 完整的统计分析数据",
            "html_report": "📄 HTML 报告 - 可视化分析报告",
            "raw_data": "🗃️ 源数据 - 原始数据预览（前50行）及统计摘要"
        }
        for ctx in st.session_state.selected_contexts:
            st.markdown(f"- {ctx_names.get(ctx, ctx)}")


# ==================== 项目对比功能 ====================

def render_compare_tab():
    """渲染项目对比标签页"""
    st.markdown("### 🔍 项目对比")
    st.caption("选择两个分析项目进行并排对比")

    projects = SessionService.list_user_projects()
    if len(projects) < 2:
        st.info("需要至少两个项目才能进行对比，请先创建更多分析项目。")
        return

    project_options = {p["session_id"]: f"{p['source_name']} ({p.get('created_at', '')[:16]})"
                       for p in projects}

    col_left, col_right = st.columns(2)

    with col_left:
        selected_left = st.selectbox(
            "选择项目 A",
            options=list(project_options.keys()),
            format_func=lambda x: project_options.get(x, x),
            key="compare_left"
        )
    with col_right:
        selected_right = st.selectbox(
            "选择项目 B",
            options=list(project_options.keys()),
            format_func=lambda x: project_options.get(x, x),
            key="compare_right"
        )

    if st.button("📊 开始对比", type="primary", use_container_width=True):
        if selected_left == selected_right:
            st.warning("请选择两个不同的项目")
            return
        # 清除旧的缓存，重新生成
        if "compare_result" in st.session_state:
            del st.session_state.compare_result
        _render_compare_result(selected_left, selected_right)

    # 检查是否有缓存的对比结果
    if st.session_state.get("compare_result"):
        data = st.session_state.compare_result
        _render_compare_result_display(
            data["source_name_a"], data["source_name_b"],
            data["data_a"], data["data_b"],
            data["diff_summary"],
            data["export_html"]
        )


def _render_compare_result(session_id_a: str, session_id_b: str):
    """渲染对比结果（保存到session_state）"""
    json_data_a = StorageService.load_json("analysis_result", session_id_a)
    json_data_b = StorageService.load_json("analysis_result", session_id_b)

    if json_data_a is None or json_data_b is None:
        st.error("无法加载分析结果，请确保两个项目都已完成分析")
        return

    metadata_a = SessionService.load_metadata(session_id_a)
    metadata_b = SessionService.load_metadata(session_id_b)
    source_name_a = metadata_a.get("source_name", "项目 A")
    source_name_b = metadata_b.get("source_name", "项目 B")

    def extract_compare_data(json_data):
        shape = json_data.get("data_shape", {})
        quality = json_data.get("quality_report", {})
        missing_list = quality.get("missing", [])
        outliers = quality.get("outliers", {})
        duplicates = quality.get("duplicates", {})
        high_corrs = json_data.get("correlations", {}).get("high_correlations", [])
        ts_diag = json_data.get("time_series_diagnostics", {})
        ts_count = sum(1 for v in ts_diag.values() if v.get("has_autocorrelation"))

        return {
            "rows": int(shape.get("rows", 0)),
            "cols": int(shape.get("columns", 0)),
            "missing_count": len(missing_list),
            "outlier_count": len(outliers),
            "duplicate_count": int(duplicates.get("count", 0)),
            "high_corr_count": len(high_corrs),
            "ts_count": ts_count
        }

    data_a = extract_compare_data(json_data_a)
    data_b = extract_compare_data(json_data_b)

    diff_summary = []
    if abs(data_a['rows'] - data_b['rows']) / max(data_a['rows'], 1) > 0.1:
        diff_summary.append(f"数据量变化：{data_a['rows']:,} → {data_b['rows']:,}")
    if data_a['missing_count'] != data_b['missing_count']:
        diff_summary.append(f"缺失字段：{data_a['missing_count']} → {data_b['missing_count']}")
    if data_a['outlier_count'] != data_b['outlier_count']:
        diff_summary.append(f"异常字段：{data_a['outlier_count']} → {data_b['outlier_count']}")
    if data_a['high_corr_count'] != data_b['high_corr_count']:
        diff_summary.append(f"强相关对：{data_a['high_corr_count']} → {data_b['high_corr_count']}")

    export_html = _generate_compare_html(
        source_name_a, source_name_b,
        data_a, data_b,
        diff_summary
    )

    st.session_state.compare_result = {
        "source_name_a": source_name_a,
        "source_name_b": source_name_b,
        "data_a": data_a,
        "data_b": data_b,
        "diff_summary": diff_summary,
        "export_html": export_html
    }


def _render_compare_result_display(source_name_a: str, source_name_b: str, data_a: dict, data_b: dict,
                                   diff_summary: list, export_html: str):
    """渲染对比结果显示"""

    def diff_pct(a, b):
        try:
            a = float(a) if a else 0
            b = float(b) if b else 0
        except (ValueError, TypeError):
            return ""

        if a == 0:
            if b == 0:
                return ""
            return f" +{b - a:.0f}"
        pct = (b - a) / a * 100
        if abs(pct) < 0.1:
            return ""
        direction = "↑" if pct > 0 else "↓"
        return f" {direction}{abs(pct):.0f}%"

    st.markdown(f"### 对比报告：{source_name_a} ↔ {source_name_b}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{source_name_a}**")
        st.metric("总行数", f"{data_a['rows']:,}")
        st.metric("总列数", data_a['cols'])
        st.metric("缺失字段数", data_a['missing_count'])
        st.metric("异常值字段数", data_a['outlier_count'])
        st.metric("重复记录数", data_a['duplicate_count'])
        st.metric("强相关对数量", data_a['high_corr_count'])
        st.metric("自相关序列数", data_a['ts_count'])

    with col2:
        st.markdown(f"**{source_name_b}**")
        st.metric("总行数", f"{data_b['rows']:,}", delta=diff_pct(data_a['rows'], data_b['rows']))
        st.metric("总列数", data_b['cols'], delta=diff_pct(data_a['cols'], data_b['cols']))
        st.metric("缺失字段数", data_b['missing_count'],
                  delta=diff_pct(data_a['missing_count'], data_b['missing_count']))
        st.metric("异常值字段数", data_b['outlier_count'],
                  delta=diff_pct(data_a['outlier_count'], data_b['outlier_count']))
        st.metric("重复记录数", data_b['duplicate_count'],
                  delta=diff_pct(data_a['duplicate_count'], data_b['duplicate_count']))
        st.metric("强相关对数量", data_b['high_corr_count'],
                  delta=diff_pct(data_a['high_corr_count'], data_b['high_corr_count']))
        st.metric("自相关序列数", data_b['ts_count'], delta=diff_pct(data_a['ts_count'], data_b['ts_count']))

    st.markdown("---")
    st.markdown("**📊 主要差异**")
    if diff_summary:
        for item in diff_summary:
            st.caption(f"• {item}")
    else:
        st.caption("无明显差异")

    st.markdown("---")
    unique_key = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    st.download_button(
        "📥 导出对比报告",
        export_html,
        f"autostat_compare_{unique_key}.html",
        "text/html",
        key=f"download_compare_{unique_key}",
        use_container_width=True
    )


def _generate_compare_html(name_a: str, name_b: str, data_a: dict, data_b: dict, diff_summary: list) -> str:
    """生成对比报告 HTML"""
    report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AutoStat 对比报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #333; border-bottom: 3px solid #667eea; }}
        .compare-row {{ display: flex; gap: 20px; margin: 20px 0; }}
        .compare-col {{ flex: 1; background: #f8f9fa; padding: 15px; border-radius: 8px; }}
        .metric {{ font-size: 24px; font-weight: bold; color: #333; }}
        .metric-label {{ font-size: 12px; color: #666; }}
        hr {{ margin: 20px 0; }}
        .footer {{ text-align: center; padding: 20px; color: #999; font-size: 12px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>📊 AutoStat 对比报告</h1>
    <p>生成时间: {report_time}</p>
    <hr>
    <div class="compare-row">
        <div class="compare-col">
            <h3>{name_a}</h3>
            <div class="metric">{data_a['rows']:,}</div>
            <div class="metric-label">总行数</div>
            <div class="metric">{data_a['cols']}</div>
            <div class="metric-label">总列数</div>
            <div class="metric">{data_a['missing_count']}</div>
            <div class="metric-label">缺失字段数</div>
            <div class="metric">{data_a['outlier_count']}</div>
            <div class="metric-label">异常值字段数</div>
            <div class="metric">{data_a['duplicate_count']}</div>
            <div class="metric-label">重复记录数</div>
            <div class="metric">{data_a['high_corr_count']}</div>
            <div class="metric-label">强相关对数量</div>
        </div>
        <div class="compare-col">
            <h3>{name_b}</h3>
            <div class="metric">{data_b['rows']:,}</div>
            <div class="metric-label">总行数</div>
            <div class="metric">{data_b['cols']}</div>
            <div class="metric-label">总列数</div>
            <div class="metric">{data_b['missing_count']}</div>
            <div class="metric-label">缺失字段数</div>
            <div class="metric">{data_b['outlier_count']}</div>
            <div class="metric-label">异常值字段数</div>
            <div class="metric">{data_b['duplicate_count']}</div>
            <div class="metric-label">重复记录数</div>
            <div class="metric">{data_b['high_corr_count']}</div>
            <div class="metric-label">强相关对数量</div>
        </div>
    </div>
    <hr>
    <h3>📊 主要差异</h3>
    <ul>
        {''.join([f'<li>{item}</li>' for item in diff_summary]) if diff_summary else '<li>无明显差异</li>'}
    </ul>
    <div class="footer">
        <p>🤖 AutoStat 智能统计分析工具 | 对比报告自动生成</p>
    </div>
</div>
</body>
</html>"""