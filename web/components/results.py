# web/components/results.py

"""分析结果展示组件 - 对比功能"""

import streamlit as st
import json
from datetime import datetime

from web.services.session_service import SessionService
from web.services.storage_service import StorageService


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
        if "compare_result" in st.session_state:
            del st.session_state.compare_result
        _render_compare_result(selected_left, selected_right)

    if st.session_state.get("compare_result"):
        data = st.session_state.compare_result
        _render_compare_result_display(
            data["source_name_a"], data["source_name_b"],
            data["data_a"], data["data_b"],
            data["diff_summary"],
            data["export_html"]
        )


def _render_compare_result(session_id_a: str, session_id_b: str):
    """渲染对比结果"""
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