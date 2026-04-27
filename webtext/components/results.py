"""
分析结果展示组件 - 报告预览
"""

import streamlit as st
import json
from datetime import datetime

from autotext.reporter import TextReporter


def render_results_tab():
    """渲染分析报告标签页"""
    st.markdown("### 📄 分析报告")

    if not st.session_state.analysis_completed or st.session_state.analyzer is None:
        st.info("暂无分析报告，请先在「数据准备」中上传文本并开始分析")
        st.caption("支持 .txt 文件或 CSV/Excel 中的文本列")
        return

    analyzer = st.session_state.analyzer

    # 导出按钮
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📥 下载 HTML 报告", use_container_width=True):
            reporter = TextReporter(analyzer)
            html = reporter.to_html()
            st.download_button(
                "确认下载",
                html,
                f"text_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                "text/html",
                key="download_html"
            )
    with col2:
        if st.button("📥 下载 JSON 结果", use_container_width=True):
            reporter = TextReporter(analyzer)
            json_str = reporter.to_json()
            st.download_button(
                "确认下载",
                json_str,
                f"text_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                key="download_json"
            )
    with col3:
        if st.button("📥 下载 Markdown", use_container_width=True):
            reporter = TextReporter(analyzer)
            md = reporter.to_markdown()
            st.download_button(
                "确认下载",
                md,
                f"text_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                "text/markdown",
                key="download_md"
            )

    st.markdown("---")

    # 显示报告摘要
    render_report_summary(analyzer)


def render_report_summary(analyzer):
    """渲染报告摘要"""
    # 基础统计
    stats = analyzer.stats_result

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总文本数", stats['total_count'])
    with col2:
        st.metric("平均长度", f"{stats['char_length']['mean']:.0f} 字")
    with col3:
        st.metric("空文本数", stats['empty_count'])
    with col4:
        pos_rate = analyzer.sentiment_distribution.get('positive_rate', 0)
        st.metric("积极文本", f"{pos_rate:.1%}")

    # 清洗建议
    if hasattr(analyzer, 'cleaning_suggestions') and analyzer.cleaning_suggestions:
        with st.expander("🧹 清洗建议", expanded=False):
            for s in analyzer.cleaning_suggestions[:5]:
                st.caption(f"• {s}")

    # 关键词
    if analyzer.keywords:
        st.markdown("### 🔑 高频关键词")
        keywords = analyzer.keywords.get("frequency", [])[:30]
        cols = st.columns(5)
        for i, (word, count) in enumerate(keywords):
            with cols[i % 5]:
                st.caption(f"{word} ({count})")

    # 情感分布
    st.markdown("### 😊 情感分布")
    dist = analyzer.sentiment_distribution
    st.progress(dist['positive_rate'], text=f"积极: {dist['positive_rate']:.1%}")
    st.progress(dist['negative_rate'], text=f"消极: {dist['negative_rate']:.1%}")
    st.progress(dist['neutral_rate'], text=f"中性: {dist['neutral_rate']:.1%}")

    # 实体统计
    if hasattr(analyzer, 'entity_stats') and analyzer.entity_stats:
        st.markdown("### 🏷️ 实体统计")
        col1, col2, col3 = st.columns(3)
        with col1:
            person_count = analyzer.entity_stats.get("person", {}).get("unique", 0)
            st.metric("人名", person_count)
        with col2:
            loc_count = analyzer.entity_stats.get("location", {}).get("unique", 0)
            st.metric("地名", loc_count)
        with col3:
            org_count = analyzer.entity_stats.get("organization", {}).get("unique", 0)
            st.metric("组织名", org_count)

    # 聚类结果
    if hasattr(analyzer, 'cluster_info') and analyzer.cluster_info:
        st.markdown("### 🔘 文本聚类")
        for cluster in analyzer.cluster_info[:5]:
            with st.expander(f"簇 {cluster['cluster_id']} ({cluster['size']} 条)"):
                st.caption(f"关键词: {', '.join(cluster['top_words'][:10])}")
                if cluster.get('name'):
                    st.info(f"📌 {cluster['name']}")
                if cluster.get('sample_texts'):
                    st.text(f"示例: {cluster['sample_texts'][0][:200]}...")

    # 主题结果
    if hasattr(analyzer, 'topics') and analyzer.topics:
        st.markdown("### 📚 主题建模")
        for topic in analyzer.topics[:5]:
            with st.expander(f"主题 {topic['topic_id']}"):
                keywords = topic.get('keywords', [])[:10]
                st.caption(f"关键词: {', '.join(keywords)}")

    # 大模型洞察
    if hasattr(analyzer, 'llm_enhancer') and analyzer.llm_enhancer and analyzer.llm_enhancer.is_available():
        st.markdown("### 🧠 AI 深度洞察")
        if st.button("生成深度洞察", use_container_width=True):
            with st.spinner("AI 正在分析..."):
                insight = analyzer.llm_enhancer.generate_insights(
                    analyzer.stats_result,
                    analyzer.sentiment_distribution,
                    analyzer.topics if hasattr(analyzer, 'topics') else [],
                    analyzer.cluster_info if hasattr(analyzer, 'cluster_info') else []
                )
                if insight:
                    st.info(insight)
                else:
                    st.warning("生成失败，请检查大模型配置")