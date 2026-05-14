# webtext/components/data_preparation.py
"""数据准备组件 - 单文本输入"""

import streamlit as st
from datetime import datetime

from webtext.services.analysis_service import TextAnalysisService
from webtext.services.session_service import TextSessionService


def render_data_preparation():
    """渲染数据准备标签页"""
    st.markdown("### 📝 输入文本")
    st.caption("输入或粘贴待分析的文本内容，支持中英文")

    # 文本输入区域
    text = st.text_area(
        "文本内容",
        height=300,
        placeholder="请输入或粘贴要分析的文本...\n\n例如：\n今天天气真好，心情很愉快。这款产品非常好用，推荐给大家。",
        key="text_input_area"
    )

    # 可选参数（仅保留标题）
    st.markdown("---")
    st.markdown("### 🔧 可选参数")

    title = st.text_input(
        "标题（可选）",
        placeholder="为文本添加一个标题",
        key="text_title_input",
        help="标题会显示在分析报告中，便于识别"
    )

    st.markdown("---")

    # 开始分析按钮
    if st.button("▶️ 开始分析", type="primary", use_container_width=True,
                 disabled=st.session_state.get("text_analysis_running", False)):

        if not text or not text.strip():
            st.error("请先输入文本内容")
            return

        st.session_state.text_analysis_running = True

        # 显示进度
        progress_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        try:
            # 创建会话
            status_placeholder.info("📁 创建分析会话...")
            progress_bar.progress(10)

            session_id = TextSessionService.create_session(text[:50])
            st.session_state.text_current_session = session_id

            # 执行分析（BERT 始终启用）
            status_placeholder.info("🔍 正在分析文本...")
            progress_bar.progress(30)

            analyzer, html_content, json_data = TextAnalysisService.analyze_text(
                text=text,
                title=title if title else None,
                use_bert=True,  # 始终启用 BERT
                quiet=True
            )

            progress_bar.progress(80)

            if analyzer and html_content and json_data:
                status_placeholder.info("💾 保存分析结果...")
                progress_bar.progress(90)

                # 保存结果
                TextSessionService.save_analysis_results(session_id, html_content, json_data)

                # 更新 session state
                st.session_state.text_analysis_completed = True
                st.session_state.text_analyzer = analyzer
                st.session_state.text_html_content = html_content
                st.session_state.text_json_data = json_data
                st.session_state.text_chat_messages = []

                progress_bar.progress(100)
                status_placeholder.success("✅ 分析完成！")

                # 显示结果摘要
                st.markdown("---")
                st.markdown("### 📊 分析结果摘要")

                # 统计信息
                stats = analyzer.stats_result
                sentiment = analyzer.sentiment_distribution

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("文本长度", f"{stats['char_length']['mean']:.0f} 字符")
                with col2:
                    pos_rate = sentiment.get('positive_rate', 0)
                    neg_rate = sentiment.get('negative_rate', 0)
                    sentiment_label = "积极" if pos_rate > neg_rate else "消极" if neg_rate > pos_rate else "中性"
                    st.metric("情感倾向", sentiment_label)
                with col3:
                    keyword_count = len(analyzer.keywords.get('frequency', []))
                    st.metric("关键词数", keyword_count)
                with col4:
                    entity_count = sum(s.get('unique', 0) for s in analyzer.entity_stats.values())
                    st.metric("实体数量", entity_count)

                # 情感分布
                st.markdown("#### 😊 情感分布")
                st.progress(pos_rate, text=f"积极: {pos_rate:.1%}")
                st.progress(neg_rate, text=f"消极: {neg_rate:.1%}")
                st.progress(1 - pos_rate - neg_rate, text=f"中性: {1 - pos_rate - neg_rate:.1%}")

                # 高频关键词
                st.markdown("#### 🔑 高频关键词")
                keywords = analyzer.keywords.get('frequency', [])[:20]
                if keywords:
                    keyword_html = " ".join([
                                                f"<span style='background:#e3f2fd; padding:4px 12px; border-radius:20px; margin:4px; display:inline-block;'>{k}({c})</span>"
                                                for k, c in keywords[:15]])
                    st.markdown(keyword_html, unsafe_allow_html=True)
                else:
                    st.caption("暂无关键词")

                # 自动跳转到报告预览
                st.info("分析完成！正在跳转到报告预览...")
                st.session_state.text_current_tab = 1
                st.rerun()
            else:
                st.error("分析失败，请重试")

        except Exception as e:
            st.error(f"分析失败: {str(e)}")
            import traceback
            with st.expander("详细错误信息"):
                st.code(traceback.format_exc())

        finally:
            progress_bar.empty()
            status_placeholder.empty()
            st.session_state.text_analysis_running = False