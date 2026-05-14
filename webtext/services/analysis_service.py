# webtext/services/analysis_service.py
"""文本分析服务 - 执行文本分析"""

import json
import time
from typing import Tuple, Optional, Dict, Any

import streamlit as st
from autotext.analyzer import TextAnalyzer


class TextAnalysisService:
    """文本分析服务类"""

    @staticmethod
    def analyze_text(
            text: str,
            title: Optional[str] = None,
            use_bert: bool = True,
            quiet: bool = False
    ) -> Tuple[Optional[TextAnalyzer], Optional[str], Optional[Dict]]:
        """
        执行文本分析

        参数:
        - text: 文本内容
        - title: 标题（可选）
        - use_bert: 是否使用BERT增强
        - quiet: 静默模式

        返回:
        - (analyzer, html_content, json_data)
        """
        if not text or not text.strip():
            return None, None, None

        try:
            texts = [text]
            analyzer = TextAnalyzer(
                texts,
                source_name="单文本分析",
                quiet=quiet,
                use_bert=use_bert
            )

            # 设置可选参数
            if title:
                analyzer.titles = [title]

            # 执行分析
            analyzer.generate_full_report()

            # 生成报告
            html_content = analyzer.to_html()
            json_data = json.loads(analyzer.to_json())

            return analyzer, html_content, json_data

        except Exception as e:
            if not quiet:
                st.error(f"分析失败: {str(e)}")
            return None, None, None