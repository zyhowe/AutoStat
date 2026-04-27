"""
文本分析服务 - 执行文本分析的核心业务逻辑
"""

import streamlit as st
from typing import List, Optional

from autotext.analyzer import TextAnalyzer
from autotext.reporter import TextReporter


class TextAnalysisService:
    """文本分析服务类"""

    @staticmethod
    def analyze_texts(texts: List[str], titles: Optional[List[str]] = None,
                      dates: Optional[List] = None, source_name: str = "文本数据") -> TextAnalyzer:
        """执行文本分析"""
        analyzer = TextAnalyzer(texts, source_name=source_name, quiet=False)
        if titles:
            analyzer.titles = titles
        if dates:
            analyzer.dates = dates

        analyzer.generate_full_report()
        return analyzer

    @staticmethod
    def generate_report_html(analyzer: TextAnalyzer) -> str:
        """生成 HTML 报告"""
        reporter = TextReporter(analyzer)
        return reporter.to_html()

    @staticmethod
    def generate_report_json(analyzer: TextAnalyzer) -> str:
        """生成 JSON 报告"""
        reporter = TextReporter(analyzer)
        return reporter.to_json()