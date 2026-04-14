"""上下文构建模块"""

import streamlit as st
import pandas as pd
import re
from web.components.chat_interface import build_json_context_prompt, build_html_context_prompt, \
    build_data_context_prompt


def build_context_prompt(selected_contexts, analysis_type, json_data, html_content, raw_data_preview, source_name):
    """根据选中的上下文构建提示词"""

    context_parts = []

    analysis_type_map = {
        "single": "单文件分析",
        "multi": "多文件分析",
        "database": "数据库分析"
    }
    context_parts.append(
        f"## 分析概况\n- 分析类型: {analysis_type_map.get(analysis_type, '未知')}\n- 数据源: {source_name}\n")

    if "json_result" in selected_contexts and json_data:
        context_parts.append(build_json_context_prompt(json_data, analysis_type))

    if "html_report" in selected_contexts and html_content:
        context_parts.append(build_html_context_prompt(html_content))

    if "raw_data" in selected_contexts and raw_data_preview:
        context_parts.append(build_data_context_prompt(raw_data_preview))

    full_prompt = f"""
你是一位专业的数据分析师。以下是根据用户选择的上下文提供的分析数据。

{chr(10).join(context_parts)}

---

## 注意事项
- 请基于上述提供的数据上下文回答问题
- 源数据包含完整的数据预览（前50行）和统计信息
- 你可以根据日期范围、数值范围等条件查询具体数据
- 如果用户询问具体数据（如某年某月的数据），请根据数据预览和统计信息回答
- 用中文回答，结构清晰，使用markdown格式
"""
    return full_prompt