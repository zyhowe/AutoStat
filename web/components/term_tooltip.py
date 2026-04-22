"""
术语解释浮层组件
"""

import streamlit as st

# 术语库
TERM_GLOSSARY = {
    "偏度": "衡量数据分布的不对称程度。偏度>0表示右偏（有较大异常值），<0表示左偏（有较小异常值）",
    "峰度": "衡量数据分布的陡峭程度。峰度>3表示分布比正态分布更陡峭，有更多极端值",
    "相关系数": "衡量两个变量相关程度的指标。绝对值越接近1，相关性越强",
    "p值": "判断结果是否具有统计学意义的指标。通常p<0.05表示结果显著",
    "Cramer's V": "衡量两个分类变量关联强度的指标。值越接近1，关联越强",
    "Eta-squared": "衡量分类变量对数值变量解释程度的指标。值越大，组间差异越显著",
    "标准差": "衡量数据离散程度的指标。标准差越大，数据越分散",
    "置信区间": "总体参数可能落入的范围，通常95%置信区间表示有95%的把握包含真实值",
    "正态分布": "数据呈钟形曲线分布，多数值集中在均值附近",
    "异常值": "显著偏离其他观测值的数值，可能是数据错误或有特殊意义",
    "缺失值": "数据中空缺或未记录的值，需要处理",
    "自相关": "时间序列与其自身滞后的相关性，用于判断是否有规律可循",
    "平稳性": "时间序列的统计特性不随时间变化，是许多时序模型的前提",
    "IQR": "四分位距，Q3与Q1的差值，用于检测异常值",
    "ANOVA": "方差分析，用于比较多个组之间是否存在显著差异",
    "卡方检验": "用于检验两个分类变量之间是否独立",
    "t检验": "用于比较两组均值是否存在显著差异"
}


def render_term_with_tooltip(term: str, value: str = None):
    """渲染带解释浮层的术语"""
    explanation = TERM_GLOSSARY.get(term, "")

    if value:
        display = f"{term}: {value}"
    else:
        display = term

    if explanation:
        # 使用 HTML abbr 标签实现悬停解释
        st.markdown(
            f'<abbr title="{explanation}" style="border-bottom: 1px dashed #999; cursor: help;">{display}</abbr>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(display)


def apply_term_tooltips_to_html(html_content: str) -> str:
    """为HTML报告中的术语添加解释浮层"""
    # 添加样式
    style = """
    <style>
    abbr {
        border-bottom: 1px dashed #999;
        cursor: help;
        text-decoration: none;
    }
    abbr:hover {
        background-color: #f0f0f0;
    }
    </style>
    """

    # 在 head 中添加样式
    if "<head>" in html_content:
        html_content = html_content.replace("</head>", f"{style}</head>")
    else:
        html_content = style + html_content

    # 为常见术语添加 abbr 标签
    for term, explanation in TERM_GLOSSARY.items():
        # 替换普通文本（避免替换标签内的内容）
        html_content = html_content.replace(
            f">{term}<",
            f'><abbr title="{explanation}">{term}</abbr><'
        )

    return html_content


def render_glossary_section():
    """渲染术语表（可折叠）"""
    with st.expander("📖 术语表", expanded=False):
        st.markdown("常用统计术语解释：")

        cols = st.columns(2)
        for i, (term, explanation) in enumerate(TERM_GLOSSARY.items()):
            with cols[i % 2]:
                st.markdown(
                    f"**{term}**：{explanation[:50]}..." if len(explanation) > 50 else f"**{term}**：{explanation}")