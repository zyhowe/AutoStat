"""
价值预览组件 - 上传后显示数据价值
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any


def render_value_preview(df: pd.DataFrame, preview_data: Dict[str, Any]):
    """渲染价值预览卡片"""

    st.markdown("### 📊 数据洞察预览")
    st.caption("基于您的数据，我们发现以下分析机会：")

    # 价值卡片
    cols = st.columns(3)

    cards = []

    # 可预测卡片
    if preview_data.get("predictable", {}).get("has"):
        cards.append({
            "icon": "🎯",
            "title": "可预测",
            "color": "#2ca02c",
            "content": preview_data["predictable"]["description"]
        })

    # 时间序列卡片
    if preview_data.get("timeseries", {}).get("has"):
        cards.append({
            "icon": "📈",
            "title": "可预测趋势",
            "color": "#1f77b4",
            "content": preview_data["timeseries"]["description"]
        })

    # 聚类卡片
    if preview_data.get("clustering", {}).get("has"):
        cards.append({
            "icon": "🔘",
            "title": "可做分群",
            "color": "#9467bd",
            "content": preview_data["clustering"]["description"]
        })

    # 清洗卡片
    if preview_data.get("needs_cleaning", {}).get("has"):
        cards.append({
            "icon": "⚠️",
            "title": "需清洗",
            "color": "#d62728",
            "content": preview_data["needs_cleaning"]["description"]
        })

    # 补全到3个卡片
    while len(cards) < 3:
        cards.append({
            "icon": "📊",
            "title": "可探索",
            "color": "#7f7f7f",
            "content": "数据包含多维度信息，可进行探索性分析"
        })

    for i, card in enumerate(cards[:3]):
        with cols[i]:
            st.markdown(f"""
            <div style="background: {card['color']}10; 
                        border-radius: 12px; 
                        padding: 16px; 
                        text-align: center;
                        border: 1px solid {card['color']}30;">
                <div style="font-size: 28px;">{card['icon']}</div>
                <div style="font-weight: bold; margin: 8px 0;">{card['title']}</div>
                <div style="font-size: 12px; color: #666;">{card['content']}</div>
            </div>
            """, unsafe_allow_html=True)

    # 推荐分析
    st.markdown("---")
    st.info(f"💡 **推荐分析**：{preview_data.get('recommended_analysis', '数据探索性分析')}")

    # 智能摘要
    from web.services.insight_service import InsightService
    summary = InsightService.generate_smart_summary(df, preview_data)
    st.caption(summary)


def render_compact_value_preview(df: pd.DataFrame, preview_data: Dict[str, Any]):
    """渲染紧凑版价值预览（用于侧边栏）"""
    from web.services.insight_service import InsightService
    summary = InsightService.generate_smart_summary(df, preview_data)
    st.info(f"📌 {summary}")