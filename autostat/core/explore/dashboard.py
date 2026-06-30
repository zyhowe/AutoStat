"""
交互式仪表板构建器

构建可交互的数据仪表板
"""

import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Widget:
    """仪表板组件"""
    id: str
    type: str  # "metric", "chart", "table", "filter"
    config: Dict[str, Any]
    position: Dict[str, int]  # {"row": 0, "col": 0, "width": 4, "height": 2}


@dataclass
class Dashboard:
    """仪表板"""
    id: str
    title: str
    description: str
    widgets: List[Widget]
    filters: List[Dict[str, Any]]
    layout: str  # "grid", "columns"


class DashboardBuilder:
    """
    仪表板构建器

    使用方式:
        builder = DashboardBuilder()
        dashboard = builder.build(df, analysis_result)
    """

    def __init__(self):
        self.widgets: List[Widget] = []
        self.filters: List[Dict] = []

    def build(
        self,
        df: pd.DataFrame,
        analysis_result: Dict[str, Any],
        title: str = "数据分析仪表板"
    ) -> Dashboard:
        """
        构建仪表板

        参数:
        - df: 数据框
        - analysis_result: 分析结果
        - title: 仪表板标题

        返回: Dashboard 对象
        """
        self.widgets = []
        self.filters = []

        # 1. 添加KPI指标
        self._add_kpi_widgets(df)

        # 2. 添加图表
        self._add_chart_widgets(df, analysis_result)

        # 3. 添加过滤条件
        self._add_filter_widgets(df)

        return Dashboard(
            id=f"dashboard_{datetime.now().timestamp()}",
            title=title,
            description=f"基于 {len(df):,} 行数据自动生成",
            widgets=self.widgets,
            filters=self.filters,
            layout="grid"
        )

    def _add_kpi_widgets(self, df: pd.DataFrame):
        """添加KPI指标"""
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

        kpis = []

        for col in numeric_cols[:4]:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            kpis.append({
                "id": f"kpi_{col}",
                "title": col,
                "value": series.mean(),
                "format": "{:,.0f}" if series.dtype == 'int64' else "{:,.2f}",
                "change": series.pct_change().iloc[-1] if len(series) > 1 else None,
                "color": "blue"
            })

        # 添加行数
        kpis.append({
            "id": "kpi_rows",
            "title": "总行数",
            "value": len(df),
            "format": "{:,}",
            "change": None,
            "color": "green"
        })

        # 添加列数
        kpis.append({
            "id": "kpi_cols",
            "title": "总列数",
            "value": len(df.columns),
            "format": "{}",
            "change": None,
            "color": "gray"
        })

        # 创建Widget
        for i, kpi in enumerate(kpis[:6]):
            self.widgets.append(Widget(
                id=kpi["id"],
                type="metric",
                config=kpi,
                position={"row": 0, "col": i % 4, "width": 3, "height": 1}
            ))

    def _add_chart_widgets(self, df: pd.DataFrame, analysis_result: Dict[str, Any]):
        """添加图表"""
        from autostat.core.explore.chart import ChartRecommender

        recommender = ChartRecommender()

        # 获取数值和分类字段
        num_fields = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_fields = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # 检测时间字段
        time_fields = df.select_dtypes(include=['datetime64']).columns.tolist()
        time_field = time_fields[0] if time_fields else None

        # 推荐图表
        recommendations = recommender.recommend(df, num_fields + cat_fields, time_field=time_field)

        # 转换为Widget
        for i, rec in enumerate(recommendations[:4]):
            self.widgets.append(Widget(
                id=f"chart_{i}",
                type="chart",
                config={
                    "chart_type": rec.chart_type.value,
                    "data": rec.data.to_dict('records') if rec.data is not None else [],
                    "title": rec.title,
                    "description": rec.description,
                    "config": rec.config,
                },
                position={"row": 1 + i // 2, "col": (i % 2) * 6, "width": 6, "height": 3}
            ))

    def _add_filter_widgets(self, df: pd.DataFrame):
        """添加过滤条件"""
        # 分类字段作为过滤器
        cat_fields = df.select_dtypes(include=['object', 'category']).columns

        for field in cat_fields[:3]:
            if df[field].nunique() <= 20:
                self.filters.append({
                    "id": f"filter_{field}",
                    "field": field,
                    "type": "dropdown",
                    "title": field,
                    "options": df[field].dropna().unique().tolist(),
                    "default": None
                })

        # 时间字段作为日期范围
        time_fields = df.select_dtypes(include=['datetime64']).columns
        for field in time_fields[:1]:
            if len(df[field].dropna()) > 0:
                self.filters.append({
                    "id": f"filter_{field}",
                    "field": field,
                    "type": "date_range",
                    "title": field,
                    "min": df[field].min(),
                    "max": df[field].max(),
                    "default": None
                })

    def to_json(self) -> Dict[str, Any]:
        """导出为JSON"""
        return {
            "id": f"dashboard_{datetime.now().timestamp()}",
            "title": "数据分析仪表板",
            "widgets": [
                {
                    "id": w.id,
                    "type": w.type,
                    "config": w.config,
                    "position": w.position
                }
                for w in self.widgets
            ],
            "filters": self.filters,
            "layout": "grid"
        }


def build_dashboard(
    df: pd.DataFrame,
    analysis_result: Dict[str, Any],
    **kwargs
) -> Dashboard:
    """便捷函数：构建仪表板"""
    builder = DashboardBuilder()
    return builder.build(df, analysis_result, **kwargs)