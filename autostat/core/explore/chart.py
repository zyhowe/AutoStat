"""
智能图表推荐模块

根据数据特征和用户意图自动推荐最佳图表类型
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass  # 🆕 添加缺失的导入
import pandas as pd
import numpy as np


class ChartType(Enum):
    """图表类型"""
    BAR = "bar"                  # 柱状图
    GROUPED_BAR = "grouped_bar"  # 分组柱状图
    STACKED_BAR = "stacked_bar"  # 堆叠柱状图
    LINE = "line"                # 折线图
    AREA = "area"                # 面积图
    PIE = "pie"                  # 饼图
    DONUT = "donut"              # 环形图
    SCATTER = "scatter"          # 散点图
    BUBBLE = "bubble"            # 气泡图
    HISTOGRAM = "histogram"      # 直方图
    BOX = "box"                  # 箱线图
    HEATMAP = "heatmap"          # 热力图
    TREEMAP = "treemap"          # 树图
    SUNBURST = "sunburst"        # 旭日图
    RADAR = "radar"              # 雷达图
    WATERFALL = "waterfall"      # 瀑布图
    FUNNEL = "funnel"            # 漏斗图
    GAUGE = "gauge"              # 仪表盘


@dataclass
class ChartRecommendation:
    """图表推荐结果"""
    chart_type: ChartType
    config: Dict[str, Any]
    title: str
    description: str
    confidence: float
    data: pd.DataFrame


class ChartRecommender:
    """
    智能图表推荐器

    使用方式:
        recommender = ChartRecommender()
        recommendation = recommender.recommend(df, fields, intent)
    """

    def __init__(self, max_categories: int = 20):
        """
        初始化

        参数:
        - max_categories: 分类变量的最大类别数
        """
        self.max_categories = max_categories

    def recommend(
        self,
        df: pd.DataFrame,
        fields: List[str],
        intent: Optional[str] = None,
        time_field: Optional[str] = None
    ) -> List[ChartRecommendation]:
        """
        推荐图表

        参数:
        - df: 数据框
        - fields: 要展示的字段
        - intent: 用户意图（趋势/对比/构成/分布/关系）
        - time_field: 时间字段

        返回: 图表推荐列表
        """
        if fields is None or len(fields) == 0:
            fields = df.columns.tolist()

        recommendations = []

        # 分类字段和数值字段
        cat_fields = []
        num_fields = []
        for f in fields:
            if f not in df.columns:
                continue
            if pd.api.types.is_numeric_dtype(df[f]):
                num_fields.append(f)
            else:
                cat_fields.append(f)

        # 根据意图推荐
        if intent == "trend":
            recommendations = self._recommend_trend(df, num_fields, time_field)
        elif intent == "compare":
            recommendations = self._recommend_compare(df, cat_fields, num_fields)
        elif intent == "composition":
            recommendations = self._recommend_composition(df, cat_fields, num_fields)
        elif intent == "distribution":
            recommendations = self._recommend_distribution(df, num_fields)
        elif intent == "relationship":
            recommendations = self._recommend_relationship(df, num_fields)
        else:
            # 自动检测
            recommendations = self._auto_recommend(df, cat_fields, num_fields, time_field)

        return recommendations

    def _recommend_trend(
        self,
        df: pd.DataFrame,
        num_fields: List[str],
        time_field: Optional[str]
    ) -> List[ChartRecommendation]:
        """趋势图表推荐"""
        recommendations = []

        if not time_field or time_field not in df.columns:
            return self._auto_recommend(df, [], num_fields, None)

        for field in num_fields[:3]:
            # 按时间聚合
            data = df[[time_field, field]].dropna()
            if len(data) < 3:
                continue

            recommendations.append(ChartRecommendation(
                chart_type=ChartType.LINE,
                config={
                    "x": time_field,
                    "y": field,
                    "markers": True,
                },
                title=f"{field} 趋势",
                description=f"{field} 随时间的变化趋势",
                confidence=0.9,
                data=data
            ))

        return recommendations

    def _recommend_compare(
        self,
        df: pd.DataFrame,
        cat_fields: List[str],
        num_fields: List[str]
    ) -> List[ChartRecommendation]:
        """对比图表推荐"""
        recommendations = []

        if not cat_fields or not num_fields:
            return recommendations

        cat_field = cat_fields[0]
        num_field = num_fields[0]

        # 检查类别数量
        n_cats = df[cat_field].nunique()

        if n_cats <= 10:
            chart_type = ChartType.BAR
        elif n_cats <= self.max_categories:
            chart_type = ChartType.BAR
        else:
            chart_type = ChartType.BOX

        recommendations.append(ChartRecommendation(
            chart_type=chart_type,
            config={
                "x": cat_field,
                "y": num_field,
                "color": cat_field if n_cats > 1 else None,
            },
            title=f"{num_field} 按 {cat_field} 对比",
            description=f"对比不同 {cat_field} 的 {num_field}",
            confidence=0.85,
            data=df[[cat_field, num_field]].dropna()
        ))

        return recommendations

    def _recommend_composition(
        self,
        df: pd.DataFrame,
        cat_fields: List[str],
        num_fields: List[str]
    ) -> List[ChartRecommendation]:
        """构成图表推荐"""
        recommendations = []

        if not cat_fields or not num_fields:
            return recommendations

        cat_field = cat_fields[0]
        num_field = num_fields[0]

        n_cats = df[cat_field].nunique()

        # 计算占比
        total = df[num_field].sum()
        if total == 0:
            return recommendations

        if n_cats <= 8:
            chart_type = ChartType.PIE
        elif n_cats <= 15:
            chart_type = ChartType.DONUT
        else:
            chart_type = ChartType.TREEMAP

        agg_data = df.groupby(cat_field)[num_field].sum().reset_index()

        recommendations.append(ChartRecommendation(
            chart_type=chart_type,
            config={
                "names": cat_field,
                "values": num_field,
                "hole": 0.4 if chart_type == ChartType.DONUT else 0,
            },
            title=f"{num_field} 构成分析",
            description=f"{num_field} 在各 {cat_field} 中的分布",
            confidence=0.8,
            data=agg_data
        ))

        return recommendations

    def _recommend_distribution(
        self,
        df: pd.DataFrame,
        num_fields: List[str]
    ) -> List[ChartRecommendation]:
        """分布图表推荐"""
        recommendations = []

        for field in num_fields[:3]:
            series = df[field].dropna()
            if len(series) < 10:
                continue

            # 检查是否近似正态
            skew = series.skew()

            if abs(skew) < 0.5:
                chart_type = ChartType.HISTOGRAM
                description = f"{field} 近似正态分布"
            else:
                chart_type = ChartType.BOX
                description = f"{field} 分布偏斜 (偏度={skew:.2f})"

            recommendations.append(ChartRecommendation(
                chart_type=chart_type,
                config={
                    "field": field,
                    "bins": min(30, int(np.sqrt(len(series)))),
                },
                title=f"{field} 分布",
                description=description,
                confidence=0.85,
                data=pd.DataFrame({field: series})
            ))

        return recommendations

    def _recommend_relationship(
        self,
        df: pd.DataFrame,
        num_fields: List[str]
    ) -> List[ChartRecommendation]:
        """关系图表推荐"""
        recommendations = []

        if len(num_fields) < 2:
            return recommendations

        # 计算相关性
        corr_matrix = df[num_fields].corr()
        high_corr_pairs = []

        for i in range(len(num_fields)):
            for j in range(i+1, len(num_fields)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.5:
                    high_corr_pairs.append((num_fields[i], num_fields[j], corr))

        # 如果有强相关，推荐散点图
        for f1, f2, corr in high_corr_pairs[:3]:
            recommendations.append(ChartRecommendation(
                chart_type=ChartType.SCATTER,
                config={
                    "x": f1,
                    "y": f2,
                    "trendline": True,
                },
                title=f"{f1} vs {f2}",
                description=f"相关系数 r={corr:.3f}",
                confidence=0.9,
                data=df[[f1, f2]].dropna()
            ))

        # 如果字段较多，推荐热力图
        if len(num_fields) >= 3:
            recommendations.append(ChartRecommendation(
                chart_type=ChartType.HEATMAP,
                config={
                    "data": corr_matrix,
                    "annot": True,
                    "cmap": "coolwarm",
                },
                title="相关性热力图",
                description="各数值变量之间的相关系数矩阵",
                confidence=0.85,
                data=corr_matrix
            ))

        return recommendations

    def _auto_recommend(
        self,
        df: pd.DataFrame,
        cat_fields: List[str],
        num_fields: List[str],
        time_field: Optional[str]
    ) -> List[ChartRecommendation]:
        """自动推荐"""
        recommendations = []

        # 1. 如果有时间+数值 → 趋势
        if time_field and num_fields:
            recommendations.extend(self._recommend_trend(df, num_fields, time_field))

        # 2. 如果有分类+数值 → 对比或构成
        if cat_fields and num_fields:
            if len(cat_fields) >= 1 and len(num_fields) >= 1:
                recommendations.extend(self._recommend_compare(df, cat_fields[:1], num_fields[:1]))
                recommendations.extend(self._recommend_composition(df, cat_fields[:1], num_fields[:1]))

        # 3. 如果只有数值 → 分布或关系
        if num_fields and not cat_fields:
            if len(num_fields) >= 2:
                recommendations.extend(self._recommend_relationship(df, num_fields))
            recommendations.extend(self._recommend_distribution(df, num_fields[:1]))

        # 限制返回数量
        return recommendations[:5]

    def get_chart_config(self, recommendation: ChartRecommendation) -> Dict[str, Any]:
        """获取ECharts配置"""
        chart_type = recommendation.chart_type.value
        config = recommendation.config

        # ECharts配置映射（简化）
        if chart_type in ["bar", "grouped_bar", "stacked_bar"]:
            return {
                "type": "bar",
                "xAxis": {"type": "category", "data": config.get("x_data", [])},
                "yAxis": {"type": "value"},
                "series": [{"type": "bar", "data": config.get("y_data", [])}]
            }
        elif chart_type in ["line", "area"]:
            return {
                "type": "line" if chart_type == "line" else "area",
                "xAxis": {"type": "category", "data": config.get("x_data", [])},
                "yAxis": {"type": "value"},
                "series": [{"type": "line", "data": config.get("y_data", [])}]
            }
        elif chart_type in ["pie", "donut"]:
            return {
                "type": "pie",
                "series": [{
                    "type": "pie",
                    "radius": ["40%", "70%"] if chart_type == "donut" else ["0%", "70%"],
                    "data": config.get("pie_data", [])
                }]
            }
        else:
            return {"type": chart_type, "config": config}


def recommend_chart(
    df: pd.DataFrame,
    fields: List[str],
    **kwargs
) -> List[ChartRecommendation]:
    """便捷函数：推荐图表"""
    recommender = ChartRecommender(**kwargs)
    return recommender.recommend(df, fields)