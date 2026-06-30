"""
自助式探索模块

提供业务人员自助探索数据的能力：
- 自然语言转SQL (NL2SQL)
- 智能图表推荐 (Chart Recommender)
- 自动故事生成 (Story Generator)
- 交互式仪表板 (Dashboard Builder)
"""

from autostat.core.explore.nl2sql import NL2SQL
from autostat.core.explore.chart import ChartRecommender, ChartType, ChartRecommendation
from autostat.core.explore.story import StoryGenerator, Story, StorySection
from autostat.core.explore.dashboard import DashboardBuilder, Dashboard, Widget

__all__ = [
    "NL2SQL",
    "ChartRecommender",
    "ChartType",
    "ChartRecommendation",
    "StoryGenerator",
    "Story",
    "StorySection",
    "DashboardBuilder",
    "Dashboard",
    "Widget",
]