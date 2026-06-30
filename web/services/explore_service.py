"""
自助探索服务
"""

from typing import Dict, Any, Optional, List
import pandas as pd

from autostat.core.explore import NL2SQL, ChartRecommender, StoryGenerator
from autostat.core.explore.chart import ChartRecommendation
from autostat.core.explore.story import Story


class ExploreService:
    """自助探索服务"""

    @staticmethod
    def nl2sql(
        question: str,
        schema: Dict[str, Any],
        llm_client=None
    ) -> Dict[str, Any]:
        """自然语言转SQL"""
        nl2sql = NL2SQL(llm_client)
        return nl2sql.convert(question, schema)

    @staticmethod
    def recommend_charts(
        data: pd.DataFrame,
        fields: List[str],
        intent: Optional[str] = None,
        time_field: Optional[str] = None
    ) -> List[ChartRecommendation]:
        """推荐图表"""
        recommender = ChartRecommender()
        return recommender.recommend(data, fields, intent, time_field)

    @staticmethod
    def generate_story(
        data: pd.DataFrame,
        analysis_result: Dict[str, Any],
        llm_client=None
    ) -> Story:
        """生成故事"""
        generator = StoryGenerator(llm_client)
        return generator.generate(data, analysis_result)

    @staticmethod
    def execute_query(sql: str, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """执行查询"""
        nl2sql = NL2SQL()
        return nl2sql.execute(sql, data)