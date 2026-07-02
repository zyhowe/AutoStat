"""报告服务 - 不依赖 web/"""
from typing import Dict, Any, List

from autostat.reporter import Reporter
from api_server.services.insight_service import InsightService


class ReportService:
    """报告服务"""

    def __init__(self):
        self.insight_service = InsightService()

    def get_full_report(self, analysis_result: Dict) -> Dict:
        """获取完整报告"""
        return {
            "analysis_time": analysis_result.get("analysis_time"),
            "source_table": analysis_result.get("source_table"),
            "data_shape": analysis_result.get("data_shape"),
            "variable_types": analysis_result.get("variable_types"),
            "variable_summaries": analysis_result.get("variable_summaries"),
            "quality_report": analysis_result.get("quality_report"),
            "correlations": analysis_result.get("correlations"),
            "time_series_diagnostics": analysis_result.get("time_series_diagnostics"),
            "model_recommendations": analysis_result.get("model_recommendations"),
            "cleaning_suggestions": analysis_result.get("cleaning_suggestions"),
            "distribution_insights": analysis_result.get("distribution_insights")
        }

    def get_summary(self, analysis_result: Dict) -> List[Dict]:
        """获取核心结论"""
        return self.insight_service.extract_top_conclusions(analysis_result)

    def get_insights(self, analysis_result: Dict) -> Dict:
        """获取智能解读"""
        return {
            "findings": self.insight_service.generate_rule_based_insights(analysis_result),
            "conclusions": self.insight_service.extract_top_conclusions(analysis_result)
        }

    def get_html(self, analyzer) -> str:
        """获取HTML报告"""
        reporter = Reporter(analyzer)
        return reporter.to_html()