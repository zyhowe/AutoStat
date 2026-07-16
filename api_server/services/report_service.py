"""报告服务 - 不依赖 web/ 或 api_server/services/insight_service"""
import math
import numpy as np
from typing import Dict, Any, List

from autostat.reporter import Reporter
from autostat.core.insight import InsightService


def clean_nan(obj):
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    elif isinstance(obj, np.ndarray):
        return [clean_nan(v) for v in obj.tolist()]
    else:
        return obj


class ReportService:
    def __init__(self):
        self.insight_service = InsightService()

    def get_full_report(self, analysis_result: Dict) -> Dict:
        return clean_nan({
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
            "distribution_insights": analysis_result.get("distribution_insights"),
            "summary": analysis_result.get("summary", []),

            # 多表信息（用于表选择器）
            "multi_table_info": analysis_result.get("multi_table_info"),
            "is_multi_table": analysis_result.get("is_multi_table", False),
            "table_names": analysis_result.get("table_names", []),
            "relationships": analysis_result.get("relationships", []),

            # ✅ 核心修复：透传 all_tables（合并表和原始表的完整数据）
            "all_tables": analysis_result.get("all_tables")
        })

    def get_summary(self, analysis_result: Dict) -> List[Dict]:
        return clean_nan(self.insight_service.extract_top_conclusions(analysis_result))

    def get_insights(self, analysis_result: Dict) -> Dict:
        return clean_nan({
            "findings": self.insight_service.generate_rule_based_insights(analysis_result),
            "conclusions": self.insight_service.extract_top_conclusions(analysis_result)
        })

    def get_html(self, analyzer) -> str:
        return Reporter(analyzer).to_html()