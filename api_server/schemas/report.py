"""报告 Schema"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class ReportResponse(BaseModel):
    analysis_time: Optional[str]
    source_table: Optional[str]
    data_shape: Dict[str, Any]
    variable_types: Dict[str, Any]
    quality_report: Dict[str, Any]
    correlations: Dict[str, Any]
    time_series_diagnostics: Dict[str, Any]
    model_recommendations: List[Dict]
    cleaning_suggestions: List[str]


class SummaryResponse(BaseModel):
    conclusions: List[Dict]


class InsightsResponse(BaseModel):
    findings: List[str]
    conclusions: List[Dict]