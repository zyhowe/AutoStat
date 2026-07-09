"""分析 Schema"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    session_id: str
    variable_types: Optional[Dict[str, str]] = None
    include_html: bool = False  # ✅ 新增：是否生成 HTML 报告，默认 False


class AnalysisResponse(BaseModel):
    task_id: str
    session_id: str
    status: str
    message: str


class AnalysisStatus(BaseModel):
    task_id: str
    status: str
    progress: int
    message: str
    error: Optional[str] = None
    result: Optional[Dict] = None