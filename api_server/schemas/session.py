"""会话 Schema"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class SessionCreateRequest(BaseModel):
    source_name: Optional[str] = "未命名"
    analysis_type: Optional[str] = "single"
    tables_info: Optional[Dict] = None


class SessionCreateResponse(BaseModel):
    session_id: str
    message: str


class SessionInfo(BaseModel):
    session_id: str
    source_name: str
    analysis_type: str
    created_at: str
    last_accessed_at: str
    data_shape: Dict[str, Any]
    variable_types: Dict[str, str]
    files: Dict[str, str]


class ProjectItem(BaseModel):
    session_id: str
    source_name: str
    analysis_type: str
    created_at: str
    last_accessed_at: str
    data_shape: Dict[str, Any]


class ProjectListResponse(BaseModel):
    total: int
    projects: List[ProjectItem]