"""数据 Schema"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class DataPreviewResponse(BaseModel):
    head: List[Dict]
    shape: Dict[str, int]
    columns: List[str]
    dtypes: Dict[str, str]


class DataUploadResponse(BaseModel):
    file_name: str
    file_path: str
    rows: int
    columns: int
    variable_types: Dict[str, str]
    preview: DataPreviewResponse


# ==================== 多表相关 Schema ====================

class TableInfo(BaseModel):
    """单表信息"""
    name: str
    rows: int
    columns: int
    preview: Dict[str, Any]
    variable_types: Dict[str, str]
    load_status: str  # 'success' | 'failed'
    error: Optional[str] = None


class CandidateRelation(BaseModel):
    """候选关系"""
    from_table: str
    from_col: str
    to_table: str
    to_col: str
    relation_type: str  # 'one_to_one' | 'one_to_many' | 'many_to_one'
    confidence: float  # 0-1
    auto_discovered: bool = True


class DatabaseLoadRequest(BaseModel):
    config: Dict[str, Any]
    table_names: List[str]  # ✅ 改为列表
    limit: int = 5000
    max_text_length: int = 100
    relationships: Optional[List[Dict]] = None


class DatabaseLoadResponse(BaseModel):
    tables: Dict[str, Any]  # 兼容旧版
    session_id: Optional[str] = None
    table_list: List[TableInfo] = []  # ✅ 新增：详细表信息
    candidate_relations: List[CandidateRelation] = []  # ✅ 新增：候选关系
    load_summary: Dict[str, Any] = {}  # ✅ 新增：加载汇总


class RelationConfirmRequest(BaseModel):
    """关系确认请求"""
    session_id: str
    relationships: List[Dict]  # 用户确认的关系列表


class RelationConfirmResponse(BaseModel):
    success: bool
    message: str