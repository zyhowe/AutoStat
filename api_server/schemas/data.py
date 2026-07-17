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
    table_names: List[str]
    limit: int = 5000
    max_text_length: int = 100
    relationships: Optional[List[Dict]] = None


class DatabaseLoadResponse(BaseModel):
    tables: Dict[str, Any]
    session_id: Optional[str] = None
    table_list: List[TableInfo] = []
    candidate_relations: List[CandidateRelation] = []
    load_summary: Dict[str, Any] = {}


class RelationConfirmRequest(BaseModel):
    session_id: str
    relationships: List[Dict]


class RelationConfirmResponse(BaseModel):
    success: bool
    message: str


# ==================== 新增：字段类型更新 ====================

class FieldTypesUpdateRequest(BaseModel):
    """字段类型更新请求"""
    session_id: str
    table_name: str  # 表名，为空或"merged"表示合并表
    field_types: Dict[str, str]  # {字段名: 类型}


class FieldTypesUpdateResponse(BaseModel):
    """字段类型更新响应"""
    success: bool
    message: str