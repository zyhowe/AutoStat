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