"""模型相关 Schema"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel


class TrainRequest(BaseModel):
    session_id: str
    task_type: str  # classification, regression, clustering, time_series
    model_key: str
    target_column: Optional[str] = None
    features: List[str]
    params: Dict[str, Any] = {}
    user_model_name: Optional[str] = None


class TrainResponse(BaseModel):
    task_id: str
    message: str


class PredictRequest(BaseModel):
    model_key: str
    session_id: str
    input_values: Dict[str, Any]


class PredictResponse(BaseModel):
    prediction: Any
    confidence: Optional[float] = None
    probabilities: Optional[List[float]] = None
    model_name: str


class TrainStatusResponse(BaseModel):
    task_id: str
    status: str  # pending, running, completed, failed
    progress: int
    message: str
    error: Optional[str] = None
    model_key: Optional[str] = None
    user_model_name: Optional[str] = None