"""模型管理路由"""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel

from api_server.dependencies import Dependencies
from api_server.services.session_service import SessionService
from api_server.services.models_service import ModelsService
from api_server.schemas.models import (
    TrainRequest, TrainResponse,
    PredictRequest, PredictResponse,
    TrainStatusResponse
)

router = APIRouter()


@router.post("/models/train", response_model=TrainResponse)
async def train_model(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    models_service: ModelsService = Depends(Dependencies.get_models_service),
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """训练模型"""
    session = session_service.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    file_info = session_service.get_data_path(request.session_id)
    if not file_info:
        raise HTTPException(status_code=400, detail="会话没有关联的数据文件")

    task_id = f"train_{request.session_id}_{int(__import__('time').time())}"

    background_tasks.add_task(
        models_service.train_model,
        request.session_id,
        file_info,
        request.task_type,
        request.model_key,
        request.target_column,
        request.features,
        request.params,
        request.user_model_name,
        task_id
    )

    return TrainResponse(task_id=task_id, message="训练任务已提交")


@router.get("/models/list")
async def list_models(
    session_id: str,
    models_service: ModelsService = Depends(Dependencies.get_models_service)
):
    """列出已训练模型"""
    return models_service.list_models(session_id)


@router.post("/models/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    models_service: ModelsService = Depends(Dependencies.get_models_service)
):
    """执行预测"""
    result = models_service.predict(
        request.model_key,
        request.input_values,
        request.session_id
    )
    if result is None:
        raise HTTPException(status_code=404, detail="模型不存在或预测失败")
    return result


@router.delete("/models/{model_key}")
async def delete_model(
    model_key: str,
    session_id: str,
    models_service: ModelsService = Depends(Dependencies.get_models_service)
):
    """删除模型"""
    success = models_service.delete_model(model_key, session_id)
    if not success:
        raise HTTPException(status_code=404, detail="模型不存在")
    return {"message": "模型已删除"}


@router.get("/models/alert/rules")
async def get_alert_rules(
    models_service: ModelsService = Depends(Dependencies.get_models_service)
):
    """获取预警规则"""
    return models_service.get_alert_rules()


@router.post("/models/alert/check")
async def check_alert(
    data: Dict[str, Any],
    models_service: ModelsService = Depends(Dependencies.get_models_service)
):
    """检查预警"""
    return models_service.check_alert(data)


@router.get("/models/train/status/{task_id}", response_model=TrainStatusResponse)
async def get_train_status(
    task_id: str,
    models_service: ModelsService = Depends(Dependencies.get_models_service)
):
    """获取训练进度"""
    status = models_service.get_train_status(task_id)
    return {
        "task_id": task_id,  # 🆕 补上 task_id
        "status": status.get("status", "not_found"),
        "progress": status.get("progress", 0),
        "message": status.get("message", ""),
        "error": status.get("error"),
        "model_key": status.get("model_key"),
        "user_model_name": status.get("user_model_name")
    }