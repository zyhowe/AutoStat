"""分析执行路由（修改版）"""

from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel

from api_server.dependencies import Dependencies
from api_server.services.analysis_service import AnalysisService
from api_server.services.session_service import SessionService
from api_server.services.recommendation_service import RecommendationService
from api_server.schemas.analysis import AnalysisRequest, AnalysisResponse, AnalysisStatus

router = APIRouter()

# 存储任务状态
task_status = {}


@router.post("/analysis/run", response_model=AnalysisResponse)
async def run_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    analysis_service: AnalysisService = Depends(Dependencies.get_analysis_service),
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """执行分析"""
    session = session_service.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    file_info = session_service.get_file(request.session_id)
    if not file_info:
        raise HTTPException(status_code=400, detail="会话没有关联的数据文件")

    task_id = f"task_{request.session_id}"

    task_status[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "任务已提交"
    }

    background_tasks.add_task(
        analysis_service.run_analysis,
        request.session_id,
        file_info["path"],
        request.variable_types or {},
        task_id
    )

    return AnalysisResponse(
        task_id=task_id,
        session_id=request.session_id,
        status="pending",
        message="分析任务已提交"
    )


@router.get("/analysis/status/{task_id}", response_model=AnalysisStatus)
async def get_analysis_status(task_id: str):
    """获取分析进度"""
    status = task_status.get(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="任务不存在")
    return AnalysisStatus(
        task_id=task_id,
        status=status.get("status", "pending"),
        progress=status.get("progress", 0),
        message=status.get("message", ""),
        error=status.get("error")
    )