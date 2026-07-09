"""分析执行路由"""
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request
from pydantic import BaseModel

from api_server.dependencies import Dependencies
from api_server.services.analysis_service import AnalysisService
from api_server.services.session_service import SessionService
from api_server.schemas.analysis import AnalysisRequest, AnalysisResponse, AnalysisStatus
from api_server.routers.session import get_client_ip

router = APIRouter()

# 存储任务状态（生产环境应使用 Redis）
task_status = {}


@router.post("/analysis/run", response_model=AnalysisResponse)
async def run_analysis(
    request: Request,
    analysis_request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    analysis_service: AnalysisService = Depends(Dependencies.get_analysis_service),
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """执行分析"""
    # 验证会话
    session = session_service.get_session(analysis_request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 获取文件路径
    file_info = session_service.get_file(analysis_request.session_id)
    if not file_info:
        raise HTTPException(status_code=400, detail="会话没有关联的数据文件")

    # 生成任务ID
    task_id = f"task_{analysis_request.session_id}"

    # 初始化任务状态
    task_status[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "任务已提交"
    }

    # 获取客户端IP，设置到 analysis_service 实例
    client_ip = get_client_ip(request)
    analysis_service.set_client_ip(client_ip)

    # 后台执行分析（传递 include_html 参数）
    background_tasks.add_task(
        analysis_service.run_analysis,
        analysis_request.session_id,
        file_info["path"],
        analysis_request.variable_types or {},
        task_id,
        analysis_request.include_html  # ✅ 新增传递
    )

    return AnalysisResponse(
        task_id=task_id,
        session_id=analysis_request.session_id,
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