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
    """
    执行分析

    【修改】不再从请求体接收 variable_types，改为从 session 缓存读取
    用户在前端调整字段类型后，通过 /data/field_types/update 保存到缓存
    """
    # 验证会话
    session = session_service.get_session(analysis_request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    # ===== 新增：从缓存读取字段类型 =====
    field_types_cache = session_service.get_field_types_cache(analysis_request.session_id)
    print(f"📋 从缓存读取字段类型: {len(field_types_cache)} 个表")

    # 如果缓存中有字段类型，合并到 variable_types
    # 注意：这里需要读取每个表的实际字段类型，然后应用缓存覆盖
    variable_types = {}

    # 获取所有表名
    table_names = session_service.get_all_table_names(analysis_request.session_id)
    if not table_names:
        # 如果没有表名，尝试从 files 获取
        file_info = session_service.get_file(analysis_request.session_id)
        if file_info:
            table_names = ["data"]  # 默认表名

    # 对于每个表，从 metadata 中获取原始类型，然后用缓存覆盖
    # 实际上，analysis_service 会从 Parquet 加载数据并重新推断类型
    # 我们只需要把缓存中的类型传给 analysis_service
    # 但 analysis_service.run_analysis 目前只接收一个 variable_types 字典
    # 对于多表，需要按表名组织

    # 简化处理：analysis_service 会从 session 读取缓存
    # 修改 analysis_service 支持从 session 读取

    # 获取文件路径
    file_info = session_service.get_data_path(analysis_request.session_id)
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

    # 获取客户端IP
    client_ip = get_client_ip(request)
    analysis_service.set_client_ip(client_ip)

    # ===== 修改：传递 session_id 和 field_types_cache，而不是 variable_types =====
    background_tasks.add_task(
        analysis_service.run_analysis_from_cache,
        analysis_request.session_id,
        file_info,
        field_types_cache,
        task_id,
        analysis_request.include_html
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