"""会话管理路由"""
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from api_server.dependencies import Dependencies
from api_server.services.session_service import SessionService
from api_server.schemas.session import (
    SessionCreateRequest,
    SessionCreateResponse,
    SessionInfo,
    ProjectListResponse
)

router = APIRouter()


@router.post("/session/create", response_model=SessionCreateResponse)
async def create_session(
    request: SessionCreateRequest,
    service: SessionService = Depends(Dependencies.get_session_service)
):
    """创建新会话"""
    session_id = service.create_session(
        source_name=request.source_name or "未命名",
        analysis_type=request.analysis_type or "single",
        tables_info=request.tables_info
    )
    return SessionCreateResponse(
        session_id=session_id,
        message="会话创建成功"
    )


@router.get("/session/list", response_model=ProjectListResponse)
async def list_projects(
    service: SessionService = Depends(Dependencies.get_session_service)
):
    """获取最近项目列表"""
    projects = service.list_projects()
    return ProjectListResponse(
        total=len(projects),
        projects=projects
    )


@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(
    session_id: str,
    service: SessionService = Depends(Dependencies.get_session_service)
):
    """获取会话信息"""
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    service: SessionService = Depends(Dependencies.get_session_service)
):
    """删除会话"""
    success = service.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"message": "会话已删除"}