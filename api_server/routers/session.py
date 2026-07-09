"""会话管理路由"""
from typing import List
from fastapi import APIRouter, HTTPException, Depends, Request
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


def get_client_ip(request: Request) -> str:
    """从请求中获取客户端IP"""
    # 直接使用 remote_addr（TCP连接IP）
    if request.client:
        return request.client.host
    return "localhost"


@router.post("/session/create", response_model=SessionCreateResponse)
async def create_session(
    request: Request,
    req: SessionCreateRequest,
    service: SessionService = Depends(Dependencies.get_session_service)
):
    """创建新会话"""
    client_ip = get_client_ip(request)
    # ✅ 设置客户端IP到service实例
    service.set_client_ip(client_ip)
    session_id = service.create_session(
        source_name=req.source_name or "未命名",
        analysis_type=req.analysis_type or "single",
        tables_info=req.tables_info
    )
    return SessionCreateResponse(
        session_id=session_id,
        message="会话创建成功"
    )


@router.get("/session/list", response_model=ProjectListResponse)
async def list_projects(
    request: Request,
    service: SessionService = Depends(Dependencies.get_session_service)
):
    """获取最近项目列表"""
    client_ip = get_client_ip(request)
    # ✅ 设置客户端IP到service实例
    service.set_client_ip(client_ip)
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
    request: Request,
    session_id: str,
    service: SessionService = Depends(Dependencies.get_session_service)
):
    """删除会话"""
    client_ip = get_client_ip(request)
    service.set_client_ip(client_ip)
    success = service.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"message": "会话已删除"}

@router.get("/session/{session_id}/recommended_questions")
async def get_recommended_questions(
    session_id: str,
    service: SessionService = Depends(Dependencies.get_session_service)
):
    """获取推荐问题"""
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return service.get_recommended_questions(session_id)

