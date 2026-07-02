"""AI对话路由"""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from api_server.dependencies import Dependencies
from api_server.services.session_service import SessionService
from api_server.services.chat_service import ChatService

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str
    question: str
    context: Optional[List[str]] = ["json_result"]


class ChatResponse(BaseModel):
    answer: str


@router.post("/chat")
async def chat(
        request: ChatRequest,
        chat_service: ChatService = Depends(Dependencies.get_chat_service),
        session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """AI对话（非流式）"""
    # 验证会话
    session = session_service.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 获取分析结果
    analysis_result = session.get("analysis_result")
    if not analysis_result:
        raise HTTPException(status_code=400, detail="请先完成数据分析")

    answer = chat_service.chat(
        request.session_id,
        request.question,
        analysis_result,
        request.context
    )

    return ChatResponse(answer=answer)


@router.post("/chat/stream")
async def chat_stream(
        request: ChatRequest,
        chat_service: ChatService = Depends(Dependencies.get_chat_service),
        session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """AI对话（流式）"""
    # 验证会话
    session = session_service.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    analysis_result = session.get("analysis_result")
    if not analysis_result:
        raise HTTPException(status_code=400, detail="请先完成数据分析")

    def generate():
        for chunk in chat_service.chat_stream(
                request.session_id,
                request.question,
                analysis_result,
                request.context
        ):
            yield f"data: {json.dumps({'content': chunk})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@router.get("/chat/scenarios")
async def get_scenarios(
        session_id: str,
        chat_service: ChatService = Depends(Dependencies.get_chat_service),
        session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """获取场景推荐"""
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    analysis_result = session.get("analysis_result")
    if not analysis_result:
        raise HTTPException(status_code=400, detail="请先完成数据分析")

    return chat_service.get_scenarios(analysis_result)


@router.get("/chat/recommended_questions")
async def get_recommended_questions(
        session_id: str,
        chat_service: ChatService = Depends(Dependencies.get_chat_service),
        session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """获取推荐问题"""
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    analysis_result = session.get("analysis_result")
    if not analysis_result:
        raise HTTPException(status_code=400, detail="请先完成数据分析")

    return chat_service.get_recommended_questions(analysis_result)