"""AI对话路由（修改版）"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from api_server.dependencies import Dependencies
from api_server.services.session_service import SessionService
from api_server.services.chat_service import ChatService
from api_server.services.prediction_agent_service import PredictionAgentService

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str
    question: str
    context: Optional[List[str]] = ["json_result"]
    context_data: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    answer: str


# ==================== 普通对话 ====================

@router.post("/chat")
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(Dependencies.get_chat_service),
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """AI对话（非流式）"""
    session = session_service.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    analysis_result = session.get("analysis_result")
    if not analysis_result:
        raise HTTPException(status_code=400, detail="请先完成数据分析")

    answer = chat_service.chat(
        request.session_id,
        request.question,
        analysis_result,
        request.context,
        request.context_data
    )

    return ChatResponse(answer=answer)


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    chat_service: ChatService = Depends(Dependencies.get_chat_service),
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """AI对话（流式）"""
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
            request.context,
            request.context_data
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


# ==================== 推荐问题 ====================

@router.get("/chat/recommended_questions")
async def get_recommended_questions(
    session_id: str,
    scene: Optional[str] = None,
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """
    获取推荐问题

    Args:
        session_id: 会话ID
        scene: 场景名称（可选），如 report_summary, data_overview, quality, data_validation, pattern_discovery, smart_prediction

    Returns:
        {
            "report_summary": [...],
            "data_overview": [...],
            ...
        }
        或指定场景的问题列表
    """
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    if scene:
        questions = session_service.get_recommended_questions_by_scene(session_id, scene)
        return questions
    else:
        questions = session_service.get_recommended_questions(session_id)
        return questions or {}


# ==================== 智能预测Agent ====================

class PredictionRequest(BaseModel):
    session_id: str
    question: str


@router.post("/chat/prediction")
async def prediction_agent(
    request: PredictionRequest,
    agent_service: PredictionAgentService = Depends(Dependencies.get_prediction_agent_service),
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """智能预测Agent - 自然语言预测"""
    session = session_service.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    result = agent_service.process(request.session_id, request.question)

    return result


@router.post("/chat/prediction/stream")
async def prediction_agent_stream(
    request: PredictionRequest,
    agent_service: PredictionAgentService = Depends(Dependencies.get_prediction_agent_service),
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """智能预测Agent流式版本"""
    session = session_service.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    def generate():
        result = agent_service.process(request.session_id, request.question)

        if result.get('success'):
            content = result.get('result', '')
            for char in content:
                yield f"data: {json.dumps({'content': char})}\n\n"
            yield f"data: {json.dumps({'done': True, 'data': result.get('data'), 'model_used': result.get('model_used'), 'confidence': result.get('confidence')})}\n\n"
        else:
            yield f"data: {json.dumps({'content': f'❌ {result.get("error", "预测失败")}'})}\n\n"
            yield f"data: {json.dumps({'done': True, 'error': result.get('error')})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


# ==================== 场景推荐 ====================

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


@router.get("/chat/recommended_questions_old")
async def get_recommended_questions_old(
    session_id: str,
    chat_service: ChatService = Depends(Dependencies.get_chat_service),
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """获取推荐问题（旧版，保留兼容）"""
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    analysis_result = session.get("analysis_result")
    if not analysis_result:
        raise HTTPException(status_code=400, detail="请先完成数据分析")

    return chat_service.get_recommended_questions(analysis_result)