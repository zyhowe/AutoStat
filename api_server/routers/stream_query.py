"""流式查询路由 - 全量数据解码"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any
import json

from api_server.dependencies import Dependencies
from api_server.services.session_service import SessionService
from api_server.services.stream_query_service import StreamQueryService

router = APIRouter()


class StreamQueryRequest(BaseModel):
    """流式查询请求"""
    session_id: str
    context: Dict[str, Any]
    batch_size: int = 100
    max_rows: int = 10000


@router.post("/data/stream-query")
async def stream_query(
    request: StreamQueryRequest,
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """
    流式查询原始数据
    """
    # 验证会话
    session = session_service.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 获取数据库配置
    db_config = StreamQueryService.get_connection_from_session(session)
    if not db_config:
        raise HTTPException(
            status_code=400,
            detail="当前会话没有关联的数据库配置，无法执行流式查询。请确认数据源为数据库加载方式。"
        )

    # 构建 SQL（传入 session 以获取表名）
    context = request.context
    sql, description = StreamQueryService.build_sql_from_context(context, session)

    if not sql:
        raise HTTPException(
            status_code=400,
            detail="无法构建有效的 SQL 查询，请检查追溯上下文是否正确。"
        )

    # 安全校验
    sql_upper = sql.upper()
    dangerous_keywords = ['DELETE', 'UPDATE', 'INSERT', 'DROP', 'TRUNCATE', 'ALTER',
                          'EXEC', 'EXECUTE', 'CREATE', 'GRANT', 'REVOKE', 'MERGE']
    for keyword in dangerous_keywords:
        if f' {keyword} ' in f' {sql_upper} ' or sql_upper.startswith(f'{keyword} '):
            raise HTTPException(
                status_code=400,
                detail=f"SQL 包含危险关键字: {keyword}，仅允许 SELECT 查询"
            )

    # 生成流式响应
    def generate():
        # 先发送描述信息
        yield json.dumps({
            "type": "info",
            "description": description,
            "sql": sql
        }, ensure_ascii=False) + '\n'

        # 执行流式查询
        yield from StreamQueryService.stream_query(
            db_config=db_config,
            sql=sql,
            batch_size=request.batch_size,
            max_rows=request.max_rows
        )

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.post("/data/stream-query/preview")
async def stream_preview(
    request: StreamQueryRequest,
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """预览流式查询（仅返回前 10 行）"""
    request.max_rows = 10
    return await stream_query(request, session_service)