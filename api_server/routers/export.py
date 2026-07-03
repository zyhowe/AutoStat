"""导出路由"""
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import Response, JSONResponse

from api_server.dependencies import Dependencies
from api_server.services.session_service import SessionService
from api_server.services.report_service import clean_nan

router = APIRouter()


@router.get("/export/{session_id}")
async def export_report(
    session_id: str,
    format: str = Query("html", regex="^(html|json)$"),
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """导出报告（支持 HTML 和 JSON）"""
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    if format == "html":
        # 🆕 直接返回缓存的 HTML（分析完成时已生成）
        html_content = session_service.get_html(session_id)
        if not html_content:
            # 如果没有缓存，尝试从 analyzer 生成
            analyzer = session_service.get_analyzer(session_id)
            if analyzer:
                from autostat.reporter import Reporter
                reporter = Reporter(analyzer)
                html_content = reporter.to_html()
                session_service.save_html(session_id, html_content)

        if not html_content:
            raise HTTPException(status_code=404, detail="HTML 报告不存在，请重新分析")

        return Response(
            content=html_content,
            media_type="text/html",
            headers={
                "Content-Disposition": f"attachment; filename=report_{session_id}.html"
            }
        )

    elif format == "json":
        analysis_result = session.get("analysis_result")
        if not analysis_result:
            raise HTTPException(status_code=400, detail="分析尚未完成")
        cleaned_result = clean_nan(analysis_result)
        return JSONResponse(content=cleaned_result)


@router.get("/export/{session_id}/log")
async def export_log(
    session_id: str,
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """导出分析日志"""
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    log_content = session_service.get_log(session_id)
    if not log_content:
        raise HTTPException(status_code=404, detail="日志不存在")

    return Response(
        content=log_content,
        media_type="text/plain",
        headers={
            "Content-Disposition": f"attachment; filename=analysis_log_{session_id}.txt"
        }
    )