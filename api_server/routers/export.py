"""导出路由"""
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import FileResponse, JSONResponse, Response
import io

from api_server.dependencies import Dependencies
from api_server.services.report_service import ReportService
from api_server.services.session_service import SessionService

router = APIRouter()


@router.get("/export/{session_id}")
async def export_report(
    session_id: str,
    format: str = Query("html", regex="^(html|json|excel)$"),
    report_service: ReportService = Depends(Dependencies.get_report_service),
    session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """导出报告"""
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    analysis_result = session.get("analysis_result")
    if not analysis_result:
        raise HTTPException(status_code=400, detail="分析尚未完成")

    analyzer = session_service.get_analyzer(session_id)
    if not analyzer:
        raise HTTPException(status_code=400, detail="分析器不存在，请重新分析")

    if format == "html":
        html_content = report_service.get_html(analyzer)
        # 🆕 使用 Response 返回字符串内容，而不是 FileResponse
        return Response(
            content=html_content,
            media_type="text/html",
            headers={
                "Content-Disposition": f"attachment; filename=report_{session_id}.html"
            }
        )
    elif format == "json":
        return JSONResponse(content=analysis_result)
    elif format == "excel":
        # TODO: 实现 Excel 导出
        raise HTTPException(status_code=501, detail="Excel 导出功能开发中")