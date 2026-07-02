"""分析报告路由"""
from fastapi import APIRouter, HTTPException, Depends

from api_server.dependencies import Dependencies
from api_server.services.report_service import ReportService
from api_server.services.session_service import SessionService
from api_server.services.data_service import DataService

router = APIRouter()


@router.get("/report/{session_id}")
async def get_report(
        session_id: str,
        report_service: ReportService = Depends(Dependencies.get_report_service),
        session_service: SessionService = Depends(Dependencies.get_session_service),
        data_service: DataService = Depends(Dependencies.get_data_service)
):
    """获取完整报告"""
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 获取分析结果
    analysis_result = session.get("analysis_result")
    if not analysis_result:
        raise HTTPException(status_code=400, detail="分析尚未完成")

    return report_service.get_full_report(analysis_result)


@router.get("/report/{session_id}/summary")
async def get_report_summary(
        session_id: str,
        report_service: ReportService = Depends(Dependencies.get_report_service),
        session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """获取报告摘要（核心结论）"""
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    analysis_result = session.get("analysis_result")
    if not analysis_result:
        raise HTTPException(status_code=400, detail="分析尚未完成")

    return report_service.get_summary(analysis_result)


@router.get("/report/{session_id}/insights")
async def get_report_insights(
        session_id: str,
        report_service: ReportService = Depends(Dependencies.get_report_service),
        session_service: SessionService = Depends(Dependencies.get_session_service)
):
    """获取智能解读"""
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    analysis_result = session.get("analysis_result")
    if not analysis_result:
        raise HTTPException(status_code=400, detail="分析尚未完成")

    return report_service.get_insights(analysis_result)