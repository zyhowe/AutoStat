"""质量报告路由"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from api_server.dependencies import Dependencies
from api_server.services.quality_service import QualityService
from api_server.services.session_service import SessionService
from api_server.services.data_service import DataService

router = APIRouter()


@router.get("/quality/{session_id}")
async def get_quality_report(
        session_id: str,
        quality_service: QualityService = Depends(Dependencies.get_quality_service),
        session_service: SessionService = Depends(Dependencies.get_session_service),
        data_service: DataService = Depends(Dependencies.get_data_service)
):
    """获取质量报告"""
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 获取数据
    file_info = session_service.get_data_path(session_id)
    if not file_info:
        raise HTTPException(status_code=400, detail="会话没有关联的数据文件")

    df = data_service.load_file(file_info)
    variable_types = session.get("variable_types", {})

    if not variable_types:
        variable_types = data_service.infer_types(df)

    # 计算质量评分
    quality_result = quality_service.get_score(df, variable_types)

    return quality_result