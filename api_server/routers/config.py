"""配置管理路由"""
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from api_server.dependencies import Dependencies
from api_server.services.config_service import ConfigService
from api_server.services.database_service import DatabaseService

router = APIRouter()


class DbConfigRequest(BaseModel):
    name: str
    server: str
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    trusted_connection: bool = False


class LlmConfigRequest(BaseModel):
    name: str
    api_base: str
    api_key: str
    model: str
    timeout: int = 60


class TestConnectionRequest(BaseModel):
    api_base: str
    api_key: str
    model: str


@router.get("/config/database")
async def get_db_configs(
    config_service: ConfigService = Depends(Dependencies.get_config_service)
):
    """获取数据库配置列表"""
    return config_service.get_db_configs()


@router.post("/config/database")
async def save_db_config(
    request: DbConfigRequest,
    config_service: ConfigService = Depends(Dependencies.get_config_service)
):
    """保存数据库配置"""
    config = request.model_dump()
    success = config_service.save_db_config(config)
    if not success:
        raise HTTPException(status_code=400, detail="配置名称已存在")
    return {"message": "配置保存成功"}


@router.delete("/config/database/{name}")
async def delete_db_config(
    name: str,
    config_service: ConfigService = Depends(Dependencies.get_config_service)
):
    """删除数据库配置"""
    success = config_service.delete_db_config(name)
    if not success:
        raise HTTPException(status_code=404, detail="配置不存在")
    return {"message": "配置已删除"}


# ✅ 新增：测试数据库连接
@router.post("/config/database/test")
async def test_database_connection(
    request: DbConfigRequest,
    database_service: DatabaseService = Depends(Dependencies.get_database_service)
):
    """测试数据库连接"""
    success, message = database_service.test_connection(request.model_dump())
    return {"success": success, "message": message}


@router.get("/config/llm")
async def get_llm_configs(
    config_service: ConfigService = Depends(Dependencies.get_config_service)
):
    """获取大模型配置列表"""
    return config_service.get_llm_configs()


@router.post("/config/llm")
async def save_llm_config(
    request: LlmConfigRequest,
    config_service: ConfigService = Depends(Dependencies.get_config_service)
):
    """保存大模型配置"""
    config = request.model_dump()
    success = config_service.save_llm_config(config)
    if not success:
        raise HTTPException(status_code=400, detail="配置名称已存在")
    return {"message": "配置保存成功"}


@router.delete("/config/llm/{name}")
async def delete_llm_config(
    name: str,
    config_service: ConfigService = Depends(Dependencies.get_config_service)
):
    """删除大模型配置"""
    success = config_service.delete_llm_config(name)
    if not success:
        raise HTTPException(status_code=404, detail="配置不存在")
    return {"message": "配置已删除"}


@router.post("/config/llm/test")
async def test_llm_connection(
    request: TestConnectionRequest,
    config_service: ConfigService = Depends(Dependencies.get_config_service)
):
    """测试大模型连接"""
    success, message = config_service.test_llm_connection(request.model_dump())
    return {"success": success, "message": message}