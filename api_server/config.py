"""配置管理"""
import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置"""

    # 应用
    APP_NAME: str = "AutoStat API"
    APP_VERSION: str = "0.2.0"
    DEBUG: bool = True

    # 服务器
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",  # Vite 开发服务器
        "http://10.17.181.188:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://localhost:8080",  # 生产环境
        "*"  # 开发环境允许所有
    ]

    # 数据目录
    DATA_DIR: Path = Path.home() / ".autostat" / "data"
    PROJECTS_DIR: Path = Path.home() / ".autostat" / "projects"
    TEMP_DIR: Path = Path("/tmp") / "autostat"

    # 文件限制
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: List[str] = [".csv", ".xlsx", ".xls", ".json", ".txt"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# 确保目录存在
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)