"""依赖注入（修改版）"""

from typing import Optional
from fastapi import HTTPException, UploadFile
from pathlib import Path
import shutil
import uuid

from api_server.config import settings
from api_server.services.session_service import SessionService
from api_server.services.data_service import DataService
from api_server.services.analysis_service import AnalysisService
from api_server.services.quality_service import QualityService
from api_server.services.report_service import ReportService
from api_server.services.models_service import ModelsService
from api_server.services.chat_service import ChatService
from api_server.services.config_service import ConfigService
from api_server.services.database_service import DatabaseService
from api_server.services.recommendation_service import RecommendationService
from api_server.services.prediction_agent_service import PredictionAgentService


class Dependencies:
    """依赖注入容器"""

    _session_service = None
    _data_service = None
    _analysis_service = None
    _quality_service = None
    _report_service = None
    _models_service = None
    _chat_service = None
    _config_service = None
    _database_service = None
    _recommendation_service = None
    _prediction_agent_service = None

    @classmethod
    def get_session_service(cls) -> SessionService:
        if cls._session_service is None:
            cls._session_service = SessionService()
        return cls._session_service

    @classmethod
    def get_data_service(cls) -> DataService:
        if cls._data_service is None:
            cls._data_service = DataService()
        return cls._data_service

    @classmethod
    def get_analysis_service(cls) -> AnalysisService:
        if cls._analysis_service is None:
            cls._analysis_service = AnalysisService()
        return cls._analysis_service

    @classmethod
    def get_quality_service(cls) -> QualityService:
        if cls._quality_service is None:
            cls._quality_service = QualityService()
        return cls._quality_service

    @classmethod
    def get_report_service(cls) -> ReportService:
        if cls._report_service is None:
            cls._report_service = ReportService()
        return cls._report_service

    @classmethod
    def get_models_service(cls) -> ModelsService:
        if cls._models_service is None:
            cls._models_service = ModelsService()
        return cls._models_service

    @classmethod
    def get_chat_service(cls) -> ChatService:
        if cls._chat_service is None:
            cls._chat_service = ChatService()
        return cls._chat_service

    @classmethod
    def get_config_service(cls) -> ConfigService:
        if cls._config_service is None:
            cls._config_service = ConfigService()
        return cls._config_service

    @classmethod
    def get_database_service(cls) -> DatabaseService:
        if cls._database_service is None:
            cls._database_service = DatabaseService()
        return cls._database_service

    @classmethod
    def get_recommendation_service(cls) -> RecommendationService:
        if cls._recommendation_service is None:
            cls._recommendation_service = RecommendationService()
        return cls._recommendation_service

    @classmethod
    def get_prediction_agent_service(cls) -> PredictionAgentService:
        if cls._prediction_agent_service is None:
            cls._prediction_agent_service = PredictionAgentService()
        return cls._prediction_agent_service

    @classmethod
    def require_session(cls, session_id: str):
        service = cls.get_session_service()
        session = service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        return session

    @classmethod
    def save_upload_file(cls, file: UploadFile) -> Path:
        ext = Path(file.filename).suffix.lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式: {ext}，支持的格式: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )

        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        if file_size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"文件大小 {file_size / 1024 / 1024:.1f}MB 超过限制 {settings.MAX_FILE_SIZE / 1024 / 1024:.0f}MB"
            )

        file_id = str(uuid.uuid4())
        file_path = settings.TEMP_DIR / f"{file_id}{ext}"

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        return file_path