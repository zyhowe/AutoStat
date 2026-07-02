# api_server/schemas/__init__.py

from api_server.schemas.session import (
    SessionCreateRequest,
    SessionCreateResponse,
    SessionInfo,
    ProjectListResponse
)
from api_server.schemas.data import (
    DataPreviewResponse,
    DataUploadResponse
)
from api_server.schemas.analysis import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisStatus
)
from api_server.schemas.report import (
    ReportResponse,
    SummaryResponse,
    InsightsResponse
)
# 🆕 添加 models 导出
from api_server.schemas.models import (
    TrainRequest,
    TrainResponse,
    PredictRequest,
    PredictResponse,
    TrainStatusResponse
)