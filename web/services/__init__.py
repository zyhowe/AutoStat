# web/services/__init__.py

"""业务服务层"""
from web.services.cache_service import CacheService
from web.services.file_service import FileService
from web.services.analysis_service import AnalysisService
from web.services.session_service import SessionService
from web.services.storage_service import StorageService
from web.services.model_training_service import (
    get_model_display_name,
    generate_model_name,
    get_available_features,
    get_model_recommendations_from_json,
    get_models_by_task_type,
    get_model_params,
    execute_training,
    list_saved_models,
    delete_model,
    load_model_for_inference,
    execute_inference
)
from web.services.agent_service import (
    AgentService,
    get_agent_tools_description,
    build_data_context,
    build_agent_system_prompt,
    process_agent_response
)

# 新增服务
from web.services.insight_service import InsightService
from web.services.recommendation_service import RecommendationService
from web.services.feature_flags import FeatureFlags

__all__ = [
    # 原有
    'CacheService',
    'FileService',
    'AnalysisService',
    'SessionService',
    'StorageService',
    'get_model_display_name',
    'generate_model_name',
    'get_available_features',
    'get_model_recommendations_from_json',
    'get_models_by_task_type',
    'get_model_params',
    'execute_training',
    'list_saved_models',
    'delete_model',
    'load_model_for_inference',
    'execute_inference',
    # Agent服务
    'AgentService',
    'get_agent_tools_description',
    'build_data_context',
    'build_agent_system_prompt',
    'process_agent_response',
    # 新增
    'InsightService',
    'RecommendationService',
    'FeatureFlags'
]