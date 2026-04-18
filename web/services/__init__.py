"""业务服务层"""
from web.services.cache_service import CacheService
from web.services.file_service import FileService
from web.services.analysis_service import AnalysisService
from web.services.model_training_service import (
    get_model_display_name,
    generate_model_name,
    get_available_features,
    get_model_recommendations_from_json,
    get_models_by_task_type,
    get_model_params,
    execute_training,
    save_trained_model,
    list_saved_models,
    delete_model,
    load_model_for_inference,
    execute_inference
)

__all__ = [
    'CacheService',
    'FileService',
    'AnalysisService',
    'get_model_display_name',
    'generate_model_name',
    'get_available_features',
    'get_model_recommendations_from_json',
    'get_models_by_task_type',
    'get_model_params',
    'execute_training',
    'save_trained_model',
    'list_saved_models',
    'delete_model',
    'load_model_for_inference',
    'execute_inference'
]