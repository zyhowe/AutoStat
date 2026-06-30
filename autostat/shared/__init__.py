"""
跨模块共享工具
"""

from autostat.shared.schemas import (
    # 质量监控
    QualityScore,
    QualitySnapshot,
    QualityTrendResult,
    QualityMonitorResult,
    # 决策支持
    Anomaly,
    RootCause,
    RootCauseResult,
    ActionSuggestion,
    ActionRecord,
    TrackerResult,
    DecisionResult,
    # 预测与预警
    ForecastResult,
    AlertRule,
    AlertEvent,
    ForecastSnapshot,
    ForecastMonitorResult,
    # 自助探索
    ChartRecommendation,
    StorySection,
    Story,
    Widget,
    Dashboard,
    NL2SQLResult,
    # 通用
    PaginatedResult,
    ApiResponse,
    # 工厂函数
    create_quality_score,
    create_anomaly,
    create_action_suggestion,
    create_alert_event,
    create_forecast_result,
    create_dashboard,
    # 转换函数
    quality_score_to_dict,
    dict_to_quality_score,
    anomaly_to_dict,
    dict_to_anomaly,
    action_suggestion_to_dict,
    dict_to_action_suggestion,
    alert_event_to_dict,
    dict_to_alert_event,
)

from autostat.shared.storage import SharedStorage
from autostat.shared.utils import (
    safe_divide,
    calculate_confidence_interval,
    detect_outliers_iqr,
    normalize_data,
    get_timestamp,
    truncate_text,
    safe_dict_get,
    group_by,
    sorted_by,
    date_range,
    safe_round,
    is_numeric,
    flatten_dict,
)

__all__ = [
    # Schemas - 质量监控
    "QualityScore",
    "QualitySnapshot",
    "QualityTrendResult",
    "QualityMonitorResult",
    # Schemas - 决策支持
    "Anomaly",
    "RootCause",
    "RootCauseResult",
    "ActionSuggestion",
    "ActionRecord",
    "TrackerResult",
    "DecisionResult",
    # Schemas - 预测与预警
    "ForecastResult",
    "AlertRule",
    "AlertEvent",
    "ForecastSnapshot",
    "ForecastMonitorResult",
    # Schemas - 自助探索
    "ChartRecommendation",
    "StorySection",
    "Story",
    "Widget",
    "Dashboard",
    "NL2SQLResult",
    # Schemas - 通用
    "PaginatedResult",
    "ApiResponse",
    # 工厂函数
    "create_quality_score",
    "create_anomaly",
    "create_action_suggestion",
    "create_alert_event",
    "create_forecast_result",
    "create_dashboard",
    # 转换函数
    "quality_score_to_dict",
    "dict_to_quality_score",
    "anomaly_to_dict",
    "dict_to_anomaly",
    "action_suggestion_to_dict",
    "dict_to_action_suggestion",
    "alert_event_to_dict",
    "dict_to_alert_event",
    # Storage
    "SharedStorage",
    # Utils
    "safe_divide",
    "calculate_confidence_interval",
    "detect_outliers_iqr",
    "normalize_data",
    "get_timestamp",
    "truncate_text",
    "safe_dict_get",
    "group_by",
    "sorted_by",
    "date_range",
    "safe_round",
    "is_numeric",
    "flatten_dict",
]