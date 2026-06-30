"""
共享数据模型定义

所有跨模块使用的数据模型统一在此定义
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime


# ==================== 质量监控 ====================

@dataclass
class QualityScore:
    """质量评分结果"""
    timestamp: str
    table_name: str
    overall_score: float
    grade: str
    grade_icon: str
    dimensions: Dict[str, float]
    field_scores: Dict[str, Dict[str, float]]
    alerts: List[Dict] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualitySnapshot:
    """质量快照"""
    timestamp: str
    overall_score: float
    dimensions: Dict[str, float]
    alert_count: int


@dataclass
class QualityTrendResult:
    """质量趋势分析结果"""
    direction: str  # "up", "down", "stable"
    change_pct: float
    is_anomaly: bool
    anomaly_type: Optional[str]  # "spike", "drop"
    message: str


@dataclass
class QualityMonitorResult:
    """质量监控结果"""
    current: QualitySnapshot
    history: List[QualitySnapshot]
    trend: QualityTrendResult
    anomalies: List[Dict]
    summary: str


# ==================== 决策支持 ====================

@dataclass
class Anomaly:
    """异常事件"""
    id: str
    type: str  # "spike", "drop", "trend_reversal", "distribution_shift", "rule_violation", "outlier"
    severity: str  # "critical", "high", "medium", "low"
    target: str
    value: Any
    expected: Any
    message: str
    timestamp: str
    data: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)


@dataclass
class RootCause:
    """根因结论"""
    description: str
    confidence: float
    evidence: List[Dict[str, Any]]
    dimensions: Dict[str, str]


@dataclass
class RootCauseResult:
    """根因分析结果"""
    anomaly: Dict[str, Any]
    root_causes: List[RootCause]
    summary: str
    drill_path: List[str]


@dataclass
class ActionSuggestion:
    """行动建议"""
    id: str
    title: str
    description: str
    priority: str  # "高", "中", "低"
    difficulty: str  # "低", "中", "高"
    expected_effect: str
    confidence: float
    steps: List[str]
    tags: List[str] = field(default_factory=list)


@dataclass
class ActionRecord:
    """行动记录"""
    id: str
    suggestion_id: str
    title: str
    status: str  # "pending", "in_progress", "completed", "cancelled", "archived"
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[str] = None
    effect_metrics: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class TrackerResult:
    """追踪结果"""
    action: ActionRecord
    before: Dict[str, Any]
    after: Dict[str, Any]
    effect: str  # "improved", "no_change", "worsened"
    effect_pct: float
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionResult:
    """决策结果 - 组合异常、根因和建议"""
    anomaly: Anomaly
    root_causes: List[RootCause]
    suggestions: List[ActionSuggestion]
    summary: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ==================== 预测与预警 ====================

@dataclass
class ForecastResult:
    """预测结果"""
    target: str
    model_name: str
    values: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    confidence: float
    metrics: Dict[str, float]
    periods: int
    timestamp: str


@dataclass
class AlertRule:
    """预警规则"""
    id: str
    name: str
    condition: str  # 条件表达式（存储用）
    level: str  # "info", "warning", "error", "critical"
    message_template: str
    enabled: bool = True
    cooldown: int = 3600  # 冷却时间（秒）


@dataclass
class AlertEvent:
    """预警事件"""
    id: str
    rule_id: str
    level: str  # "info", "warning", "error", "critical"
    title: str
    message: str
    data: Dict[str, Any]
    triggered_at: str
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[str] = None


@dataclass
class ForecastSnapshot:
    """预测快照（用于监控）"""
    timestamp: str
    target: str
    actual: float
    predicted: float
    error: float
    error_pct: float


@dataclass
class ForecastMonitorResult:
    """预测监控结果"""
    status: str  # "good", "warning", "critical"
    message: str
    metrics: Dict[str, float]
    recent_errors: List[float]
    drift_detected: bool


# ==================== 自助探索 ====================

@dataclass
class ChartRecommendation:
    """图表推荐结果"""
    chart_type: str  # "bar", "line", "pie", "scatter", "histogram", "box", "heatmap", etc.
    config: Dict[str, Any]
    title: str
    description: str
    confidence: float
    data: List[Dict[str, Any]]


@dataclass
class StorySection:
    """故事章节"""
    title: str
    content: str
    chart_ref: Optional[str] = None
    data_ref: Optional[str] = None


@dataclass
class Story:
    """生成的故事"""
    title: str
    summary: str
    sections: List[StorySection]
    key_findings: List[str]
    recommendations: List[str]
    generated_at: str


@dataclass
class Widget:
    """仪表板组件"""
    id: str
    type: str  # "metric", "chart", "table", "filter"
    config: Dict[str, Any]
    position: Dict[str, int]  # {"row": 0, "col": 0, "width": 4, "height": 2}


@dataclass
class Dashboard:
    """仪表板"""
    id: str
    title: str
    description: str
    widgets: List[Widget]
    filters: List[Dict[str, Any]]
    layout: str  # "grid", "columns"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class NL2SQLResult:
    """自然语言转SQL结果"""
    sql: str
    parsed: Dict[str, Any]  # 解析后的意图
    confidence: float
    error: Optional[str] = None


# ==================== 通用 ====================

@dataclass
class PaginatedResult:
    """分页结果"""
    items: List[Any]
    total: int
    page: int
    page_size: int
    has_more: bool


@dataclass
class ApiResponse:
    """API响应"""
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ==================== 工厂函数 ====================

def create_quality_score(
    table_name: str,
    overall_score: float,
    grade: str,
    grade_icon: str,
    dimensions: Dict[str, float],
    field_scores: Dict[str, Dict[str, float]],
    alerts: Optional[List[Dict]] = None,
    details: Optional[Dict[str, Any]] = None
) -> QualityScore:
    """创建质量评分对象"""
    return QualityScore(
        timestamp=datetime.now().isoformat(),
        table_name=table_name,
        overall_score=overall_score,
        grade=grade,
        grade_icon=grade_icon,
        dimensions=dimensions,
        field_scores=field_scores,
        alerts=alerts or [],
        details=details or {}
    )


def create_anomaly(
    anomaly_type: str,
    severity: str,
    target: str,
    value: Any,
    expected: Any,
    message: str,
    evidence: Optional[List[str]] = None,
    data: Optional[Dict[str, Any]] = None
) -> Anomaly:
    """创建异常对象"""
    return Anomaly(
        id=f"anomaly_{int(datetime.now().timestamp())}",
        type=anomaly_type,
        severity=severity,
        target=target,
        value=value,
        expected=expected,
        message=message,
        timestamp=datetime.now().isoformat(),
        evidence=evidence or [],
        data=data or {}
    )


def create_action_suggestion(
    title: str,
    description: str,
    priority: str = "中",
    difficulty: str = "中",
    expected_effect: str = "",
    confidence: float = 0.7,
    steps: Optional[List[str]] = None,
    tags: Optional[List[str]] = None
) -> ActionSuggestion:
    """创建行动建议对象"""
    return ActionSuggestion(
        id=f"suggestion_{int(datetime.now().timestamp())}",
        title=title,
        description=description,
        priority=priority,
        difficulty=difficulty,
        expected_effect=expected_effect,
        confidence=confidence,
        steps=steps or [],
        tags=tags or []
    )


def create_alert_event(
    rule_id: str,
    level: str,
    title: str,
    message: str,
    data: Dict[str, Any]
) -> AlertEvent:
    """创建预警事件"""
    return AlertEvent(
        id=f"alert_{int(datetime.now().timestamp())}",
        rule_id=rule_id,
        level=level,
        title=title,
        message=message,
        data=data,
        triggered_at=datetime.now().isoformat()
    )


def create_forecast_result(
    target: str,
    model_name: str,
    values: List[float],
    lower_bound: List[float],
    upper_bound: List[float],
    confidence: float = 0.95,
    metrics: Optional[Dict[str, float]] = None,
    periods: int = 12
) -> ForecastResult:
    """创建预测结果对象"""
    return ForecastResult(
        target=target,
        model_name=model_name,
        values=values,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        confidence=confidence,
        metrics=metrics or {},
        periods=periods,
        timestamp=datetime.now().isoformat()
    )


def create_dashboard(
    title: str,
    description: str,
    widgets: List[Widget],
    filters: Optional[List[Dict[str, Any]]] = None,
    layout: str = "grid"
) -> Dashboard:
    """创建仪表板对象"""
    return Dashboard(
        id=f"dashboard_{int(datetime.now().timestamp())}",
        title=title,
        description=description,
        widgets=widgets,
        filters=filters or [],
        layout=layout
    )


# ==================== 工具函数 ====================

def quality_score_to_dict(score: QualityScore) -> Dict[str, Any]:
    """将 QualityScore 转换为字典"""
    return {
        "timestamp": score.timestamp,
        "table_name": score.table_name,
        "overall_score": score.overall_score,
        "grade": score.grade,
        "grade_icon": score.grade_icon,
        "dimensions": score.dimensions,
        "field_scores": score.field_scores,
        "alerts": score.alerts,
        "details": score.details
    }


def dict_to_quality_score(data: Dict[str, Any]) -> QualityScore:
    """从字典创建 QualityScore"""
    return QualityScore(
        timestamp=data.get("timestamp", datetime.now().isoformat()),
        table_name=data.get("table_name", "unknown"),
        overall_score=data.get("overall_score", 0.0),
        grade=data.get("grade", "未知"),
        grade_icon=data.get("grade_icon", "⚪"),
        dimensions=data.get("dimensions", {}),
        field_scores=data.get("field_scores", {}),
        alerts=data.get("alerts", []),
        details=data.get("details", {})
    )


def anomaly_to_dict(anomaly: Anomaly) -> Dict[str, Any]:
    """将 Anomaly 转换为字典"""
    return {
        "id": anomaly.id,
        "type": anomaly.type,
        "severity": anomaly.severity,
        "target": anomaly.target,
        "value": anomaly.value,
        "expected": anomaly.expected,
        "message": anomaly.message,
        "timestamp": anomaly.timestamp,
        "data": anomaly.data,
        "evidence": anomaly.evidence
    }


def dict_to_anomaly(data: Dict[str, Any]) -> Anomaly:
    """从字典创建 Anomaly"""
    return Anomaly(
        id=data.get("id", f"anomaly_{int(datetime.now().timestamp())}"),
        type=data.get("type", "unknown"),
        severity=data.get("severity", "medium"),
        target=data.get("target", ""),
        value=data.get("value"),
        expected=data.get("expected"),
        message=data.get("message", ""),
        timestamp=data.get("timestamp", datetime.now().isoformat()),
        data=data.get("data", {}),
        evidence=data.get("evidence", [])
    )


def action_suggestion_to_dict(suggestion: ActionSuggestion) -> Dict[str, Any]:
    """将 ActionSuggestion 转换为字典"""
    return {
        "id": suggestion.id,
        "title": suggestion.title,
        "description": suggestion.description,
        "priority": suggestion.priority,
        "difficulty": suggestion.difficulty,
        "expected_effect": suggestion.expected_effect,
        "confidence": suggestion.confidence,
        "steps": suggestion.steps,
        "tags": suggestion.tags
    }


def dict_to_action_suggestion(data: Dict[str, Any]) -> ActionSuggestion:
    """从字典创建 ActionSuggestion"""
    return ActionSuggestion(
        id=data.get("id", f"suggestion_{int(datetime.now().timestamp())}"),
        title=data.get("title", ""),
        description=data.get("description", ""),
        priority=data.get("priority", "中"),
        difficulty=data.get("difficulty", "中"),
        expected_effect=data.get("expected_effect", ""),
        confidence=data.get("confidence", 0.7),
        steps=data.get("steps", []),
        tags=data.get("tags", [])
    )


def alert_event_to_dict(event: AlertEvent) -> Dict[str, Any]:
    """将 AlertEvent 转换为字典"""
    return {
        "id": event.id,
        "rule_id": event.rule_id,
        "level": event.level,
        "title": event.title,
        "message": event.message,
        "data": event.data,
        "triggered_at": event.triggered_at,
        "acknowledged": event.acknowledged,
        "resolved": event.resolved,
        "resolved_at": event.resolved_at
    }


def dict_to_alert_event(data: Dict[str, Any]) -> AlertEvent:
    """从字典创建 AlertEvent"""
    return AlertEvent(
        id=data.get("id", f"alert_{int(datetime.now().timestamp())}"),
        rule_id=data.get("rule_id", ""),
        level=data.get("level", "warning"),
        title=data.get("title", ""),
        message=data.get("message", ""),
        data=data.get("data", {}),
        triggered_at=data.get("triggered_at", datetime.now().isoformat()),
        acknowledged=data.get("acknowledged", False),
        resolved=data.get("resolved", False),
        resolved_at=data.get("resolved_at")
    )