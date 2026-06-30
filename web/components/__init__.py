# web/components/__init__.py

"""UI组件模块"""
from web.components.sidebar import render_sidebar
from web.components.data_preparation import render_data_preparation
from web.components.preview_report import render_preview_report
from web.components.model_center import render_model_center
from web.components.ai_assistant import render_ai_assistant
from web.components.results import render_compare_tab
from web.components.quality_dashboard import render_quality_dashboard, render_quality_dashboard_inline

# 原有组件（保留）
from web.components.chat_interface import render_chat_interface, render_recommended_questions
from web.components.scenario_recommendation import render_scenario_recommendation
from web.components.natural_query import render_natural_query
from web.components.sql_generator import render_sql_generator
from web.components.model_training import render_model_training
from web.components.agent_inference import render_agent_inference
from web.components.demo_data import render_demo_section
from web.components.empty_state import render_empty_state
from web.components.value_preview import render_value_preview
from web.components.progress_stage import ProgressStage, render_simple_progress
from web.components.term_tooltip import render_term_with_tooltip, apply_term_tooltips_to_html
from web.components.tips import TipsManager
from web.components.audit_rule import render_audit_rule_tab

# 预测预警组件（被模型中心引用）
from web.components.forecast_panel import render_forecast_tab, render_alert_tab, render_monitor_tab

__all__ = [
    # 主组件
    'render_sidebar',
    'render_data_preparation',
    'render_preview_report',
    'render_model_center',
    'render_ai_assistant',
    'render_compare_tab',

    # 质量看板
    'render_quality_dashboard',
    'render_quality_dashboard_inline',

    # 原有组件
    'render_chat_interface',
    'render_recommended_questions',
    'render_scenario_recommendation',
    'render_natural_query',
    'render_sql_generator',
    'render_model_training',
    'render_agent_inference',
    'render_demo_section',
    'render_empty_state',
    'render_value_preview',
    'ProgressStage',
    'render_simple_progress',
    'render_term_with_tooltip',
    'apply_term_tooltips_to_html',
    'TipsManager',
    'render_audit_rule_tab',

    # 预测预警
    'render_forecast_tab',
    'render_alert_tab',
    'render_monitor_tab',
]