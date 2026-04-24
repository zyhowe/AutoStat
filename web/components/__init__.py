# web/components/__init__.py

"""UI组件模块"""
from web.components.sidebar import render_sidebar
from web.components.tabs import render_tabs, scroll_to_top
from web.components.data_preparation import render_data_preparation
from web.components.results import render_preview_tab, render_ai_tab, render_compare_tab
from web.components.chat_interface import render_chat_interface, render_recommended_questions
from web.components.scenario_recommendation import render_scenario_recommendation
from web.components.natural_query import render_natural_query
from web.components.sql_generator import render_sql_generator
from web.components.model_training import render_model_training
from web.components.agent_inference import render_agent_inference

# 新增组件
from web.components.demo_data import render_demo_section
from web.components.empty_state import render_empty_state
from web.components.value_preview import render_value_preview
from web.components.progress_stage import ProgressStage, render_simple_progress
from web.components.term_tooltip import render_term_with_tooltip, apply_term_tooltips_to_html
from web.components.tips import TipsManager

__all__ = [
    # 原有导出
    'render_sidebar',
    'render_tabs',
    'scroll_to_top',
    'render_data_preparation',
    'render_preview_tab',
    'render_ai_tab',
    'render_compare_tab',
    'render_chat_interface',
    'render_recommended_questions',
    'render_scenario_recommendation',
    'render_natural_query',
    'render_sql_generator',
    'render_model_training',
    'render_agent_inference',

    # 新增导出
    'render_demo_section',
    'render_empty_state',
    'render_value_preview',
    'ProgressStage',
    'render_simple_progress',
    'render_term_with_tooltip',
    'apply_term_tooltips_to_html',
    'TipsManager'
]