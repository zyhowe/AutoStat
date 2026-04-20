# web/components/__init__.py

"""UI组件模块"""
from web.components.sidebar import render_sidebar
from web.components.tabs import render_tabs, scroll_to_top
from web.components.data_preparation import render_data_preparation
from web.components.results import render_preview_tab, render_ai_tab
from web.components.chat_interface import render_chat_interface, render_recommended_questions
from web.components.scenario_recommendation import render_scenario_recommendation
from web.components.natural_query import render_natural_query
from web.components.sql_generator import render_sql_generator
from web.components.model_training import render_model_training
from web.components.agent_inference import render_agent_inference

__all__ = [
    'render_sidebar',
    'render_tabs',
    'scroll_to_top',
    'render_data_preparation',
    'render_preview_tab',
    'render_ai_tab',
    'render_chat_interface',
    'render_recommended_questions',
    'render_scenario_recommendation',
    'render_natural_query',
    'render_sql_generator',
    'render_model_training',
    'render_agent_inference'
]