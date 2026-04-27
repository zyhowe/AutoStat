"""文本分析 UI 组件模块"""

from webtext.components.sidebar import render_sidebar
from webtext.components.tabs import render_tabs, scroll_to_top
from webtext.components.data_preparation import render_data_preparation
from webtext.components.results import render_results_tab
from webtext.components.model_training import render_model_training
from webtext.components.chat_interface import render_chat_tab

__all__ = [
    "render_sidebar",
    "render_tabs",
    "scroll_to_top",
    "render_data_preparation",
    "render_results_tab",
    "render_model_training",
    "render_chat_tab",
]