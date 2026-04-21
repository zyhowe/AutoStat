# web/services/cache_service.py

"""缓存服务 - 管理 session_state 中的数据缓存"""

import streamlit as st


class CacheService:
    """缓存服务类"""

    @staticmethod
    def init_session_state():
        """初始化所有 session_state 变量"""
        # 分析器
        if 'single_analyzer' not in st.session_state:
            st.session_state.single_analyzer = None
        if 'single_output' not in st.session_state:
            st.session_state.single_output = None
        if 'multi_analyzer' not in st.session_state:
            st.session_state.multi_analyzer = None
        if 'multi_output' not in st.session_state:
            st.session_state.multi_output = None
        if 'db_analyzer' not in st.session_state:
            st.session_state.db_analyzer = None
        if 'db_output' not in st.session_state:
            st.session_state.db_output = None

        # 当前分析结果
        if 'current_html' not in st.session_state:
            st.session_state.current_html = None
        if 'current_json_data' not in st.session_state:
            st.session_state.current_json_data = None
        if 'current_analysis_type' not in st.session_state:
            st.session_state.current_analysis_type = None
        if 'current_source_name' not in st.session_state:
            st.session_state.current_source_name = None

        # 大模型相关
        if 'llm_client' not in st.session_state:
            st.session_state.llm_client = None
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        if 'selected_contexts' not in st.session_state:
            st.session_state.selected_contexts = ["json_result"]
        if 'raw_data_preview' not in st.session_state:
            st.session_state.raw_data_preview = None

        # UI 状态
        if 'selected_db_config' not in st.session_state:
            st.session_state.selected_db_config = None
        if 'selected_llm_config' not in st.session_state:
            st.session_state.selected_llm_config = None

        # 日期特征级别
        if 'date_features_level' not in st.session_state:
            st.session_state.date_features_level = "basic"

        # 分析完成标志
        if 'analysis_completed' not in st.session_state:
            st.session_state.analysis_completed = False

        # 当前标签页
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = 0

        # 单文件缓存
        if 'single_cached_df' not in st.session_state:
            st.session_state.single_cached_df = None
        if 'single_cached_name' not in st.session_state:
            st.session_state.single_cached_name = None
        if 'single_cached_ext' not in st.session_state:
            st.session_state.single_cached_ext = None

        # 多文件缓存
        if 'multi_cached_tables' not in st.session_state:
            st.session_state.multi_cached_tables = None
        if 'multi_tmp_dir' not in st.session_state:
            st.session_state.multi_tmp_dir = None

        # 数据库缓存
        if 'db_cached_tables' not in st.session_state:
            st.session_state.db_cached_tables = None
        if 'db_config' not in st.session_state:
            st.session_state.db_config = None

    @staticmethod
    def clear_single_cache():
        """清除单文件缓存"""
        st.session_state.single_cached_df = None
        st.session_state.single_cached_name = None
        st.session_state.single_cached_ext = None
        if 'saved_variable_types' in st.session_state:
            del st.session_state.saved_variable_types
        if 'field_selector_refresh_ts' in st.session_state:
            del st.session_state.field_selector_refresh_ts

    @staticmethod
    def clear_multi_cache():
        """清除多文件缓存"""
        st.session_state.multi_cached_tables = None
        if st.session_state.multi_tmp_dir:
            import shutil
            shutil.rmtree(st.session_state.multi_tmp_dir, ignore_errors=True)
            st.session_state.multi_tmp_dir = None
        if "multi_relationships" in st.session_state:
            del st.session_state.multi_relationships
        if "relationship_refresh_ts" in st.session_state:
            del st.session_state.relationship_refresh_ts
        if "multi_table_type_keys" in st.session_state:
            del st.session_state.multi_table_type_keys
        keys_to_delete = [k for k in st.session_state.keys() if k.startswith("saved_variable_types_")]
        for key in keys_to_delete:
            del st.session_state[key]
        if "field_selector_refresh_ts" in st.session_state:
            del st.session_state.field_selector_refresh_ts

    @staticmethod
    def clear_db_cache():
        """清除数据库缓存"""
        st.session_state.db_cached_tables = None
        if "multi_relationships" in st.session_state:
            del st.session_state.multi_relationships
        if "relationship_refresh_ts" in st.session_state:
            del st.session_state.relationship_refresh_ts
        if "multi_table_type_keys" in st.session_state:
            del st.session_state.multi_table_type_keys
        keys_to_delete = [k for k in st.session_state.keys() if k.startswith("saved_variable_types_")]
        for key in keys_to_delete:
            del st.session_state[key]
        if "field_selector_refresh_ts" in st.session_state:
            del st.session_state.field_selector_refresh_ts

    @staticmethod
    def reset_analysis_state():
        """重置分析状态（模式切换时调用）"""
        st.session_state.analysis_completed = False
        st.session_state.current_tab = 0
        st.session_state.current_html = None
        st.session_state.current_json_data = None
        st.session_state.chat_messages = []
        st.session_state.single_analyzer = None
        st.session_state.multi_analyzer = None
        st.session_state.db_analyzer = None