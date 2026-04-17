"""数据准备组件 - UI 渲染（上传、字段选择、开始分析）"""

import streamlit as st
import pandas as pd
from web.services.file_service import FileService
from web.services.analysis_service import AnalysisService
from web.services.cache_service import CacheService
from web.utils.data_preprocessor import render_field_selector, render_relationship_manager, auto_discover_relationships


def render_data_preparation(analysis_mode: str):
    """渲染数据准备标签页"""

    if analysis_mode == "📁 单文件分析":
        render_single_file_preparation()
    elif analysis_mode == "📚 多文件分析":
        render_multi_file_preparation()
    elif analysis_mode == "🗄️ 数据库分析":
        render_database_preparation()


# ==================== 单文件分析 ====================

def render_single_file_preparation():
    """单文件分析的数据准备"""
    st.markdown("### 📁 上传数据文件")

    with st.expander("ℹ️ 使用说明", expanded=False):
        st.markdown("""
        **支持的文件格式：** CSV、Excel(.xlsx/.xls)、JSON、TXT
        **限制建议：** 行数 < 10万，列数 < 200，文件大小 < 100MB
        """)

    uploaded = st.file_uploader(
        "选择数据文件", type=['csv', 'xlsx', 'xls', 'json', 'txt'],
        help="支持 CSV、Excel、JSON、TXT 格式",
        key="single_uploader"
    )

    # 处理新上传的文件
    if uploaded:
        file_size = len(uploaded.getvalue()) / (1024 * 1024)
        if file_size > 100:
            st.warning(f"⚠️ 文件大小 {file_size:.1f}MB，超过建议限制")
        elif file_size > 50:
            st.info(f"📊 文件大小 {file_size:.1f}MB，加载可能需要一些时间")

        ext = uploaded.name.split('.')[-1].lower()

        # 检查文件是否变化
        current_file_key = f"{uploaded.name}_{uploaded.size}"
        if current_file_key != st.session_state.get("last_single_file_key"):
            st.session_state.last_single_file_key = current_file_key
            CacheService.clear_single_cache()
            st.session_state.analysis_completed = False

        # 加载并缓存数据
        if st.session_state.single_cached_df is None:
            try:
                df = FileService.load_file(uploaded, ext)
                if df is not None and not df.empty:
                    df.columns = [str(col).strip().replace('\n', '_').replace('\r', '_') for col in df.columns]
                    df = df.replace(r'^\s*$', pd.NA, regex=True)
                    st.session_state.single_cached_df = df
                    st.session_state.single_cached_name = uploaded.name
                    st.session_state.single_cached_ext = ext
                else:
                    st.error("文件加载失败")
                    return
            except Exception as e:
                st.error(f"读取失败: {str(e)}")
                return

    # 显示已缓存的数据
    if st.session_state.single_cached_df is not None:
        render_single_file_content(
            st.session_state.single_cached_df,
            st.session_state.single_cached_ext,
            st.session_state.single_cached_name
        )
    else:
        st.info("请上传数据文件开始分析")


def render_single_file_content(df: pd.DataFrame, ext: str, file_name: str):
    """渲染单文件内容（数据预览、字段选择、开始分析）"""
    rows, cols = df.shape
    st.subheader("📄 原始数据预览")
    st.dataframe(df.head(100))
    st.write(f"形状: {rows:,} 行 × {cols} 列")

    # 数据质量快速提示
    col_info = []
    for col in df.columns:
        missing_pct = df[col].isna().mean() * 100
        if missing_pct > 50:
            col_info.append(f"⚠️ {col}: 缺失率 {missing_pct:.1f}%")
        elif missing_pct > 20:
            col_info.append(f"📌 {col}: 缺失率 {missing_pct:.1f}%")
    if col_info:
        with st.expander("📊 数据质量提醒", expanded=False):
            for info in col_info[:5]:
                st.caption(info)
            if len(col_info) > 5:
                st.caption(f"... 还有 {len(col_info) - 5} 个字段")

    # 字段选择器
    st.markdown("---")
    selected_columns, variable_types = render_field_selector(
        df, initial_types=None, prefix="single", save_key="saved_variable_types"
    )

    if not selected_columns:
        st.warning("⚠️ 请至少保留一个字段")
        return

    filtered_df = df[selected_columns].copy()

    # 开始分析按钮
    st.markdown("---")
    if st.button("▶️ 开始分析", type="primary", use_container_width=True):
        AnalysisService.analyze_single_file(file_name, ext, filtered_df, variable_types)


# ==================== 多文件分析 ====================

def render_multi_file_preparation():
    """多文件分析的数据准备"""
    st.markdown("### 📚 上传多个数据文件")

    with st.expander("ℹ️ 使用说明", expanded=False):
        st.markdown("""
        **适用场景：** 多个相关表需要联合分析（如订单表、用户表、产品表）
        **文件要求：** 至少上传 2 个文件
        **限制建议：** 表数量 < 10，每个表行数 < 5万
        """)

    files = st.file_uploader(
        "选择多个数据文件", type=['csv', 'xlsx', 'xls', 'json', 'txt'],
        accept_multiple_files=True, key="multi_uploader",
        help="按住 Ctrl 或 Cmd 键可选择多个文件"
    )

    # 处理新上传的文件
    if files and len(files) >= 2:
        current_files_key = tuple(sorted([f.name for f in files]))
        if current_files_key != st.session_state.get("last_multi_files_key"):
            st.session_state.last_multi_files_key = current_files_key
            CacheService.clear_multi_cache()
            st.session_state.analysis_completed = False

        if len(files) > 10:
            st.warning(f"⚠️ 表数量 {len(files)} 超过建议限制")

        st.success(f"已选择 {len(files)} 个文件")

        if st.session_state.multi_cached_tables is None:
            with st.spinner("正在加载文件..."):
                tmp_dir, tables = FileService.load_tables(files)
                if tables:
                    st.session_state.multi_cached_tables = tables
                    st.session_state.multi_tmp_dir = tmp_dir
                else:
                    st.error("没有成功加载任何表")
                    return

    # 显示已缓存的数据
    if st.session_state.multi_cached_tables is not None:
        render_multi_file_content(st.session_state.multi_cached_tables)
    else:
        if files and len(files) == 1:
            st.warning("多文件分析需要至少2个文件，请继续添加")
        else:
            st.info("请上传至少2个数据文件开始分析")


def render_multi_file_content(tables: dict):
    """渲染多文件内容（字段管理、关系管理、开始分析）"""
    st.subheader("📊 已加载的表")
    for name, df in tables.items():
        st.caption(f"  📋 {name}: {len(df)}行 x {len(df.columns)}列")

    st.markdown("---")
    st.markdown("### 🔧 字段管理")

    filtered_tables = {}
    variable_types_dict = {}

    table_names = list(tables.keys())

    # 检查表是否变化
    current_table_key = tuple(sorted(table_names))
    if current_table_key != st.session_state.get("last_multi_table_key"):
        st.session_state.last_multi_table_key = current_table_key
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

    if "multi_table_type_keys" not in st.session_state:
        st.session_state.multi_table_type_keys = {}

    for table_name, df in tables.items():
        with st.expander(f"📋 表: {table_name}", expanded=True):
            save_key = f"saved_variable_types_{table_name}"
            st.session_state.multi_table_type_keys[table_name] = save_key

            selected_columns, variable_types = render_field_selector(
                df, initial_types=None, prefix=table_name, save_key=save_key
            )

            if selected_columns:
                filtered_tables[table_name] = df[selected_columns].copy()
                variable_types_dict[table_name] = variable_types
            else:
                st.warning(f"⚠️ 表 {table_name} 没有保留任何字段")
                filtered_tables[table_name] = pd.DataFrame()
                variable_types_dict[table_name] = {}

    valid_tables = {k: v for k, v in filtered_tables.items() if not v.empty}

    if not valid_tables:
        st.error("没有保留任何有效字段，请至少为每个表保留一个字段")
        return

    st.markdown("---")
    st.markdown("### 🔗 表间关系管理")

    if "multi_relationships" not in st.session_state:
        st.session_state.multi_relationships = auto_discover_relationships(valid_tables)

    if "relationship_refresh_ts" not in st.session_state:
        st.session_state.relationship_refresh_ts = int(__import__('time').time())

    relationships = st.session_state.multi_relationships
    if relationships:
        st.info(f"当前有 {len(relationships)} 个表间关系")
        for rel in relationships:
            if rel.get('from_col') and rel.get('to_col'):
                st.caption(f"  • {rel.get('from_table')}.{rel.get('from_col')} → {rel.get('to_table')}.{rel.get('to_col')}")
    else:
        st.info("暂无表间关系，可手动添加")

    render_relationship_manager(list(valid_tables.keys()))

    final_relationships = st.session_state.get("multi_relationships", [])
    valid_relationships = [rel for rel in final_relationships if rel.get('from_col') and rel.get('to_col')]

    st.markdown("---")
    if st.button("▶️ 开始分析", type="primary", use_container_width=True):
        AnalysisService.analyze_multi_file(valid_tables, variable_types_dict, valid_relationships)


# ==================== 数据库分析 ====================

def render_database_preparation():
    """数据库分析的数据准备"""
    st.markdown("### 🗄️ 连接数据库")

    with st.expander("ℹ️ 使用说明", expanded=False):
        st.markdown("""
        **适用场景：** 需要分析 SQL Server 数据库中的数据
        **前置要求：** 安装 pyodbc 驱动
        **限制建议：** 表数量 < 10，每个表记录数 < 5万
        """)

    if st.session_state.selected_db_config is None:
        st.warning("请先在侧边栏配置并选择数据库")
        return

    config = st.session_state.selected_db_config

    with st.form("db_form"):
        st.info(f"📡 使用配置: {config.get('name')}")
        st.caption(f"服务器: {config.get('server')} | 数据库: {config.get('database')}")

        tables_input = st.text_area(
            "表名列表（每行一个）",
            placeholder="users\norders\nproducts",
            key="db_tables",
            help="输入要分析的表名，每行一个"
        )

        col1, col2 = st.columns(2)
        with col1:
            limit = st.number_input(
                "每个表最大加载行数",
                min_value=100,
                max_value=100000,
                value=5000,
                key="db_limit"
            )
        with col2:
            max_text_length = st.number_input(
                "文本字段最大长度",
                min_value=50,
                max_value=500,
                value=100,
                key="db_text_length"
            )

        submitted = st.form_submit_button("🔌 连接数据库", type="primary")

    if submitted:
        if not tables_input.strip():
            st.error("请至少输入一个表名")
            return

        table_names = [t.strip() for t in tables_input.strip().split('\n') if t.strip()]

        if len(table_names) > 10:
            st.error(f"表数量 {len(table_names)} 超过限制")
            return

        load_and_prepare_database(config, table_names, limit, max_text_length)

    if st.session_state.db_cached_tables is not None:
        render_database_content(st.session_state.db_cached_tables)


def load_and_prepare_database(config: dict, table_names: list, limit: int, max_text_length: int):
    """加载数据库表并准备预处理"""
    import traceback

    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    try:
        status_placeholder.info("🔌 正在连接数据库...")
        progress_bar.progress(10)

        conn = FileService.test_db_connection(config)
        if conn is None:
            st.error("数据库连接失败，请检查配置")
            return
        conn.close()

        status_placeholder.success("✅ 数据库连接成功")
        progress_bar.progress(20)
        status_placeholder.info("📊 正在加载表数据...")

        success = FileService.load_db_tables(config, table_names, limit, max_text_length)

        if not success:
            st.error("没有成功加载任何表")
            return

        progress_bar.empty()
        status_placeholder.empty()

        CacheService.clear_db_cache()

        st.session_state.db_cached_tables = success
        st.session_state.db_config = config

        st.rerun()

    except Exception as e:
        progress_bar.empty()
        status_placeholder.empty()
        st.error(f"连接失败: {str(e)}")
        with st.expander("📋 详细错误信息", expanded=False):
            st.code(traceback.format_exc(), language='python')


def render_database_content(tables: dict):
    """渲染数据库内容（字段管理、关系管理、开始分析）"""
    st.subheader("📊 已加载的表")
    for name, df in tables.items():
        st.caption(f"  📋 {name}: {len(df)}行 x {len(df.columns)}列")

    st.markdown("---")
    st.markdown("### 🔧 字段管理")

    filtered_tables = {}
    variable_types_dict = {}

    table_names = list(tables.keys())

    current_table_key = tuple(sorted(table_names))
    if current_table_key != st.session_state.get("last_db_table_key"):
        st.session_state.last_db_table_key = current_table_key
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

    if "multi_table_type_keys" not in st.session_state:
        st.session_state.multi_table_type_keys = {}

    for table_name, df in tables.items():
        with st.expander(f"📋 表: {table_name}", expanded=True):
            save_key = f"saved_variable_types_{table_name}"
            st.session_state.multi_table_type_keys[table_name] = save_key

            selected_columns, variable_types = render_field_selector(
                df, initial_types=None, prefix=table_name, save_key=save_key
            )

            if selected_columns:
                filtered_tables[table_name] = df[selected_columns].copy()
                variable_types_dict[table_name] = variable_types
            else:
                st.warning(f"⚠️ 表 {table_name} 没有保留任何字段")
                filtered_tables[table_name] = pd.DataFrame()
                variable_types_dict[table_name] = {}

    valid_tables = {k: v for k, v in filtered_tables.items() if not v.empty}

    if not valid_tables:
        st.error("没有保留任何有效字段，请至少为每个表保留一个字段")
        return

    st.markdown("---")
    st.markdown("### 🔗 表间关系管理")

    if "multi_relationships" not in st.session_state:
        st.session_state.multi_relationships = auto_discover_relationships(valid_tables)

    if "relationship_refresh_ts" not in st.session_state:
        st.session_state.relationship_refresh_ts = int(__import__('time').time())

    relationships = st.session_state.multi_relationships
    if relationships:
        st.info(f"当前有 {len(relationships)} 个表间关系")
        for rel in relationships:
            if rel.get('from_col') and rel.get('to_col'):
                st.caption(f"  • {rel.get('from_table')}.{rel.get('from_col')} → {rel.get('to_table')}.{rel.get('to_col')}")
    else:
        st.info("暂无表间关系，可手动添加")

    render_relationship_manager(list(valid_tables.keys()))

    final_relationships = st.session_state.get("multi_relationships", [])
    valid_relationships = [rel for rel in final_relationships if rel.get('from_col') and rel.get('to_col')]

    st.markdown("---")
    if st.button("▶️ 开始分析", type="primary", use_container_width=True):
        AnalysisService.analyze_database(valid_tables, variable_types_dict, valid_relationships, st.session_state.db_config)