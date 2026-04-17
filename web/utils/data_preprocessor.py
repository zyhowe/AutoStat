"""
数据预处理模块 - 字段选择、类型修改、关系管理
"""

import streamlit as st
import pandas as pd
import time
from typing import Dict, List, Any, Tuple, Optional


# 变量类型选项
VARIABLE_TYPE_OPTIONS = {
    "continuous": "连续变量",
    "categorical": "分类变量",
    "ordinal": "有序分类",
    "datetime": "日期时间",
    "identifier": "标识符",
    "text": "文本",
    "exclude": "排除"
}

# 变量类型反向映射
TYPE_DISPLAY_TO_VALUE = {v: k for k, v in VARIABLE_TYPE_OPTIONS.items()}

# 默认排除的关键词（MCP模式自动排除，Web模式默认打叉）
DEFAULT_EXCLUDE_KEYWORDS = [
    'tmstamp', 'entrydt', 'transferdt', 'entrydate', 'entrytime',
    'examine', 'isdel', 'synchronize', 'isdelete', 'deleted',
    'createuser', 'updateuser', 'createip', 'updateip',
    'temp', 'tmp', 'bak', 'backup', 'sys', 'system'
]


def should_exclude_by_default(col_name: str) -> bool:
    """判断是否默认排除该字段"""
    col_lower = col_name.lower()
    for keyword in DEFAULT_EXCLUDE_KEYWORDS:
        if keyword.lower() == col_lower or keyword.lower() in col_lower:
            return True
    return False


def get_default_variable_type(col: str, df: pd.DataFrame) -> str:
    """根据列内容推断默认变量类型"""
    if should_exclude_by_default(col):
        return "exclude"

    if 'date' in col.lower() or '时间' in col or '日期' in col:
        try:
            pd.to_datetime(df[col], errors='coerce')
            return "datetime"
        except:
            pass

    if col.lower() in ['id', '编号', 'code', 'key'] or col.lower().endswith('_id') or col.lower().endswith('id'):
        return "identifier"

    if pd.api.types.is_numeric_dtype(df[col]):
        unique_count = df[col].nunique()
        total_count = len(df[col].dropna())
        if total_count > 0 and unique_count / total_count < 0.05 and unique_count < 20:
            return "categorical"
        if unique_count <= 10:
            return "ordinal"
        return "continuous"

    if df[col].dtype == 'object':
        unique_count = df[col].nunique()
        if unique_count <= 20:
            return "categorical"
        return "text"

    return "exclude"


def render_field_selector(df: pd.DataFrame, initial_types: Dict[str, str] = None, prefix: str = "", save_key: str = None) -> Tuple[List[str], Dict[str, str]]:
    """渲染字段选择器

    参数:
    - df: 数据框
    - initial_types: 初始类型字典
    - prefix: 组件 key 前缀
    - save_key: session_state 中保存类型的 key（用于多表模式独立保存）
    """
    st.markdown("#### 📋 字段管理")
    st.caption("✅ 勾选要保留的字段，并可调整每个字段的变量类型（排除的字段不参与分析）")

    # 获取刷新时间戳
    refresh_ts = st.session_state.get("field_selector_refresh_ts", 0)

    # 确定保存类型的 key
    type_save_key = save_key if save_key else "saved_variable_types"

    # 从 session_state 读取保存的类型
    saved_types = st.session_state.get(type_save_key, None)
    if saved_types is not None and initial_types is None:
        initial_types = saved_types

    # 准备当前类型
    current_types = {}
    if initial_types:
        current_types = initial_types.copy()
    else:
        for col in df.columns:
            current_types[col] = get_default_variable_type(col, df)

    # 创建表头
    col1, col2, col3 = st.columns([0.5, 3, 2])
    with col1:
        st.markdown("**✅ 保留**")
    with col2:
        st.markdown("**字段名**")
    with col3:
        st.markdown("**变量类型**")

    st.divider()

    selected_columns = []
    new_types = {}

    for col in df.columns:
        cols = st.columns([0.5, 3, 2])
        unique_key = f"{prefix}_{col}" if prefix else col

        with cols[0]:
            default_selected = current_types.get(col, "continuous") != "exclude"
            selected = st.checkbox(
                f"保留 {col}",
                value=default_selected,
                key=f"field_{unique_key}_{refresh_ts}",
                label_visibility="collapsed"
            )
            if selected:
                selected_columns.append(col)

        with cols[1]:
            st.write(f"`{col}`")

        with cols[2]:
            current_type = current_types.get(col, "continuous")
            display_value = VARIABLE_TYPE_OPTIONS.get(current_type, "连续变量")
            new_type_display = st.selectbox(
                f"类型 {col}",
                options=list(VARIABLE_TYPE_OPTIONS.values()),
                index=list(VARIABLE_TYPE_OPTIONS.values()).index(display_value),
                key=f"type_{unique_key}_{refresh_ts}",
                label_visibility="collapsed"
            )
            new_type_value = TYPE_DISPLAY_TO_VALUE[new_type_display]
            new_types[col] = new_type_value

            # 如果类型发生变化，立即更新 session_state 中的类型，然后刷新
            if new_type_value != current_type:
                # 更新保存的类型
                saved = st.session_state.get(type_save_key, {})
                saved[col] = new_type_value
                st.session_state[type_save_key] = saved
                # 更新时间戳强制刷新
                st.session_state.field_selector_refresh_ts = int(time.time())
                st.rerun()

    # 保存当前类型到 session_state
    st.session_state[type_save_key] = new_types

    return selected_columns, new_types


def render_preprocessing_interface(
    df: pd.DataFrame,
    title: str = "数据预处理",
    initial_types: Dict[str, str] = None
) -> Tuple[bool, pd.DataFrame, Dict[str, str]]:
    """渲染预处理界面（单表模式）"""
    st.markdown(f"### 🔧 {title}")

    selected_columns, variable_types = render_field_selector(df, initial_types, prefix="", save_key="saved_variable_types")

    if not selected_columns:
        st.warning("⚠️ 请至少保留一个字段")
        return False, None, None

    filtered_df = df[selected_columns].copy()

    if st.button("▶️ 开始分析", type="primary", width="stretch", key="preprocess_confirm"):
        return True, filtered_df, variable_types

    return False, None, None


def render_relationship_manager(table_names: List[str]) -> List[Dict]:
    """渲染表间关系管理器 - 支持增删改"""
    st.markdown("#### 🔗 表间关系管理")
    st.caption("定义表之间的关联关系，用于多表联合分析")
    st.caption("💡 提示：从表（外键）在前 → 主表（主键）在后")

    # 直接从 session_state 读取
    relationships = st.session_state.get("multi_relationships", [])

    # 获取刷新时间戳
    refresh_ts = st.session_state.get("relationship_refresh_ts", int(time.time()))

    # 显示关系摘要（放在最上面）
    if relationships:
        st.markdown("---")
        st.markdown("**关系摘要：**")
        for rel in relationships:
            if rel.get('from_col') and rel.get('to_col'):
                st.caption(f"  • {rel.get('from_table')}.{rel.get('from_col')} → {rel.get('to_table')}.{rel.get('to_col')}")
            else:
                st.caption(f"  • ⚠️ 不完整关系: {rel.get('from_table')}.{rel.get('from_col')} → {rel.get('to_table')}.{rel.get('to_col')}")

    # 显示现有关系（支持修改和删除）
    if relationships:
        st.markdown("**已有关系：**")
        for i, rel in enumerate(relationships):
            from_idx = table_names.index(rel.get('from_table', table_names[0])) if rel.get('from_table') in table_names else 0
            to_idx = table_names.index(rel.get('to_table', table_names[0])) if rel.get('to_table') in table_names else 0

            col1, col2, col3, col4, col5, col7, col6 = st.columns([1.2, 1.2, 0.8, 1.2, 1.2, 0.5, 0.5])

            with col1:
                from_table = st.selectbox(
                    "源表",
                    options=table_names,
                    index=from_idx,
                    key=f"rel_from_table_{i}_{refresh_ts}",
                    label_visibility="collapsed"
                )
            with col2:
                from_col = st.text_input(
                    "源列",
                    value=rel.get('from_col', ''),
                    key=f"rel_from_col_{i}_{refresh_ts}",
                    placeholder="列名",
                    label_visibility="collapsed"
                )

            with col4:
                to_table = st.selectbox(
                    "目标表",
                    options=table_names,
                    index=to_idx,
                    key=f"rel_to_table_{i}_{refresh_ts}",
                    label_visibility="collapsed"
                )
            with col5:
                to_col = st.text_input(
                    "目标列",
                    value=rel.get('to_col', ''),
                    key=f"rel_to_col_{i}_{refresh_ts}",
                    placeholder="列名",
                    label_visibility="collapsed"
                )

            with col3:
                st.markdown("<div style='text-align: center; line-height: 1.2;'>→</div>", unsafe_allow_html=True)
                if st.button("🔄", key=f"swap_rel_{i}_{refresh_ts}", width="stretch"):
                    current_relationships = st.session_state.get("multi_relationships", [])
                    if i < len(current_relationships):
                        current = current_relationships[i]
                        current_relationships[i] = {
                            'from_table': current.get('to_table', ''),
                            'from_col': current.get('to_col', ''),
                            'to_table': current.get('from_table', ''),
                            'to_col': current.get('from_col', '')
                        }
                        st.session_state.multi_relationships = current_relationships
                        st.session_state.relationship_refresh_ts = int(time.time())
                        st.rerun()

            with col7:
                if st.button("修改", key=f"update_rel_{i}_{refresh_ts}", width="stretch"):
                    current_relationships = st.session_state.get("multi_relationships", [])
                    if i < len(current_relationships):
                        current_relationships[i] = {
                            'from_table': from_table,
                            'from_col': from_col.strip(),
                            'to_table': to_table,
                            'to_col': to_col.strip()
                        }
                        st.session_state.multi_relationships = current_relationships
                        st.session_state.relationship_refresh_ts = int(time.time())
                        st.rerun()

            with col6:
                if st.button("删除", key=f"del_rel_{i}_{refresh_ts}", width="stretch"):
                    current_relationships = st.session_state.get("multi_relationships", [])
                    if i < len(current_relationships):
                        current_relationships.pop(i)
                        st.session_state.multi_relationships = current_relationships
                        st.session_state.relationship_refresh_ts = int(time.time())
                        st.rerun()

        st.markdown("---")

    # 添加新关系
    with st.form(key=f"add_relation_form_{refresh_ts}"):
        st.markdown("**添加新关系：**")
        col1, col2, col3, col4, col5, col6 = st.columns([1.2, 1.2, 0.5, 1.2, 1.2, 0.5])
        with col1:
            new_from_table = st.selectbox(
                "源表",
                options=table_names,
                key=f"new_rel_from_table_{refresh_ts}",
                label_visibility="collapsed"
            )
        with col2:
            new_from_col = st.text_input(
                "源列",
                key=f"new_rel_from_col_{refresh_ts}",
                placeholder="列名",
                label_visibility="collapsed"
            )
        with col3:
            st.markdown("→")
        with col4:
            new_to_table = st.selectbox(
                "目标表",
                options=table_names,
                key=f"new_rel_to_table_{refresh_ts}",
                label_visibility="collapsed"
            )
        with col5:
            new_to_col = st.text_input(
                "目标列",
                key=f"new_rel_to_col_{refresh_ts}",
                placeholder="列名",
                label_visibility="collapsed"
            )
        with col6:
            submitted = st.form_submit_button("➕ 添加", width="stretch")

        if submitted:
            if new_from_col and new_to_col:
                current_relationships = st.session_state.get("multi_relationships", [])
                is_duplicate = False
                for rel in current_relationships:
                    if (rel.get('from_table') == new_from_table and
                            rel.get('from_col') == new_from_col.strip() and
                            rel.get('to_table') == new_to_table and
                            rel.get('to_col') == new_to_col.strip()):
                        is_duplicate = True
                        break
                    if (rel.get('from_table') == new_to_table and
                            rel.get('from_col') == new_to_col.strip() and
                            rel.get('to_table') == new_from_table and
                            rel.get('to_col') == new_from_col.strip()):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    current_relationships.append({
                        'from_table': new_from_table,
                        'from_col': new_from_col.strip(),
                        'to_table': new_to_table,
                        'to_col': new_to_col.strip()
                    })
                    st.session_state.multi_relationships = current_relationships
                    st.session_state.relationship_refresh_ts = int(time.time())
                    st.rerun()
                else:
                    st.warning("该关系已存在，请勿重复添加")
            else:
                st.warning("请填写源列和目标列")

    # 显示关系摘要（再次显示）
    final_relationships = st.session_state.get("multi_relationships", [])
    if final_relationships:
        st.markdown("---")
        st.markdown("**关系摘要：**")
        for rel in final_relationships:
            if rel.get('from_col') and rel.get('to_col'):
                st.caption(f"  • {rel.get('from_table')}.{rel.get('from_col')} → {rel.get('to_table')}.{rel.get('to_col')}")
            else:
                st.caption(f"  • ⚠️ 不完整关系: {rel.get('from_table')}.{rel.get('from_col')} → {rel.get('to_table')}.{rel.get('to_col')}")

    return st.session_state.get("multi_relationships", [])


def auto_discover_relationships(tables: Dict[str, pd.DataFrame]) -> List[Dict]:
    """自动发现表间关系（通过相同列名）- 智能判断方向"""
    relationships = []
    table_names = list(tables.keys())

    column_to_tables = {}
    for table_name, df in tables.items():
        for col in df.columns:
            col_lower = col.lower()
            if col_lower not in column_to_tables:
                column_to_tables[col_lower] = []
            column_to_tables[col_lower].append(table_name)

    for col, tbls in column_to_tables.items():
        if len(tbls) >= 2:
            table_info = []
            for tbl in tbls:
                df = tables[tbl]
                # 检查列是否存在（防止 KeyError）
                if col not in df.columns:
                    # 尝试用原始列名（不转小写）查找
                    original_col = None
                    for c in df.columns:
                        if c.lower() == col:
                            original_col = c
                            break
                    if original_col is None:
                        continue
                    actual_col = original_col
                else:
                    actual_col = col

                unique_count = df[actual_col].nunique()
                total_count = len(df[actual_col].dropna())
                unique_ratio = unique_count / total_count if total_count > 0 else 0
                table_info.append({
                    'table': tbl,
                    'col': actual_col,
                    'unique_count': unique_count,
                    'total_count': total_count,
                    'unique_ratio': unique_ratio
                })

            if len(table_info) >= 2:
                table_info.sort(key=lambda x: x['unique_ratio'], reverse=True)

                from_table = table_info[-1]['table']
                to_table = table_info[0]['table']
                from_col = table_info[-1]['col']
                to_col = table_info[0]['col']

                # 检查是否已存在相同关系
                is_duplicate = False
                for rel in relationships:
                    if (rel['from_table'] == from_table and rel['to_table'] == to_table and
                            rel['from_col'] == from_col and rel['to_col'] == to_col):
                        is_duplicate = True
                        break
                    # 检查反向关系
                    if (rel['from_table'] == to_table and rel['to_table'] == from_table and
                            rel['from_col'] == to_col and rel['to_col'] == from_col):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    relationships.append({
                        'from_table': from_table,
                        'from_col': from_col,
                        'to_table': to_table,
                        'to_col': to_col
                    })

    return relationships


def render_multi_preprocessing_interface(
    tables: Dict[str, pd.DataFrame],
    relationships: List[Dict] = None,
    initial_types_dict: Dict[str, Dict[str, str]] = None
) -> Tuple[bool, Dict[str, pd.DataFrame], Dict[str, Dict[str, str]], List[Dict]]:
    """渲染多表预处理界面 - 自动发现关系"""
    st.markdown(f"### 🔧 数据预处理")

    filtered_tables = {}
    variable_types_dict = {}

    table_names = list(tables.keys())

    # 检查表是否变化，如果变化则清除缓存
    current_table_key = tuple(sorted(table_names))
    if current_table_key != st.session_state.get("last_table_key"):
        st.session_state.last_table_key = current_table_key
        if "multi_relationships" in st.session_state:
            del st.session_state.multi_relationships
        if "relationship_refresh_ts" in st.session_state:
            del st.session_state.relationship_refresh_ts
        if "multi_table_type_keys" in st.session_state:
            del st.session_state.multi_table_type_keys
        # 清除每个表的类型缓存
        keys_to_delete = [k for k in st.session_state.keys() if k.startswith("saved_variable_types_")]
        for key in keys_to_delete:
            del st.session_state[key]
        if "field_selector_refresh_ts" in st.session_state:
            del st.session_state.field_selector_refresh_ts

    tabs = st.tabs(table_names)

    # 初始化每个表的类型保存 key
    if "multi_table_type_keys" not in st.session_state:
        st.session_state.multi_table_type_keys = {}

    for i, (table_name, df) in enumerate(tables.items()):
        with tabs[i]:
            st.markdown(f"#### 📋 表: {table_name}")
            initial_types = initial_types_dict.get(table_name) if initial_types_dict else None

            # 每个表使用独立的 save_key
            save_key = f"saved_variable_types_{table_name}"
            st.session_state.multi_table_type_keys[table_name] = save_key

            selected_columns, variable_types = render_field_selector(
                df, initial_types, prefix=table_name, save_key=save_key
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
        return False, None, None, None

    # 初始化 session_state 中的关系
    if "multi_relationships" not in st.session_state:
        if relationships is None:
            st.session_state.multi_relationships = auto_discover_relationships(valid_tables)
        else:
            st.session_state.multi_relationships = relationships.copy()

    if "relationship_refresh_ts" not in st.session_state:
        st.session_state.relationship_refresh_ts = int(time.time())

    if st.session_state.multi_relationships:
        st.info(f"当前有 {len(st.session_state.multi_relationships)} 个表间关系")
    else:
        st.info("暂无表间关系，可手动添加")

    # 关系管理（直接从 session_state 读取和写入）
    render_relationship_manager(list(valid_tables.keys()))

    # 开始分析按钮
    if st.button("▶️ 开始分析", type="primary", width="stretch", key="multi_preprocess_confirm"):
        valid_relationships = [rel for rel in st.session_state.multi_relationships if rel.get('from_col') and rel.get('to_col')]
        return True, valid_tables, variable_types_dict, valid_relationships

    return False, None, None, None