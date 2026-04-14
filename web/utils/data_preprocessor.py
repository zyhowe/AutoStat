"""
数据预处理模块 - 字段选择、类型修改、关系管理
"""

import streamlit as st
import pandas as pd
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
    """判断是否默认排除该字段（MCP模式自动排除，Web模式默认打叉）"""
    col_lower = col_name.lower()
    for keyword in DEFAULT_EXCLUDE_KEYWORDS:
        if keyword.lower() == col_lower or keyword.lower() in col_lower:
            return True
    return False


def get_default_variable_type(col: str, df: pd.DataFrame) -> str:
    """根据列内容推断默认变量类型"""
    # 排除关键词优先
    if should_exclude_by_default(col):
        return "exclude"

    # 日期列检测
    if 'date' in col.lower() or '时间' in col or '日期' in col:
        try:
            pd.to_datetime(df[col], errors='coerce')
            return "datetime"
        except:
            pass

    # ID类列检测
    if col.lower() in ['id', '编号', 'code', 'key'] or col.lower().endswith('_id') or col.lower().endswith('id'):
        return "identifier"

    # 数值列检测
    if pd.api.types.is_numeric_dtype(df[col]):
        unique_count = df[col].nunique()
        total_count = len(df[col].dropna())
        if total_count > 0 and unique_count / total_count < 0.05 and unique_count < 20:
            return "categorical"
        if unique_count <= 10:
            return "ordinal"
        return "continuous"

    # 文本列检测
    if df[col].dtype == 'object':
        unique_count = df[col].nunique()
        if unique_count <= 20:
            return "categorical"
        return "text"

    return "exclude"


def render_field_selector(df: pd.DataFrame, initial_types: Dict[str, str] = None, prefix: str = "") -> Tuple[
    List[str], Dict[str, str]]:
    """渲染字段选择器"""
    st.markdown("#### 📋 字段管理")
    st.caption("✅ 勾选要保留的字段，并可调整每个字段的变量类型（排除的字段不参与分析）")

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

        # 使用前缀_列名作为唯一键
        unique_key = f"{prefix}_{col}" if prefix else col

        with cols[0]:
            default_selected = current_types.get(col, "continuous") != "exclude"
            selected = st.checkbox(
                f"保留 {col}",
                value=default_selected,
                key=f"field_{unique_key}",
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
                key=f"type_{unique_key}",
                label_visibility="collapsed"
            )
            new_types[col] = TYPE_DISPLAY_TO_VALUE[new_type_display]

    return selected_columns, new_types


def render_preprocessing_interface(
        df: pd.DataFrame,
        title: str = "数据预处理",
        initial_types: Dict[str, str] = None
) -> Tuple[bool, pd.DataFrame, Dict[str, str]]:
    """渲染预处理界面（单表模式）"""
    st.markdown(f"### 🔧 {title}")

    # 字段选择（单表模式不需要前缀）
    selected_columns, variable_types = render_field_selector(df, initial_types, prefix="")

    if not selected_columns:
        st.warning("⚠️ 请至少保留一个字段")
        return False, None, None

    # 根据当前选中的列过滤数据
    filtered_df = df[selected_columns].copy()

    # 开始分析按钮
    if st.button("▶️ 开始分析", type="primary", width="stretch", key="preprocess_confirm"):
        return True, filtered_df, variable_types

    return False, None, None


def render_multi_preprocessing_interface(
        tables: Dict[str, pd.DataFrame],
        relationships: List[Dict] = None,
        initial_types_dict: Dict[str, Dict[str, str]] = None
) -> Tuple[bool, Dict[str, pd.DataFrame], Dict[str, Dict[str, str]], List[Dict]]:
    """渲染多表预处理界面 - 自动发现关系"""
    st.markdown(f"### 🔧 数据预处理")

    filtered_tables = {}
    variable_types_dict = {}

    # 使用 tabs 分别处理每个表
    table_names = list(tables.keys())
    tabs = st.tabs(table_names)

    for i, (table_name, df) in enumerate(tables.items()):
        with tabs[i]:
            st.markdown(f"#### 📋 表: {table_name}")
            initial_types = initial_types_dict.get(table_name) if initial_types_dict else None
            selected_columns, variable_types = render_field_selector(df, initial_types, prefix=table_name)

            if selected_columns:
                filtered_tables[table_name] = df[selected_columns].copy()
                variable_types_dict[table_name] = variable_types
            else:
                st.warning(f"⚠️ 表 {table_name} 没有保留任何字段")
                filtered_tables[table_name] = pd.DataFrame()
                variable_types_dict[table_name] = {}

    # 检查是否有有效表
    valid_tables = {k: v for k, v in filtered_tables.items() if not v.empty}
    if not valid_tables:
        st.error("没有保留任何有效字段，请至少为每个表保留一个字段")
        return False, None, None, None

    # 使用 session_state 保存关系列表
    rel_key = "multi_relationships"
    if rel_key not in st.session_state:
        if relationships is None:
            st.session_state[rel_key] = auto_discover_relationships(valid_tables)
        else:
            st.session_state[rel_key] = relationships.copy()

    # 显示当前关系数量
    if st.session_state[rel_key]:
        st.info(f"当前有 {len(st.session_state[rel_key])} 个表间关系")
    else:
        st.info("暂无表间关系，可手动添加")

    # 关系管理（增删改）- 直接操作 session_state 中的列表
    updated_relationships = render_relationship_manager(st.session_state[rel_key], list(valid_tables.keys()))

    # 更新 session_state
    st.session_state[rel_key] = updated_relationships

    # 开始分析按钮
    if st.button("▶️ 开始分析", type="primary", width="stretch", key="multi_preprocess_confirm"):
        valid_relationships = [rel for rel in updated_relationships if rel['from_col'] and rel['to_col']]
        return True, valid_tables, variable_types_dict, valid_relationships

    return False, None, None, None


def auto_discover_relationships(tables: Dict[str, pd.DataFrame]) -> List[Dict]:
    """自动发现表间关系（通过相同列名）- 智能判断方向"""
    relationships = []
    table_names = list(tables.keys())

    # 收集所有列及其所属表
    column_to_tables = {}
    for table_name, df in tables.items():
        for col in df.columns:
            col_lower = col.lower()
            if col_lower not in column_to_tables:
                column_to_tables[col_lower] = []
            column_to_tables[col_lower].append(table_name)

    # 找出出现在多个表中的列
    for col, tbls in column_to_tables.items():
        if len(tbls) >= 2:
            # 对每个表分析该列的唯一值比例
            table_info = []
            for tbl in tbls:
                df = tables[tbl]
                unique_count = df[col].nunique()
                total_count = len(df[col].dropna())
                unique_ratio = unique_count / total_count if total_count > 0 else 0
                table_info.append({
                    'table': tbl,
                    'unique_count': unique_count,
                    'total_count': total_count,
                    'unique_ratio': unique_ratio
                })

            # 按唯一值比例排序：唯一值比例高的（接近1）可能是主键（主表）
            # 唯一值比例低的（接近0）可能是外键（从表）
            # 从表（外键）在前，主表（主键）在后
            table_info.sort(key=lambda x: x['unique_ratio'], reverse=True)

            # 唯一值比例最高的作为主表（目标表），最低的作为从表（源表）
            if len(table_info) >= 2:
                from_table = table_info[-1]['table']  # 唯一值比例最低的（从表）
                to_table = table_info[0]['table']  # 唯一值比例最高的（主表）

                # 检查是否已存在
                is_duplicate = False
                for rel in relationships:
                    if rel['from_table'] == from_table and rel['to_table'] == to_table and rel['from_col'] == col:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    relationships.append({
                        'from_table': from_table,
                        'from_col': col,
                        'to_table': to_table,
                        'to_col': col
                    })

    return relationships


def render_relationship_manager(relationships: List[Dict], table_names: List[str]) -> List[Dict]:
    """渲染表间关系管理器 - 支持增删改"""
    st.markdown("#### 🔗 表间关系管理")
    st.caption("定义表之间的关联关系，用于多表联合分析")
    st.caption("💡 提示：从表（外键）在前 → 主表（主键）在后")

    if relationships is None:
        relationships = []

    # 显示现有关系（支持修改和删除）
    if relationships:
        st.markdown("**已有关系：**")
        for i, rel in enumerate(relationships):
            # 确保 from_table 和 to_table 在 table_names 中
            from_idx = table_names.index(rel.get('from_table', table_names[0])) if rel.get(
                'from_table') in table_names else 0
            to_idx = table_names.index(rel.get('to_table', table_names[0])) if rel.get('to_table') in table_names else 0

            # 布局：源表 | 源列 | 箭头+交换 | 目标表 | 目标列 | 删除
            col1, col2, col3, col4, col5, col6 = st.columns([1.2, 1.2, 0.8, 1.2, 1.2, 0.5])

            with col1:
                from_table = st.selectbox(
                    "源表",
                    options=table_names,
                    index=from_idx,
                    key=f"rel_from_table_{i}",
                    label_visibility="collapsed"
                )
            with col2:
                from_col = st.text_input(
                    "源列",
                    value=rel.get('from_col', ''),
                    key=f"rel_from_col_{i}",
                    placeholder="列名",
                    label_visibility="collapsed"
                )
            with col3:
                # 箭头和交换按钮垂直排列
                st.markdown("<div style='text-align: center; line-height: 1.2;'>→</div>", unsafe_allow_html=True)
                if st.button("🔄 交换", key=f"swap_rel_{i}", use_container_width=True):
                    # 交换源和目标
                    relationships[i] = {
                        'from_table': to_table if 'to_table' in locals() else rel.get('to_table', ''),
                        'from_col': to_col if 'to_col' in locals() else rel.get('to_col', ''),
                        'to_table': from_table,
                        'to_col': from_col if 'from_col' in locals() else rel.get('from_col', '')
                    }
                    st.rerun()
            with col4:
                to_table = st.selectbox(
                    "目标表",
                    options=table_names,
                    index=to_idx,
                    key=f"rel_to_table_{i}",
                    label_visibility="collapsed"
                )
            with col5:
                to_col = st.text_input(
                    "目标列",
                    value=rel.get('to_col', ''),
                    key=f"rel_to_col_{i}",
                    placeholder="列名",
                    label_visibility="collapsed"
                )
            with col6:
                if st.button("🗑️", key=f"del_rel_{i}", use_container_width=True):
                    relationships.pop(i)
                    st.rerun()

            # 更新关系（如果用户手动修改）
            relationships[i] = {
                'from_table': from_table,
                'from_col': from_col.strip(),
                'to_table': to_table,
                'to_col': to_col.strip()
            }

        st.markdown("---")

    # 添加新关系
    with st.form(key="add_relation_form"):
        st.markdown("**添加新关系：**")
        col1, col2, col3, col4, col5, col6 = st.columns([1.2, 1.2, 0.5, 1.2, 1.2, 0.5])
        with col1:
            new_from_table = st.selectbox("源表", options=table_names, key="new_rel_from_table",
                                          label_visibility="collapsed")
        with col2:
            new_from_col = st.text_input("源列", key="new_rel_from_col", placeholder="列名",
                                         label_visibility="collapsed")
        with col3:
            st.markdown("→")
        with col4:
            new_to_table = st.selectbox("目标表", options=table_names, key="new_rel_to_table",
                                        label_visibility="collapsed")
        with col5:
            new_to_col = st.text_input("目标列", key="new_rel_to_col", placeholder="列名", label_visibility="collapsed")
        with col6:
            submitted = st.form_submit_button("➕ 添加", width="stretch")

        if submitted:
            if new_from_col and new_to_col:
                # 检查是否重复
                is_duplicate = False
                for rel in relationships:
                    if (rel['from_table'] == new_from_table and
                            rel['from_col'] == new_from_col.strip() and
                            rel['to_table'] == new_to_table and
                            rel['to_col'] == new_to_col.strip()):
                        is_duplicate = True
                        break
                    # 也检查反向
                    if (rel['from_table'] == new_to_table and
                            rel['from_col'] == new_to_col.strip() and
                            rel['to_table'] == new_from_table and
                            rel['to_col'] == new_from_col.strip()):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    relationships.append({
                        'from_table': new_from_table,
                        'from_col': new_from_col.strip(),
                        'to_table': new_to_table,
                        'to_col': new_to_col.strip()
                    })
                    st.success(f"已添加关系: {new_from_table}.{new_from_col} → {new_to_table}.{new_to_col}")
                    st.rerun()
                else:
                    st.warning("该关系已存在，请勿重复添加")
            else:
                st.warning("请填写源列和目标列")

    # 显示关系摘要
    if relationships:
        st.markdown("---")
        st.markdown("**关系摘要：**")
        for rel in relationships:
            if rel['from_col'] and rel['to_col']:
                st.caption(f"  • {rel['from_table']}.{rel['from_col']} → {rel['to_table']}.{rel['to_col']}")
            else:
                st.caption(
                    f"  • ⚠️ 不完整关系: {rel['from_table']}.{rel['from_col']} → {rel['to_table']}.{rel['to_col']}")

    return relationships