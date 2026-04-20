# web/components/sql_generator.py

"""SQL生成组件 - 纯内容，用于标签页（仅数据库模式）"""

import streamlit as st


def render_sql_generator():
    """渲染SQL生成Tab（纯内容，不含外层标签页）"""
    st.markdown("#### 📝 SQL查询生成")
    st.caption("用中文描述你想查询的数据，AI会基于表结构和关联关系生成SQL语句")

    # 检查是否有JSON结果和源数据
    if st.session_state.current_json_data is None:
        st.warning("⚠️ 请先完成数据分析，生成JSON报告")
        return

    if "raw_data" not in st.session_state.selected_contexts:
        st.info("💡 提示：SQL生成需要基于源数据，建议在「选择分析上下文」中勾选「源数据」")

    # 显示表结构信息
    with st.expander("📋 表结构信息", expanded=False):
        json_data = st.session_state.current_json_data
        variable_types = json_data.get('variable_types', {})

        st.markdown("**字段列表：**")
        for col, info in variable_types.items():
            type_desc = info.get('type_desc', info.get('type', 'unknown'))
            icon = "📊" if type_desc == "连续变量" else "🏷️" if "分类" in type_desc else "📅" if "日期" in type_desc else "🔑"
            st.caption(f"  {icon} {col}: {type_desc}")

        # 显示表间关系（多表模式）
        if st.session_state.current_analysis_type in ["multi", "database"]:
            multi_info = json_data.get('multi_table_info', {})
            relationships = multi_info.get('relationships', [])
            tables = multi_info.get('tables', {})

            if tables:
                st.markdown("\n**表名列表：**")
                for name in tables.keys():
                    st.caption(f"  - {name}")

            if relationships:
                st.markdown("\n**表间关系：**")
                for rel in relationships:
                    st.caption(f"  - {rel.get('from_table')}.{rel.get('from_col')} → {rel.get('to_table')}.{rel.get('to_col')}")

    # 动态生成示例SQL查询（基于实际字段）
    json_data = st.session_state.current_json_data
    variable_types = json_data.get('variable_types', {})
    multi_info = json_data.get('multi_table_info', {})
    tables = multi_info.get('tables', {})

    main_table = list(tables.keys())[0] if tables else "your_table"

    numeric_cols = [col for col, info in variable_types.items() if info.get('type') == 'continuous']
    cat_cols = [col for col, info in variable_types.items()
                if info.get('type') in ['categorical', 'categorical_numeric', 'ordinal']]
    date_cols = [col for col, info in variable_types.items() if info.get('type') == 'datetime']

    st.markdown("**💡 示例SQL查询（点击使用）：**")

    sql_examples = []

    if date_cols:
        date_field = date_cols[0]
        sql_examples.append(f"查询{date_field}在2024年的所有记录")
        sql_examples.append(f"按{date_field}分组统计每天的记录数")

    if numeric_cols:
        num_field = numeric_cols[0]
        sql_examples.append(f"查询{num_field}大于1000的记录")
        sql_examples.append(f"按{num_field}降序排序，取前10条")

    if cat_cols:
        cat_field = cat_cols[0]
        sql_examples.append(f"按{cat_field}分组统计记录数")
        if numeric_cols:
            sql_examples.append(f"按{cat_field}分组计算{numeric_cols[0]}的平均值")

    if len(numeric_cols) >= 2:
        sql_examples.append(f"计算{numeric_cols[0]}和{numeric_cols[1]}的相关性")

    relationships = multi_info.get('relationships', [])
    if relationships:
        rel = relationships[0]
        sql_examples.append(f"关联{rel.get('from_table')}和{rel.get('to_table')}，查询完整信息")

    for ex in sql_examples[:6]:
        if st.button(f"📝 {ex}", key=f"sql_ex_{hash(ex)}", use_container_width=True):
            table_info = "\n".join([f"  - {col}: {info.get('type_desc', info.get('type', 'unknown'))}"
                                    for col, info in variable_types.items()])

            rel_info = ""
            if relationships:
                rel_info = "\n\n**表间关系：**\n"
                for rel in relationships:
                    rel_info += f"  - {rel.get('from_table')}.{rel.get('from_col')} = {rel.get('to_table')}.{rel.get('to_col')}\n"

            question = f"""请根据以下表结构生成SQL查询语句：

## 用户需求
{ex}

## 表名
{main_table}

## 表结构
{table_info}
{rel_info}

## 要求
1. 只输出SQL语句，用```sql```代码块包裹
2. 添加必要的注释说明
3. 考虑性能优化（如索引使用）
4. 如果涉及多表，请正确使用JOIN"""
            st.session_state.pending_question = question
            st.rerun()