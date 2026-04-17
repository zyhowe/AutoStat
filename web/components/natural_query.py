"""自然语言查询组件 - 纯内容，用于标签页"""

import streamlit as st


def render_natural_query():
    """渲染自然语言查询Tab（纯内容，不含外层标签页）"""
    st.markdown("#### 🔍 自然语言查询")
    st.caption("用中文描述你想查询的数据，AI会基于JSON分析结果和源数据回答")

    # 检查是否有JSON结果和源数据
    if st.session_state.current_json_data is None:
        st.warning("⚠️ 请先完成数据分析，生成JSON报告")
        return

    if "raw_data" not in st.session_state.selected_contexts:
        st.info("💡 提示：自然语言查询需要基于源数据，建议在「选择分析上下文」中勾选「源数据」")

    # 显示数据摘要，帮助用户理解可查询的内容
    with st.expander("📋 数据摘要（可查询的字段）", expanded=False):
        json_data = st.session_state.current_json_data
        variable_types = json_data.get('variable_types', {})

        # 按类型分组显示字段
        fields_by_type = {
            'continuous': '📊 数值字段（可计算均值、总和、最大/最小值）',
            'categorical': '🏷️ 分类字段（可按类别分组统计）',
            'datetime': '📅 日期字段（可按时间范围筛选）',
            'identifier': '🔑 标识符（唯一标识）'
        }

        for typ, display_name in fields_by_type.items():
            cols = [col for col, info in variable_types.items() if info.get('type') == typ]
            if cols:
                st.markdown(f"**{display_name}**")
                # 每行显示5个字段
                for i in range(0, len(cols), 5):
                    st.caption("  " + ", ".join(cols[i:i + 5]))

    # 动态生成示例查询（基于实际字段）
    json_data = st.session_state.current_json_data
    variable_types = json_data.get('variable_types', {})

    # 获取日期字段
    date_cols = [col for col, info in variable_types.items() if info.get('type') == 'datetime']

    # 获取数值字段
    numeric_cols = [col for col, info in variable_types.items() if info.get('type') == 'continuous']

    # 获取分类字段
    cat_cols = [col for col, info in variable_types.items()
                if info.get('type') in ['categorical', 'categorical_numeric', 'ordinal']]

    st.markdown("**💡 示例查询（点击使用）：**")

    # 生成示例列表
    examples = []

    # 时间范围查询示例
    if date_cols:
        date_field = date_cols[0]
        examples.append(f"查询{date_field}在最近30天的数据")
        examples.append(f"按{date_field}统计每天的记录数")

    # 数值统计示例
    if numeric_cols:
        num_field = numeric_cols[0]
        examples.append(f"统计{num_field}的平均值、最大值和最小值")
        examples.append(f"{num_field}最高的前10条记录")
        if len(numeric_cols) >= 2:
            examples.append(f"分析{numeric_cols[0]}和{numeric_cols[1]}的相关性")

    # 分类统计示例
    if cat_cols:
        cat_field = cat_cols[0]
        examples.append(f"按{cat_field}分组统计记录数")
        if numeric_cols:
            examples.append(f"按{cat_field}分组统计{numeric_cols[0]}的平均值")

    # 综合查询示例
    if date_cols and numeric_cols:
        examples.append(f"查询{date_cols[0]}在2024年且{numeric_cols[0]}大于平均值的记录")

    # 显示示例按钮
    for ex in examples[:8]:
        if st.button(f"🔍 {ex}", key=f"ex_{hash(ex)}", use_container_width=True):
            # 获取数据范围信息
            data_shape = json_data.get('data_shape', {})
            rows = data_shape.get('rows', 0)

            question = f"""请根据JSON分析结果和源数据回答以下查询：

用户查询：{ex}

## 数据概况
- 总行数: {rows}
- 字段数: {len(variable_types)}
- 主要字段: {', '.join(list(variable_types.keys())[:15])}

## 字段类型
{chr(10).join([f"- {col}: {info.get('type_desc', info.get('type', 'unknown'))}"
               for col, info in list(variable_types.items())[:20]])}

## 要求
1. 如果查询的是具体数据，请列出相关记录
2. 如果查询的是统计信息，请给出计算结果
3. 请用中文回答，格式清晰
4. 如果无法从数据中获取，请说明原因"""
            st.session_state.pending_question = question
            st.rerun()