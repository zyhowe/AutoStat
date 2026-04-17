"""场景推荐组件 - 纯内容，用于标签页"""

import streamlit as st


def render_scenario_recommendation():
    """渲染业务场景推荐Tab（纯内容，不含外层标签页）"""
    st.markdown("#### 🎯 场景推荐")
    st.caption("基于实际字段特征，自动推荐分析视角")

    # 检查是否有JSON结果
    if st.session_state.current_json_data is None:
        st.warning("⚠️ 请先完成数据分析，生成JSON报告")
        return

    # 提取数据特征
    json_data = st.session_state.current_json_data
    variable_types = json_data.get('variable_types', {})
    quality_report = json_data.get('quality_report', {})
    time_series_diagnostics = json_data.get('time_series_diagnostics', {})

    # 获取各类型字段
    numeric_cols = [col for col, info in variable_types.items() if info.get('type') == 'continuous']
    categorical_cols = [col for col, info in variable_types.items()
                        if info.get('type') in ['categorical', 'categorical_numeric', 'ordinal']]
    date_cols = [col for col, info in variable_types.items() if info.get('type') == 'datetime']

    # 显示数据特征摘要
    with st.expander("📊 数据特征摘要", expanded=False):
        type_names = {
            'continuous': '连续变量',
            'categorical': '分类变量',
            'datetime': '日期时间',
            'identifier': '标识符',
            'text': '文本'
        }
        type_counts = {}
        for col, info in variable_types.items():
            typ = info.get('type', 'unknown')
            type_counts[typ] = type_counts.get(typ, 0) + 1

        st.caption("**变量类型分布：**")
        for typ, name in type_names.items():
            if typ in type_counts:
                st.caption(f"  - {name}: {type_counts[typ]}个")

        if time_series_diagnostics:
            st.caption("\n**时间序列检测：**")
            for key, diag in list(time_series_diagnostics.items())[:3]:
                st.caption(f"  - {key}: 平稳性={'是' if diag.get('is_stationary') else '否'}, "
                           f"自相关={'有' if diag.get('has_autocorrelation') else '无'}")

        missing_count = len(quality_report.get('missing', []))
        outlier_count = len(quality_report.get('outliers', {}))
        if missing_count > 0 or outlier_count > 0:
            st.caption("\n**数据质量提示：**")
            if missing_count > 0:
                st.caption(f"  - 存在 {missing_count} 个字段有缺失值")
            if outlier_count > 0:
                st.caption(f"  - 存在 {outlier_count} 个字段有异常值")

    st.divider()

    # ==================== 动态生成分析视角 ====================
    st.markdown("**📌 推荐分析视角（点击使用）：**")

    # 根据实际字段生成视角
    perspectives = []

    # 1. 如果有日期字段 + 数值字段 → 趋势分析
    if date_cols and numeric_cols:
        date_field = date_cols[0]
        num_field = numeric_cols[0]
        perspectives.append({
            "icon": "📈",
            "name": "趋势分析",
            "question": f"请分析 {num_field} 随时间（{date_field}）的变化趋势，包括季节性、周期性和异常波动。"
        })

    # 2. 如果有分类字段 + 数值字段 → 对比分析
    if categorical_cols and numeric_cols:
        cat_field = categorical_cols[0]
        num_field = numeric_cols[0]
        perspectives.append({
            "icon": "📊",
            "name": "对比分析",
            "question": f"请按 {cat_field} 分组分析 {num_field} 的分布差异，找出显著差异的类别。"
        })

    # 3. 如果有多个数值字段 → 相关性分析
    if len(numeric_cols) >= 2:
        num_field1 = numeric_cols[0]
        num_field2 = numeric_cols[1]
        perspectives.append({
            "icon": "🔗",
            "name": "相关性分析",
            "question": f"请分析 {num_field1} 与 {num_field2} 的相关性，并解释业务含义。"
        })

    # 4. 如果有多个分类字段 → 关联分析
    if len(categorical_cols) >= 2:
        cat_field1 = categorical_cols[0]
        cat_field2 = categorical_cols[1]
        perspectives.append({
            "icon": "🔍",
            "name": "关联分析",
            "question": f"请分析 {cat_field1} 与 {cat_field2} 之间的关联关系，使用交叉分析和卡方检验。"
        })

    # 5. 如果有日期字段 → 时间序列预测
    if date_cols and numeric_cols and time_series_diagnostics:
        has_auto = any(d.get('has_autocorrelation') for d in time_series_diagnostics.values())
        if has_auto:
            num_field = numeric_cols[0]
            perspectives.append({
                "icon": "🔮",
                "name": "预测分析",
                "question": f"请基于历史数据预测 {num_field} 的未来趋势，并给出置信区间。"
            })

    # 6. 如果有高缺失率字段 → 数据质量分析
    high_missing = [m for m in quality_report.get('missing', []) if m.get('percent', 0) > 20]
    if high_missing:
        missing_fields = ", ".join([m['column'] for m in high_missing[:3]])
        perspectives.append({
            "icon": "⚠️",
            "name": "数据质量分析",
            "question": f"请分析以下字段的缺失值问题：{missing_fields}，并给出填充建议。"
        })

    # 7. 如果有异常值 → 异常分析
    outliers = quality_report.get('outliers', {})
    if outliers:
        outlier_fields = ", ".join(list(outliers.keys())[:3])
        perspectives.append({
            "icon": "🚨",
            "name": "异常值分析",
            "question": f"请分析 {outlier_fields} 的异常值，判断是否合理以及如何处理。"
        })

    # 显示视角按钮（每行2个）
    if perspectives:
        for i in range(0, len(perspectives), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(perspectives):
                    p = perspectives[idx]
                    if col.button(f"{p['icon']} {p['name']}", key=f"perspective_{idx}", use_container_width=True):
                        st.session_state.pending_question = p['question']
                        st.rerun()
    else:
        st.info("💡 根据当前数据特征，暂无自动生成的分析视角。请尝试自然语言查询。")

    st.divider()

    # ==================== 自动识别业务场景 ====================
    st.markdown("**🎯 业务场景识别**")
    st.caption("AI自动识别当前数据属于什么业务场景")

    if st.button("🔍 自动识别业务场景", key="detect_scenario", use_container_width=True):
        question = f"""请根据以下数据特征，自动识别业务场景：

## 数据特征
- 连续变量: {len(numeric_cols)}个 ({', '.join(numeric_cols[:8])})
- 分类变量: {len(categorical_cols)}个 ({', '.join(categorical_cols[:8])})
- 日期变量: {len(date_cols)}个 ({', '.join(date_cols[:8])})
- 是否包含时间序列: {'是' if time_series_diagnostics else '否'}

## 主要字段
{', '.join(list(variable_types.keys())[:20])}

## 任务
1. 根据字段特征判断这是什么业务场景（如：销售分析、用户分析、财务分析、运营分析、风控分析等）
2. 给出判断依据（哪些字段支持这个判断）
3. 推荐该场景下的核心分析维度和关键指标
4. 提供2-3个具体的分析问题和业务洞察建议

请用中文回答，结构清晰。"""
        st.session_state.pending_question = question
        st.rerun()