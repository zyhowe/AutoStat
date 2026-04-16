"""AI智能解读组件"""

import streamlit as st
import pandas as pd
import re
from autostat.prompts import get_recommended_questions


def get_initial_question(analysis_type: str) -> str:
    """根据分析类型获取初始提问内容"""
    if analysis_type == "single":
        return "请根据这份数据分析报告，帮我解读一下数据的主要特征、质量问题和建模建议。"
    elif analysis_type == "multi":
        return "请根据这份多表关联分析报告，帮我解读一下表间关系、数据特征和跨表分析建议。"
    elif analysis_type == "database":
        return "请根据这份数据库分析报告，帮我解读一下数据模型、性能优化和BI建设建议。"
    else:
        return "请根据这份数据分析报告，帮我解读一下主要内容。"


def extract_json_context(json_data: dict, analysis_type: str) -> str:
    """从 JSON 数据提取上下文信息（纯数据提取，不含提示词前缀）"""

    data_shape = json_data.get('data_shape', {})
    variable_types = json_data.get('variable_types', {})
    quality_report = json_data.get('quality_report', {})
    cleaning_suggestions = json_data.get('cleaning_suggestions', [])
    correlations = json_data.get('correlations', {})
    model_recommendations = json_data.get('model_recommendations', [])
    variable_summaries = json_data.get('variable_summaries', {})

    type_counts = {}
    type_display = {
        'continuous': '连续变量',
        'categorical': '分类变量',
        'categorical_numeric': '数值型分类',
        'ordinal': '有序分类',
        'datetime': '日期时间',
        'identifier': '标识符',
        'text': '文本'
    }
    for col, info in variable_types.items():
        typ = info.get('type', 'unknown')
        type_counts[typ] = type_counts.get(typ, 0) + 1

    type_summary = ", ".join([f"{type_display.get(t, t)}: {c}" for t, c in type_counts.items()])

    missing_list = quality_report.get('missing', [])
    missing_summary = "\n".join([f"  - {m['column']}: {m['percent']:.1f}%" for m in missing_list[:10]])
    if len(missing_list) > 10:
        missing_summary += f"\n  - ... 还有 {len(missing_list) - 10} 个字段"

    outliers = quality_report.get('outliers', {})
    outlier_summary = "\n".join([f"  - {col}: {info.get('count', 0)}个 ({info.get('percent', 0):.1f}%)"
                                 for col, info in list(outliers.items())[:5]])

    high_corrs = correlations.get('high_correlations', [])
    corr_summary = "\n".join([f"  - {c['var1']} ↔ {c['var2']}: r={c['value']}" for c in high_corrs[:5]])

    model_summary = ""
    for rec in model_recommendations[:3]:
        model_summary += f"  - {rec.get('task_type', '')}: {rec.get('ml', '')}\n"

    date_range_info = ""
    for col, summary in variable_summaries.items():
        if summary.get('type') == 'datetime':
            min_date = summary.get('min_date')
            max_date = summary.get('max_date')
            if min_date and max_date:
                date_range_info += f"  - {col}: {min_date} 至 {max_date}\n"

    numeric_summary = ""
    for col, summary in list(variable_summaries.items())[:10]:
        if summary.get('type') == 'continuous':
            min_val = summary.get('min', 'N/A')
            max_val = summary.get('max', 'N/A')
            mean_val = summary.get('mean', 'N/A')
            median_val = summary.get('median', 'N/A')

            if isinstance(min_val, (int, float)):
                min_str = f"{min_val:.2f}"
            else:
                min_str = str(min_val)
            if isinstance(max_val, (int, float)):
                max_str = f"{max_val:.2f}"
            else:
                max_str = str(max_val)
            if isinstance(mean_val, (int, float)):
                mean_str = f"{mean_val:.2f}"
            else:
                mean_str = str(mean_val)
            if isinstance(median_val, (int, float)):
                median_str = f"{median_val:.2f}"
            else:
                median_str = str(median_val)

            numeric_summary += f"  - {col}: 均值={mean_str}, 中位数={median_str}, 范围=[{min_str}, {max_str}]\n"

    multi_info = ""
    if analysis_type in ["multi", "database"]:
        multi_tables = json_data.get('multi_table_info', {}).get('tables', {})
        relationships = json_data.get('multi_table_info', {}).get('relationships', [])
        if multi_tables:
            multi_info = "\n### 多表信息\n"
            multi_info += "**包含的表:**\n"
            for name, info in multi_tables.items():
                shape = info.get('shape', {})
                multi_info += f"  - {name}: {shape.get('rows', 0)}行 x {shape.get('columns', 0)}列\n"
            if relationships:
                multi_info += "\n**表间关系:**\n"
                for rel in relationships[:5]:
                    multi_info += f"  - {rel.get('from_table')}.{rel.get('from_col')} → {rel.get('to_table')}.{rel.get('to_col')}\n"

    return f"""
### JSON 分析结果数据

**数据概览**
- 总行数: {data_shape.get('rows', 0):,}
- 总列数: {data_shape.get('columns', 0)}
- 变量类型分布: {type_summary}

**日期范围**
{date_range_info if date_range_info else '  无日期列'}

**数值变量分布**
{numeric_summary if numeric_summary else '  无数值变量'}

**数据质量**
缺失值:
{missing_summary if missing_summary else '  无缺失值'}

异常值:
{outlier_summary if outlier_summary else '  无异常值'}

重复记录: {quality_report.get('duplicates', {}).get('count', 0)}条

**清洗建议**
{chr(10).join([f"  - {s}" for s in cleaning_suggestions[:5]]) if cleaning_suggestions else '  无清洗建议'}

**相关性分析**
{corr_summary if corr_summary else '  无强相关对'}

**建模建议**
{model_summary if model_summary else '  无建模建议'}

{multi_info}
"""


def extract_html_context(html_content: str) -> str:
    """从 HTML 报告提取关键信息（纯数据提取，不含提示词前缀）"""

    context_parts = []

    title_match = re.search(r'<h1>(.*?)</h1>', html_content)
    if title_match:
        context_parts.append(f"报告标题: {title_match.group(1)}")

    stat_cards = re.findall(r'<div class="stat-card"><div class="value">(.*?)</div><div class="label">(.*?)</div>', html_content)
    if stat_cards:
        stats_summary = "**关键指标:**\n"
        for value, label in stat_cards:
            stats_summary += f"  - {label}: {value}\n"
        context_parts.append(stats_summary)

    insight_match = re.search(r'<h2>💡 核心洞察</h2>(.*?)</div>', html_content, re.DOTALL)
    if insight_match:
        insights = re.findall(r'<li>(.*?)</li>', insight_match.group(1))
        if insights:
            context_parts.append("**核心洞察:**\n" + "\n".join([f"  - {i}" for i in insights[:3]]))

    type_match = re.search(r'变量类型分布</strong><br>(.*?)</div>', html_content, re.DOTALL)
    if type_match:
        type_lines = [line.strip() for line in type_match.group(1).split('<br>') if line.strip()]
        if type_lines:
            context_parts.append("**变量类型:** " + ", ".join(type_lines[:5]))

    cleaning_match = re.search(r'<h2>🧹 数据清洗建议</h2>(.*?)</div>', html_content, re.DOTALL)
    if cleaning_match:
        cleaning_items = re.findall(r'<td>(.*?)</td>', cleaning_match.group(1))
        if cleaning_items:
            context_parts.append("**清洗建议:** " + ", ".join(cleaning_items[:3]))

    return "\n".join(context_parts) if context_parts else "无法提取HTML摘要信息"


def extract_data_context(raw_data_preview: dict) -> str:
    """从原始数据提取上下文信息（纯数据提取，不含提示词前缀）"""
    if raw_data_preview is None:
        return "无源数据信息"

    shape = raw_data_preview.get('shape', (0, 0))
    dtypes = raw_data_preview.get('dtypes', {})
    summary_stats = raw_data_preview.get('summary_stats', {})
    preview = raw_data_preview.get('preview', pd.DataFrame())
    numeric_cols = raw_data_preview.get('numeric_cols', [])
    cat_cols = raw_data_preview.get('cat_cols', [])
    date_cols = raw_data_preview.get('date_cols', [])

    context_parts = []
    context_parts.append(f"- 数据形状: {shape[0]} 行 × {shape[1]} 列")

    type_counts = {}
    for col, dtype in dtypes.items():
        dtype_str = str(dtype)
        if 'int' in dtype_str or 'float' in dtype_str:
            type_counts['数值型'] = type_counts.get('数值型', 0) + 1
        elif 'datetime' in dtype_str:
            type_counts['日期型'] = type_counts.get('日期型', 0) + 1
        elif 'object' in dtype_str:
            type_counts['文本型'] = type_counts.get('文本型', 0) + 1
        else:
            type_counts['其他'] = type_counts.get('其他', 0) + 1

    type_summary = ", ".join([f"{t}: {c}" for t, c in type_counts.items()])
    context_parts.append(f"- 字段类型分布: {type_summary}")

    columns = list(dtypes.keys())[:30]
    context_parts.append(f"- 字段列表: {', '.join(columns)}")
    if len(dtypes) > 30:
        context_parts.append(f"  ... 还有 {len(dtypes) - 30} 个字段")

    if date_cols:
        context_parts.append("\n### 日期范围")
        for col in date_cols:
            stats = summary_stats.get(col, {})
            if stats.get('type') == 'datetime':
                context_parts.append(f"- {col}: {stats.get('min', 'N/A')} 至 {stats.get('max', 'N/A')} "
                                   f"(共{stats.get('unique_dates', 0)}个日期)")

    if numeric_cols:
        context_parts.append("\n### 数值列统计")
        for col in numeric_cols:
            stats = summary_stats.get(col, {})
            if stats.get('type') == 'numeric':
                context_parts.append(f"- {col}: 范围=[{stats.get('min', 'N/A'):.2f}, {stats.get('max', 'N/A'):.2f}], "
                                   f"均值={stats.get('mean', 'N/A'):.2f}, "
                                   f"中位数={stats.get('median', 'N/A'):.2f}")

    if cat_cols:
        context_parts.append("\n### 分类列统计")
        for col in cat_cols:
            stats = summary_stats.get(col, {})
            if stats.get('type') == 'categorical':
                top_vals = stats.get('top_values', {})
                top_str = ", ".join([f"{k}:{v}" for k, v in list(top_vals.items())[:3]])
                context_parts.append(f"- {col}: {stats.get('unique_count', 0)}个唯一值, "
                                   f"最高频占比={stats.get('top_percent', 0):.1f}%, "
                                   f"前3: {top_str}")

    if len(preview) > 0:
        preview_str = preview.head(50).to_string()
        context_parts.append(f"\n### 数据预览 (前50行)\n```\n{preview_str}\n```")

    return "\n".join(context_parts)


def build_context_prompt(selected_contexts, analysis_type, json_data, html_content, raw_data_preview, source_name):
    """根据选中的上下文构建完整提示词"""

    context_parts = []

    analysis_type_map = {
        "single": "单文件分析",
        "multi": "多文件分析",
        "database": "数据库分析"
    }
    context_parts.append(f"**分析类型:** {analysis_type_map.get(analysis_type, '未知')}")
    context_parts.append(f"**数据源:** {source_name}")
    context_parts.append("")

    if "json_result" in selected_contexts and json_data:
        context_parts.append(extract_json_context(json_data, analysis_type))

    if "html_report" in selected_contexts and html_content:
        html_summary = extract_html_context(html_content)
        if html_summary:
            context_parts.append(f"### HTML 报告摘要\n{html_summary}")

    if "raw_data" in selected_contexts and raw_data_preview:
        context_parts.append(f"### 源数据信息\n{extract_data_context(raw_data_preview)}")

    full_context = "\n\n".join(context_parts)

    full_prompt = f"""
你是一位专业的数据分析师。以下是根据用户选择的上下文提供的分析数据。

{full_context}

---

## 注意事项
- 请基于上述提供的数据上下文回答问题
- 源数据包含完整的数据预览（前50行）和统计信息
- 你可以根据日期范围、数值范围等条件查询具体数据
- 如果用户询问具体数据（如某年某月的数据），请根据数据预览和统计信息回答
- 用中文回答，结构清晰
"""
    return full_prompt


def send_message(question: str, analysis_type: str = None):
    """发送消息到 AI（使用当前选中的上下文）"""
    if not question or not question.strip():
        return False

    if st.session_state.llm_client is None:
        st.error("请先在侧边栏配置大模型")
        return False

    if st.session_state.current_json_data is None:
        st.error("请先完成数据分析")
        return False

    st.session_state.chat_messages.append({"role": "user", "content": question})

    system_prompt = build_context_prompt(
        st.session_state.selected_contexts,
        st.session_state.current_analysis_type,
        st.session_state.current_json_data,
        st.session_state.current_html,
        st.session_state.raw_data_preview,
        st.session_state.current_source_name
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state.chat_messages[:-1]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    with st.spinner("AI 思考中..."):
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in st.session_state.llm_client.chat_stream(messages):
                if chunk:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

    st.session_state.chat_messages.append({"role": "assistant", "content": full_response})
    return True


def render_scenario_recommendation():
    """渲染业务场景推荐Tab"""
    st.markdown("#### 🎯 业务场景推荐")
    st.caption("基于JSON分析结果和源数据，自动识别业务场景并提供分析建议")

    # 检查是否有JSON结果
    if st.session_state.current_json_data is None:
        st.warning("⚠️ 请先完成数据分析，生成JSON报告")
        return

    # 检查是否有源数据
    if "raw_data" not in st.session_state.selected_contexts:
        st.info("💡 提示：场景推荐需要基于JSON分析结果和源数据，建议勾选「JSON结果」和「源数据」")

    # 提取数据特征用于展示
    json_data = st.session_state.current_json_data
    variable_types = json_data.get('variable_types', {})
    quality_report = json_data.get('quality_report', {})
    time_series_diagnostics = json_data.get('time_series_diagnostics', {})

    # 统计变量类型
    type_counts = {}
    for col, info in variable_types.items():
        typ = info.get('type', 'unknown')
        type_counts[typ] = type_counts.get(typ, 0) + 1

    type_names = {
        'continuous': '连续变量',
        'categorical': '分类变量',
        'datetime': '日期时间',
        'identifier': '标识符',
        'text': '文本'
    }

    # 显示数据特征摘要
    with st.expander("📊 数据特征摘要", expanded=False):
        st.caption("**变量类型分布：**")
        for typ, name in type_names.items():
            if typ in type_counts:
                st.caption(f"  - {name}: {type_counts[typ]}个")

        # 时间序列信息
        if time_series_diagnostics:
            st.caption("\n**时间序列检测：**")
            for key, diag in list(time_series_diagnostics.items())[:3]:
                st.caption(f"  - {key}: 平稳性={'是' if diag.get('is_stationary') else '否'}, "
                           f"自相关={'有' if diag.get('has_autocorrelation') else '无'}")

        # 数据质量
        missing_count = len(quality_report.get('missing', []))
        outlier_count = len(quality_report.get('outliers', {}))
        if missing_count > 0 or outlier_count > 0:
            st.caption("\n**数据质量提示：**")
            if missing_count > 0:
                st.caption(f"  - 存在 {missing_count} 个字段有缺失值")
            if outlier_count > 0:
                st.caption(f"  - 存在 {outlier_count} 个字段有异常值")

    if st.button("🔍 自动识别业务场景", key="detect_scenario", use_container_width=True):
        with st.spinner("正在分析数据特征，识别业务场景..."):
            # 提取关键信息用于场景识别
            numeric_cols = [col for col, info in variable_types.items()
                            if info.get('type') == 'continuous']
            categorical_cols = [col for col, info in variable_types.items()
                                if info.get('type') in ['categorical', 'categorical_numeric', 'ordinal']]
            date_cols = [col for col, info in variable_types.items()
                         if info.get('type') == 'datetime']

            # 构建场景识别问题
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
            send_message(question)
            st.rerun()

    st.divider()

    # 动态生成快速开始按钮（基于实际字段）
    st.markdown("**快速开始：**")

    # 根据实际字段生成推荐视角
    col1, col2 = st.columns(2)

    # 检测是否有销售相关字段
    sales_keywords = ['销售', '销售额', '销量', '价格', '金额', '收入', 'sales', 'revenue', 'amount', 'price']
    has_sales = any(any(kw in col.lower() for kw in sales_keywords) for col in variable_types.keys())

    # 检测是否有用户相关字段
    user_keywords = ['用户', '会员', '客户', 'user', 'customer', 'member']
    has_user = any(any(kw in col.lower() for kw in user_keywords) for col in variable_types.keys())

    # 检测是否有财务相关字段
    finance_keywords = ['成本', '利润', '费用', '支出', '预算', 'cost', 'profit', 'expense', 'budget']
    has_finance = any(any(kw in col.lower() for kw in finance_keywords) for col in variable_types.keys())

    # 检测是否有运营相关字段
    operation_keywords = ['点击', '曝光', '转化', 'pv', 'uv', 'click', 'view', 'conversion']
    has_operation = any(any(kw in col.lower() for kw in operation_keywords) for col in variable_types.keys())

    with col1:
        if has_sales:
            if st.button("📊 销售分析视角", key="scene_sales", use_container_width=True):
                # 提取销售相关字段
                sales_fields = [col for col in variable_types.keys()
                                if any(kw in col.lower() for kw in sales_keywords)]
                fields_str = ", ".join(sales_fields[:5]) if sales_fields else "相关字段"
                question = f"""请从销售分析的角度解读数据，重点关注以下字段：
{fields_str}

请分析：
1. 销售额/销量的时间趋势（如有日期字段）
2. 产品/类别排行分析
3. 客户贡献度分析（如有客户字段）
4. 销售预测建议

请基于实际数据特征回答。"""
                send_message(question)
                st.rerun()
        else:
            st.button("📊 销售分析视角", key="scene_sales_disabled", disabled=True, use_container_width=True,
                      help="未检测到销售相关字段")

    with col2:
        if has_user:
            if st.button("👥 用户分析视角", key="scene_user", use_container_width=True):
                user_fields = [col for col in variable_types.keys()
                               if any(kw in col.lower() for kw in user_keywords)]
                fields_str = ", ".join(user_fields[:5]) if user_fields else "相关字段"
                question = f"""请从用户分析的角度解读数据，重点关注以下字段：
{fields_str}

请分析：
1. 用户画像特征
2. 用户行为模式
3. 用户分层和价值分析
4. 用户留存/流失建议

请基于实际数据特征回答。"""
                send_message(question)
                st.rerun()
        else:
            st.button("👥 用户分析视角", key="scene_user_disabled", disabled=True, use_container_width=True,
                      help="未检测到用户相关字段")

    col1, col2 = st.columns(2)
    with col1:
        if has_finance:
            if st.button("💰 财务分析视角", key="scene_finance", use_container_width=True):
                finance_fields = [col for col in variable_types.keys()
                                  if any(kw in col.lower() for kw in finance_keywords)]
                fields_str = ", ".join(finance_fields[:5]) if finance_fields else "相关字段"
                question = f"""请从财务分析的角度解读数据，重点关注以下字段：
{fields_str}

请分析：
1. 收入成本结构
2. 利润率分析
3. 预算执行情况（如有预算字段）
4. 财务风险提示

请基于实际数据特征回答。"""
                send_message(question)
                st.rerun()
        else:
            st.button("💰 财务分析视角", key="scene_finance_disabled", disabled=True, use_container_width=True,
                      help="未检测到财务相关字段")

    with col2:
        if has_operation:
            if st.button("📈 运营分析视角", key="scene_operation", use_container_width=True):
                op_fields = [col for col in variable_types.keys()
                             if any(kw in col.lower() for kw in operation_keywords)]
                fields_str = ", ".join(op_fields[:5]) if op_fields else "相关字段"
                question = f"""请从运营分析的角度解读数据，重点关注以下字段：
{fields_str}

请分析：
1. 核心运营指标
2. 效率分析
3. 渠道效果评估（如有渠道字段）
4. 优化建议

请基于实际数据特征回答。"""
                send_message(question)
                st.rerun()
        else:
            st.button("📈 运营分析视角", key="scene_operation_disabled", disabled=True, use_container_width=True,
                      help="未检测到运营相关字段")


def render_natural_query():
    """渲染自然语言查询Tab"""
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
    variable_summaries = json_data.get('variable_summaries', {})

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
            send_message(ex)
            st.rerun()

    st.divider()

    query = st.text_input("输入您的查询", key="natural_query",
                          placeholder="例如：查询销售额大于1000的订单，或统计每个月的销售额变化趋势")

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("查询", key="run_natural_query", use_container_width=True):
            if "raw_data" not in st.session_state.selected_contexts:
                st.error("请先在「选择分析上下文」中勾选「源数据」")
            elif st.session_state.current_json_data is None:
                st.error("请先完成数据分析")
            elif query.strip():
                # 获取数据范围信息
                data_shape = json_data.get('data_shape', {})
                rows = data_shape.get('rows', 0)

                question = f"""请根据JSON分析结果和源数据回答以下查询：

用户查询：{query}

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
                send_message(question)
                st.rerun()
            else:
                st.warning("请输入查询内容")


def render_sql_generator():
    """渲染SQL生成Tab（仅数据库模式）"""
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
            # 根据类型添加图标
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
                    st.caption(
                        f"  - {rel.get('from_table')}.{rel.get('from_col')} → {rel.get('to_table')}.{rel.get('to_col')}")

    # 动态生成示例SQL查询（基于实际字段）
    json_data = st.session_state.current_json_data
    variable_types = json_data.get('variable_types', {})
    multi_info = json_data.get('multi_table_info', {})
    tables = multi_info.get('tables', {})

    # 获取主表名
    main_table = list(tables.keys())[0] if tables else "your_table"

    # 获取字段
    numeric_cols = [col for col, info in variable_types.items() if info.get('type') == 'continuous']
    cat_cols = [col for col, info in variable_types.items()
                if info.get('type') in ['categorical', 'categorical_numeric', 'ordinal']]
    date_cols = [col for col, info in variable_types.items() if info.get('type') == 'datetime']

    st.markdown("**💡 示例SQL查询（点击使用）：**")

    # 生成示例
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

    # 多表关联示例
    relationships = multi_info.get('relationships', [])
    if relationships:
        rel = relationships[0]
        sql_examples.append(f"关联{rel.get('from_table')}和{rel.get('to_table')}，查询完整信息")

    # 显示示例按钮
    for ex in sql_examples[:6]:
        if st.button(f"📝 {ex}", key=f"sql_ex_{hash(ex)}", use_container_width=True):
            # 构建表结构信息
            table_info = "\n".join([f"  - {col}: {info.get('type_desc', info.get('type', 'unknown'))}"
                                    for col, info in variable_types.items()])

            # 获取表间关系
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
            send_message(question)
            st.rerun()

    st.divider()

    query = st.text_input("输入您的查询", key="sql_query",
                          placeholder="例如：查询销售额最高的前10个产品，或关联订单表和用户表")

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("生成SQL", key="generate_sql", use_container_width=True):
            if "raw_data" not in st.session_state.selected_contexts:
                st.error("请先在「选择分析上下文」中勾选「源数据」")
            elif st.session_state.current_json_data is None:
                st.error("请先完成数据分析")
            elif query.strip():
                # 构建表结构信息
                variable_types = st.session_state.current_json_data.get('variable_types', {})
                table_info = "\n".join([f"  - {col}: {info.get('type_desc', info.get('type', 'unknown'))}"
                                        for col, info in variable_types.items()])

                # 获取表间关系
                multi_info = st.session_state.current_json_data.get('multi_table_info', {})
                relationships = multi_info.get('relationships', [])
                rel_info = ""
                if relationships:
                    rel_info = "\n\n**表间关系：**\n"
                    for rel in relationships:
                        rel_info += f"  - {rel.get('from_table')}.{rel.get('from_col')} = {rel.get('to_table')}.{rel.get('to_col')}\n"

                question = f"""请根据以下表结构生成SQL查询语句：

## 用户需求
{query}

## 表结构
{table_info}
{rel_info}

## 要求
1. 只输出SQL语句，用```sql```代码块包裹
2. 添加必要的注释说明
3. 考虑性能优化（如索引使用）
4. 如果涉及多表，请正确使用JOIN
5. 表名使用实际表名"""
                send_message(question)
                st.rerun()
            else:
                st.warning("请输入查询内容")

def render_chat_interface():
    """渲染聊天界面"""

    st.markdown("#### 📚 选择分析上下文")
    st.caption("勾选要提供给AI的上下文信息，多选可获得更全面的分析")

    col_ctx1, col_ctx2, col_ctx3 = st.columns(3)

    with col_ctx1:
        ctx_json = st.checkbox(
            "📊 JSON 结果",
            value="json_result" in st.session_state.selected_contexts,
            key="ctx_json",
            help="包含完整的统计分析数据（变量类型、相关性、质量报告等）"
        )

    with col_ctx2:
        ctx_html = st.checkbox(
            "📄 HTML 报告",
            value="html_report" in st.session_state.selected_contexts,
            key="ctx_html",
            help="包含可视化的分析报告（图表、统计摘要等）"
        )

    with col_ctx3:
        ctx_data = st.checkbox(
            "🗃️ 源数据",
            value="raw_data" in st.session_state.selected_contexts,
            key="ctx_data",
            help="包含原始数据预览（前50行）及完整统计摘要"
        )

    selected = []
    if ctx_json:
        selected.append("json_result")
    if ctx_html:
        selected.append("html_report")
    if ctx_data:
        selected.append("raw_data")

    if set(selected) != set(st.session_state.selected_contexts):
        st.session_state.selected_contexts = selected

    if not st.session_state.selected_contexts:
        st.warning("⚠️ 请至少选择一种上下文信息")
        return

    with st.expander(f"📋 已选上下文 ({len(st.session_state.selected_contexts)}项)", expanded=False):
        ctx_names = {
            "json_result": "📊 JSON 结果 - 完整的统计分析数据",
            "html_report": "📄 HTML 报告 - 可视化分析报告",
            "raw_data": "🗃️ 源数据 - 原始数据预览（前50行）及统计摘要"
        }
        for ctx in st.session_state.selected_contexts:
            st.markdown(f"- {ctx_names.get(ctx, ctx)}")

    st.divider()

    # 历史消息显示
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    # Tab页
    is_database = st.session_state.current_analysis_type == "database"

    if is_database:
        tab_labels = ["💬 对话", "🎯 场景推荐", "🔍 自然查询", "📝 SQL生成"]
    else:
        tab_labels = ["💬 对话", "🎯 场景推荐", "🔍 自然查询"]

    tabs = st.tabs(tab_labels)

    with tabs[0]:
        if st.session_state.current_json_data:
            has_datetime = False
            if st.session_state.current_json_data.get('variable_types'):
                for info in st.session_state.current_json_data.get('variable_types', {}).values():
                    if info.get('type') == 'datetime':
                        has_datetime = True
                        break

            recommended_qs = get_recommended_questions(
                st.session_state.current_analysis_type,
                has_datetime
            )

            st.markdown("#### 💡 推荐问题")
            st.caption("点击下方问题快速提问")

            cols = st.columns(min(len(recommended_qs), 3))
            for i, q in enumerate(recommended_qs):
                col_idx = i % 3
                if cols[col_idx].button(q, key=f"rec_q_{i}", use_container_width=True):
                    send_message(q)
                    st.rerun()

            st.divider()

        prompt = st.chat_input("输入您的问题...", key="chat_input")
        if prompt and prompt.strip():
            send_message(prompt.strip())
            st.rerun()

        if len(st.session_state.chat_messages) == 0:
            st.info("💡 提示：点击上方推荐问题快速提问，或在下方输入框输入您的问题。")

    with tabs[1]:
        render_scenario_recommendation()

    with tabs[2]:
        render_natural_query()

    if is_database:
        with tabs[3]:
            render_sql_generator()