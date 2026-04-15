"""
提示词模板模块 - 纯提示词模板，无Streamlit依赖
供MCP服务、CLI和Web界面调用
"""

from typing import Dict, Any, List


def get_recommended_questions(analysis_type: str, has_datetime: bool = False) -> List[str]:
    """
    根据分析类型获取推荐问题列表

    参数:
    - analysis_type: 分析类型 (single/multi/database)
    - has_datetime: 是否包含日期列（用于单文件分析）

    返回:
    - 推荐问题列表
    """
    if analysis_type == "single":
        questions = [
            "📊 解读数据的主要特征和业务含义",
            "⚠️ 分析数据质量问题并给出清洗建议",
            "🔗 分析变量之间的相关性和关联关系",
            "🤖 推荐适合的建模方案和特征选择",
            "🎯 识别数据中的异常值和离群点",
            "📋 总结数据的关键洞察和行动建议"
        ]
        if has_datetime:
            questions.insert(3, "📈 时间序列分析和预测建议")
        return questions

    elif analysis_type == "multi":
        return [
            "🔗 解读表间关系和关联方式",
            "📊 各表的数据质量对比分析",
            "🏗️ 建议构建什么样的数据模型（星型/雪花型）",
            "🔍 跨表关联分析建议和特征工程",
            "📈 多表联合的时间序列分析建议"
        ]

    elif analysis_type == "database":
        return [
            "🏗️ 评估当前数据库设计的合理性",
            "⚡ 索引优化和查询性能建议",
            "📈 BI分析和核心指标体系建议",
            "🛡️ 数据治理和质量管理建议",
            "🔗 表间关系优化建议"
        ]

    else:
        return [
            "📊 解读数据的主要特征",
            "⚠️ 分析数据质量问题",
            "🤖 推荐建模方案"
        ]


def build_single_table_prompt(report_data: Dict[str, Any]) -> str:
    """
    构建单文件分析提示词（供MCP/CLI使用）

    参数:
    - report_data: JSON格式的分析结果数据
    """
    data_shape = report_data.get('data_shape', {})
    variable_types = report_data.get('variable_types', {})
    quality_report = report_data.get('quality_report', {})
    cleaning_suggestions = report_data.get('cleaning_suggestions', [])
    correlations = report_data.get('correlations', {})
    distribution_insights = report_data.get('distribution_insights', {})
    time_series_diagnostics = report_data.get('time_series_diagnostics', {})
    model_recommendations = report_data.get('model_recommendations', [])
    variable_summaries = report_data.get('variable_summaries', {})

    # 变量类型统计
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

    # 缺失值
    missing_list = quality_report.get('missing', [])
    missing_summary = "\n".join([f"  - {m['column']}: {m['percent']:.1f}%" for m in missing_list[:10]])
    if len(missing_list) > 10:
        missing_summary += f"\n  - ... 还有 {len(missing_list) - 10} 个字段"

    # 异常值
    outliers = quality_report.get('outliers', {})
    outlier_summary = "\n".join([f"  - {col}: {info.get('count', 0)}个 ({info.get('percent', 0):.1f}%)"
                                 for col, info in list(outliers.items())[:5]])

    # 强相关对
    high_corrs = correlations.get('high_correlations', [])
    corr_summary = "\n".join([f"  - {c['var1']} ↔ {c['var2']}: r={c['value']}" for c in high_corrs[:5]])

    # 偏态变量
    skewed = distribution_insights.get('skewed_variables', [])
    skewed_summary = "\n".join([f"  - {s['name']}: 偏度={s['skew']}" for s in skewed[:5]])

    # 时间序列
    ts_summary = ""
    if time_series_diagnostics:
        for key, diag in list(time_series_diagnostics.items())[:3]:
            ts_summary += f"  - {key}: 平稳性={'是' if diag.get('is_stationary') else '否'}, "
            ts_summary += f"自相关={'有' if diag.get('has_autocorrelation') else '无'}\n"

    # 日期范围
    date_range_info = ""
    for col, summary in variable_summaries.items():
        if summary.get('type') == 'datetime':
            min_date = summary.get('min_date')
            max_date = summary.get('max_date')
            if min_date and max_date:
                date_range_info = f"  - {col}: {min_date} 至 {max_date}"

    # 建模建议摘要
    model_summary = ""
    for rec in model_recommendations[:3]:
        model_summary += f"  - {rec.get('task_type', '')}: {rec.get('ml', '')}\n"
        if rec.get('feature_columns'):
            features = rec.get('feature_columns', [])[:3]
            model_summary += f"    推荐特征: {', '.join(features)}\n"

    return f"""
你是一位资深专业的数据分析师，拥有10年以上的数据分析经验。请根据以下数据分析报告，提供详细的解读和建议。

## 数据概览
- 数据源: {report_data.get('source_table', '未知')}
- 总行数: {data_shape.get('rows', 0):,}
- 总列数: {data_shape.get('columns', 0)}
- 分析时间: {report_data.get('analysis_time', '未知')}

## 变量类型分布
{type_summary}

## 日期范围
{date_range_info if date_range_info else '  无日期列'}

## 数据质量报告
### 缺失值情况
{missing_summary if missing_summary else '  无缺失值'}

### 异常值情况
{outlier_summary if outlier_summary else '  无异常值'}

### 重复记录
重复记录数: {quality_report.get('duplicates', {}).get('count', 0)} ({quality_report.get('duplicates', {}).get('percent', 0):.1f}%)

## 清洗建议
{chr(10).join([f"  - {s}" for s in cleaning_suggestions[:5]]) if cleaning_suggestions else '  无清洗建议'}

## 相关性分析
### 强相关对 (|r| > 0.7)
{corr_summary if corr_summary else '  无强相关对'}

## 分布洞察
### 偏态变量（偏度 > 2）
{skewed_summary if skewed_summary else '  无偏态变量'}

## 时间序列诊断
{ts_summary if ts_summary else '  未检测到时间序列数据'}

## 建模建议
{model_summary if model_summary else '  无建模建议'}

---

## 请回答以下问题：

1. **数据概览**：这个数据集的核心特征是什么？主要包含哪些类型的变量？

2. **数据质量**：数据质量存在哪些主要问题？建议如何处理？

3. **关键发现**：从相关性和分布中发现了哪些有意义的模式或规律？

4. **建模建议**：如果要进行预测分析，推荐使用什么模型？为什么？

5. **业务建议**：请给出3-5条具体可执行的业务建议或下一步行动计划。

请用中文回答，结构清晰，重点突出。
"""


def build_multi_table_prompt(report_data: Dict[str, Any], multi_info: Dict[str, Any]) -> str:
    """
    构建多文件分析提示词（供MCP/CLI使用）

    参数:
    - report_data: JSON格式的分析结果数据
    - multi_info: 多表信息
    """
    base_prompt = build_single_table_prompt(report_data)

    tables_info = multi_info.get('tables', {})
    relationships = multi_info.get('relationships', [])
    table_groups = multi_info.get('table_groups', [])

    tables_summary = ""
    for name, info in tables_info.items():
        shape = info.get('shape', {})
        tables_summary += f"  - {name}: {shape.get('rows', 0)}行 x {shape.get('columns', 0)}列\n"

    rel_summary = ""
    for rel in relationships[:10]:
        rel_summary += f"  - {rel.get('from_table')}.{rel.get('from_col')} → {rel.get('to_table')}.{rel.get('to_col')} ({rel.get('type', 'unknown')})\n"

    groups_summary = ""
    for group in table_groups:
        if group.get('type') == 'related':
            groups_summary += f"  - 关联表组: {', '.join(group.get('tables', []))} ({len(group.get('relationships', []))}个关系)\n"
        else:
            groups_summary += f"  - 独立表: {group.get('tables', [])[0]}\n"

    multi_section = f"""

## 多表关系信息

### 各表概览
{tables_summary}

### 表间关系
{rel_summary if rel_summary else "  未发现表间关系"}

### 表组信息
{groups_summary}

---

## 额外问题（多表分析）

6. **数据模型**：这些表之间是如何关联的？可以构建什么样的数据模型（星型/雪花型）？

7. **跨表洞察**：跨表分析发现了哪些单表分析无法发现的关联或规律？

8. **整合建议**：如果要进行多表联合分析，推荐使用什么方法？是否需要构建数据仓库？

请继续回答以上问题，与前面的问题一起输出。
"""
    return base_prompt + multi_section


def build_database_prompt(report_data: Dict[str, Any], multi_info: Dict[str, Any],
                          server: str = "", database: str = "") -> str:
    """
    构建数据库分析提示词（供MCP/CLI使用）

    参数:
    - report_data: JSON格式的分析结果数据
    - multi_info: 多表信息
    - server: 数据库服务器地址
    - database: 数据库名称
    """
    base_prompt = build_multi_table_prompt(report_data, multi_info)

    db_section = f"""

## 数据库信息
- 服务器: {server}
- 数据库: {database}

---

## 额外问题（数据库分析）

9. **数据库设计**：当前数据库的数据模型是否合理？有什么优化建议？

10. **性能优化**：建议建立哪些索引来优化查询性能？

11. **BI建议**：如果要进行BI分析，推荐构建哪些核心指标和维度？

12. **数据治理**：请给出数据质量管理和治理建议。

请继续回答以上问题，与前面的问题一起输出。
"""
    return base_prompt + db_section


def build_chat_prompt(report_data: Dict[str, Any], analysis_type: str,
                      user_question: str, chat_history: List[Dict[str, str]]) -> str:
    """
    构建对话提示词（包含报告上下文）- 供MCP/CLI使用

    参数:
    - report_data: JSON格式的分析结果数据
    - analysis_type: 分析类型 (single/multi/database)
    - user_question: 用户当前问题
    - chat_history: 对话历史
    """
    # 根据分析类型构建上下文
    if analysis_type == "single":
        context = build_single_table_prompt(report_data)
    elif analysis_type == "multi":
        multi_info = report_data.get('multi_table_info', {})
        context = build_multi_table_prompt(report_data, multi_info)
    elif analysis_type == "database":
        multi_info = report_data.get('multi_table_info', {})
        server = report_data.get('db_server', '未知')
        database = report_data.get('db_database', '未知')
        context = build_database_prompt(report_data, multi_info, server, database)
    else:
        context = build_single_table_prompt(report_data)

    # 构建对话历史
    history_text = ""
    for msg in chat_history[-10:]:  # 保留最近10条
        role = "用户" if msg["role"] == "user" else "助手"
        history_text += f"{role}: {msg['content']}\n\n"

    return f"""
你是专业的数据分析师，正在回答用户关于刚才分析的数据集的问题。

## 数据集分析报告（作为上下文参考）

{context}

---

## 对话历史
{history_text}

---

## 当前问题
用户: {user_question}

请基于上述数据集的分析结果，回答用户的问题。如果问题与数据集无关，请礼貌地引导用户询问数据相关问题。
用中文回答，结构清晰。
"""