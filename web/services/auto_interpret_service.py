"""
自动解读服务 - 分析完成后自动生成综合解读
"""

import streamlit as st
from typing import Dict, Any, Optional

from web.services.session_service import SessionService
from web.services.storage_service import StorageService


def get_interpretation_prompt(json_data: Dict[str, Any]) -> str:
    """生成综合解读提示词"""
    data_shape = json_data.get('data_shape', {})
    variable_types = json_data.get('variable_types', {})
    quality_report = json_data.get('quality_report', {})
    correlations = json_data.get('correlations', {})
    time_series_diagnostics = json_data.get('time_series_diagnostics', {})
    model_recommendations = json_data.get('model_recommendations', [])

    # 统计变量类型
    type_counts = {}
    type_display = {
        'continuous': '连续变量',
        'categorical': '分类变量',
        'datetime': '日期时间',
        'identifier': '标识符'
    }
    for info in variable_types.values():
        typ = info.get('type', 'unknown')
        type_counts[typ] = type_counts.get(typ, 0) + 1
    type_summary = "、".join([f"{type_display.get(t, t)} {c}个" for t, c in type_counts.items() if t in type_display])

    # 数据质量
    missing_count = len(quality_report.get('missing', []))
    outlier_count = len(quality_report.get('outliers', {}))
    dup_count = quality_report.get('duplicates', {}).get('count', 0)

    # 强相关（全部列出）
    high_corrs = correlations.get('high_correlations', [])
    top_corrs = [f"{c['var1']}↔{c['var2']}({c['value']})" for c in high_corrs]

    # 时间序列（全部列出）
    ts_vars = [k for k, v in time_series_diagnostics.items() if v.get('has_autocorrelation')]
    ts_summary = "、".join(ts_vars) if ts_vars else "无"

    # 模型机会
    model_types = list(set([r.get('task_type', '') for r in model_recommendations]))
    model_summary = "、".join(model_types) if model_types else "暂无"

    return f"""请对以下数据提供综合解读，限制在1000字以内：

## 数据概览
- 总行数: {data_shape.get('rows', 0):,}
- 总列数: {data_shape.get('columns', 0)}
- 变量类型: {type_summary}

## 数据质量
- 存在缺失值的字段: {missing_count}个
- 存在异常值的字段: {outlier_count}个
- 重复记录: {dup_count}条

## 关键发现
- 强相关关系: {', '.join(top_corrs) if top_corrs else '无明显强相关'}
- 时间序列规律: {ts_summary}
- 建模机会: {model_summary}

## 要求
1. 用中文回答，结构清晰
2. 包含：数据整体评估、主要发现、数据质量建议、分析价值
3. 控制在1000字以内
"""


def auto_interpret(session_id: str, llm_client) -> Optional[str]:
    """
    自动解读数据

    返回: 解读文本，失败返回 None
    """
    try:
        # 加载分析结果
        json_data = StorageService.load_json("analysis_result", session_id)
        if not json_data:
            return None

        # 生成提示词
        prompt = get_interpretation_prompt(json_data)

        # 调用大模型
        messages = [
            {"role": "system", "content": "你是专业的数据分析师，提供简洁、专业的综合解读。"},
            {"role": "user", "content": prompt}
        ]

        interpretation = llm_client.chat(messages, temperature=0.5)

        # 限制长度
        if len(interpretation) > 1000:
            interpretation = interpretation[:1000] + "..."

        # 保存解读结果
        StorageService.save_text("auto_interpretation", interpretation, session_id)

        return interpretation

    except Exception as e:
        print(f"自动解读失败: {e}")
        return None