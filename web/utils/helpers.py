"""辅助函数模块"""

import streamlit as st
import sys
import io
import pandas as pd
import numpy as np


def capture_and_run(func, *args, **kwargs):
    """捕获输出并运行函数"""
    f = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = f
        sys.stderr = f
        result = func(*args, **kwargs)
        output = f.getvalue()
        return result, output
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def get_raw_data_preview(data):
    """获取原始数据预览信息（包含完整数据）"""
    if data is None or data.empty:
        return None

    full_data = data
    preview = data.head(100) if len(data) > 100 else data

    dtypes = {col: str(dtype) for col, dtype in data.dtypes.items()}

    summary_stats = {}

    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in numeric_cols[:15]:
        series = data[col].dropna()
        if len(series) > 0:
            summary_stats[col] = {
                'type': 'numeric',
                'min': float(series.min()),
                'max': float(series.max()),
                'mean': float(series.mean()),
                'median': float(series.median()),
                'std': float(series.std()),
                'q1': float(series.quantile(0.25)),
                'q3': float(series.quantile(0.75))
            }

    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols[:15]:
        vc = data[col].value_counts()
        if len(vc) > 0:
            summary_stats[col] = {
                'type': 'categorical',
                'unique_count': len(vc),
                'top_values': {str(k): int(v) for k, v in vc.head(5).items()},
                'top_percent': float(vc.head(1).values[0] / len(data) * 100) if len(vc) > 0 else 0
            }

    date_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
    for col in date_cols[:5]:
        if len(data[col].dropna()) > 0:
            summary_stats[col] = {
                'type': 'datetime',
                'min': str(data[col].min()),
                'max': str(data[col].max()),
                'unique_dates': data[col].nunique()
            }

    return {
        'preview': preview,
        'full_data': full_data,
        'shape': (len(data), len(data.columns)),
        'dtypes': dtypes,
        'summary_stats': summary_stats,
        'numeric_cols': numeric_cols[:15],
        'cat_cols': cat_cols[:15],
        'date_cols': date_cols[:5]
    }


def detect_scenario_from_fields(variable_types: dict, column_names: list) -> dict:
    """
    根据字段特征自动识别业务场景（供MCP/CLI使用）

    参数:
    - variable_types: 变量类型字典
    - column_names: 列名列表

    返回:
    - 场景识别结果字典
    """
    col_lower = [c.lower() for c in column_names]

    score = {
        "sales": 0,
        "user": 0,
        "finance": 0,
        "operation": 0,
        "risk": 0
    }

    sales_keywords = ['销售', '销售额', '销量', '产品', '商品', '订单', 'price', 'amount', 'revenue']
    for kw in sales_keywords:
        if any(kw in c for c in col_lower):
            score["sales"] += 1

    user_keywords = ['用户', '会员', 'customer', 'user', '注册', '登录', '活跃', '留存']
    for kw in user_keywords:
        if any(kw in c for c in col_lower):
            score["user"] += 1

    finance_keywords = ['收入', '成本', '利润', '预算', '支出', 'income', 'cost', 'profit', 'expense']
    for kw in finance_keywords:
        if any(kw in c for c in col_lower):
            score["finance"] += 1

    operation_keywords = ['pv', 'uv', '点击', '转化', '渠道', '曝光', 'click', 'conversion']
    for kw in operation_keywords:
        if any(kw in c for c in col_lower):
            score["operation"] += 1

    risk_keywords = ['风险', '逾期', '坏账', '欺诈', '评分', 'risk', 'default', 'fraud', 'score']
    for kw in risk_keywords:
        if any(kw in c for c in col_lower):
            score["risk"] += 1

    max_score = max(score.values()) if score else 0
    if max_score == 0:
        return {"scenario": "general", "confidence": 0, "reason": "无法识别特定场景"}

    scenario_map = {
        "sales": "销售分析",
        "user": "用户分析",
        "finance": "财务分析",
        "operation": "运营分析",
        "risk": "风控分析"
    }

    detected = [k for k, v in score.items() if v == max_score][0]

    return {
        "scenario": scenario_map.get(detected, "通用分析"),
        "confidence": max_score / (len(sales_keywords) * 0.3),
        "reason": f"检测到 {max_score} 个相关关键词"
    }


def generate_api_example(analysis_type: str, file_path: str = "data.csv") -> str:
    """
    生成API调用示例代码

    参数:
    - analysis_type: 分析类型 (single/multi/database)
    - file_path: 文件路径（单文件模式使用）

    返回:
    - Python代码字符串
    """
    if analysis_type == "single":
        return f'''from autostat import AutoStatisticalAnalyzer
from autostat import Reporter

# 分析数据
analyzer = AutoStatisticalAnalyzer("{file_path}")
analyzer.generate_full_report()

# 导出报告
reporter = Reporter(analyzer)
reporter.to_html("report.html")
reporter.to_json("result.json")'''

    elif analysis_type == "multi":
        return '''from autostat import MultiTableStatisticalAnalyzer
import pandas as pd

# 加载多个表
tables = {{
    "orders": pd.read_csv("orders.csv"),
    "users": pd.read_csv("users.csv"),
    "products": pd.read_csv("products.csv")
}}

# 分析
analyzer = MultiTableStatisticalAnalyzer(tables)
analyzer.analyze_all_tables()

# 导出结果
analyzer.to_json("result.json")'''

    elif analysis_type == "database":
        return '''from autostat import MultiTableStatisticalAnalyzer
from autostat.loader import DataLoader

# 从数据库加载数据
tables = DataLoader.load_multiple_tables(
    server="your_server",
    database="your_database",
    table_names=["users", "orders"],
    username="your_username",
    password="your_password"
)

# 分析
analyzer = MultiTableStatisticalAnalyzer(tables)
analyzer.analyze_all_tables()'''

    else:
        return "# 请选择分析模式"