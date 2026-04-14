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

    # 数值列统计
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

    # 分类列统计
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

    # 日期列统计
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