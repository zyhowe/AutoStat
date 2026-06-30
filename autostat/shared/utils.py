"""
通用工具函数
"""

import numpy as np
from typing import List, Tuple, Optional, Any, Dict  # 🆕 添加 Any, Dict
from datetime import datetime


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法"""
    if b == 0:
        return default
    return a / b


def calculate_confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    计算置信区间

    参数:
    - values: 数值列表
    - confidence: 置信度 (0-1)

    返回: (lower, upper)
    """
    if len(values) < 2:
        return (values[0], values[0]) if values else (0, 0)

    import scipy.stats as stats

    mean = np.mean(values)
    std = np.std(values, ddof=1)
    n = len(values)

    z_score = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = z_score * std / np.sqrt(n)

    return (mean - margin, mean + margin)


def detect_outliers_iqr(
    values: List[float],
    multiplier: float = 1.5
) -> List[int]:
    """
    使用IQR检测异常值索引

    参数:
    - values: 数值列表
    - multiplier: IQR倍数

    返回: 异常值索引列表
    """
    if len(values) < 4:
        return []

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    if iqr == 0:
        return []

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    return [i for i, v in enumerate(values) if v < lower or v > upper]


def normalize_data(
    values: List[float],
    method: str = "minmax"
) -> List[float]:
    """
    数据归一化

    参数:
    - values: 数值列表
    - method: "minmax" 或 "zscore"

    返回: 归一化后的数值列表
    """
    if not values:
        return []

    if method == "minmax":
        min_val = min(values)
        max_val = max(values)
        if max_val == min_val:
            return [0.5] * len(values)
        return [(v - min_val) / (max_val - min_val) for v in values]

    elif method == "zscore":
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return [0] * len(values)
        return [(v - mean) / std for v in values]

    else:
        return values


def get_timestamp() -> str:
    """获取当前时间戳字符串"""
    return datetime.now().isoformat()


def truncate_text(text: str, max_len: int = 100, suffix: str = "...") -> str:
    """截断文本"""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len - len(suffix)] + suffix


def safe_dict_get(d: Dict, key: str, default: Any = None) -> Any:
    """
    安全获取字典值，支持点号分隔的嵌套键

    参数:
    - d: 字典
    - key: 键名，支持点号分隔（如 "a.b.c"）
    - default: 默认值

    返回: 对应的值或默认值
    """
    if not d or not isinstance(d, dict):
        return default

    keys = key.split(".")
    current = d

    for k in keys:
        if isinstance(current, dict):
            current = current.get(k)
        else:
            return default

        if current is None:
            return default

    return current


def group_by(
    items: List[Dict],
    key: str
) -> Dict[str, List[Dict]]:
    """按指定键分组"""
    result = {}
    for item in items:
        value = item.get(key)
        if value is None:
            value = "unknown"
        if value not in result:
            result[value] = []
        result[value].append(item)
    return result


def sorted_by(
    items: List[Dict],
    key: str,
    reverse: bool = False
) -> List[Dict]:
    """按指定键排序"""
    return sorted(items, key=lambda x: x.get(key, 0), reverse=reverse)


def date_range(start: str, end: str, freq: str = "D") -> List[str]:
    """生成日期范围"""
    from datetime import datetime, timedelta

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    if freq == "D":
        delta = timedelta(days=1)
    elif freq == "W":
        delta = timedelta(weeks=1)
    elif freq == "M":
        # 简化：每月1日
        dates = []
        current = start_dt.replace(day=1)
        while current <= end_dt:
            dates.append(current.strftime("%Y-%m-%d"))
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        return dates
    else:
        delta = timedelta(days=1)

    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current.strftime("%Y-%m-%d"))
        current += delta

    return dates


def safe_round(value: float, decimals: int = 2) -> float:
    """安全四舍五入"""
    if value is None:
        return 0.0
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return 0.0


def is_numeric(value: Any) -> bool:
    """检查值是否为数值类型"""
    if value is None:
        return False
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    """
    扁平化嵌套字典

    参数:
    - d: 嵌套字典
    - parent_key: 父键名
    - sep: 分隔符

    返回: 扁平化后的字典
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)