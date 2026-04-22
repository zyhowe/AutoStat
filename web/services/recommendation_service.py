"""
推荐服务 - 智能参数推荐
"""

import pandas as pd
import os
from typing import Dict, Any, Tuple, Optional


class RecommendationService:
    """智能推荐服务"""

    @staticmethod
    def get_recommended_params(file_path: str = None, df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        根据文件/数据特征推荐最优参数

        返回:
        {
            "sample_rate": float,
            "date_features_level": str,
            "auto_clean": bool,
            "output_level": str
        }
        """
        # 获取数据大小
        if file_path and os.path.exists(file_path):
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        elif df is not None:
            # 估算DataFrame内存占用
            file_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        else:
            file_size_mb = 0

        # 获取数据特征
        has_datetime = False
        has_numeric = False
        n_rows = 0

        if df is not None:
            has_datetime = any(df[col].dtype == 'datetime64' for col in df.columns)
            has_numeric = any(df[col].dtype in ['int64', 'float64'] for col in df.columns)
            n_rows = len(df)

        # 推荐采样率
        if file_size_mb < 10:
            sample_rate = 1.0
        elif file_size_mb < 100:
            sample_rate = 0.5
        else:
            sample_rate = 0.1

        # 推荐日期特征级别
        if has_datetime and has_numeric and n_rows > 100:
            date_features_level = "full"
        elif has_datetime:
            date_features_level = "basic"
        else:
            date_features_level = "none"

        # 推荐自动清洗
        auto_clean = file_size_mb < 50  # 小文件自动清洗

        # 推荐输出级别
        if n_rows > 10000:
            output_level = "compact"
        else:
            output_level = "full"

        return {
            "sample_rate": sample_rate,
            "date_features_level": date_features_level,
            "auto_clean": auto_clean,
            "output_level": output_level
        }

    @staticmethod
    def get_recommended_target(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """
        推荐最佳目标列

        返回: (列名, 推荐任务类型)
        """
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        if not numeric_cols:
            return None, None

        # 优先选择方差最大的列（信息量最大）
        variances = df[numeric_cols].var()
        if not variances.empty:
            best_target = variances.idxmax()
        else:
            best_target = numeric_cols[0]

        # 判断任务类型
        unique_count = df[best_target].nunique()
        if unique_count <= 10:
            task_type = "classification"
        else:
            task_type = "regression"

        return best_target, task_type

    @staticmethod
    def get_recommended_features(df: pd.DataFrame, target_col: str = None, max_features: int = 20) -> list:
        """
        推荐特征列

        返回: 特征列名列表
        """
        # 排除目标列
        exclude_cols = [target_col] if target_col else []

        # 排除标识符列
        id_keywords = ['id', '_id', '编号', 'code', 'key', '用户ID', '订单号']
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower or col_lower.endswith(kw) for kw in id_keywords):
                exclude_cols.append(col)

        # 排除高缺失率列
        high_missing = df.columns[df.isna().sum() / len(df) > 0.5].tolist()
        exclude_cols.extend(high_missing)

        # 获取数值列和分类列
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # 过滤
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
        cat_cols = [c for c in cat_cols if c not in exclude_cols and df[c].nunique() <= 20]

        # 按相关性排序（如果有目标列）
        if target_col and target_col in df.columns:
            numeric_corrs = []
            for col in numeric_cols:
                if col != target_col:
                    corr = df[[target_col, col]].corr().iloc[0, 1]
                    if not pd.isna(corr):
                        numeric_corrs.append((col, abs(corr)))
            numeric_corrs.sort(key=lambda x: x[1], reverse=True)
            numeric_cols = [c for c, _ in numeric_corrs[:15]]

        # 合并，优先数值列
        features = numeric_cols + cat_cols

        return features[:max_features]

    @staticmethod
    def should_auto_analyze(file_size_mb: float, n_rows: int) -> bool:
        """
        判断是否应该自动开始分析
        """
        # 小文件自动分析，大文件需要用户确认
        return file_size_mb < 50 and n_rows < 100000