"""
字段检测模块 - 自动识别标题列、正文列、时间列、互动指标列
"""

import pandas as pd
import re
from typing import Dict, List, Optional, Tuple, Any


class FieldDetector:
    """字段检测器 - 自动识别文本数据中的特殊字段"""

    # 标题列关键词
    TITLE_KEYWORDS = ['title', '标题', 'subject', '主题', 'headline', '名称', 'name']

    # 时间列关键词
    TIME_KEYWORDS = ['time', '时间', 'date', '日期', 'created_at', '创建时间', 'updated_at', '更新时间',
                     'publish', '发布时间', 'post_time', '发文时间']

    # 互动指标关键词
    METRIC_KEYWORDS = {
        'views': ['view', '阅读', '浏览', 'pv', '曝光'],
        'comments': ['comment', '评论', '回复', 'reply', '留言'],
        'likes': ['like', '点赞', '好评', '赞同', '赞'],
        'shares': ['share', '分享', '转发', 'retweet', '转载']
    }

    def __init__(self, df: pd.DataFrame):
        """
        初始化字段检测器

        参数:
        - df: 数据框
        """
        self.df = df
        self._detected = None

    def detect_all(self) -> Dict[str, Any]:
        """
        检测所有字段

        返回:
        {
            "title_col": str or None,      # 标题列名
            "content_col": str or None,    # 正文字段名
            "time_col": str or None,       # 时间列名
            "metric_cols": {               # 互动指标列
                "views": str or None,
                "comments": str or None,
                "likes": str or None,
                "shares": str or None
            },
            "other_text_cols": List[str]   # 其他文本列
        }
        """
        if self._detected is not None:
            return self._detected

        result = {
            "title_col": None,
            "content_col": None,
            "time_col": None,
            "metric_cols": {k: None for k in self.METRIC_KEYWORDS.keys()},
            "other_text_cols": []
        }

        # 识别时间列
        result["time_col"] = self._detect_time_column()

        # 识别标题列
        result["title_col"] = self._detect_title_column()

        # 识别互动指标列
        result["metric_cols"] = self._detect_metric_columns()

        # 识别正文字段（内容最长的文本列）
        result["content_col"] = self._detect_content_column(result["title_col"])

        # 其他文本列
        result["other_text_cols"] = self._detect_other_text_columns(
            result["title_col"],
            result["content_col"],
            result["time_col"],
            result["metric_cols"]
        )

        self._detected = result
        return result

    def _detect_time_column(self) -> Optional[str]:
        """检测时间列"""
        for col in self.df.columns:
            col_lower = col.lower()
            # 检查列名
            for kw in self.TIME_KEYWORDS:
                if kw in col_lower:
                    return col

            # 检查数据类型
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                return col

            # 检查是否能转换为日期时间
            try:
                sample = self.df[col].dropna().head(10)
                if len(sample) > 0:
                    converted = pd.to_datetime(sample, errors='coerce')
                    if converted.notna().sum() > len(sample) * 0.8:
                        return col
            except:
                pass

        return None

    def _detect_title_column(self) -> Optional[str]:
        """检测标题列"""
        candidates = []

        for col in self.df.columns:
            col_lower = col.lower()
            # 关键词匹配
            for kw in self.TITLE_KEYWORDS:
                if kw in col_lower:
                    candidates.append((col, 100))
                    break
            else:
                # 文本长度特征
                sample = self.df[col].dropna()
                if len(sample) > 0:
                    avg_len = sample.astype(str).str.len().mean()
                    # 标题通常较短（10-50字符）
                    if 10 < avg_len < 50:
                        candidates.append((col, 50))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        return None

    def _detect_metric_columns(self) -> Dict[str, Optional[str]]:
        """检测互动指标列"""
        result = {k: None for k in self.METRIC_KEYWORDS.keys()}

        for metric_type, keywords in self.METRIC_KEYWORDS.items():
            for col in self.df.columns:
                col_lower = col.lower()
                for kw in keywords:
                    if kw in col_lower:
                        # 检查是否为数值类型
                        if pd.api.types.is_numeric_dtype(self.df[col]):
                            result[metric_type] = col
                            break
                if result[metric_type] is not None:
                    break

        return result

    def _detect_content_column(self, title_col: Optional[str]) -> Optional[str]:
        """检测正文字段（内容最长的文本列）"""
        text_cols = []
        for col in self.df.columns:
            # 排除已识别的标题列和时间列
            if col == title_col:
                continue

            # 检查是否为文本类型
            if pd.api.types.is_string_dtype(self.df[col]) or pd.api.types.is_object_dtype(self.df[col]):
                sample = self.df[col].dropna()
                if len(sample) > 0:
                    avg_len = sample.astype(str).str.len().mean()
                    if avg_len > 20:  # 正文通常较长
                        text_cols.append((col, avg_len))

        if text_cols:
            text_cols.sort(key=lambda x: x[1], reverse=True)
            return text_cols[0][0]

        return None

    def _detect_other_text_columns(self, title_col: Optional[str],
                                    content_col: Optional[str],
                                    time_col: Optional[str],
                                    metric_cols: Dict) -> List[str]:
        """检测其他文本列"""
        exclude = set()
        if title_col:
            exclude.add(title_col)
        if content_col:
            exclude.add(content_col)
        if time_col:
            exclude.add(time_col)
        for v in metric_cols.values():
            if v:
                exclude.add(v)

        other_cols = []
        for col in self.df.columns:
            if col in exclude:
                continue
            if pd.api.types.is_string_dtype(self.df[col]) or pd.api.types.is_object_dtype(self.df[col]):
                sample = self.df[col].dropna()
                if len(sample) > 0:
                    avg_len = sample.astype(str).str.len().mean()
                    if avg_len > 10:
                        other_cols.append(col)

        return other_cols

    def get_summary(self) -> Dict:
        """获取检测结果摘要"""
        result = self.detect_all()
        return {
            "has_title": result["title_col"] is not None,
            "has_time": result["time_col"] is not None,
            "has_content": result["content_col"] is not None,
            "has_metrics": any(v is not None for v in result["metric_cols"].values()),
            "other_text_cols_count": len(result["other_text_cols"])
        }