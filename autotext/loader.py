"""
文本加载器模块 - 支持从文件、文件夹、DataFrame 加载文本
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from pathlib import Path


class TextLoader:
    """文本加载器 - 支持多种数据源"""

    @staticmethod
    def from_file(file_path: str, encoding: str = 'utf-8') -> List[str]:
        """
        从文本文件加载文本

        参数:
        - file_path: 文件路径
        - encoding: 编码

        返回: 文本列表（每行作为一条文本）
        """
        texts = []
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
        return texts

    @staticmethod
    def from_folder(folder_path: str, extensions: List[str] = None, encoding: str = 'utf-8') -> Dict[str, List[str]]:
        """
        从文件夹加载所有文本文件

        参数:
        - folder_path: 文件夹路径
        - extensions: 文件扩展名列表，默认 ['.txt']
        - encoding: 编码

        返回: {"file_name": [文本列表]}
        """
        if extensions is None:
            extensions = ['.txt']

        result = {}
        folder = Path(folder_path)

        for ext in extensions:
            for file_path in folder.glob(f"*{ext}"):
                texts = TextLoader.from_file(str(file_path), encoding)
                if texts:
                    result[file_path.stem] = texts

        return result

    @staticmethod
    def from_dataframe(df: pd.DataFrame, text_col: str, title_col: Optional[str] = None,
                       time_col: Optional[str] = None, metric_cols: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        从 DataFrame 加载文本列

        参数:
        - df: DataFrame
        - text_col: 文本列名
        - title_col: 标题列名（可选）
        - time_col: 时间列名（可选）
        - metric_cols: 互动指标列名 {"views": "col1", "comments": "col2", ...}

        返回: {
            "texts": List[str],
            "titles": List[str] (可选),
            "dates": List[Any] (可选),
            "metrics": Dict[str, List[Any]] (可选)
        }
        """
        result = {"texts": []}

        # 提取文本列
        if text_col not in df.columns:
            raise ValueError(f"文本列 '{text_col}' 不存在于 DataFrame 中")

        result["texts"] = df[text_col].fillna("").astype(str).tolist()

        # 提取标题列
        if title_col and title_col in df.columns:
            result["titles"] = df[title_col].fillna("").astype(str).tolist()

        # 提取时间列
        if time_col and time_col in df.columns:
            result["dates"] = df[time_col].tolist()

        # 提取互动指标列
        if metric_cols:
            result["metrics"] = {}
            for key, col in metric_cols.items():
                if col and col in df.columns:
                    result["metrics"][key] = df[col].tolist()

        return result

    @staticmethod
    def from_text_list(texts: List[str], titles: Optional[List[str]] = None,
                       dates: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        从文本列表加载

        参数:
        - texts: 文本列表
        - titles: 标题列表（可选）
        - dates: 时间列表（可选）

        返回: {"texts": List[str], "titles": List[str] (可选), "dates": List[Any] (可选)}
        """
        result = {"texts": texts}
        if titles:
            result["titles"] = titles
        if dates:
            result["dates"] = dates
        return result

    @staticmethod
    def detect_encoding(file_path: str) -> str:
        """检测文件编码"""
        import chardet

        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            return result.get('encoding', 'utf-8')