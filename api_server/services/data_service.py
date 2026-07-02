"""数据服务 - 不依赖 web/"""
import pandas as pd
from typing import Dict, Any, List

from autostat.loader import DataLoader
from autostat.core.base import BaseAnalyzer


class DataService:
    """数据处理服务"""

    @staticmethod
    def load_file(file_path: str) -> pd.DataFrame:
        """加载文件"""
        return DataLoader.load_from_file(file_path)

    @staticmethod
    def get_preview(df: pd.DataFrame, rows: int = 100) -> Dict:
        """获取数据预览"""
        return {
            "head": df.head(rows).to_dict(orient="records"),
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict()
        }

    @staticmethod
    def infer_types(df: pd.DataFrame) -> Dict[str, str]:
        """推断变量类型"""
        analyzer = BaseAnalyzer(df, quiet=True)
        analyzer._infer_variable_types()
        return analyzer.variable_types