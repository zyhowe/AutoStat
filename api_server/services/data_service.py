"""数据服务 - 不依赖 web/"""
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path

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

    # ==================== ✅ 新增：保存 Parquet ====================
    @staticmethod
    def save_to_parquet(df: pd.DataFrame, path: Path) -> bool:
        """
        将 DataFrame 保存为 Parquet 格式

        参数:
        - df: 要保存的 DataFrame
        - path: 保存路径

        返回:
        - bool: 是否保存成功
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False, compression='zstd')
            print(f"✅ Parquet 缓存已保存: {path}")
            return True
        except Exception as e:
            print(f"❌ Parquet 保存失败: {e}")
            return False

    @staticmethod
    def load_parquet(path: Path) -> pd.DataFrame:
        """
        从 Parquet 文件加载数据

        参数:
        - path: Parquet 文件路径

        返回:
        - pd.DataFrame
        """
        return pd.read_parquet(path)