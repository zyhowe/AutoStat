"""数据服务 - 不依赖 web/"""
import pandas as pd
from typing import Dict, Any, List, Optional
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

    # ==================== ✅ 统一 Parquet 保存/加载 ====================

    @staticmethod
    def save_to_parquet(df: pd.DataFrame, session_service, session_id: str, table_name: str) -> bool:
        """
        将 DataFrame 保存为 Parquet 格式（按表名）

        参数:
        - df: 要保存的 DataFrame
        - session_service: SessionService 实例
        - session_id: 会话ID
        - table_name: 表名

        返回:
        - bool: 是否保存成功
        """
        try:
            path = session_service.get_table_parquet_path(session_id, table_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False, compression='zstd')
            print(f"✅ 表 {table_name} 已保存到 Parquet: {path}")
            return True
        except Exception as e:
            print(f"❌ Parquet 保存失败 ({table_name}): {e}")
            return False

    @staticmethod
    def load_parquet(session_service, session_id: str, table_name: str) -> Optional[pd.DataFrame]:
        """
        从 Parquet 文件加载指定表

        参数:
        - session_service: SessionService 实例
        - session_id: 会话ID
        - table_name: 表名

        返回:
        - pd.DataFrame 或 None
        """
        path = session_service.get_table_parquet_path(session_id, table_name)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception as e:
                print(f"❌ 加载 Parquet 失败 ({table_name}): {e}")
                return None
        return None

    @staticmethod
    def load_all_tables_from_parquet(session_service, session_id: str) -> Dict[str, pd.DataFrame]:
        """
        加载会话的所有表（从 Parquet）

        返回:
        - {表名: DataFrame}
        """
        table_names = session_service.get_all_table_names(session_id)
        tables = {}
        for name in table_names:
            df = DataService.load_parquet(session_service, session_id, name)
            if df is not None and not df.empty:
                tables[name] = df
        return tables

    # ==================== 多表支持（旧版兼容） ====================

    @staticmethod
    def save_tables_to_parquet(tables: Dict[str, pd.DataFrame], session_service, session_id: str) -> Dict[str, bool]:
        """
        保存多个表到 Parquet（每个表独立保存）

        参数:
        - tables: {表名: DataFrame}
        - session_service: SessionService 实例
        - session_id: 会话ID

        返回:
        - {表名: 是否成功}
        """
        results = {}
        for name, df in tables.items():
            if df is None or df.empty:
                results[name] = False
                continue
            results[name] = DataService.save_to_parquet(df, session_service, session_id, name)
            if results[name]:
                session_service.save_table_info(session_id, name, {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "saved_at": pd.Timestamp.now().isoformat()
                })
        return results

    @staticmethod
    def load_table_from_parquet(session_service, session_id: str, table_name: str) -> Optional[pd.DataFrame]:
        """从 Parquet 加载单个表"""
        return DataService.load_parquet(session_service, session_id, table_name)