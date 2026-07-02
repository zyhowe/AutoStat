"""数据库服务"""
import pandas as pd
from typing import Dict, Any, List, Optional
from autostat.loader import DataLoader
from autostat.core.base import BaseAnalyzer


class DatabaseService:
    """数据库服务 - 连接 SQL Server"""

    @staticmethod
    def load_tables(
        config: Dict[str, Any],
        table_names: List[str],
        limit: int = 5000,
        max_text_length: int = 100,
        relationships: Optional[List[Dict]] = None
    ) -> Dict[str, pd.DataFrame]:
        """从数据库加载表"""
        tables = DataLoader.load_multiple_tables(
            server=config.get('server'),
            database=config.get('database'),
            table_names=table_names,
            username=config.get('username'),
            password=config.get('password'),
            trusted_connection=config.get('trusted_connection', False),
            limit=limit,
            relationships=relationships,
            max_text_length=max_text_length
        )
        return tables

    @staticmethod
    def infer_types(df: pd.DataFrame) -> Dict[str, str]:
        """推断变量类型"""
        analyzer = BaseAnalyzer(df, quiet=True)
        analyzer._infer_variable_types()
        return analyzer.variable_types

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
    def test_connection(config: Dict[str, Any]) -> tuple:
        """测试数据库连接"""
        try:
            import pyodbc
            server = config.get('server')
            database = config.get('database')
            username = config.get('username')
            password = config.get('password')
            trusted_connection = config.get('trusted_connection', False)

            possible_drivers = [
                'ODBC Driver 17 for SQL Server',
                'ODBC Driver 13 for SQL Server',
                'SQL Server Native Client 11.0',
                'SQL Server'
            ]

            available_drivers = pyodbc.drivers()

            for driver in possible_drivers:
                if driver in available_drivers:
                    if trusted_connection or not username:
                        conn_str = f'DRIVER={{{driver}}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'
                    else:
                        conn_str = f'DRIVER={{{driver}}};SERVER={server};DATABASE={database};UID={username};PWD={password};'

                    conn = pyodbc.connect(conn_str, timeout=10)
                    conn.close()
                    return True, f"连接成功 (驱动: {driver})"

            return False, "未找到可用驱动"
        except ImportError:
            return False, "pyodbc 未安装，请运行: pip install pyodbc"
        except Exception as e:
            return False, f"连接失败: {str(e)}"