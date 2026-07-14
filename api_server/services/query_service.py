"""查询服务 - 使用 Pandas 执行查询"""
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from api_server.services.session_service import SessionService
from api_server.services.config_service import ConfigService
from autostat.loader import DataLoader


class QueryService:
    """查询服务 - 使用 Pandas 执行筛选查询"""

    def __init__(self):
        self.session_service = SessionService()
        self.config_service = ConfigService()
        self._schema_cache = {}

    def get_data_source(self, session_id: str) -> Tuple[str, Optional[Dict]]:
        """获取会话的数据源类型"""
        # 1. 检查 Parquet 缓存
        parquet_path = self.session_service.get_data_parquet_path(session_id)
        if parquet_path.exists():
            return 'parquet', {'path': str(parquet_path)}

        # 2. 检查是否从数据库加载
        session = self.session_service.get_session(session_id)
        if session:
            tables_info = session.get('tables_info', {})
            if tables_info:
                db_config = tables_info.get('db_config')
                table_names = tables_info.get('tables', [])
                if db_config and table_names:
                    return 'sql_server', {
                        'db_config': db_config,
                        'table_names': table_names
                    }

        # 3. 检查是否有原始文件
        file_info = self.session_service.get_file(session_id)
        if file_info:
            return 'file', {'path': file_info['path']}

        return 'none', {}

    def query_data(
        self,
        session_id: str,
        filters: List[Dict[str, Any]],
        limit: int = 100,
        fields: Optional[List[str]] = None,
        order_by: Optional[str] = None
    ) -> Tuple[bool, str, pd.DataFrame]:
        """
        查询数据 - 使用 Pandas 布尔索引

        参数:
        - filters: [{"field": "age", "operator": "eq", "value": 30}]
        - limit: 返回行数
        - fields: 返回字段列表

        返回:
        - (success, message, dataframe)
        """
        source_type, info = self.get_data_source(session_id)

        if source_type == 'none':
            return False, "没有可用的数据源", pd.DataFrame()

        try:
            # 加载数据
            if source_type == 'parquet':
                df = pd.read_parquet(info['path'])
            elif source_type == 'sql_server':
                db_config = info['db_config']
                table_names = info['table_names']
                table_name = table_names[0] if isinstance(table_names, list) else table_names
                df = DataLoader.load_sql_server(
                    server=db_config.get('server'),
                    database=db_config.get('database'),
                    table_name=table_name,
                    username=db_config.get('username'),
                    password=db_config.get('password'),
                    trusted_connection=db_config.get('trusted_connection', False),
                    limit=limit
                )
                if df is None:
                    return False, "加载数据失败", pd.DataFrame()
            elif source_type == 'file':
                df = DataLoader.load_from_file(info['path'])
            else:
                return False, "未知数据源", pd.DataFrame()

            # 应用筛选条件
            for f in filters:
                field = f.get('field')
                operator = f.get('operator', 'eq')
                value = f.get('value')

                if field not in df.columns:
                    continue

                if operator == 'eq':
                    df = df[df[field] == value]
                elif operator == 'gt':
                    df = df[df[field] > value]
                elif operator == 'lt':
                    df = df[df[field] < value]
                elif operator == 'gte':
                    df = df[df[field] >= value]
                elif operator == 'lte':
                    df = df[df[field] <= value]
                elif operator == 'contains':
                    df = df[df[field].astype(str).str.contains(str(value), na=False, case=False)]

            total = len(df)

            # 排序
            if order_by:
                try:
                    df = df.sort_values(order_by.split()[0], ascending='DESC' not in order_by.upper())
                except:
                    pass

            # 选择字段
            if fields:
                valid_fields = [f for f in fields if f in df.columns]
                if valid_fields:
                    df = df[valid_fields]

            # 限制行数
            df = df.head(limit)

            if len(df) == 0:
                return True, "查询结果为空", pd.DataFrame()

            return True, f"查询成功，返回 {len(df)} 行（共 {total} 条匹配）", df

        except Exception as e:
            return False, f"查询失败: {e}", pd.DataFrame()

    def get_schema_info(self, session_id: str) -> Dict[str, Any]:
        """获取表结构信息（用于 Prompt）"""
        if session_id in self._schema_cache:
            return self._schema_cache[session_id]

        source_type, info = self.get_data_source(session_id)
        result = {
            'source_type': source_type,
            'table_name': '未知表',
            'columns': [],
            'total_rows': 0,
            'sample_data': []
        }

        try:
            if source_type == 'parquet':
                df = pd.read_parquet(info['path'])
                result['table_name'] = Path(info['path']).stem
                result['columns'] = [{'name': c, 'type': str(df[c].dtype)} for c in df.columns]
                result['total_rows'] = len(df)
                result['sample_data'] = df.head(5).to_dict('records')

            elif source_type == 'sql_server':
                db_config = info['db_config']
                table_names = info['table_names']
                table_name = table_names[0] if isinstance(table_names, list) else table_names
                result['table_name'] = table_name

                # 获取列信息
                schema = DataLoader.get_table_schema(
                    server=db_config.get('server'),
                    database=db_config.get('database'),
                    table_name=table_name,
                    username=db_config.get('username'),
                    password=db_config.get('password'),
                    trusted_connection=db_config.get('trusted_connection', False)
                )
                if schema:
                    result['columns'] = [{'name': s.get('名称'), 'type': s.get('类型')} for s in schema]

                # 获取样例数据
                try:
                    df = DataLoader.load_sql_server(
                        server=db_config.get('server'),
                        database=db_config.get('database'),
                        table_name=table_name,
                        username=db_config.get('username'),
                        password=db_config.get('password'),
                        trusted_connection=db_config.get('trusted_connection', False),
                        limit=10
                    )
                    if df is not None and len(df) > 0:
                        result['sample_data'] = df.head(5).to_dict('records')
                        result['total_rows'] = len(df)
                except:
                    pass

            elif source_type == 'file':
                df = DataLoader.load_from_file(info['path'])
                result['table_name'] = Path(info['path']).stem
                result['columns'] = [{'name': c, 'type': str(df[c].dtype)} for c in df.columns]
                result['total_rows'] = len(df)
                result['sample_data'] = df.head(5).to_dict('records')

        except Exception as e:
            print(f"获取 schema 失败: {e}")

        self._schema_cache[session_id] = result
        return result