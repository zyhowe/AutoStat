"""
数据加载器模块
支持 CSV、Excel、JSON、TXT、SQL Server
"""

import os
import numpy as np
import pandas as pd
import pyodbc
from typing import List, Dict, Optional, Any


class DataLoader:
    """数据加载器 - 支持多种数据源"""

    DEFAULT_DATE_COLUMNS = [
        '日期', 'date', 'Date', '时间', 'time', 'Time',
        '销售日期', '创建时间', '更新时间', 'datetime', 'DateTime',
        'order_date', 'OrderDate', 'created_at', 'updated_at'
    ]

    LARGE_FIELD_TYPES = [
        'text', 'ntext', 'image',
        'varchar(max)', 'nvarchar(max)', 'varbinary(max)', 'xml'
    ]

    @staticmethod
    def load_csv(file_path, encoding='utf-8-sig', parse_dates=True, date_columns=None, **kwargs):
        """加载CSV文件"""
        if not parse_dates:
            return pd.read_csv(file_path, encoding=encoding, **kwargs)

        try:
            sample_df = pd.read_csv(file_path, encoding=encoding, nrows=5, **kwargs)
            existing_columns = list(sample_df.columns)

            if date_columns is not None:
                if isinstance(date_columns, str):
                    date_columns = [date_columns]
                valid_date_cols = [col for col in date_columns if col in existing_columns]
            else:
                valid_date_cols = [col for col in existing_columns if col in DataLoader.DEFAULT_DATE_COLUMNS]

            if valid_date_cols:
                print(f"  📅 自动识别日期列: {valid_date_cols}")
                return pd.read_csv(file_path, encoding=encoding, parse_dates=valid_date_cols, **kwargs)
            else:
                return pd.read_csv(file_path, encoding=encoding, **kwargs)
        except Exception as e:
            print(f"  ⚠️ 日期解析失败: {e}")
            return pd.read_csv(file_path, encoding=encoding, **kwargs)

    @staticmethod
    def load_excel(file_path, sheet_name=0, **kwargs):
        """加载Excel文件"""
        return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)

    @staticmethod
    def load_json(file_path, encoding='utf-8-sig', **kwargs):
        """加载JSON文件"""
        import json
        with open(file_path, 'r', encoding=encoding) as f:
            data = json.load(f)
        return pd.DataFrame(data)

    @staticmethod
    def load_txt(file_path, delimiter='\t', encoding='utf-8-sig', parse_dates=True, date_columns=None, **kwargs):
        """加载TXT文件"""
        if not parse_dates:
            return pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, **kwargs)

        try:
            sample_df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, nrows=5, **kwargs)
            existing_columns = list(sample_df.columns)

            if date_columns is not None:
                if isinstance(date_columns, str):
                    date_columns = [date_columns]
                valid_date_cols = [col for col in date_columns if col in existing_columns]
            else:
                valid_date_cols = [col for col in existing_columns if col in DataLoader.DEFAULT_DATE_COLUMNS]

            if valid_date_cols:
                return pd.read_csv(file_path, delimiter=delimiter, encoding=encoding,
                                   parse_dates=valid_date_cols, **kwargs)
            else:
                return pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, **kwargs)
        except Exception as e:
            return pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, **kwargs)

    @staticmethod
    def load_from_file(file_path, parse_dates=True, date_columns=None, **kwargs):
        """根据文件扩展名自动加载"""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.csv':
            return DataLoader.load_csv(file_path, parse_dates=parse_dates, date_columns=date_columns, **kwargs)
        elif ext in ['.xlsx', '.xls']:
            return DataLoader.load_excel(file_path, **kwargs)
        elif ext == '.txt':
            return DataLoader.load_txt(file_path, parse_dates=parse_dates, date_columns=date_columns, **kwargs)
        elif ext == '.json':
            return DataLoader.load_json(file_path, **kwargs)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    @staticmethod
    def load_sql_server(server, database, table_name=None, query=None,
                        username=None, password=None, trusted_connection=True,
                        exclude_columns=None, limit=1000, **kwargs):
        """加载SQL Server数据"""
        # 选择ODBC驱动
        possible_drivers = [
            'SQL Server',
            'SQL Server Native Client 11.0',
            'ODBC Driver 17 for SQL Server',
            'ODBC Driver 13 for SQL Server'
        ]

        available_drivers = pyodbc.drivers()
        selected_driver = None

        for driver in possible_drivers:
            if driver in available_drivers:
                selected_driver = driver
                break

        if selected_driver is None and available_drivers:
            selected_driver = available_drivers[0]

        if selected_driver is None:
            raise Exception("未找到任何ODBC驱动")

        # 构建连接字符串
        if trusted_connection or username is None:
            conn_str = f"DRIVER={{{selected_driver}}};SERVER={server};DATABASE={database};Trusted_Connection=yes;"
        else:
            conn_str = f"DRIVER={{{selected_driver}}};SERVER={server};DATABASE={database};UID={username};PWD={password};"

        conn_str += "Connect Timeout=30;"

        if 'ODBC Driver' in selected_driver:
            conn_str += "Encrypt=yes;TrustServerCertificate=yes;"

        conn = pyodbc.connect(conn_str)

        if query:
            df = pd.read_sql(query, conn)
            conn.close()
            return df

        if '.' in table_name:
            full_table_name = table_name
        else:
            full_table_name = f"[{table_name}]"

        # 处理排除列
        if exclude_columns:
            if isinstance(exclude_columns, list):
                exclude_set = set([col.lower() for col in exclude_columns])
            else:
                exclude_set = set([exclude_columns.lower()])
        else:
            exclude_set = set()

        # 获取列名（简化版，完整版见原代码）
        try:
            columns_query = f"SELECT TOP 0 * FROM {full_table_name}"
            sample_df = pd.read_sql(columns_query, conn)
            keep_columns = [col for col in sample_df.columns if col.lower() not in exclude_set]
            columns_str = ', '.join([f'[{col}]' for col in keep_columns])
        except:
            columns_str = '*'

        final_query = f"SELECT TOP {limit} {columns_str} FROM {full_table_name}"
        df = pd.read_sql(final_query, conn)
        conn.close()

        return df

    @staticmethod
    def load_multiple_tables(server, database, table_names,
                             username=None, password=None, trusted_connection=True,
                             exclude_columns=None, limit=1000, relationships=None, **kwargs):
        """批量加载多个SQL Server表"""
        tables = {}

        if isinstance(table_names, list):
            table_dict = {name: name for name in table_names}
        elif isinstance(table_names, dict):
            table_dict = table_names
        else:
            raise ValueError("table_names 必须是列表或字典")

        for display_name, actual_name in table_dict.items():
            print(f"📊 加载表: {display_name}")
            try:
                df = DataLoader.load_sql_server(
                    server=server, database=database, table_name=actual_name,
                    username=username, password=password, trusted_connection=trusted_connection,
                    exclude_columns=exclude_columns, limit=limit, **kwargs
                )
                tables[display_name] = df
                print(f"  ✅ {len(df)}行 x {len(df.columns)}列")
            except Exception as e:
                print(f"  ❌ 加载失败: {e}")
                tables[display_name] = None

        return tables