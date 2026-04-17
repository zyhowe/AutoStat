"""文件服务 - 文件加载、保存、临时文件管理"""

import streamlit as st
import pandas as pd
import tempfile
import os
import io
import json
import re
from autostat.loader import DataLoader


class FileService:
    """文件服务类"""

    @staticmethod
    def load_file(uploaded, ext):
        """加载文件"""
        if ext == 'csv':
            return FileService._load_csv(uploaded)
        elif ext in ['xlsx', 'xls']:
            return FileService._load_excel(uploaded, ext)
        elif ext == 'json':
            return FileService._load_json(uploaded)
        else:
            return pd.read_csv(uploaded, delimiter='\t', engine='python', on_bad_lines='skip')

    @staticmethod
    def _load_csv(uploaded):
        """加载 CSV 文件"""
        try:
            content = uploaded.read()
            content = content.replace(b'\x00', b'')
            content_str = content.decode('utf-8', errors='ignore')
            return pd.read_csv(io.StringIO(content_str))
        except:
            uploaded.seek(0)
            return pd.read_csv(uploaded, encoding='utf-8', engine='python', on_bad_lines='skip')

    @staticmethod
    def _load_excel(uploaded, ext):
        """加载 Excel 文件"""
        with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as tmp_excel:
            tmp_excel.write(uploaded.getbuffer())
            tmp_excel_path = tmp_excel.name

        try:
            df = pd.read_excel(tmp_excel_path, engine='openpyxl')
        except:
            try:
                df = pd.read_excel(tmp_excel_path, engine='xlrd')
            except:
                df = pd.read_excel(tmp_excel_path)

        os.unlink(tmp_excel_path)
        return df

    @staticmethod
    def _load_json(uploaded):
        """加载 JSON 文件"""
        content = uploaded.read()
        content_str = content.decode('utf-8', errors='ignore')
        content_str = re.sub(r'[\x00-\x1f\x7f]', '', content_str)

        try:
            data = json.loads(content_str)
            return pd.DataFrame(data)
        except:
            lines = content_str.strip().split('\n')
            data_list = []
            for line in lines:
                if line.strip():
                    data_list.append(json.loads(line))
            return pd.DataFrame(data_list)

    @staticmethod
    def save_temp_file(df, tmp, ext):
        """保存文件到临时文件"""
        if ext == 'csv':
            df.to_csv(tmp.name, index=False, encoding='utf-8')
        elif ext in ['xlsx', 'xls']:
            df.to_excel(tmp.name, index=False)
        elif ext == 'json':
            df.to_json(tmp.name, orient='records', force_ascii=False)
        else:
            df.to_csv(tmp.name, sep='\t', index=False, encoding='utf-8')

    @staticmethod
    def load_tables(files):
        """加载所有表"""
        tmp_dir = tempfile.mkdtemp()
        paths = {}
        for f in files:
            p = os.path.join(tmp_dir, f.name)
            with open(p, 'wb') as w:
                w.write(f.getbuffer())
            paths[os.path.splitext(f.name)[0]] = p

        tables = {}
        failed = []
        for name, p in paths.items():
            try:
                df = DataLoader.load_from_file(p)
                if df is not None and not df.empty:
                    tables[name] = df
                else:
                    failed.append(f"{name} (空文件)")
            except Exception as e:
                failed.append(f"{name} ({str(e)[:50]})")

        if failed:
            st.warning(f"以下表加载失败: {', '.join(failed)}")

        return tmp_dir, tables

    @staticmethod
    def test_db_connection(config):
        """测试数据库连接"""
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

                try:
                    conn = pyodbc.connect(conn_str, timeout=10)
                    return conn
                except:
                    continue

        return None

    @staticmethod
    def load_db_tables(config, table_names, limit, max_text_length):
        """加载数据库表"""
        tables = DataLoader.load_multiple_tables(
            server=config.get('server'),
            database=config.get('database'),
            table_names=table_names,
            username=config.get('username') if not config.get('trusted_connection') else None,
            password=config.get('password') if not config.get('trusted_connection') else None,
            trusted_connection=config.get('trusted_connection', False),
            limit=limit,
            relationships=None,
            max_text_length=max_text_length
        )
        return {n: df for n, df in tables.items() if df is not None and not df.empty}