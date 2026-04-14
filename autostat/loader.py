"""
数据加载器模块
支持 CSV、Excel、JSON、TXT、SQL Server
"""

import numpy as np
import pandas as pd
import pyodbc
import os
import json
import io
import re
from typing import List, Dict, Optional, Any

class DataLoader:
    """数据加载器 - 支持多种数据源"""

    # 常见的日期列名（用于自动识别）
    DEFAULT_DATE_COLUMNS = [
        '日期', 'date', 'Date', '时间', 'time', 'Time',
        '销售日期', '创建时间', '更新时间', 'datetime', 'DateTime',
        'order_date', 'OrderDate', 'created_at', 'updated_at',
        'birth_date', 'BirthDate', '注册日期', '交易日期',
        '就诊日期', '住院日期', '出院日期', '出生日期'
    ]

    # 需要直接剔除的大字段类型（只剔除真正的大字段，不剔除普通varchar/nvarchar）
    LARGE_FIELD_TYPES = [
        'text', 'ntext', 'image', 'xml', 'geography', 'geometry', 'hierarchyid'
    ]

    # 需要剔除的MAX类型（长度=-1表示MAX）
    MAX_TYPES = ['varchar', 'nvarchar', 'varbinary']

    @staticmethod
    def load_csv(file_path, encoding='utf-8-sig', parse_dates=True, date_columns=None, **kwargs):
        """加载CSV文件（增强健壮性）"""
        print(f"  📂 开始加载CSV文件: {file_path}")

        df = None

        # 方法1：二进制模式读取，处理空字符和特殊字符
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                # 替换空字符
                if b'\x00' in content:
                    print(f"  ⚠️ 发现空字符，正在清理...")
                    content = content.replace(b'\x00', b'')
                # 解码，忽略无法解码的字符
                content_str = content.decode('utf-8', errors='ignore')
                # 清理其他特殊控制字符（保留换行和回车）
                content_str = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', content_str)
                df = pd.read_csv(io.StringIO(content_str), **kwargs)
                print(f"  ✅ 二进制模式读取成功")
        except Exception as e:
            print(f"  ⚠️ 二进制模式读取失败: {e}")

        # 方法2：普通方式读取，尝试多种编码
        if df is None:
            encodings = [encoding, 'utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1', 'cp1252']
            for enc in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=enc, **kwargs)
                    print(f"  ✅ 使用编码 {enc} 读取成功")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"  ⚠️ 编码 {enc} 失败: {e}")
                    continue

        # 方法3：使用 python 引擎，跳过错误行
        if df is None:
            try:
                df = pd.read_csv(file_path, encoding='utf-8', engine='python',
                                on_bad_lines='skip', **kwargs)
                print(f"  ✅ Python引擎读取成功（已跳过错误行）")
            except Exception as e:
                print(f"  ❌ 所有读取方式都失败: {e}")
                raise ValueError(f"无法读取CSV文件: {e}")

        if df is None:
            raise ValueError("无法读取CSV文件")

        # 清理列名（去除首尾空格，替换特殊字符）
        df.columns = [str(col).strip().replace('\n', '_').replace('\r', '_').replace('\t', '_') for col in df.columns]

        # 处理空字符串为NaN
        df = df.replace(r'^\s*$', np.nan, regex=True)

        if not parse_dates:
            return df

        try:
            existing_columns = list(df.columns)
            if date_columns is not None:
                if isinstance(date_columns, str):
                    date_columns = [date_columns]
                valid_date_cols = [col for col in date_columns if col in existing_columns]
            else:
                valid_date_cols = [col for col in existing_columns if col in DataLoader.DEFAULT_DATE_COLUMNS]

            if valid_date_cols:
                print(f"  📅 自动识别日期列: {valid_date_cols}")
                for col in valid_date_cols:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        pass
        except Exception as e:
            print(f"  ⚠️ 日期解析失败: {e}")

        return df

    @staticmethod
    def load_excel(file_path, sheet_name=0, **kwargs):
        """加载Excel文件（增强健壮性）"""
        print(f"  📂 开始加载Excel文件: {file_path}")

        df = None
        engines = ['openpyxl', 'xlrd', 'calamine', None]

        for engine in engines:
            try:
                if engine is None:
                    df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
                else:
                    df = pd.read_excel(file_path, sheet_name=sheet_name, engine=engine, **kwargs)
                print(f"  ✅ 使用引擎 {engine or 'auto'} 读取成功")
                break
            except Exception as e:
                print(f"  ⚠️ 引擎 {engine or 'auto'} 失败: {e}")
                continue

        if df is None:
            # 尝试作为CSV读取（可能是错误后缀名）
            try:
                df = DataLoader.load_csv(file_path, **kwargs)
                print(f"  ✅ 作为CSV文件读取成功")
            except:
                pass

        if df is None:
            raise ValueError("无法读取Excel文件，请确认文件格式正确（支持 .xlsx, .xls）")

        # 清理列名
        df.columns = [str(col).strip().replace('\n', '_').replace('\r', '_').replace('\t', '_') for col in df.columns]

        # 处理空字符串为NaN
        df = df.replace(r'^\s*$', np.nan, regex=True)

        return df

    @staticmethod
    def load_json(file_path, encoding='utf-8-sig', orient='records',
                  parse_dates=True, date_columns=None, **kwargs):
        """加载JSON文件（增强健壮性）"""
        print(f"  📂 开始加载JSON文件: {file_path}")

        df = None

        # 方法1：尝试多种JSON格式
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
        except:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

        # 清理控制字符
        content = re.sub(r'[\x00-\x1f\x7f]', '', content)

        json_formats = [None, 'records', 'index', 'columns', 'values', 'table']
        for fmt in json_formats:
            try:
                data = json.loads(content)
                df = pd.DataFrame(data)
                print(f"  ✅ JSON读取成功")
                break
            except:
                continue

        if df is None:
            # 尝试逐行读取JSON
            try:
                lines = content.strip().split('\n')
                data_list = []
                for line in lines:
                    if line.strip():
                        data_list.append(json.loads(line))
                df = pd.DataFrame(data_list)
                print(f"  ✅ 逐行JSON读取成功")
            except:
                pass

        if df is None:
            raise ValueError("无法解析JSON文件")

        # 清理列名
        df.columns = [str(col).strip().replace('\n', '_').replace('\r', '_').replace('\t', '_') for col in df.columns]

        # 处理空字符串为NaN
        df = df.replace(r'^\s*$', np.nan, regex=True)

        if parse_dates:
            if date_columns is None:
                date_columns = DataLoader.DEFAULT_DATE_COLUMNS
            elif isinstance(date_columns, str):
                date_columns = [date_columns]

            for col in date_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        print(f"  📅 转换日期列: {col}")
                    except:
                        pass

        return df

    @staticmethod
    def load_txt(file_path, delimiter='\t', encoding='utf-8-sig', parse_dates=True, date_columns=None, **kwargs):
        """加载TXT文件（增强健壮性）"""
        print(f"  📂 开始加载TXT文件: {file_path}")

        df = None

        # 尝试多种分隔符
        delimiters = [delimiter, ',', '\t', ';', '|', ' ']
        for delim in delimiters:
            try:
                df = pd.read_csv(file_path, delimiter=delim, encoding=encoding,
                               on_bad_lines='skip', engine='python', **kwargs)
                print(f"  ✅ 使用分隔符 '{delim}' 读取成功")
                break
            except:
                continue

        if df is None:
            raise ValueError("无法读取TXT文件")

        # 清理列名
        df.columns = [str(col).strip().replace('\n', '_').replace('\r', '_').replace('\t', '_') for col in df.columns]

        # 处理空字符串为NaN
        df = df.replace(r'^\s*$', np.nan, regex=True)

        if not parse_dates:
            return df

        try:
            existing_columns = list(df.columns)
            if date_columns is not None:
                if isinstance(date_columns, str):
                    date_columns = [date_columns]
                valid_date_cols = [col for col in date_columns if col in existing_columns]
            else:
                valid_date_cols = [col for col in existing_columns if col in DataLoader.DEFAULT_DATE_COLUMNS]

            if valid_date_cols:
                print(f"  📅 自动识别日期列: {valid_date_cols}")
                for col in valid_date_cols:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        pass
        except Exception as e:
            print(f"  ⚠️ 日期解析失败: {e}")

        return df

    @staticmethod
    def load_json_string(json_str, parse_dates=True, date_columns=None, **kwargs):
        """从JSON字符串加载数据（增强健壮性）"""
        print("📄 从JSON字符串加载数据...")

        # 清理控制字符
        json_str = re.sub(r'[\x00-\x1f\x7f]', '', json_str)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            # 尝试修复常见问题
            try:
                # 替换单引号为双引号
                json_str = json_str.replace("'", '"')
                data = json.loads(json_str)
            except:
                raise ValueError(f"JSON字符串解析失败: {e}")

        df = pd.DataFrame(data)

        # 清理列名
        df.columns = [str(col).strip().replace('\n', '_').replace('\r', '_').replace('\t', '_') for col in df.columns]

        # 处理空字符串为NaN
        df = df.replace(r'^\s*$', np.nan, regex=True)

        if parse_dates:
            if date_columns is None:
                date_columns = DataLoader.DEFAULT_DATE_COLUMNS
            elif isinstance(date_columns, str):
                date_columns = [date_columns]

            for col in date_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        print(f"  📅 转换日期列: {col}")
                    except:
                        pass

        return df

    @staticmethod
    def load_sql_server(server, database, table_name=None, query=None,
                        username=None, password=None, trusted_connection=True,
                        parse_dates=True, date_columns=None,
                        exclude_columns=None, limit=1000,
                        smart_sampling=False, join_key=None,
                        main_table_df=None, match_key=None,
                        max_text_length=100, **kwargs):
        """加载SQL Server数据"""

        possible_drivers = [
            'SQL Server',
            'SQL Server Native Client 11.0',
            'SQL Server Native Client 10.0',
            'ODBC Driver 17 for SQL Server',
            'ODBC Driver 13 for SQL Server',
            'ODBC Driver 11 for SQL Server'
        ]

        available_drivers = pyodbc.drivers()
        selected_driver = None

        for driver in possible_drivers:
            if driver in available_drivers:
                selected_driver = driver
                print(f"✅ 找到可用驱动: {driver}")
                break

        if selected_driver is None:
            if available_drivers:
                selected_driver = available_drivers[0]
                print(f"✅ 使用驱动: {selected_driver}")
            else:
                raise Exception("未找到任何ODBC驱动")

        if trusted_connection or username is None or str(username).lower() in ['windows', 'trusted_connection', '']:
            conn_str = f"DRIVER={{{selected_driver}}};SERVER={server};DATABASE={database};Trusted_Connection=yes;"
        else:
            conn_str = f"DRIVER={{{selected_driver}}};SERVER={server};DATABASE={database};UID={username};PWD={password};"

        conn_str += "Connect Timeout=30;"

        if 'ODBC Driver' in selected_driver:
            conn_str += "Encrypt=yes;TrustServerCertificate=yes;"

        conn = pyodbc.connect(conn_str)

        if query:
            final_query = query
            df = pd.read_sql(final_query, conn)
            conn.close()

            df.columns = [col.lower() for col in df.columns]

            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        if len(df[col]) > 0 and isinstance(df[col].iloc[0], bytes):
                            df[col] = df[col].apply(
                                lambda x: x.decode('gbk', errors='ignore') if isinstance(x, bytes) else x)
                    except:
                        pass

            if parse_dates:
                if date_columns is None:
                    date_columns = [col for col in df.columns if col in DataLoader.DEFAULT_DATE_COLUMNS]
                elif isinstance(date_columns, str):
                    date_columns = [date_columns]

                for col in date_columns:
                    if col in df.columns:
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        except:
                            pass

            print(f"✅ 最终加载: {len(df)}行 x {len(df.columns)}列")
            return df

        if '.' in table_name:
            full_table_name = table_name
        else:
            full_table_name = f"[{table_name}]"

        keep_columns = DataLoader._get_filtered_columns(
            conn, table_name, exclude_columns, max_text_length
        )

        if not keep_columns:
            print(f"  ⚠️ 警告: 表 {table_name} 所有字段均被过滤，返回空DataFrame")
            conn.close()
            return pd.DataFrame()

        columns_str = ', '.join([f'[{col}]' for col in keep_columns])

        if smart_sampling and join_key and main_table_df is not None and match_key:
            key_values = main_table_df[match_key].dropna().unique().tolist()

            if not key_values:
                print(f"  ⚠️ 主表关联键 {match_key} 无有效值")
                df = pd.DataFrame()
            else:
                print(f"  📊 智能采样: 基于主表的 {len(key_values)} 个关联键值")

                all_dfs = []
                batch_size = 500

                for i in range(0, len(key_values), batch_size):
                    batch_keys = key_values[i:i + batch_size]
                    keys_str = ','.join([f"'{k}'" if isinstance(k, str) else str(k) for k in batch_keys])

                    batch_query = f"""
                        SELECT {columns_str} 
                        FROM {full_table_name} 
                        WHERE [{join_key}] IN ({keys_str})
                    """
                    batch_df = pd.read_sql(batch_query, conn)
                    all_dfs.append(batch_df)
                    print(f"  📦 批次 {i // batch_size + 1}: 加载 {len(batch_df)} 行")

                df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        else:
            final_query = f"SELECT TOP {limit} {columns_str} FROM {full_table_name}"
            print(f"📊 普通采样: {final_query}")
            df = pd.read_sql(final_query, conn)

        conn.close()

        df.columns = [col.lower() for col in df.columns]

        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    if len(df[col]) > 0 and isinstance(df[col].iloc[0], bytes):
                        df[col] = df[col].apply(
                            lambda x: x.decode('gbk', errors='ignore') if isinstance(x, bytes) else x)
                except:
                    pass

        if parse_dates:
            if date_columns is None:
                date_columns = [col for col in df.columns if col in DataLoader.DEFAULT_DATE_COLUMNS]
            elif isinstance(date_columns, str):
                date_columns = [date_columns]

            for col in date_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        pass

        print(f"✅ 最终加载: {len(df)}行 x {len(df.columns)}列")
        return df

    @staticmethod
    def _get_filtered_columns(conn, table_name, exclude_columns, max_text_length=100):
        """获取过滤后的列名列表

        过滤规则：
        1. 大字段类型（text/ntext/image/xml等）直接剔除
        2. MAX类型（varchar(max)/nvarchar(max)/varbinary(max)）直接剔除
        3. 普通文本字段（varchar/nvarchar）长度 > max_text_length 的剔除
        4. 用户指定的排除列剔除
        """
        try:
            schema_query = f"""
            SELECT 
                c.name AS column_name,
                t.name AS data_type,
                c.max_length AS max_length,
                c.is_nullable AS is_nullable
            FROM sys.columns c
            INNER JOIN sys.types t ON c.user_type_id = t.user_type_id
            WHERE c.object_id = OBJECT_ID('{table_name}')
            ORDER BY c.column_id
            """

            schema_df = pd.read_sql(schema_query, conn)

            exclude_set = set()
            if exclude_columns:
                if isinstance(exclude_columns, list):
                    exclude_set.update([col.lower() for col in exclude_columns])
                elif isinstance(exclude_columns, str):
                    exclude_set.update([exclude_columns.lower()])

            keep_columns = []
            removed_large_fields = []
            removed_max_fields = []
            removed_long_text = []
            removed_by_user = []

            for _, row in schema_df.iterrows():
                col_name = row['column_name']
                data_type = row['data_type'].lower()
                max_len = row['max_length']

                # 1. 用户指定排除
                if col_name.lower() in exclude_set:
                    removed_by_user.append(col_name)
                    continue

                # 2. 大字段类型直接剔除（text/ntext/image/xml等）
                is_large_field = False
                for large_type in DataLoader.LARGE_FIELD_TYPES:
                    if data_type == large_type or large_type in data_type:
                        is_large_field = True
                        break

                if is_large_field:
                    removed_large_fields.append(f"{col_name}({data_type})")
                    continue

                # 3. MAX类型剔除（varchar(max)/nvarchar(max)/varbinary(max)）
                is_max_type = False
                for max_type in DataLoader.MAX_TYPES:
                    if data_type == max_type and max_len == -1:
                        is_max_type = True
                        break

                if is_max_type:
                    removed_max_fields.append(f"{col_name}({data_type},MAX)")
                    continue

                # 4. 普通文本字段长度检查
                if data_type in ['varchar', 'nvarchar', 'char', 'nchar']:
                    if max_len > max_text_length:
                        removed_long_text.append(f"{col_name}({data_type},{max_len})")
                        continue

                # 通过所有检查，保留该列
                keep_columns.append(col_name)

            # 打印过滤日志
            if removed_large_fields:
                print(f"    🗑️ 剔除大字段: {', '.join(removed_large_fields[:5])}")
                if len(removed_large_fields) > 5:
                    print(f"       ... 还有{len(removed_large_fields) - 5}个")

            if removed_max_fields:
                print(f"    🗑️ 剔除MAX字段: {', '.join(removed_max_fields[:5])}")
                if len(removed_max_fields) > 5:
                    print(f"       ... 还有{len(removed_max_fields) - 5}个")

            if removed_long_text:
                print(f"    🗑️ 剔除超长文本(>{max_text_length}): {', '.join(removed_long_text[:5])}")
                if len(removed_long_text) > 5:
                    print(f"       ... 还有{len(removed_long_text) - 5}个")

            if removed_by_user:
                print(f"    🗑️ 剔除用户指定列: {', '.join(removed_by_user[:5])}")
                if len(removed_by_user) > 5:
                    print(f"       ... 还有{len(removed_by_user) - 5}个")

            print(f"    ✅ 保留列数: {len(keep_columns)}")

            return keep_columns

        except Exception as e:
            print(f"    ⚠️ 获取表结构失败: {e}，将尝试加载所有列")
            return None

    @staticmethod
    def load_multiple_tables(server, database, table_names,
                             username=None, password=None, trusted_connection=True,
                             exclude_columns=None, limit=1000,
                             relationships=None, max_text_length=100, **kwargs):
        """批量加载多个SQL Server表"""

        tables = {}

        if isinstance(table_names, list):
            table_dict = {name: name for name in table_names}
        elif isinstance(table_names, dict):
            table_dict = table_names
        else:
            raise ValueError("table_names 必须是列表或字典")

        print(f"\n📂 批量加载 {len(table_dict)} 个表...")
        print(f"   文本字段最大保留长度: {max_text_length}")
        print(f"{'─' * 50}")

        if relationships:
            from collections import defaultdict
            degree = defaultdict(int)
            for rel in relationships:
                degree[rel['from_table']] += 1
                degree[rel['to_table']] += 1

            main_table = max(degree.items(), key=lambda x: x[1])[0] if degree else list(table_dict.keys())[0]
            print(f"🎯 主表: {main_table} (根据关系定义)")
        else:
            main_table = list(table_dict.keys())[0]
            print(f"🎯 主表: {main_table} (无关系定义)")

        relation_map = {}
        if relationships:
            for rel in relationships:
                if rel['from_table'] == main_table:
                    relation_map[rel['to_table']] = {
                        'join_key': rel['to_col'],
                        'match_key': rel['from_col']
                    }
                elif rel['to_table'] == main_table:
                    relation_map[rel['from_table']] = {
                        'join_key': rel['from_col'],
                        'match_key': rel['to_col']
                    }

        main_df = None
        for display_name, actual_name in table_dict.items():
            if display_name != main_table:
                continue

            print(f"\n📊 加载主表: {display_name}")

            if isinstance(exclude_columns, dict):
                table_exclude = exclude_columns.get(display_name, exclude_columns.get(actual_name, []))
            else:
                table_exclude = exclude_columns

            try:
                df = DataLoader.load_sql_server(
                    server=server,
                    database=database,
                    table_name=actual_name,
                    username=username,
                    password=password,
                    trusted_connection=trusted_connection,
                    exclude_columns=table_exclude,
                    limit=limit,
                    smart_sampling=False,
                    max_text_length=max_text_length,
                    **kwargs
                )
                tables[display_name] = df
                main_df = df
                print(f"  ✅ 加载主表: {len(df)}行 x {len(df.columns)}列")
            except Exception as e:
                print(f"  ❌ 加载失败: {e}")
                tables[display_name] = None
                return tables

        for display_name, actual_name in table_dict.items():
            if display_name == main_table:
                continue

            print(f"\n📊 加载从表: {display_name}")

            rel_info = relation_map.get(display_name)
            if not rel_info:
                print(f"  ⚠️ 未找到 {display_name} 与主表的关联关系，使用普通采样")
                smart_sampling = False
                join_key = None
                match_key = None
            else:
                smart_sampling = True
                join_key = rel_info['join_key']
                match_key = rel_info['match_key']
                print(f"  🔗 关联键: {display_name}.{join_key} = {main_table}.{match_key}")

            if isinstance(exclude_columns, dict):
                table_exclude = exclude_columns.get(display_name, exclude_columns.get(actual_name, []))
            else:
                table_exclude = exclude_columns

            try:
                df = DataLoader.load_sql_server(
                    server=server,
                    database=database,
                    table_name=actual_name,
                    username=username,
                    password=password,
                    trusted_connection=trusted_connection,
                    exclude_columns=table_exclude,
                    limit=limit,
                    smart_sampling=smart_sampling,
                    join_key=join_key,
                    main_table_df=main_df if smart_sampling else None,
                    match_key=match_key if smart_sampling else None,
                    max_text_length=max_text_length,
                    **kwargs
                )
                tables[display_name] = df
                print(f"  ✅ 加载从表: {len(df)}行 x {len(df.columns)}列")
            except Exception as e:
                print(f"  ❌ 加载失败: {e}")
                tables[display_name] = None

        return tables

    @staticmethod
    def get_table_schema(server, database, table_name, username=None, password=None, trusted_connection=True):
        """获取SQL Server表结构信息"""
        if trusted_connection:
            conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'
        else:
            conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password};'

        try:
            conn = pyodbc.connect(conn_str)
        except pyodbc.Error:
            try:
                conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection={1 if trusted_connection else 0};'
                if not trusted_connection:
                    conn_str += f'UID={username};PWD={password};'
                conn = pyodbc.connect(conn_str)
            except pyodbc.Error as e:
                raise Exception(f"连接SQL Server失败: {e}")

        schema_query = f"""
        SELECT 
            ROW_NUMBER() OVER (ORDER BY c.colid) AS 序号,
            c.name AS 名称,
            t.name AS 类型,
            CASE 
                WHEN t.name IN ('varchar', 'nvarchar', 'char', 'nchar') AND c.length > 0 THEN
                     CASE WHEN c.length = -1 THEN 'MAX' ELSE CAST(c.length AS VARCHAR) END
                WHEN t.name IN ('decimal', 'numeric') AND c.prec > 0 THEN
                     CAST(c.prec AS VARCHAR) + 
                     CASE WHEN c.scale > 0 THEN ',' + CAST(c.scale AS VARCHAR) ELSE '' END
                ELSE ''
            END AS 长度精度,
            CASE c.isnullable WHEN 1 THEN 1 ELSE 0 END AS 可空,
            CASE WHEN EXISTS (
                SELECT 1 FROM sysindexkeys ik 
                INNER JOIN sysindexes i ON ik.id = i.id AND ik.indid = i.indid
                WHERE ik.id = a.id AND ik.colid = c.colid AND i.name LIKE 'PK_%'
            ) THEN 1 ELSE 0 END AS 主键,
            CASE WHEN EXISTS (
                SELECT 1 FROM sysforeignkeys fk 
                WHERE fk.fkeyid = a.id AND fk.fkey = c.colid
            ) THEN 1 ELSE 0 END AS 外键,
            com.text AS 默认值
        FROM (
            SELECT id FROM sysobjects WHERE name = '{table_name}' AND xtype = 'U'
        ) a
        LEFT JOIN syscolumns c ON a.id = c.id
        LEFT JOIN systypes t ON c.xtype = t.xusertype
        LEFT JOIN syscomments com ON c.cdefault = com.id
        ORDER BY c.colid
        """

        try:
            schema_df = pd.read_sql(schema_query, conn)
            conn.close()
            return schema_df.to_dict('records')
        except Exception as e:
            conn.close()
            raise Exception(f"获取表结构失败: {e}")

    @staticmethod
    def load_from_file(file_path, parse_dates=True, date_columns=None, **kwargs):
        """根据文件扩展名自动加载数据（增强健壮性）"""
        ext = os.path.splitext(file_path)[1].lower()
        print(f"  📂 文件扩展名: {ext}")

        if ext == '.csv':
            return DataLoader.load_csv(file_path, parse_dates=parse_dates, date_columns=date_columns, **kwargs)
        elif ext in ['.xlsx', '.xls']:
            return DataLoader.load_excel(file_path, **kwargs)
        elif ext == '.txt':
            return DataLoader.load_txt(file_path, parse_dates=parse_dates, date_columns=date_columns, **kwargs)
        elif ext == '.json':
            return DataLoader.load_json(file_path, parse_dates=parse_dates, date_columns=date_columns, **kwargs)
        else:
            raise ValueError(f"不支持的文件格式: {ext}，支持格式: .csv, .xlsx, .xls, .txt, .json")