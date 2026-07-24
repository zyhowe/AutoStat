"""流式查询服务 - 从 SQL Server 流式读取数据"""
import json
import pyodbc
import re
from decimal import Decimal
from typing import Dict, Any, Generator, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class StreamQueryService:
    """流式查询服务 - 支持从 SQL Server 逐批读取数据"""

    @staticmethod
    def _safe_json_serialize(value):
        """安全地将值转换为 JSON 可序列化的类型"""
        if value is None:
            return None
        if isinstance(value, Decimal):
            if value.is_nan():
                return None
            if value.is_infinite():
                return None if value.is_nan() else float('inf') if value > 0 else float('-inf')
            return float(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, bytes):
            try:
                return value.decode('utf-8', errors='ignore')
            except:
                return str(value)
        return value

    @staticmethod
    def _get_table_name(session: Dict[str, Any]) -> Optional[str]:
        """从会话中获取实际的表名"""
        # 1. 从 tables_meta 获取
        tables_meta = session.get('tables_meta', {})
        if tables_meta:
            table_names = list(tables_meta.keys())
            if table_names:
                logger.info(f"[流式查询] 从 tables_meta 获取表名: {table_names[0]}")
                return table_names[0]

        # 2. 从 tables_info 获取
        tables_info = session.get('tables_info', {})
        tables = tables_info.get('tables', [])
        if tables and isinstance(tables, list) and len(tables) > 0:
            logger.info(f"[流式查询] 从 tables_info.tables 获取表名: {tables[0]}")
            return tables[0]
        if tables and isinstance(tables, dict):
            table_names = list(tables.keys())
            if table_names:
                logger.info(f"[流式查询] 从 tables_info.tables 获取表名: {table_names[0]}")
                return table_names[0]

        # 3. 从 files 获取
        files = session.get('files', {})
        if files:
            file_names = list(files.keys())
            if file_names:
                table_name = file_names[0]
                if '.' in table_name:
                    table_name = table_name.rsplit('.', 1)[0]
                logger.info(f"[流式查询] 从 files 获取表名: {table_name}")
                return table_name

        # 4. 从 analysis_result.source_table 获取
        analysis_result = session.get('analysis_result', {})
        source_table = analysis_result.get('source_table')
        if source_table:
            logger.info(f"[流式查询] 从 analysis_result.source_table 获取表名: {source_table}")
            return source_table

        logger.error("[流式查询] 无法从会话中获取表名")
        return None

    @staticmethod
    def _extract_fields_from_rule(rule: str) -> List[str]:
        """从规则表达式中提取所有字段名"""
        if not rule:
            return []

        # 匹配所有 companyfixasset 开头的字段
        field_matches = re.findall(r'companyfixasset\d+', rule)
        if field_matches:
            return list(set(field_matches))

        # 匹配其他可能的字段模式
        other_matches = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]+\b', rule)
        keywords = {'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT', 'IN', 'IS', 'NULL',
                    'ABS', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'AS', 'ON', 'JOIN',
                    'LEFT', 'RIGHT', 'INNER', 'OUTER', 'GROUP', 'ORDER', 'BY', 'HAVING',
                    'UNION', 'ALL', 'DISTINCT', 'TOP', 'WITH', 'OVER', 'PARTITION'}
        result = []
        for f in other_matches:
            if f.upper() in keywords:
                continue
            if f.isdigit():
                continue
            if f in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                continue
            result.append(f)
        return list(set(result))

    @staticmethod
    def _build_rule_where_clause(rule: str, fields: List[str]) -> Optional[str]:
        """根据规则表达式构建 SQL WHERE 条件"""
        if not rule:
            return None

        # 解析规则: left = right
        if ' = ' not in rule:
            if '=' in rule:
                left, right = rule.split('=', 1)
            else:
                return None
        else:
            left, right = rule.split(' = ', 1)

        left = left.strip()
        right = right.strip()

        # 提取所有字段用于 NOT NULL 检查
        if not fields:
            fields = StreamQueryService._extract_fields_from_rule(rule)

        if not fields:
            return None

        not_null_parts = [f"{f} IS NOT NULL" for f in fields]
        not_null_clause = " AND ".join(not_null_parts) if not_null_parts else "1=1"

        # 构建差值比较条件
        diff_condition = f"ABS(({left}) - ({right})) > 0.01"

        where_clause = f"({not_null_clause}) AND ({diff_condition})"
        return where_clause

    @staticmethod
    def build_sql_from_context(context: Dict[str, Any], session: Dict[str, Any]) -> tuple:
        """根据上下文和会话信息构建 SQL 查询"""
        limit = context.get('limit', 10000)
        description = context.get('description', '数据追溯')

        table_name = StreamQueryService._get_table_name(session)
        if not table_name:
            logger.error("[流式查询] 无法获取表名")
            return None, None

        logger.info(f"[流式查询] 使用表名: {table_name}")

        # ===== 方式1：规则追溯 =====
        if context.get('rule'):
            rule = context.get('rule')
            fields = context.get('fields', [])
            sample_rows = context.get('sample_rows', [])

            # 提取字段
            if not fields:
                fields = StreamQueryService._extract_fields_from_rule(rule)

            # 构建 SELECT 字段列表：标识字段 + 规则涉及的字段
            identity_fields = ['id', 'companycode', 'reportdate', 'declaredate']
            select_fields = list(identity_fields)
            for f in fields:
                if f not in select_fields:
                    select_fields.append(f)

            if len(select_fields) > 50:
                select_fields = select_fields[:50]

            select_str = ', '.join(select_fields)

            where_clause = StreamQueryService._build_rule_where_clause(rule, fields)

            if where_clause:
                sql = f"SELECT {select_str} FROM {table_name} WHERE {where_clause} ORDER BY id"
                description = f"规则「{rule[:80]}{'...' if len(rule) > 80 else ''}」的全量违反记录"
                logger.info(f"[流式查询] 规则追溯SQL: {sql[:300]}...")
                return sql, description
            else:
                if sample_rows:
                    ids_str = ','.join(str(id) for id in sample_rows[:100])
                    sql = f"SELECT {select_str} FROM {table_name} WHERE id IN ({ids_str}) ORDER BY id"
                    description = f"规则「{rule[:80]}」的样本违反记录（{len(sample_rows)} 条）"
                    return sql, description
                else:
                    return None, None

        # ===== 方式2：按行号列表 =====
        if context.get('row_ids'):
            ids = context['row_ids']
            if len(ids) > 1000:
                ids = ids[:1000]
            ids_str = ','.join(str(id) for id in ids)
            sql = f"SELECT * FROM {table_name} WHERE id IN ({ids_str}) ORDER BY id"
            description = f"追溯 {len(ids)} 条诊断记录对应的原始数据"
            return sql, description

        # ===== 方式3：按行号范围 =====
        if context.get('id_range'):
            start = context['id_range'].get('start', 1)
            end = context['id_range'].get('end', 5000)
            if end - start > limit:
                end = start + limit
            sql = f"SELECT * FROM {table_name} WHERE id BETWEEN {start} AND {end} ORDER BY id"
            description = f"数据段 {start}-{end}"
            return sql, description

        # ===== 方式4：按公司代码 =====
        if context.get('company_code'):
            company_code = context['company_code']
            sql = f"SELECT * FROM {table_name} WHERE companycode = '{company_code}' ORDER BY reportdate DESC, id"
            description = f"公司 {company_code} 的全部数据"
            return sql, description

        # ===== 方式5：组合条件 =====
        if context.get('filters'):
            conditions = []
            for key, value in context['filters'].items():
                if value is None:
                    continue
                if isinstance(value, str):
                    conditions.append(f"{key} = '{value}'")
                elif isinstance(value, (int, float)):
                    conditions.append(f"{key} = {value}")
                elif isinstance(value, list):
                    if len(value) == 1:
                        conditions.append(f"{key} = {value[0]}")
                    else:
                        values_str = ','.join(str(v) for v in value)
                        conditions.append(f"{key} IN ({values_str})")
            if conditions:
                where_clause = ' AND '.join(conditions)
                sql = f"SELECT * FROM {table_name} WHERE {where_clause} ORDER BY id"
                desc_parts = [f"{k}={v}" for k, v in context['filters'].items() if v is not None]
                description = f"筛选: {', '.join(desc_parts)}"
                return sql, description

        # ===== 默认 =====
        sql = f"SELECT TOP {limit} * FROM {table_name} ORDER BY id DESC"
        description = f"最近 {limit} 条数据"
        return sql, description

    @staticmethod
    def stream_query(
        db_config: Dict[str, Any],
        sql: str,
        batch_size: int = 100,
        max_rows: int = 100000
    ) -> Generator[str, None, None]:
        """流式执行 SQL 查询，逐行返回 NDJSON 格式数据"""
        if not sql:
            yield json.dumps({
                "type": "error",
                "message": "无法构建有效的SQL查询"
            }, ensure_ascii=False) + '\n'
            return

        server = db_config.get('server')
        database = db_config.get('database')
        username = db_config.get('username')
        password = db_config.get('password')
        trusted_connection = db_config.get('trusted_connection', False)

        logger.info(f"[流式查询] 开始执行 SQL: {sql[:200]}...")

        available_drivers = pyodbc.drivers()
        possible_drivers = [
            'ODBC Driver 17 for SQL Server',
            'ODBC Driver 13 for SQL Server',
            'SQL Server Native Client 11.0',
            'SQL Server'
        ]
        selected_driver = None
        for driver in possible_drivers:
            if driver in available_drivers:
                selected_driver = driver
                break
        if selected_driver is None and available_drivers:
            selected_driver = available_drivers[0]

        if selected_driver is None:
            error_msg = "未找到任何 ODBC 驱动，请安装 SQL Server ODBC 驱动"
            logger.error(f"[流式查询] {error_msg}")
            yield json.dumps({
                "type": "error",
                "message": error_msg
            }, ensure_ascii=False) + '\n'
            return

        if trusted_connection or not username:
            conn_str = f"DRIVER={{{selected_driver}}};SERVER={server};DATABASE={database};Trusted_Connection=yes;"
        else:
            conn_str = f"DRIVER={{{selected_driver}}};SERVER={server};DATABASE={database};UID={username};PWD={password};"

        conn_str += "Connect Timeout=30;"

        if 'ODBC Driver' in selected_driver:
            conn_str += "Encrypt=yes;TrustServerCertificate=yes;"

        conn = None
        cursor = None

        try:
            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()
            cursor.arraysize = batch_size

            logger.info(f"[流式查询] 执行查询...")
            cursor.execute(sql)

            columns = [desc[0] for desc in cursor.description]
            logger.info(f"[流式查询] 列数: {len(columns)}")

            if not columns:
                yield json.dumps({
                    "type": "error",
                    "message": "查询结果无列"
                }, ensure_ascii=False) + '\n'
                return

            row_count = 0

            yield json.dumps({
                "type": "meta",
                "columns": columns,
                "total_estimate": "unknown",
                "timestamp": datetime.now().isoformat()
            }, ensure_ascii=False) + '\n'

            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    logger.info(f"[流式查询] 取数完成，共 {row_count} 行")
                    break

                for row in rows:
                    if row_count >= max_rows:
                        yield json.dumps({
                            "type": "warning",
                            "message": f"已达到最大行数限制 ({max_rows} 行)"
                        }, ensure_ascii=False) + '\n'
                        return

                    row_dict = {}
                    for i, col in enumerate(columns):
                        val = row[i]
                        row_dict[col] = StreamQueryService._safe_json_serialize(val)

                    row_count += 1
                    yield json.dumps(row_dict, ensure_ascii=False) + '\n'

            yield json.dumps({
                "type": "complete",
                "row_count": row_count,
                "timestamp": datetime.now().isoformat()
            }, ensure_ascii=False) + '\n'

        except pyodbc.Error as e:
            error_msg = f"数据库错误: {str(e)}"
            logger.error(f"[流式查询] {error_msg}")
            yield json.dumps({
                "type": "error",
                "message": error_msg
            }, ensure_ascii=False) + '\n'
        except Exception as e:
            error_msg = f"查询失败: {str(e)}"
            logger.error(f"[流式查询] {error_msg}")
            yield json.dumps({
                "type": "error",
                "message": error_msg
            }, ensure_ascii=False) + '\n'
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    @staticmethod
    def get_connection_from_session(session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """从会话信息中提取数据库配置"""
        tables_info = session.get('tables_info', {})
        if not tables_info:
            return None

        db_config = tables_info.get('db_config')
        if not db_config:
            return None

        required_fields = ['server', 'database']
        for field in required_fields:
            if not db_config.get(field):
                return None

        return {
            'server': db_config.get('server'),
            'database': db_config.get('database'),
            'username': db_config.get('username'),
            'password': db_config.get('password'),
            'trusted_connection': db_config.get('trusted_connection', False)
        }