"""
MCP服务模块 - 供AI Agent调用
"""

import json
import os
import time
import pandas as pd
from typing import Dict, Any

from fastmcp import FastMCP

from autostat.analyzer import AutoStatisticalAnalyzer
from autostat.multi_analyzer import MultiTableStatisticalAnalyzer
from autostat.loader import DataLoader
from autostat.reporter import Reporter

mcp = FastMCP("DataAnalyzerMCP")


@mcp.tool
def analyze_from_file(
    file_path: str,
    table_name: str = None,
    output_level: str = "compact",
    date_features: str = None,
    categorical_top_k: int = 10,
    correlation_threshold: float = 0.3,
    max_columns: int = 30
) -> str:
    """
    分析单个文件（支持CSV、Excel、TXT、JSON）

    参数:
    - file_path: 文件路径
    - table_name: 表名（可选，默认使用文件名）
    - output_level: 输出级别，可选 minimal/compact/full
    - date_features: 日期派生列级别，可选 none/basic/full
    - categorical_top_k: 分类变量输出前K个类别
    - correlation_threshold: 相关性阈值
    - max_columns: 详细输出的最大列数

    返回:
    - JSON格式的分析结果
    """
    try:
        start_time = time.time()

        if table_name is None:
            table_name = os.path.splitext(os.path.basename(file_path))[0]

        date_features_level = date_features or "basic"

        analyzer = AutoStatisticalAnalyzer(
            file_path,
            source_table_name=table_name,
            auto_clean=False,
            quiet=True,
            date_features_level=date_features_level
        )

        result = {
            'success': True,
            'analysis_time': pd.Timestamp.now().isoformat(),
            'analysis_duration_seconds': time.time() - start_time,
            'file_name': table_name,
            'data_shape': {
                'rows': len(analyzer.data),
                'columns': len(analyzer.data.columns)
            },
            'column_names': list(analyzer.data.columns)[:50],
            'variable_types': {
                col: {
                    'type': var_type,
                    'type_desc': analyzer._get_type_description(var_type)
                }
                for col, var_type in list(analyzer.variable_types.items())[:max_columns]
            },
            'quality_report': {
                'missing': analyzer.quality_report.get('missing', [])[:5],
                'duplicates': analyzer.quality_report.get('duplicates', {})
            }
        }

        return json.dumps(result, ensure_ascii=False, indent=2, default=str)

    except Exception as e:
        import traceback
        return json.dumps({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }, ensure_ascii=False, indent=2)


@mcp.tool
def analyze_multiple_files(
    file_paths_json: str,
    relationships_json: str = None,
    output_level: str = "compact"
) -> str:
    """
    批量分析多个文件，智能判断单表/多表关联

    参数:
    - file_paths_json: JSON格式的文件列表，如 '["file1.csv", "file2.csv"]' 或 '{"name1": "path1"}'
    - relationships_json: JSON格式的关系定义，如 '[{"from_table": "t1", "from_col": "id", "to_table": "t2", "to_col": "id"}]'
    - output_level: 输出级别

    返回:
    - JSON格式的分析结果
    """
    try:
        start_time = time.time()
        file_config = json.loads(file_paths_json)

        if isinstance(file_config, list):
            files_dict = {}
            for path in file_config:
                normalized_path = path.replace('\\', '/')
                name = os.path.splitext(os.path.basename(normalized_path))[0]
                files_dict[name] = normalized_path
        else:
            files_dict = {k: v.replace('\\', '/') for k, v in file_config.items()}

        tables = {}
        for name, path in files_dict.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"文件不存在: {path}")
            df = DataLoader.load_from_file(path)
            tables[name] = df

        if len(tables) == 1:
            name = list(tables.keys())[0]
            analyzer = AutoStatisticalAnalyzer(tables[name], source_table_name=name, quiet=True)
            result = {
                'success': True,
                'analysis_type': 'single_table',
                'file_name': name,
                'data_shape': {'rows': len(tables[name]), 'columns': len(tables[name].columns)},
                'column_names': list(tables[name].columns)[:50]
            }
        else:
            relationships = None
            if relationships_json:
                relationships = json.loads(relationships_json)
                if not isinstance(relationships, list):
                    relationships = [relationships]

            analyzer = MultiTableStatisticalAnalyzer(
                tables,
                relationships={'foreign_keys': relationships} if relationships else None
            )
            result = {
                'success': True,
                'analysis_type': 'multi_table',
                'tables': {
                    name: {'rows': len(df), 'columns': len(df.columns), 'columns': list(df.columns)[:30]}
                    for name, df in tables.items()
                },
                'relationships': analyzer.all_relationships.get('foreign_keys', [])
            }

        result['analysis_time'] = pd.Timestamp.now().isoformat()
        result['analysis_duration_seconds'] = time.time() - start_time

        return json.dumps(result, ensure_ascii=False, indent=2, default=str)

    except Exception as e:
        return json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False, indent=2)


@mcp.tool
def analyze_from_db(
    server: str,
    database: str,
    table_names_json: str,
    username: str = None,
    password: str = None,
    trusted_connection: bool = False,
    relationships_json: str = None,
    limit: int = 10000,
    max_text_length: int = 100
) -> str:
    """
    从SQL Server数据库分析数据

    参数:
    - server: 数据库服务器地址
    - database: 数据库名称
    - table_names_json: 表名列表JSON格式
    - username: 用户名
    - password: 密码
    - trusted_connection: 是否使用Windows认证
    - relationships_json: 关系定义JSON
    - limit: 最大加载行数
    - max_text_length: 文本字段最大保留长度

    返回:
    - JSON格式的分析结果
    """
    try:
        start_time = time.time()
        table_config = json.loads(table_names_json)

        if isinstance(table_config, list):
            tables_dict = {name: name for name in table_config}
        else:
            tables_dict = table_config

        relationships = None
        if relationships_json:
            relationships = json.loads(relationships_json)

        tables = DataLoader.load_multiple_tables(
            server=server,
            database=database,
            table_names=tables_dict,
            username=username,
            password=password,
            trusted_connection=trusted_connection,
            limit=limit,
            relationships=relationships,
            max_text_length=max_text_length
        )

        successful_tables = {name: df for name, df in tables.items() if df is not None and len(df) > 0}

        if len(successful_tables) == 0:
            return json.dumps({'success': False, 'error': '没有成功加载任何表'}, ensure_ascii=False, indent=2)

        if len(successful_tables) == 1:
            name = list(successful_tables.keys())[0]
            df = successful_tables[name]
            result = {
                'success': True,
                'analysis_type': 'single_table_from_db',
                'table_name': name,
                'data_shape': {'rows': len(df), 'columns': len(df.columns)},
                'column_names': list(df.columns)[:50]
            }
        else:
            result = {
                'success': True,
                'analysis_type': 'multi_table_from_db',
                'tables': {name: {'rows': len(df), 'columns': len(df.columns)} for name, df in successful_tables.items()}
            }

        result['analysis_time'] = pd.Timestamp.now().isoformat()
        result['analysis_duration_seconds'] = time.time() - start_time

        return json.dumps(result, ensure_ascii=False, indent=2, default=str)

    except Exception as e:
        return json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=6011)