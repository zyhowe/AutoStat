"""
MCP服务模块
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
    """分析单个文件"""
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
                for col, var_type in analyzer.variable_types.items()
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
    """批量分析多个文件"""
    try:
        start_time = time.time()
        file_config = json.loads(file_paths_json)

        if isinstance(file_config, list):
            files_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in file_config}
        else:
            files_dict = file_config

        tables = {}
        for name, path in files_dict.items():
            df = DataLoader.load_from_file(path)
            tables[name] = df

        if len(tables) == 1:
            # 单表分析
            name = list(tables.keys())[0]
            analyzer = AutoStatisticalAnalyzer(tables[name], source_table_name=name, quiet=True)
            reporter = Reporter(analyzer)
            result = {
                'success': True,
                'analysis_type': 'single_table',
                'tables': {name: {'shape': tables[name].shape, 'columns': list(tables[name].columns)}}
            }
        else:
            # 多表分析
            relationships = None
            if relationships_json:
                relationships = json.loads(relationships_json)
            analyzer = MultiTableStatisticalAnalyzer(tables, relationships={'foreign_keys': relationships})
            result = {
                'success': True,
                'analysis_type': 'multi_table',
                'tables': {name: {'shape': df.shape, 'columns': list(df.columns)} for name, df in tables.items()},
                'relationships': analyzer.all_relationships.get('foreign_keys', [])
            }

        result['analysis_time'] = pd.Timestamp.now().isoformat()
        result['analysis_duration_seconds'] = time.time() - start_time

        return json.dumps(result, ensure_ascii=False, indent=2, default=str)

    except Exception as e:
        return json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=6011)