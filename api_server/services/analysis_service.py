"""分析服务 - 不依赖 web/"""
import json
import sys
import io
import traceback

import logging
import warnings
warnings.filterwarnings('ignore')

# 设置 matplotlib 日志级别为 WARNING
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.category').setLevel(logging.WARNING)

from typing import Dict, Any, Optional

from autostat.core.analyzer import AutoStatisticalAnalyzer
from autostat.loader import DataLoader
from autostat.core.base import BaseAnalyzer
from autostat.reporter import Reporter
from api_server.services.session_service import SessionService


class AnalysisService:
    """分析执行服务"""

    def __init__(self):
        self.session_service = SessionService()

    def run_analysis(self, session_id: str, file_path: str, variable_types: Dict, task_id: str):
        from api_server.routers.analysis import task_status

        try:
            task_status[task_id] = {"status": "running", "progress": 10, "message": "加载数据中..."}

            df = DataLoader.load_from_file(file_path)

            task_status[task_id] = {"status": "running", "progress": 30, "message": "分析数据中..."}

            filtered_types = {}
            for k, v in variable_types.items():
                if v != 'exclude' and k in df.columns:
                    filtered_types[k] = v

            if not filtered_types:
                base = BaseAnalyzer(df, quiet=True)
                base._infer_variable_types()
                filtered_types = base.variable_types

            analyzer = AutoStatisticalAnalyzer(
                df,
                source_table_name=session_id,
                predefined_types=filtered_types,
                skip_auto_inference=bool(filtered_types),
                quiet=True
            )

            task_status[task_id] = {"status": "running", "progress": 60, "message": "生成报告中..."}

            # 捕获日志
            log_capture = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = log_capture

            try:
                analyzer.generate_full_report()
                log_content = log_capture.getvalue()
            finally:
                sys.stdout = old_stdout
                log_capture.close()

            self.session_service.save_log(session_id, log_content)

            # 保存 JSON 结果
            json_result = json.loads(analyzer.to_json())
            self.session_service.save_analysis_result(session_id, json_result)

            # 🆕 生成并保存 HTML 报告（一次生成，导出直接返回）
            reporter = Reporter(analyzer)
            html_content = reporter.to_html()
            self.session_service.save_html(session_id, html_content)

            self.session_service.save_variable_types(session_id, analyzer.variable_types)
            self.session_service.save_analyzer(session_id, analyzer)

            task_status[task_id] = {
                "status": "completed",
                "progress": 100,
                "message": "分析完成",
                "result": json_result
            }

        except Exception as e:
            task_status[task_id] = {
                "status": "failed",
                "progress": 0,
                "message": f"分析失败: {str(e)}",
                "error": traceback.format_exc()
            }