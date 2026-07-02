# api_server/services/analysis_service.py

"""分析服务 - 不依赖 web/"""
import json
import traceback
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 🆕 使用非交互式后端，不弹窗
from typing import Dict, Any, Optional

from autostat.core.analyzer import AutoStatisticalAnalyzer
from autostat.loader import DataLoader
from api_server.services.session_service import SessionService


class AnalysisService:
    """分析执行服务"""

    def __init__(self):
        self.session_service = SessionService()

    def run_analysis(self, session_id: str, file_path: str, variable_types: Dict, task_id: str):
        from api_server.routers.analysis import task_status

        try:
            task_status[task_id] = {
                "status": "running",
                "progress": 10,
                "message": "加载数据中..."
            }

            df = DataLoader.load_from_file(file_path)

            task_status[task_id] = {
                "status": "running",
                "progress": 30,
                "message": "分析数据中..."
            }

            # 🆕 过滤：只保留存在于 df 中的字段
            filtered_types = {}
            for k, v in variable_types.items():
                if v != 'exclude' and k in df.columns:
                    filtered_types[k] = v

            # 如果没有有效的类型，自动推断
            if not filtered_types:
                from autostat.core.base import BaseAnalyzer
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

            task_status[task_id] = {
                "status": "running",
                "progress": 60,
                "message": "生成报告中..."
            }

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.switch_backend('Agg')

            analyzer.generate_full_report()
            json_result = json.loads(analyzer.to_json())

            task_status[task_id] = {
                "status": "running",
                "progress": 80,
                "message": "保存结果中..."
            }

            self.session_service.save_analysis_result(session_id, json_result)
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