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
from api_server.services.recommendation_service import RecommendationService


class AnalysisService:
    """分析执行服务"""

    def __init__(self):
        self.session_service = SessionService()
        self.recommendation_service = RecommendationService()
        self._client_ip = None

    def set_client_ip(self, client_ip: str):
        self._client_ip = client_ip

    def run_analysis(self, session_id: str, file_path: str, variable_types: Dict, task_id: str, include_html: bool = False):  # ✅ 新增参数
        from api_server.routers.analysis import task_status

        if self._client_ip:
            self.session_service.set_client_ip(self._client_ip)

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

            # 获取真实表名（带调试日志）
            print("\n" + "=" * 70)
            print("[DEBUG] ===== analysis_service.py: 获取真实表名 =====")
            print("=" * 70)

            real_table_name = None
            session_meta = self.session_service.get_session(session_id)

            print(f"[DEBUG] session_id: {session_id}")
            print(f"[DEBUG] session_meta: {session_meta is not None}")

            if session_meta:
                tables_info = session_meta.get('tables_info', {})
                print(f"[DEBUG] tables_info: {tables_info}")

                if isinstance(tables_info, dict):
                    tables = tables_info.get('tables', [])
                    print(f"[DEBUG] tables (dict): {tables}")
                    if tables and len(tables) > 0:
                        real_table_name = tables[0]
                elif isinstance(tables_info, list):
                    print(f"[DEBUG] tables (list): {tables_info}")
                    if tables_info:
                        real_table_name = tables_info[0]

            if not real_table_name and session_meta:
                source_name = session_meta.get('source_name', '')
                print(f"[DEBUG] source_name: {source_name}")
                if source_name.endswith('_db'):
                    real_table_name = source_name[:-3]
                elif source_name.endswith('_demo'):
                    real_table_name = source_name[:-5]
                else:
                    real_table_name = source_name
                print(f"[DEBUG] real_table_name from source_name: {real_table_name}")

            if not real_table_name:
                real_table_name = session_id
                print(f"[DEBUG] real_table_name fallback to session_id: {real_table_name}")

            print(f"[DEBUG] ✅ 最终 real_table_name: {real_table_name}")
            print("=" * 70 + "\n")

            analyzer = AutoStatisticalAnalyzer(
                df,
                source_table_name=real_table_name,
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
                # ✅ 传递 include_html 参数给 generate_full_report
                analyzer.generate_full_report(include_html=include_html)
                log_content = log_capture.getvalue()
            finally:
                sys.stdout = old_stdout
                log_capture.close()

            self.session_service.save_log(session_id, log_content)

            # 保存 JSON 结果
            json_result = json.loads(analyzer.to_json())
            self.session_service.save_analysis_result(session_id, json_result)

            print(f"[DEBUG] ✅ analysis_result.source_table = {json_result.get('source_table')}")

            # 补充时间序列真实数据点（最近30个）
            try:
                data = analyzer.data
                variable_types_dict = analyzer.variable_types

                date_cols = [col for col, typ in variable_types_dict.items() if typ == 'datetime' and col in data.columns]
                if date_cols:
                    date_col = date_cols[0]
                    numeric_cols = [col for col, typ in variable_types_dict.items() if typ == 'continuous' and col in data.columns]

                    ts_diag = json_result.get('time_series_diagnostics', {})

                    for col in numeric_cols:
                        try:
                            ts_data = data.groupby(date_col)[col].mean().reset_index()
                            ts_data = ts_data.dropna()
                            if len(ts_data) > 0:
                                points = ts_data.tail(30)
                                data_points = []
                                for _, row in points.iterrows():
                                    date_val = row[date_col]
                                    if hasattr(date_val, 'strftime'):
                                        date_str = date_val.strftime('%Y-%m-%d')
                                    else:
                                        date_str = str(date_val)
                                    data_points.append({
                                        'date': date_str,
                                        'value': float(row[col])
                                    })

                                found = False
                                for key in list(ts_diag.keys()):
                                    if key == col or key.startswith(col + '_'):
                                        ts_diag[key]['data_points'] = data_points
                                        found = True
                                        break
                                if not found:
                                    for key in list(ts_diag.keys()):
                                        if col in key:
                                            ts_diag[key]['data_points'] = data_points
                                            break
                        except Exception as e:
                            print(f"⚠️ 保存时间序列数据点失败 ({col}): {e}")
                            continue

                    json_result['time_series_diagnostics'] = ts_diag
                    self.session_service.save_analysis_result(session_id, json_result)
                    print(f"✅ 已保存时间序列真实数据点")
            except Exception as e:
                print(f"⚠️ 补充时间序列数据点失败: {e}")
                import traceback as tb
                tb.print_exc()

            # 生成核心结论
            try:
                from autostat.core.insight import InsightService
                insight_service = InsightService()
                conclusions = insight_service.extract_top_conclusions(json_result)
                json_result['summary'] = conclusions
                self.session_service.save_analysis_result(session_id, json_result)
                print(f"✅ 已生成核心结论: {len(conclusions)} 条")
            except Exception as e:
                print(f"⚠️ 生成核心结论失败: {e}")
                import traceback as tb
                tb.print_exc()

            # ============================================================
            # HTML 报告：仅在 include_html=True 时生成
            # ============================================================
            if include_html:
                reporter = Reporter(analyzer)
                html_content = reporter.to_html()
                self.session_service.save_html(session_id, html_content)
                print(f"✅ HTML 报告已生成并保存")
            else:
                print(f"⏩ 跳过 HTML 报告生成 (include_html=False)")

            self.session_service.save_variable_types(session_id, analyzer.variable_types)
            self.session_service.save_analyzer(session_id, analyzer)

            # 生成个性化推荐问题
            try:
                questions = self.recommendation_service.generate(json_result)
                self.session_service.save_recommended_questions(session_id, questions)
                print(f"✅ 已生成推荐问题: {sum(len(v) for v in questions.values())} 条")
            except Exception as e:
                print(f"⚠️ 生成推荐问题失败: {e}")
                import traceback as tb
                tb.print_exc()

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