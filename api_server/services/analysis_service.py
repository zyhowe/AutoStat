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

    def run_analysis(self, session_id: str, file_path: str, variable_types: Dict, task_id: str):
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


            # ============================================================
            # 🆕 补充时间序列真实数据点（最近30个）
            # ============================================================
            try:
                # 获取数据框
                data = analyzer.data
                variable_types_dict = analyzer.variable_types

                # 找出日期列
                date_cols = [col for col, typ in variable_types_dict.items() if typ == 'datetime' and col in data.columns]
                if date_cols:
                    date_col = date_cols[0]
                    numeric_cols = [col for col, typ in variable_types_dict.items() if typ == 'continuous' and col in data.columns]

                    # 获取时间序列诊断结果
                    ts_diag = json_result.get('time_series_diagnostics', {})

                    for col in numeric_cols:
                        try:
                            # 按日期分组取均值
                            ts_data = data.groupby(date_col)[col].mean().reset_index()
                            ts_data = ts_data.dropna()
                            if len(ts_data) > 0:
                                # 取最近30个数据点
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

                                # 找到对应的诊断条目并添加 data_points
                                # 诊断条目的 key 可能是 col 或 col_分组
                                found = False
                                for key in list(ts_diag.keys()):
                                    if key == col or key.startswith(col + '_'):
                                        ts_diag[key]['data_points'] = data_points
                                        found = True
                                        break
                                # 如果没有精确匹配，将数据点添加到第一个包含该字段名的条目
                                if not found:
                                    for key in list(ts_diag.keys()):
                                        if col in key:
                                            ts_diag[key]['data_points'] = data_points
                                            break
                        except Exception as e:
                            print(f"⚠️ 保存时间序列数据点失败 ({col}): {e}")
                            continue

                    # 更新 json_result 并重新保存
                    json_result['time_series_diagnostics'] = ts_diag
                    self.session_service.save_analysis_result(session_id, json_result)
                    print(f"✅ 已保存时间序列真实数据点")
            except Exception as e:
                print(f"⚠️ 补充时间序列数据点失败: {e}")
                import traceback as tb
                tb.print_exc()

            # ============================================================
            # 🆕 生成核心结论
            # ============================================================
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

            # 🆕 生成并保存 HTML 报告
            reporter = Reporter(analyzer)
            html_content = reporter.to_html()
            self.session_service.save_html(session_id, html_content)

            self.session_service.save_variable_types(session_id, analyzer.variable_types)
            self.session_service.save_analyzer(session_id, analyzer)

            # ============================================================
            # 🆕 生成个性化推荐问题
            # ============================================================
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