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

            # ==================== 🔍 调试：核心结论数据检查 ====================
            print("\n" + "=" * 70)
            print("🔍 核心结论数据检查")
            print("=" * 70)

            ts_diag = json_result.get('time_series_diagnostics', {})
            print(f"\n【1】time_series_diagnostics 数量: {len(ts_diag)}")
            if ts_diag:
                sample_key = list(ts_diag.keys())[0]
                print(f"    示例: {sample_key} -> {ts_diag[sample_key]}")

            high_corrs = json_result.get('correlations', {}).get('high_correlations', [])
            print(f"\n【2】high_correlations 数量: {len(high_corrs)}")
            if high_corrs:
                print(f"    示例: {high_corrs[0]}")

            missing = json_result.get('quality_report', {}).get('missing', [])
            print(f"\n【3】missing 数量: {len(missing)}")
            if missing:
                print(f"    示例: {missing[0]}")

            outliers = json_result.get('quality_report', {}).get('outliers', {})
            print(f"\n【4】outliers 数量: {len(outliers)}")
            if outliers:
                sample_key = list(outliers.keys())[0]
                print(f"    示例: {sample_key} -> {outliers[sample_key]}")

            data_shape = json_result.get('data_shape', {})
            print(f"\n【5】data_shape: {data_shape}")

            variable_types = json_result.get('variable_types', {})
            print(f"\n【6】variable_types 数量: {len(variable_types)}")
            if variable_types:
                sample_key = list(variable_types.keys())[0]
                print(f"    示例: {sample_key} -> {variable_types[sample_key]}")

            distribution = json_result.get('distribution_insights', {})
            skewed = distribution.get('skewed_variables', [])
            imbalanced = distribution.get('imbalanced_categoricals', [])
            print(f"\n【7】distribution_insights: skewed={len(skewed)}, imbalanced={len(imbalanced)}")

            model_recs = json_result.get('model_recommendations', [])
            print(f"\n【8】model_recommendations 数量: {len(model_recs)}")

            print("\n" + "=" * 70)
            print("✅ 调试数据打印完成")
            print("=" * 70 + "\n")

            # ============================================================
            # 🆕 生成核心结论（调用 InsightService）
            # ============================================================
            try:
                from autostat.core.insight import InsightService
                insight_service = InsightService()
                conclusions = insight_service.extract_top_conclusions(json_result)
                json_result['summary'] = conclusions
                # ✅ 重新保存（现在包含 summary）
                self.session_service.save_analysis_result(session_id, json_result)
                print(f"✅ 已生成核心结论: {len(conclusions)} 条")
                for c in conclusions:
                    print(f"    - {c.get('title', '')}")
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