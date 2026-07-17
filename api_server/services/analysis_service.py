"""分析服务 - 统一入口，单表/多表均走 MultiTableStatisticalAnalyzer"""
import json
import sys
import io
import traceback
import logging
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.category').setLevel(logging.WARNING)

from typing import Dict, Any, Optional

from autostat.multi_analyzer import MultiTableStatisticalAnalyzer
from autostat.loader import DataLoader
from autostat.core.base import BaseAnalyzer
from autostat.reporter import Reporter
from api_server.services.session_service import SessionService
from api_server.services.recommendation_service import RecommendationService
from api_server.services.data_service import DataService


class AnalysisService:
    """分析执行服务 - 统一入口"""

    def __init__(self):
        self.session_service = SessionService()
        self.recommendation_service = RecommendationService()
        self._client_ip = None

    def set_client_ip(self, client_ip: str):
        self._client_ip = client_ip

    # ==================== 原有方法（保留兼容） ====================

    def run_analysis(self, session_id: str, file_path: str, variable_types: Dict, task_id: str, include_html: bool = False):
        """原有方法，保留兼容"""
        # 调用新方法，但将 variable_types 包装成缓存格式
        field_types_cache = {"data": variable_types} if variable_types else {}
        self.run_analysis_from_cache(session_id, file_path, field_types_cache, task_id, include_html)

    # ==================== 新增方法：从缓存读取字段类型 ====================

    def run_analysis_from_cache(
        self,
        session_id: str,
        file_path: str,
        field_types_cache: Dict[str, Dict[str, str]],
        task_id: str,
        include_html: bool = False
    ):
        """
        从缓存读取字段类型执行分析

        参数:
        - session_id: 会话ID
        - file_path: 数据文件路径
        - field_types_cache: {表名: {字段名: 类型}} 从 session 缓存读取
        - task_id: 任务ID
        - include_html: 是否生成HTML
        """
        from api_server.routers.analysis import task_status

        if self._client_ip:
            self.session_service.set_client_ip(self._client_ip)

        try:
            task_status[task_id] = {"status": "running", "progress": 10, "message": "加载数据中..."}

            # ==================== 统一加载所有表 ====================
            tables_data = self._load_all_tables(session_id, file_path)
            if not tables_data:
                raise ValueError("没有可用的表数据")

            task_status[task_id] = {"status": "running", "progress": 30, "message": "分析数据中..."}

            # ==================== 应用缓存的字段类型 ====================
            # 将缓存的字段类型应用到每个表
            for table_name, df in tables_data.items():
                cached_types = field_types_cache.get(table_name, {})
                if cached_types:
                    # 过滤掉不在当前表中的字段
                    valid_types = {k: v for k, v in cached_types.items() if k in df.columns}
                    if valid_types:
                        # 保存到 session 的 variable_types（分析时使用）
                        self.session_service.save_variable_types(session_id, valid_types)
                        print(f"✅ 应用字段类型缓存到 {table_name}: {len(valid_types)} 个字段")

            # ==================== 统一使用 MultiTableStatisticalAnalyzer ====================
            relationships = self.session_service.get_relationships(session_id)
            table_names = list(tables_data.keys())

            # 1. 创建分析器（只初始化，不执行分析）
            analyzer = MultiTableStatisticalAnalyzer(
                tables=tables_data,
                relationships=relationships,
                date_features_level="basic"
            )

            # 2. 执行分析（只执行一次，内部完成所有表的分析）
            analyzer.analyze_all()

            # 3. 获取合并分析器（此时已缓存）
            merged_analyzer = analyzer.get_merged_analyzer()
            self.session_service.save_analyzer(session_id, merged_analyzer)

            # ==================== 生成 JSON 结果 ====================
            json_result = json.loads(analyzer.to_json())
            json_result['is_multi_table'] = len(tables_data) > 1
            json_result['table_names'] = table_names
            json_result['relationships'] = relationships

            # ==================== 保存合并表 Parquet ====================
            merged_df = merged_analyzer.data
            if merged_df is not None and not merged_df.empty:
                DataService.save_to_parquet(merged_df, self.session_service, session_id, "merged")
                print(f"✅ 合并表已保存为 Parquet: merged.parquet")

            # ==================== HTML 报告 ====================
            if include_html:
                html_content = analyzer.to_html()
                self.session_service.save_html(session_id, html_content)

            self.session_service.save_variable_types(session_id, merged_analyzer.variable_types)

            # ==================== 生成推荐问题（所有表） ====================
            try:
                all_questions = {}
                all_tables = json_result.get('all_tables', {})
                for table_name, table_data in all_tables.items():
                    temp_result = {
                        'data_shape': table_data.get('data_shape', {}),
                        'variable_types': table_data.get('variable_types', {}),
                        'variable_summaries': table_data.get('variable_summaries', {}),
                        'quality_report': table_data.get('quality_report', {}),
                        'correlations': table_data.get('correlations', {}),
                        'time_series_diagnostics': table_data.get('time_series_diagnostics', {}),
                        'model_recommendations': table_data.get('model_recommendations', []),
                        'cleaning_suggestions': table_data.get('cleaning_suggestions', []),
                        'source_table': table_name
                    }
                    questions = self.recommendation_service.generate(temp_result)
                    all_questions[table_name] = questions
                    print(f"✅ 已生成 {table_name} 的推荐问题: {sum(len(v) for v in questions.values())} 条")

                # 按规范格式保存推荐问题
                new_questions = {}
                new_questions["merged"] = all_questions.get("merged", {})
                new_questions["all_tables"] = {}
                for table_name, questions in all_questions.items():
                    if table_name != "merged":
                        new_questions["all_tables"][table_name] = questions

                if len(tables_data) == 1:
                    new_questions["merged"] = {}

                self.session_service.save_recommended_questions(session_id, new_questions)
                print(f"✅ 所有表的推荐问题已保存")

            except Exception as e:
                print(f"⚠️ 生成推荐问题失败: {e}")
                import traceback as tb
                tb.print_exc()

            # ==================== 保存分析结果 ====================
            self.session_service.save_analysis_result(session_id, json_result)

            # ==================== 补充时间序列真实数据点 ====================
            self._enrich_timeseries_data(session_id, merged_analyzer, json_result)

            # ==================== 生成核心结论 ====================
            self._generate_conclusions(session_id, json_result)

            # ===== 清除字段类型缓存（分析完成后不再需要） =====
            self.session_service.clear_field_types_cache(session_id)

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

    # ==================== 私有方法 ====================

    def _load_all_tables(self, session_id: str, file_path: str) -> Dict[str, pd.DataFrame]:
        """加载所有表数据（优先 Parquet，回退 CSV）"""
        import pandas as pd

        tables = {}
        table_names = self.session_service.get_all_table_names(session_id)

        if table_names:
            # 从 Parquet 加载
            for name in table_names:
                df = DataService.load_parquet(self.session_service, session_id, name)
                if df is not None and not df.empty:
                    tables[name] = df
                    print(f"📂 从 Parquet 加载表 {name}: {len(df)} 行 x {len(df.columns)} 列")

        # 如果没有 Parquet 缓存，回退到 CSV
        if not tables:
            file_info = self.session_service.get_file(session_id)
            if file_info:
                print(f"📂 Parquet 不存在，从 CSV 加载: {file_info['path']}")
                df = DataLoader.load_from_file(file_info['path'])
                if df is not None and not df.empty:
                    import os
                    base_name = os.path.splitext(os.path.basename(file_info['name']))[0]
                    tables[base_name] = df
                    print(f"✅ 从 CSV 加载: {base_name} ({len(df)} 行 x {len(df.columns)} 列)")
                    DataService.save_to_parquet(df, self.session_service, session_id, base_name)
                    self.session_service.save_table_info(session_id, base_name, {
                        "rows": len(df),
                        "columns": len(df.columns),
                        "saved_at": pd.Timestamp.now().isoformat()
                    })

        return tables

    def _enrich_timeseries_data(self, session_id: str, analyzer, json_result: Dict):
        """补充时间序列真实数据点（最近30个）"""
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

    def _generate_conclusions(self, session_id: str, json_result: Dict):
        """生成核心结论"""
        try:
            from autostat.core.insight import InsightService
            insight_service = InsightService()
            conclusions = insight_service.extract_top_conclusions(json_result)
            json_result['summary'] = conclusions
            self.session_service.save_analysis_result(session_id, json_result)
            print(f"✅ 已生成核心结论: {len(conclusions)} 条")
        except Exception as e:
            print(f"⚠️ 生成核心结论失败: {e}")