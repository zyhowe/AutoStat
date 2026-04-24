# web/services/analysis_service.py

"""分析服务 - 执行数据分析的核心业务逻辑"""

import streamlit as st
import json
import tempfile
import os
import time
import pickle
from autostat import AutoStatisticalAnalyzer, MultiTableStatisticalAnalyzer
from autostat.reporter import Reporter
from web.utils.helpers import capture_and_run, get_raw_data_preview
from web.services.file_service import FileService
from web.services.session_service import SessionService
from web.services.storage_service import StorageService
from web.services.feature_flags import FeatureFlags
from web.components.progress_stage import ProgressStage
from web.services.error_handler import ErrorHandler


class AnalysisService:
    """分析服务类"""

    @staticmethod
    def analyze_single_file(file_name: str, ext: str, filtered_df, variable_types: dict):
        """执行单文件分析"""
        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        # 初始化分阶段进度
        progress = ProgressStage()
        progress.start()

        try:
            # 阶段1：加载数据
            progress.start_stage("loading")
            status_placeholder.info("📁 正在准备数据...")
            progress.complete_stage("loading", f"已完成 {len(filtered_df)}行×{len(filtered_df.columns)}列")
            progress_bar.progress(progress.get_overall_progress())

            # 创建会话 - 单文件模式
            session_id = SessionService.create_session(file_name, "single")
            SessionService.set_current_session(session_id)
            st.session_state.current_session_id = session_id

            # 保存原始数据
            StorageService.save_dataframe("raw_data", filtered_df, session_id)
            StorageService.save_json("variable_types", variable_types, session_id)

            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{ext}', delete=False) as tmp:
                FileService.save_temp_file(filtered_df, tmp, ext)
                path = tmp.name

            # 阶段2：类型识别（使用用户定义类型，跳过）
            progress.start_stage("type")
            progress.complete_stage("type", f"已完成 {len(variable_types)}个变量")
            progress_bar.progress(progress.get_overall_progress())

            # 阶段3：特征提取
            progress.start_stage("feature")
            status_placeholder.info("🔍 正在分析数据...")
            progress.complete_stage("feature", "日期特征提取完成")
            progress_bar.progress(progress.get_overall_progress())

            date_level = st.session_state.date_features_level

            def run():
                a = AutoStatisticalAnalyzer(
                    path,
                    auto_clean=False,
                    quiet=False,
                    date_features_level=date_level,
                    predefined_types=variable_types,
                    skip_auto_inference=True
                )
                a.generate_full_report()
                return a

            analyzer, output = capture_and_run(run)

            # 阶段4-7：质量检查、关系分析、时间序列、生成报告
            progress.start_stage("quality")
            progress.complete_stage("quality", "数据质量检查完成")
            progress_bar.progress(progress.get_overall_progress())

            progress.start_stage("relation")
            progress.complete_stage("relation", "变量关系分析完成")
            progress_bar.progress(progress.get_overall_progress())

            progress.start_stage("timeseries")
            progress.complete_stage("timeseries", f"发现{len(analyzer.time_series_diagnostics)}个自相关序列")
            progress_bar.progress(progress.get_overall_progress())

            progress.start_stage("report")
            status_placeholder.info("📝 正在生成报告...")
            progress.complete_stage("report", "HTML/JSON报告生成完成")
            progress_bar.progress(progress.get_overall_progress())

            try:
                os.unlink(path)
            except:
                pass

            st.session_state.single_analyzer = analyzer
            st.session_state.single_output = output
            st.session_state.current_analysis_type = "single"
            st.session_state.current_source_name = file_name

            reporter = Reporter(analyzer)
            html_content = reporter.to_html()
            json_data = json.loads(analyzer.to_json())

            st.session_state.current_html = html_content
            st.session_state.current_json_data = json_data
            st.session_state.raw_data_preview = get_raw_data_preview(analyzer.data)
            st.session_state.chat_messages = []

            # 保存分析结果到存储
            StorageService.save_analysis_results(
                session_id, analyzer, html_content, json_data, output, filtered_df
            )

            # 保存分析器对象
            session_path = SessionService.get_session_path(session_id)
            with open(session_path / "analyzer.pkl", "wb") as f:
                pickle.dump(analyzer, f)

            # 标记分析完成并跳转到预览报告
            st.session_state.analysis_completed = True
            st.session_state.current_tab = 1
            st.session_state.scroll_to_top = True

            progress.complete()
            progress_bar.progress(1.0)
            status_placeholder.success("✅ 分析完成！")

            # 自动训练（如果开启）
            if FeatureFlags.is_auto_training_enabled() and FeatureFlags.is_auto_analysis_enabled():
                status_placeholder.info("🤖 正在自动训练推荐模型...")
                from web.services.auto_train_service import auto_train_from_recommendation
                auto_train_from_recommendation(session_id)
                status_placeholder.empty()

            # 自动解读（如果开启）
            if FeatureFlags.is_auto_interpretation_enabled() and st.session_state.llm_client is not None:
                status_placeholder.info("🧠 正在生成综合解读...")
                from web.services.auto_interpret_service import auto_interpret
                interpretation = auto_interpret(session_id, st.session_state.llm_client)
                if interpretation:
                    st.session_state.auto_interpretation = interpretation
                status_placeholder.empty()

            time.sleep(0.5)
            status_placeholder.empty()
            progress_bar.empty()

            st.rerun()

        except Exception as e:
            progress_bar.empty()
            status_placeholder.empty()
            error_msg = ErrorHandler.handle_error(e, "单文件分析", "请检查数据文件格式")
            st.error(f"分析失败: {error_msg}")
            with st.expander("📋 查看详情", expanded=False):
                st.code(str(e))

    @staticmethod
    def analyze_multi_file(filtered_tables: dict, variable_types_dict: dict, filtered_relationships: list):
        """执行多文件分析"""
        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        # 初始化分阶段进度
        progress = ProgressStage()
        progress.start()

        try:
            # 阶段1：加载数据
            progress.start_stage("loading")
            status_placeholder.info("📁 正在准备数据...")
            progress.complete_stage("loading", f"已加载{len(filtered_tables)}个表")
            progress_bar.progress(progress.get_overall_progress())

            # 创建会话 - 多文件模式
            first_name = list(filtered_tables.keys())[0] if filtered_tables else "unknown"
            source_name = f"{first_name}等{len(filtered_tables)}个文件"
            session_id = SessionService.create_session(source_name, "multi", filtered_tables)
            SessionService.set_current_session(session_id)
            st.session_state.current_session_id = session_id

            # 阶段2：类型识别
            progress.start_stage("type")
            progress.complete_stage("type", "类型识别完成")
            progress_bar.progress(progress.get_overall_progress())

            # 阶段3：特征提取
            progress.start_stage("feature")
            status_placeholder.info("🔍 正在分析数据...")
            progress.complete_stage("feature", "特征提取完成")
            progress_bar.progress(progress.get_overall_progress())

            def run():
                if filtered_relationships:
                    a = MultiTableStatisticalAnalyzer(
                        filtered_tables,
                        relationships={'foreign_keys': filtered_relationships},
                        predefined_types=variable_types_dict
                    )
                else:
                    a = MultiTableStatisticalAnalyzer(
                        filtered_tables,
                        predefined_types=variable_types_dict
                    )
                a.analyze_all_tables()
                return a

            analyzer, output = capture_and_run(run)

            # 阶段4-7
            progress.start_stage("quality")
            progress.complete_stage("quality", "质量检查完成")
            progress_bar.progress(progress.get_overall_progress())

            progress.start_stage("relation")
            progress.complete_stage("relation", "关系分析完成")
            progress_bar.progress(progress.get_overall_progress())

            progress.start_stage("timeseries")
            progress.complete_stage("timeseries", "时间序列分析完成")
            progress_bar.progress(progress.get_overall_progress())

            progress.start_stage("report")
            status_placeholder.info("📝 正在生成报告...")
            progress.complete_stage("report", "报告生成完成")
            progress_bar.progress(progress.get_overall_progress())

            st.session_state.multi_analyzer = analyzer
            st.session_state.multi_output = output
            st.session_state.current_analysis_type = "multi"
            st.session_state.current_source_name = source_name

            merged_analyzer = analyzer.get_merged_analyzer()
            reporter = Reporter(merged_analyzer)
            html_content = reporter.to_html()
            json_data = json.loads(merged_analyzer.to_json())

            st.session_state.current_html = html_content
            st.session_state.current_json_data = json_data
            st.session_state.raw_data_preview = get_raw_data_preview(merged_analyzer.data)
            st.session_state.chat_messages = []

            # 保存分析结果
            first_table_name = list(filtered_tables.keys())[0]
            StorageService.save_dataframe("raw_data", filtered_tables[first_table_name], session_id)
            StorageService.save_dataframe("processed_data", merged_analyzer.data, session_id)
            StorageService.save_json("variable_types", merged_analyzer.variable_types, session_id)
            StorageService.save_text("analysis_report", html_content, session_id)
            StorageService.save_json("analysis_result", json_data, session_id)
            StorageService.save_text("analysis_log", output, session_id)
            SessionService.update_data_shape(len(merged_analyzer.data), len(merged_analyzer.data.columns), session_id)
            SessionService.update_variable_types(merged_analyzer.variable_types, session_id)

            # 保存分析器对象
            session_path = SessionService.get_session_path(session_id)
            with open(session_path / "analyzer.pkl", "wb") as f:
                pickle.dump(merged_analyzer, f)

            # 标记分析完成并跳转到预览报告
            st.session_state.analysis_completed = True
            st.session_state.current_tab = 1
            st.session_state.scroll_to_top = True

            progress.complete()
            progress_bar.progress(1.0)
            status_placeholder.success("✅ 分析完成！")

            # 自动训练（如果开启）
            if FeatureFlags.is_auto_training_enabled() and FeatureFlags.is_auto_analysis_enabled():
                status_placeholder.info("🤖 正在自动训练推荐模型...")
                from web.services.auto_train_service import auto_train_from_recommendation
                auto_train_from_recommendation(session_id)
                status_placeholder.empty()

            # 自动解读（如果开启）
            if FeatureFlags.is_auto_interpretation_enabled() and st.session_state.llm_client is not None:
                status_placeholder.info("🧠 正在生成综合解读...")
                from web.services.auto_interpret_service import auto_interpret
                interpretation = auto_interpret(session_id, st.session_state.llm_client)
                if interpretation:
                    st.session_state.auto_interpretation = interpretation
                status_placeholder.empty()

            time.sleep(0.5)
            status_placeholder.empty()
            progress_bar.empty()

            st.rerun()

        except Exception as e:
            progress_bar.empty()
            status_placeholder.empty()
            error_msg = ErrorHandler.handle_error(e, "多文件分析", "请检查文件格式和表间关系")
            st.error(f"分析失败: {error_msg}")
            with st.expander("📋 查看详情", expanded=False):
                st.code(str(e))

    @staticmethod
    def analyze_database(filtered_tables: dict, variable_types_dict: dict, filtered_relationships: list, config: dict):
        """执行数据库分析"""
        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        # 初始化分阶段进度
        progress = ProgressStage()
        progress.start()

        try:
            # 阶段1：加载数据
            progress.start_stage("loading")
            status_placeholder.info("📁 正在准备数据...")
            progress.complete_stage("loading", f"已加载{len(filtered_tables)}个表")
            progress_bar.progress(progress.get_overall_progress())

            # 创建会话 - 数据库模式
            first_name = list(filtered_tables.keys())[0] if filtered_tables else "unknown"
            source_name = f"{first_name}等{len(filtered_tables)}个表"
            session_id = SessionService.create_session(source_name, "database", filtered_tables)
            SessionService.set_current_session(session_id)
            st.session_state.current_session_id = session_id

            # 阶段2：类型识别
            progress.start_stage("type")
            progress.complete_stage("type", "类型识别完成")
            progress_bar.progress(progress.get_overall_progress())

            # 阶段3：特征提取
            progress.start_stage("feature")
            status_placeholder.info("🔍 正在分析数据...")
            progress.complete_stage("feature", "特征提取完成")
            progress_bar.progress(progress.get_overall_progress())

            def run():
                if filtered_relationships:
                    a = MultiTableStatisticalAnalyzer(
                        filtered_tables,
                        relationships={'foreign_keys': filtered_relationships},
                        predefined_types=variable_types_dict
                    )
                else:
                    a = MultiTableStatisticalAnalyzer(
                        filtered_tables,
                        predefined_types=variable_types_dict
                    )
                a.analyze_all_tables()
                return a

            analyzer, output = capture_and_run(run)

            # 阶段4-7
            progress.start_stage("quality")
            progress.complete_stage("quality", "质量检查完成")
            progress_bar.progress(progress.get_overall_progress())

            progress.start_stage("relation")
            progress.complete_stage("relation", "关系分析完成")
            progress_bar.progress(progress.get_overall_progress())

            progress.start_stage("timeseries")
            progress.complete_stage("timeseries", "时间序列分析完成")
            progress_bar.progress(progress.get_overall_progress())

            progress.start_stage("report")
            status_placeholder.info("📝 正在生成报告...")
            progress.complete_stage("report", "报告生成完成")
            progress_bar.progress(progress.get_overall_progress())

            st.session_state.db_analyzer = analyzer
            st.session_state.db_output = output
            st.session_state.current_analysis_type = "database"
            st.session_state.current_source_name = source_name

            merged_analyzer = analyzer.get_merged_analyzer()
            reporter = Reporter(merged_analyzer)
            html_content = reporter.to_html()
            json_data = json.loads(merged_analyzer.to_json())
            json_data['db_server'] = config.get('server')
            json_data['db_database'] = config.get('database')

            st.session_state.current_html = html_content
            st.session_state.current_json_data = json_data
            st.session_state.raw_data_preview = get_raw_data_preview(merged_analyzer.data)
            st.session_state.chat_messages = []

            # 保存分析结果
            first_table_name = list(filtered_tables.keys())[0]
            StorageService.save_dataframe("raw_data", filtered_tables[first_table_name], session_id)
            StorageService.save_dataframe("processed_data", merged_analyzer.data, session_id)
            StorageService.save_json("variable_types", merged_analyzer.variable_types, session_id)
            StorageService.save_text("analysis_report", html_content, session_id)
            StorageService.save_json("analysis_result", json_data, session_id)
            StorageService.save_text("analysis_log", output, session_id)
            SessionService.update_data_shape(len(merged_analyzer.data), len(merged_analyzer.data.columns), session_id)
            SessionService.update_variable_types(merged_analyzer.variable_types, session_id)

            # 保存分析器对象
            session_path = SessionService.get_session_path(session_id)
            with open(session_path / "analyzer.pkl", "wb") as f:
                pickle.dump(merged_analyzer, f)

            # 标记分析完成并跳转到预览报告
            st.session_state.analysis_completed = True
            st.session_state.current_tab = 1
            st.session_state.scroll_to_top = True

            progress.complete()
            progress_bar.progress(1.0)
            status_placeholder.success("✅ 分析完成！")

            # 自动训练（如果开启）
            if FeatureFlags.is_auto_training_enabled() and FeatureFlags.is_auto_analysis_enabled():
                status_placeholder.info("🤖 正在自动训练推荐模型...")
                from web.services.auto_train_service import auto_train_from_recommendation
                auto_train_from_recommendation(session_id)
                status_placeholder.empty()

            # 自动解读（如果开启）
            if FeatureFlags.is_auto_interpretation_enabled() and st.session_state.llm_client is not None:
                status_placeholder.info("🧠 正在生成综合解读...")
                from web.services.auto_interpret_service import auto_interpret
                interpretation = auto_interpret(session_id, st.session_state.llm_client)
                if interpretation:
                    st.session_state.auto_interpretation = interpretation
                status_placeholder.empty()

            time.sleep(0.5)
            status_placeholder.empty()
            progress_bar.empty()

            st.rerun()

        except Exception as e:
            progress_bar.empty()
            status_placeholder.empty()
            error_msg = ErrorHandler.handle_error(e, "数据库分析", "请检查数据库连接和表结构")
            st.error(f"分析失败: {error_msg}")
            with st.expander("📋 查看详情", expanded=False):
                st.code(str(e))