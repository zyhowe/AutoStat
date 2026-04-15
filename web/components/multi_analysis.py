"""多文件分析组件"""

import streamlit as st
import pandas as pd
import tempfile
import os
import json
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autostat import MultiTableStatisticalAnalyzer
from autostat.loader import DataLoader
from autostat.reporter import Reporter
from web.components.chat_interface import render_chat_interface
from web.utils.helpers import capture_and_run, get_raw_data_preview
from web.utils.data_preprocessor import render_multi_preprocessing_interface


def multi_file_mode():
    """多文件分析模式"""
    st.markdown("### 📚 多文件分析")

    # 注意事项提示
    with st.expander("ℹ️ 使用说明与注意事项", expanded=False):
        st.markdown("""
        **适用场景：** 多个相关表需要联合分析（如订单表、用户表、产品表）
        
        **文件要求：** 至少上传 2 个文件，支持 CSV、Excel、JSON、TXT
        
        **限制建议：** 表数量 < 10，每个表行数 < 5万，总数据量 < 50万行
        
        **预处理功能：** 
        - 可以勾选要保留的字段，调整变量类型
        - 系统会自动发现表间关系，你可以在"表间关系管理"中修改或删除
        - 也可以手动添加新的关系
        """)

    files = st.file_uploader(
        "选择多个数据文件", type=['csv', 'xlsx', 'xls', 'json', 'txt'],
        accept_multiple_files=True, key="multi_uploader",
        help="按住 Ctrl 或 Cmd 键可选择多个文件，建议不超过10个"
    )

    if files and len(files) >= 2:
        # 表数量检查
        if len(files) > 10:
            st.warning(f"⚠️ 表数量 {len(files)} 超过建议限制 10个，分析可能较慢，建议减少表数量。")
        elif len(files) > 5:
            st.info(f"📊 表数量 {len(files)}，分析可能需要一些时间。")

        st.success(f"已选择 {len(files)} 个文件")

        # 显示文件列表和预估大小
        info = []
        total_size_mb = 0
        for f in files:
            size_mb = len(f.getvalue()) / (1024 * 1024)
            total_size_mb += size_mb
            info.append({
                "文件名": f.name,
                "大小": f"{size_mb:.1f} KB" if size_mb < 1 else f"{size_mb:.1f} MB"
            })

        st.dataframe(pd.DataFrame(info), use_container_width=True)

        # 总量警告
        if total_size_mb > 200:
            st.warning(f"⚠️ 总文件大小 {total_size_mb:.1f}MB，超过建议限制，分析可能较慢或内存不足。")
        elif total_size_mb > 100:
            st.info(f"📊 总文件大小 {total_size_mb:.1f}MB，分析可能需要一些时间。")

        # 单个文件大小警告
        large_files = [f for f in files if len(f.getvalue()) > 50 * 1024 * 1024]
        if large_files:
            st.warning(f"⚠️ 以下文件较大（>50MB），加载可能较慢：{', '.join([f.name for f in large_files])}")

        # 加载表
        preprocess_key = "multi_preprocess"

        if preprocess_key not in st.session_state:
            st.session_state[preprocess_key] = {
                "tables": None,
                "tmp_dir": None
            }

        # 加载表（只加载一次）
        if st.session_state[preprocess_key]["tables"] is None:
            with st.spinner("正在加载文件..."):
                tmp_dir, tables = load_tables(files)
                if tables:
                    st.session_state[preprocess_key]["tables"] = tables
                    st.session_state[preprocess_key]["tmp_dir"] = tmp_dir
                else:
                    st.error("没有成功加载任何表")
                    return

        tables = st.session_state[preprocess_key]["tables"]

        # 显示加载结果统计
        st.subheader("📊 已加载的表")
        for name, df in tables.items():
            st.caption(f"  📋 {name}: {len(df)}行 x {len(df.columns)}列")

        # 显示预处理界面（包含自动关系发现）
        confirmed, filtered_tables, variable_types_dict, filtered_relationships = render_multi_preprocessing_interface(
            tables,
            relationships=None,
            initial_types_dict=None
        )

        if confirmed:
            # 开始分析
            with st.spinner("分析中..."):
                status_placeholder = st.empty()
                progress_bar = st.progress(0)

                try:
                    status_placeholder.info("📁 正在准备数据...")
                    progress_bar.progress(20)

                    status_placeholder.info("🔍 正在分析数据...")
                    progress_bar.progress(50)

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

                    status_placeholder.info("📝 正在生成报告...")
                    progress_bar.progress(80)

                    # 清理临时文件
                    if "tmp_dir" in st.session_state[preprocess_key]:
                        shutil.rmtree(st.session_state[preprocess_key]["tmp_dir"], ignore_errors=True)

                    st.session_state.multi_analyzer = analyzer
                    st.session_state.multi_output = output
                    st.session_state.current_analysis_type = "multi"
                    st.session_state.current_source_name = f"{len(files)}个文件"

                    merged_analyzer = analyzer.get_merged_analyzer()
                    reporter = Reporter(merged_analyzer)
                    st.session_state.current_html = reporter.to_html()
                    st.session_state.current_json_data = json.loads(merged_analyzer.to_json())
                    st.session_state.raw_data_preview = get_raw_data_preview(merged_analyzer.data)
                    st.session_state.chat_messages = []

                    progress_bar.progress(100)
                    status_placeholder.success("✅ 分析完成！")

                    import time
                    time.sleep(0.5)
                    status_placeholder.empty()
                    progress_bar.empty()

                    st.rerun()

                except Exception as e:
                    progress_bar.empty()
                    status_placeholder.empty()
                    st.error(f"分析失败: {str(e)}")

    elif files and len(files) == 1:
        st.warning("多文件分析需要至少2个文件，请继续添加")

    # 显示结果
    display_results()


def load_tables(files):
    """加载所有表"""
    tmp_dir = tempfile.mkdtemp()
    paths = {}
    for f in files:
        p = os.path.join(tmp_dir, f.name)
        with open(p, 'wb') as w:
            w.write(f.getbuffer())
        paths[os.path.splitext(f.name)[0]] = p

    tables = {}
    failed = []
    for name, p in paths.items():
        try:
            df = DataLoader.load_from_file(p)
            if df is not None and not df.empty:
                tables[name] = df
            else:
                failed.append(f"{name} (空文件)")
        except Exception as e:
            failed.append(f"{name} ({str(e)[:50]})")

    if failed:
        st.warning(f"以下表加载失败: {', '.join(failed)}")

    return tmp_dir, tables


def display_results():
    """显示分析结果"""
    if st.session_state.multi_analyzer is not None:
        analyzer = st.session_state.multi_analyzer
        output = st.session_state.multi_output

        st.success("✅ 分析完成！")

        col_success, col_dl1, col_dl2 = st.columns([3, 1, 1])
        with col_dl1:
            st.download_button("📥 下载 HTML 报告", st.session_state.current_html,
                              "autostat_multi_report.html", "text/html", width="stretch")
        with col_dl2:
            st.download_button("📥 下载 JSON 结果", analyzer.to_json(),
                              "autostat_multi_result.json", "application/json", width="stretch")

        with st.expander("📝 分析过程日志", expanded=False):
            st.code(output, language='text')

        with st.expander("📄 预览报告", expanded=False):
            if st.session_state.current_html:
                st.html(st.session_state.current_html)
            else:
                st.info("暂无报告预览")

        with st.expander("🤖 AI 智能解读", expanded=False):
            if st.session_state.llm_client is None:
                st.warning("请先在侧边栏配置大模型")
            elif st.session_state.current_json_data is None:
                st.info("请先完成数据分析")
            else:
                render_chat_interface()