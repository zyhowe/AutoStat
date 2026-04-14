"""单文件分析组件"""

import streamlit as st
import pandas as pd
import tempfile
import os
import io
import json
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autostat import AutoStatisticalAnalyzer
from autostat.reporter import Reporter
from web.components.chat_interface import render_chat_interface
from web.utils.helpers import capture_and_run, get_raw_data_preview
from web.utils.data_preprocessor import render_preprocessing_interface, get_default_variable_type


def single_file_mode():
    """单文件分析模式"""
    st.markdown("### 📁 单文件分析")

    # 注意事项提示（保持原有完整内容）
    with st.expander("ℹ️ 使用说明与注意事项", expanded=False):
        st.markdown("""
        **支持的文件格式：** CSV、Excel(.xlsx/.xls)、JSON、TXT

        **数据要求：**
        - 文件编码建议使用 UTF-8（会自动尝试其他编码）
        - 首行应为列名（表头）
        - 避免包含特殊控制字符（会自动清理）

        **限制建议：**
        - 行数：建议 < 10万行（大文件建议先采样）
        - 列数：建议 < 200列
        - 文件大小：建议 < 100MB

        **分析内容：**
        - 自动识别变量类型（连续/分类/日期/标识符）
        - 数据质量检查（缺失值、异常值、重复值）
        - 相关性分析
        - 时间序列分析（如有日期列）
        - 生成 HTML/JSON/Markdown/Excel 报告

        **预处理功能：**
        - 可以勾选要保留的字段（排除的字段不参与分析）
        - 可以调整每个字段的变量类型
        - 系统字段（如 tmstamp、entrydt 等）默认会被标记为"排除"

        **注意事项：**
        - 大文件（>100MB）建议先采样后再分析
        - 日期列建议命名为包含"date""时间""日期"等关键词
        - 标识符列（如ID）会被自动识别并排除分析
        """)

    uploaded = st.file_uploader(
        "选择数据文件", type=['csv', 'xlsx', 'xls', 'json', 'txt'],
        help="支持 CSV、Excel、JSON、TXT 格式",
        key="single_uploader"
    )

    if uploaded:
        file_size = len(uploaded.getvalue()) / (1024 * 1024)
        if file_size > 100:
            st.warning(f"⚠️ 文件大小 {file_size:.1f}MB，超过建议限制")
        elif file_size > 50:
            st.info(f"📊 文件大小 {file_size:.1f}MB，加载可能需要一些时间")

        ext = uploaded.name.split('.')[-1].lower()
        date_level = st.selectbox(
            "日期特征级别",
            ['basic', 'none'],
            index=0,
            key="single_date",
            help="basic: 提取年/月/季度；none: 不提取日期特征"
        )

        # 加载数据
        try:
            df = load_file(uploaded, ext)
            if df is None or df.empty:
                st.error("文件加载失败")
                return

            # 清理列名
            df.columns = [str(col).strip().replace('\n', '_').replace('\r', '_') for col in df.columns]
            df = df.replace(r'^\s*$', pd.NA, regex=True)

            # 显示原始数据预览
            rows, cols = df.shape
            st.subheader("📄 原始数据预览")
            st.dataframe(df.head(100))
            st.write(f"形状: {rows:,} 行 × {cols} 列")

            # 数据质量快速提示
            col_info = []
            for col in df.columns:
                missing_pct = df[col].isna().mean() * 100
                if missing_pct > 50:
                    col_info.append(f"⚠️ {col}: 缺失率 {missing_pct:.1f}%")
                elif missing_pct > 20:
                    col_info.append(f"📌 {col}: 缺失率 {missing_pct:.1f}%")
            if col_info:
                with st.expander("📊 数据质量提醒", expanded=False):
                    for info in col_info[:5]:
                        st.caption(info)
                    if len(col_info) > 5:
                        st.caption(f"... 还有 {len(col_info) - 5} 个字段")

            # 预处理界面
            confirmed, filtered_df, variable_types = render_preprocessing_interface(df, "数据预处理")

            if confirmed:
                # 采样建议（大文件时）
                sampled_df = filtered_df
                if len(filtered_df) > 50000:
                    sample_btn = st.checkbox("📌 建议：对数据进行采样分析（随机抽取 10000 行）", value=False)
                    if sample_btn:
                        sampled_df = filtered_df.sample(n=min(10000, len(filtered_df)), random_state=42)
                        st.info(f"已采样至 {len(sampled_df)} 行")

                # 开始分析
                with st.spinner("分析中..."):
                    status_placeholder = st.empty()
                    progress_bar = st.progress(0)

                    try:
                        status_placeholder.info("📁 正在准备数据...")
                        progress_bar.progress(20)

                        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{ext}', delete=False) as tmp:
                            save_file(sampled_df, tmp, ext)
                            path = tmp.name

                        status_placeholder.info("🔍 正在分析数据...")
                        progress_bar.progress(50)

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

                        status_placeholder.info("📝 正在生成报告...")
                        progress_bar.progress(80)

                        try:
                            os.unlink(path)
                        except:
                            pass

                        st.session_state.single_analyzer = analyzer
                        st.session_state.single_output = output
                        st.session_state.current_analysis_type = "single"
                        st.session_state.current_source_name = uploaded.name

                        reporter = Reporter(analyzer)
                        st.session_state.current_html = reporter.to_html()
                        st.session_state.current_json_data = json.loads(analyzer.to_json())
                        st.session_state.raw_data_preview = get_raw_data_preview(sampled_df)
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

        except Exception as e:
            st.error(f"读取失败: {str(e)}")

    # 显示结果
    display_results()


def load_file(uploaded, ext):
    """加载文件"""
    if ext == 'csv':
        return load_csv(uploaded)
    elif ext in ['xlsx', 'xls']:
        return load_excel(uploaded, ext)
    elif ext == 'json':
        return load_json(uploaded)
    else:
        return pd.read_csv(uploaded, delimiter='\t', engine='python', on_bad_lines='skip')


def load_csv(uploaded):
    """加载 CSV 文件"""
    try:
        content = uploaded.read()
        content = content.replace(b'\x00', b'')
        content_str = content.decode('utf-8', errors='ignore')
        return pd.read_csv(io.StringIO(content_str))
    except:
        uploaded.seek(0)
        return pd.read_csv(uploaded, encoding='utf-8', engine='python', on_bad_lines='skip')


def load_excel(uploaded, ext):
    """加载 Excel 文件"""
    with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as tmp_excel:
        tmp_excel.write(uploaded.getbuffer())
        tmp_excel_path = tmp_excel.name

    try:
        df = pd.read_excel(tmp_excel_path, engine='openpyxl')
    except:
        try:
            df = pd.read_excel(tmp_excel_path, engine='xlrd')
        except:
            df = pd.read_excel(tmp_excel_path)

    os.unlink(tmp_excel_path)
    return df


def load_json(uploaded):
    """加载 JSON 文件"""
    content = uploaded.read()
    content_str = content.decode('utf-8', errors='ignore')
    content_str = re.sub(r'[\x00-\x1f\x7f]', '', content_str)

    try:
        data = json.loads(content_str)
        return pd.DataFrame(data)
    except:
        lines = content_str.strip().split('\n')
        data_list = []
        for line in lines:
            if line.strip():
                data_list.append(json.loads(line))
        return pd.DataFrame(data_list)


def save_file(df, tmp, ext):
    """保存文件到临时文件"""
    if ext == 'csv':
        df.to_csv(tmp.name, index=False, encoding='utf-8')
    elif ext in ['xlsx', 'xls']:
        df.to_excel(tmp.name, index=False)
    elif ext == 'json':
        df.to_json(tmp.name, orient='records', force_ascii=False)
    else:
        df.to_csv(tmp.name, sep='\t', index=False, encoding='utf-8')


def display_results():
    """显示分析结果"""
    if st.session_state.single_analyzer is not None:
        analyzer = st.session_state.single_analyzer
        output = st.session_state.single_output

        st.success("✅ 分析完成！")

        col_success, col_dl1, col_dl2 = st.columns([3, 1, 1])
        with col_dl1:
            st.download_button("📥 下载 HTML 报告", st.session_state.current_html,
                              "autostat_report.html", "text/html", width="stretch")
        with col_dl2:
            st.download_button("📥 下载 JSON 结果", analyzer.to_json(),
                              "autostat_result.json", "application/json", width="stretch")

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