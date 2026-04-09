"""
Streamlit Web界面 - 支持单文件、多文件、数据库连接
"""

import streamlit as st
import pandas as pd
import sys
import os
import tempfile
import contextlib
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autostat import AutoStatisticalAnalyzer, MultiTableStatisticalAnalyzer
from autostat.loader import DataLoader
from autostat.reporter import Reporter

st.set_page_config(
    page_title="AutoStat 智能数据分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 session_state
if 'single_analyzer' not in st.session_state:
    st.session_state.single_analyzer = None
if 'single_output' not in st.session_state:
    st.session_state.single_output = None
if 'multi_analyzer' not in st.session_state:
    st.session_state.multi_analyzer = None
if 'multi_output' not in st.session_state:
    st.session_state.multi_output = None
if 'db_analyzer' not in st.session_state:
    st.session_state.db_analyzer = None
if 'db_output' not in st.session_state:
    st.session_state.db_output = None
if 'single_file_key' not in st.session_state:
    st.session_state.single_file_key = 0
if 'multi_file_key' not in st.session_state:
    st.session_state.multi_file_key = 0

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 AutoStat 智能数据分析助手</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">自动识别数据类型、检测数据质量、选择统计方法、生成分析报告</div>', unsafe_allow_html=True)

st.sidebar.title("⚙️ 分析模式")
analysis_mode = st.sidebar.selectbox(
    "选择分析模式",
    ["📁 单文件分析", "📚 多文件分析", "🗄️ 数据库分析"],
    key="analysis_mode"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 使用说明")
st.sidebar.markdown("""
**单文件分析**：上传 CSV、Excel、JSON、TXT 文件
**多文件分析**：上传多个相关文件，自动发现表间关系
**数据库分析**：连接 SQL Server 数据库
""")


def capture_and_run(func, *args, **kwargs):
    """捕获输出并运行函数"""
    f = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = f
        sys.stderr = f
        result = func(*args, **kwargs)
        output = f.getvalue()
        return result, output
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def clear_single():
    st.session_state.single_analyzer = None
    st.session_state.single_output = None


def clear_multi():
    st.session_state.multi_analyzer = None
    st.session_state.multi_output = None


def clear_db():
    st.session_state.db_analyzer = None
    st.session_state.db_output = None


def display_single_results():
    if st.session_state.single_analyzer is None:
        return

    analyzer = st.session_state.single_analyzer
    output = st.session_state.single_output

    st.success("✅ 分析完成！")

    with st.expander("📝 分析过程日志", expanded=True):
        st.code(output, language='text')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总行数", f"{len(analyzer.data):,}")
    with col2:
        st.metric("总列数", len(analyzer.data.columns))
    with col3:
        missing_cnt = len([m for m in analyzer.quality_report.get('missing', []) if m.get('percent', 0) > 20])
        st.metric("高缺失字段", missing_cnt)
    with col4:
        dup_cnt = analyzer.quality_report.get('duplicates', {}).get('count', 0)
        st.metric("重复记录", dup_cnt)

    st.subheader("📋 变量类型分布")
    type_counts = {}
    type_desc = {
        'continuous': '连续变量', 'categorical': '分类变量',
        'categorical_numeric': '数值型分类', 'ordinal': '有序分类',
        'datetime': '日期时间', 'identifier': '标识符', 'text': '文本'
    }
    for typ in analyzer.variable_types.values():
        type_counts[typ] = type_counts.get(typ, 0) + 1

    col1, col2 = st.columns(2)
    items = list(type_counts.items())
    mid = len(items) // 2
    with col1:
        for typ, cnt in items[:mid]:
            st.write(f"- {type_desc.get(typ, typ)}: {cnt} 列")
    with col2:
        for typ, cnt in items[mid:]:
            st.write(f"- {type_desc.get(typ, typ)}: {cnt} 列")

    if analyzer.cleaning_suggestions:
        with st.expander("💡 清洗建议"):
            for s in analyzer.cleaning_suggestions[:5]:
                st.write(f"- {s}")

    reporter = Reporter(analyzer)

    html = reporter.to_html()
    json_str = reporter.to_json()

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("📥 下载 HTML 报告", html, "autostat_report.html", "text/html")
    with c2:
        st.download_button("📥 下载 JSON 结果", json_str, "autostat_result.json", "application/json")


def display_multi_results():
    if st.session_state.multi_analyzer is None:
        return

    analyzer = st.session_state.multi_analyzer
    output = st.session_state.multi_output

    st.success("✅ 分析完成！")

    with st.expander("📝 分析过程日志", expanded=True):
        st.code(output, language='text')

    st.subheader("📊 表组信息")
    for g in analyzer.table_groups:
        if g['type'] == 'related':
            st.write(f"🔗 关联表组: {', '.join(g['tables'])} ({len(g['relationships'])}个关系)")
        else:
            st.write(f"📄 独立表: {g['tables'][0]}")

    st.subheader("📋 各表信息")
    info = []
    for name, df in analyzer.tables.items():
        info.append({"表名": name, "行数": len(df), "列数": len(df.columns)})
    st.dataframe(pd.DataFrame(info))

    # 获取合并后的分析器，用于显示时间序列诊断
    merged_analyzer = analyzer.get_merged_analyzer()

    # 显示时间序列诊断（如果有）
    if hasattr(merged_analyzer, 'time_series_diagnostics') and merged_analyzer.time_series_diagnostics:
        st.subheader("📈 时间序列诊断")
        diag_data = []
        for key, diag in merged_analyzer.time_series_diagnostics.items():
            diag_data.append({
                "变量/分组": key,
                "平稳性": "✅ 平稳" if diag.get('is_stationary') else "⚠️ 非平稳",
                "自相关性": "✅ 存在" if diag.get('has_autocorrelation') else "❌ 无",
                "样本量": diag.get('n_samples', 0)
            })
        st.dataframe(pd.DataFrame(diag_data))

    # 生成报告
    html = analyzer.to_html()
    json_str = analyzer.to_json()

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("📥 下载 HTML 报告", html, "autostat_multi_report.html", "text/html")
    with c2:
        st.download_button("📥 下载 JSON 结果", json_str, "autostat_multi_result.json", "application/json")


def display_db_results():
    if st.session_state.db_analyzer is None:
        return

    analyzer = st.session_state.db_analyzer
    output = st.session_state.db_output

    st.success("✅ 分析完成！")

    with st.expander("📝 分析过程日志", expanded=True):
        st.code(output, language='text')

    st.subheader("📊 表组信息")
    for g in analyzer.table_groups:
        if g['type'] == 'related':
            st.write(f"🔗 关联表组: {', '.join(g['tables'])} ({len(g['relationships'])}个关系)")
        else:
            st.write(f"📄 独立表: {g['tables'][0]}")

    st.subheader("📋 各表信息")
    info = []
    for name, df in analyzer.tables.items():
        info.append({"表名": name, "行数": len(df), "列数": len(df.columns)})
    st.dataframe(pd.DataFrame(info))

    # 获取合并后的分析器，用于显示时间序列诊断
    merged_analyzer = analyzer.get_merged_analyzer()

    # 显示时间序列诊断（如果有）
    if hasattr(merged_analyzer, 'time_series_diagnostics') and merged_analyzer.time_series_diagnostics:
        st.subheader("📈 时间序列诊断")
        diag_data = []
        for key, diag in merged_analyzer.time_series_diagnostics.items():
            diag_data.append({
                "变量/分组": key,
                "平稳性": "✅ 平稳" if diag.get('is_stationary') else "⚠️ 非平稳",
                "自相关性": "✅ 存在" if diag.get('has_autocorrelation') else "❌ 无",
                "样本量": diag.get('n_samples', 0)
            })
        st.dataframe(pd.DataFrame(diag_data))

    # 生成报告
    html = analyzer.to_html()
    json_str = analyzer.to_json()

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("📥 下载 HTML 报告", html, "autostat_db_report.html", "text/html")
    with c2:
        st.download_button("📥 下载 JSON 结果", json_str, "autostat_db_result.json", "application/json")


# ==================== 单文件分析 ====================
def single_file_mode():
    st.markdown("### 📁 单文件分析")

    uploaded = st.file_uploader(
        "选择数据文件", type=['csv', 'xlsx', 'xls', 'json', 'txt'],
        help="支持 CSV、Excel、JSON、TXT 格式",
        key="single_uploader"
    )

    if uploaded:
        ext = uploaded.name.split('.')[-1].lower()
        date_level = st.selectbox("日期特征级别", ['basic', 'none'], index=0, key="single_date")

        try:
            if ext == 'csv':
                df = pd.read_csv(uploaded)
            elif ext in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded)
            elif ext == 'json':
                df = pd.read_json(uploaded)
            else:
                df = pd.read_csv(uploaded, delimiter='\t')

            st.subheader("📄 数据预览")
            st.dataframe(df.head(100))
            st.write(f"形状: {df.shape[0]} 行 × {df.shape[1]} 列")

            if st.button("🚀 开始分析", type="primary", key="single_start"):
                with st.spinner("分析中..."):
                    with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{ext}', delete=False) as tmp:
                        if ext == 'csv':
                            df.to_csv(tmp.name, index=False)
                        elif ext in ['xlsx', 'xls']:
                            df.to_excel(tmp.name, index=False)
                        elif ext == 'json':
                            df.to_json(tmp.name, orient='records', force_ascii=False)
                        else:
                            df.to_csv(tmp.name, sep='\t', index=False)
                        path = tmp.name

                    def run():
                        a = AutoStatisticalAnalyzer(path, auto_clean=False, quiet=False, date_features_level=date_level)
                        a.generate_full_report()
                        return a

                    analyzer, output = capture_and_run(run)

                    try:
                        os.unlink(path)
                    except:
                        pass

                    st.session_state.single_analyzer = analyzer
                    st.session_state.single_output = output
                    st.rerun()

        except Exception as e:
            st.error(f"读取失败: {str(e)}")

    display_single_results()


# ==================== 多文件分析 ====================
def multi_file_mode():
    st.markdown("### 📚 多文件分析")

    files = st.file_uploader(
        "选择多个数据文件", type=['csv', 'xlsx', 'xls', 'json', 'txt'],
        accept_multiple_files=True, key="multi_uploader"
    )

    if files and len(files) >= 2:
        st.success(f"已选择 {len(files)} 个文件")

        info = [{"文件名": f.name, "大小": f"{len(f.getvalue())/1024:.1f} KB"} for f in files]
        st.dataframe(pd.DataFrame(info))

        with st.expander("🔗 手动定义表间关系（可选）"):
            rel_text = st.text_area("关系定义（每行一个）", placeholder="orders.user_id = users.user_id", key="multi_rel")
            relationships = []
            if rel_text.strip():
                for line in rel_text.strip().split('\n'):
                    if '=' in line:
                        parts = line.split('=')
                        if len(parts) == 2:
                            from_part, to_part = parts[0].strip(), parts[1].strip()
                            if '.' in from_part and '.' in to_part:
                                ft, fc = from_part.split('.')
                                tt, tc = to_part.split('.')
                                relationships.append({'from_table': ft, 'from_col': fc, 'to_table': tt, 'to_col': tc})

        if st.button("🚀 开始分析", type="primary", key="multi_start"):
            with st.spinner("分析中..."):
                tmp_dir = tempfile.mkdtemp()
                paths = {}
                for f in files:
                    p = os.path.join(tmp_dir, f.name)
                    with open(p, 'wb') as w:
                        w.write(f.getbuffer())
                    paths[os.path.splitext(f.name)[0]] = p

                tables = {}
                for name, p in paths.items():
                    tables[name] = DataLoader.load_from_file(p)

                def run():
                    if relationships:
                        a = MultiTableStatisticalAnalyzer(tables, relationships={'foreign_keys': relationships})
                    else:
                        a = MultiTableStatisticalAnalyzer(tables)
                    a.analyze_all_tables()
                    return a

                analyzer, output = capture_and_run(run)

                import shutil
                shutil.rmtree(tmp_dir)

                st.session_state.multi_analyzer = analyzer
                st.session_state.multi_output = output
                st.rerun()

    elif files and len(files) == 1:
        st.warning("多文件分析需要至少2个文件")

    display_multi_results()


# ==================== 数据库分析 ====================
def database_mode():
    st.markdown("### 🗄️ 数据库分析")

    with st.form("db_form"):
        c1, c2 = st.columns(2)
        with c1:
            server = st.text_input("服务器地址", key="db_server")
            database = st.text_input("数据库名称", key="db_name")
        with c2:
            username = st.text_input("用户名", placeholder="sa", key="db_user")
            password = st.text_input("密码", type="password", key="db_pwd")
            trusted = st.checkbox("Windows身份认证", value=False, key="db_trusted")

        tables_input = st.text_area("表名列表（每行一个）", placeholder="users\norders", key="db_tables")
        rel_input = st.text_area("关系定义（可选）", placeholder="orders.user_id = users.user_id", key="db_rels")
        limit = st.number_input("最大加载行数", min_value=100, max_value=100000, value=5000, key="db_limit")

        submitted = st.form_submit_button("🔌 连接并分析", type="primary")

    if submitted:
        if not server or not database:
            st.error("请填写服务器和数据库名称")
            return
        if not trusted and (not username or not password):
            st.error("请填写用户名密码或使用Windows认证")
            return
        if not tables_input.strip():
            st.error("请至少输入一个表名")
            return

        table_names = [t.strip() for t in tables_input.strip().split('\n') if t.strip()]

        relationships = []
        if rel_input.strip():
            for line in rel_input.strip().split('\n'):
                if '=' in line:
                    parts = line.split('=')
                    if len(parts) == 2:
                        from_part, to_part = parts[0].strip(), parts[1].strip()
                        if '.' in from_part and '.' in to_part:
                            ft, fc = from_part.split('.')
                            tt, tc = to_part.split('.')
                            relationships.append({'from_table': ft, 'from_col': fc, 'to_table': tt, 'to_col': tc})

        with st.spinner("连接并分析..."):
            try:
                tables = DataLoader.load_multiple_tables(
                    server=server, database=database,
                    table_names=table_names,
                    username=username if not trusted else None,
                    password=password if not trusted else None,
                    trusted_connection=trusted,
                    limit=limit,
                    relationships=relationships if relationships else None
                )

                success = {n: df for n, df in tables.items() if df is not None and len(df) > 0}

                if success:
                    st.success(f"成功加载 {len(success)} 个表")

                    def run():
                        if relationships:
                            a = MultiTableStatisticalAnalyzer(success, relationships={'foreign_keys': relationships})
                        else:
                            a = MultiTableStatisticalAnalyzer(success)
                        a.analyze_all_tables()
                        return a

                    analyzer, output = capture_and_run(run)

                    st.session_state.db_analyzer = analyzer
                    st.session_state.db_output = output
                    st.rerun()
                else:
                    st.error("没有成功加载任何表")

            except Exception as e:
                st.error(f"连接失败: {str(e)}")

    display_db_results()


# ==================== 关于 ====================
st.sidebar.markdown("---")
if st.sidebar.button("📖 关于"):
    with st.expander("📖 关于 AutoStat", expanded=True):
        st.markdown("""
        **AutoStat** 智能统计分析工具
        
        - 自动识别数据类型
        - 数据质量体检
        - 智能统计方法选择
        - 多表关联分析
        - 时间序列分析
        - 多种输出格式 (HTML/JSON/Markdown/Excel)
        
        版本: 0.1.0 | 许可证: MIT
        """)


# ==================== 主入口 ====================
if analysis_mode == "📁 单文件分析":
    single_file_mode()
elif analysis_mode == "📚 多文件分析":
    multi_file_mode()
elif analysis_mode == "🗄️ 数据库分析":
    database_mode()