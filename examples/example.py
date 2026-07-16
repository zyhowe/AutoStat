"""
使用示例
"""

import pandas as pd
import numpy as np
import sys
import os
#import time
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autostat import AutoStatisticalAnalyzer, MultiTableStatisticalAnalyzer
from autostat.reporter import Reporter
from autostat.loader import DataLoader


def generate_sample_data():
    """生成示例数据"""
    np.random.seed(42)
    n = 300

    data = pd.DataFrame({
        '产品类别': np.random.choice(['A', 'B', 'C'], n),
        '销售额': np.random.normal(100, 15, n),
        '销售日期': pd.date_range('2023-01-01', periods=n, freq='D'),
        '订单编号': range(1, n + 1),
        '客户等级': np.random.choice(['低', '中', '高'], n)
    })
    return data


def generate_sample_data2():
    """生成具有自相关性的示例数据"""
    np.random.seed(42)
    n = 300

    phi = 0.7
    sigma = 8

    sales = np.zeros(n)
    sales[0] = np.random.normal(100, 15)

    for t in range(1, n):
        c = 100 * (1 - phi)
        sales[t] = c + phi * sales[t - 1] + np.random.normal(0, sigma)

    sales = np.maximum(sales, 20)
    sales = sales.round(2)

    month_effects = {
        1: 5, 2: 8, 3: 3, 4: -2, 5: -3, 6: -4,
        7: -5, 8: -4, 9: 0, 10: 2, 11: 4, 12: 6
    }

    start_date = pd.Timestamp('2023-01-01')
    dates = [start_date + pd.Timedelta(days=i) for i in range(n)]

    sales_with_seasonality = sales.copy()
    for i, date in enumerate(dates):
        month = date.month
        sales_with_seasonality[i] += month_effects.get(month, 0)

    sales_with_seasonality = np.maximum(sales_with_seasonality, 20)
    sales_with_seasonality = sales_with_seasonality.round(2)

    data = pd.DataFrame({
        '产品类别': np.random.choice(['A', 'B', 'C'], n),
        '销售额': sales_with_seasonality,
        '销售日期': dates,
        '订单编号': range(1, n + 1),
        '客户等级': np.random.choice(['低', '中', '高'], n)
    })

    level_adjustments = {'低': -5, '中': 0, '高': 8}
    for level, adjustment in level_adjustments.items():
        mask = data['客户等级'] == level
        data.loc[mask, '销售额'] = data.loc[mask, '销售额'] + adjustment

    data['销售额'] = np.maximum(data['销售额'], 20).round(2)

    from statsmodels.tsa.stattools import acf
    acf_values = acf(data['销售额'], nlags=20, fft=False)
    print(f"\n📊 生成的自相关序列统计:")
    print(f"  - 序列长度: {n}")
    print(f"  - 均值: {data['销售额'].mean():.2f}")
    print(f"  - 标准差: {data['销售额'].std():.2f}")
    print(f"  - 一阶自相关系数: {acf_values[1]:.4f}")

    monthly_avg = data.groupby(data['销售日期'].dt.month)['销售额'].mean()
    print(f"\n📅 月度平均销售额:")
    for month, avg in monthly_avg.items():
        print(f"  {month}月: {avg:.2f}")

    level_avg = data.groupby('客户等级')['销售额'].mean()
    print(f"\n👥 客户等级平均销售额:")
    for level, avg in level_avg.items():
        print(f"  {level}: {avg:.2f}")

    return data


def example_single_table():
    """单表分析示例"""
    print("=" * 60)
    print("示例1: 单表分析")
    print("=" * 60)

    data = generate_sample_data2()
    print(f"数据形状: {data.shape}")
    print(data.head())

    analyzer = AutoStatisticalAnalyzer(data, auto_clean=False, quiet=False)
    analyzer.generate_full_report()


def example_html_report():
    """生成HTML报告示例"""
    print("=" * 60)
    print("示例2: 生成HTML报告")
    print("=" * 60)

    data = generate_sample_data()
    analyzer = AutoStatisticalAnalyzer(data, quiet=True)
    reporter = Reporter(analyzer)
    reporter.to_html("sample_report.html")
    print("✅ 报告已生成: sample_report.html")


def example_json_output():
    """生成JSON输出示例"""
    print("=" * 60)
    print("示例3: 生成JSON输出")
    print("=" * 60)

    data = generate_sample_data()
    analyzer = AutoStatisticalAnalyzer(data, quiet=True)
    reporter = Reporter(analyzer)
    reporter.to_json("sample_result.json")
    print("✅ JSON已生成: sample_result.json")


def generate_multi_table_sample_data():
    """生成多表示例数据"""
    np.random.seed(42)
    n_patients = 500
    n_visits = 1200
    n_meds = 800

    patients = pd.DataFrame({
        'patient_id': range(1, n_patients + 1),
        '性别': np.random.choice(['男', '女'], n_patients, p=[0.48, 0.52]),
        '年龄': np.random.normal(45, 15, n_patients).round().astype(int),
        '身高': np.random.normal(165, 8, n_patients).round(1),
        '体重': np.random.normal(65, 12, n_patients).round(1),
        '吸烟史': np.random.choice(['无', '有', '已戒烟'], n_patients, p=[0.6, 0.3, 0.1])
    })

    visits = pd.DataFrame({
        'visit_id': range(1, n_visits + 1),
        'patient_id': np.random.choice(patients['patient_id'], n_visits),
        '就诊日期': pd.date_range('2020-01-01', periods=n_visits, freq='D'),
        '收缩压': np.random.normal(125, 15, n_visits).round(1),
        '舒张压': np.random.normal(80, 10, n_visits).round(1),
        '心率': np.random.normal(75, 10, n_visits).round().astype(int),
        '就诊类型': np.random.choice(['门诊', '急诊', '住院'], n_visits, p=[0.7, 0.2, 0.1])
    })

    medications = pd.DataFrame({
        'med_id': range(1, n_meds + 1),
        'visit_id': np.random.choice(visits['visit_id'], n_meds),
        '药品名称': np.random.choice(['阿司匹林', '布洛芬', '对乙酰氨基酚', '阿莫西林', '头孢'], n_meds),
        '剂量': np.random.choice(['100mg', '200mg', '500mg', '1g'], n_meds),
        '天数': np.random.poisson(5, n_meds)
    })

    hospitals = pd.DataFrame({
        'hospital_id': [1, 2, 3],
        '医院名称': ['市一医院', '中心医院', '妇幼保健院'],
        '床位数': [800, 1200, 500],
        '等级': ['三甲', '三甲', '二甲']
    })

    return {
        'patients': patients,
        'visits': visits,
        'medications': medications,
        'hospitals': hospitals
    }


def example_multi_table():
    """多表分析示例"""
    print("=" * 60)
    print("示例4: 多表分析（使用生成的数据）")
    print("=" * 60)

    tables = generate_multi_table_sample_data()

    print("加载的表:")
    for name, df in tables.items():
        print(f"  - {name}: {df.shape}")

    analyzer = MultiTableStatisticalAnalyzer(tables)
    analyzer.analyze_all_tables()


def example_multi_table_from_files():
    """多表分析示例 - 从文件加载"""
    print("=" * 60)
    print("示例5: 多表分析（从文件加载）")
    print("=" * 60)

    # 先生成示例数据并保存
    tables_data = generate_multi_table_sample_data()

    # 保存到临时CSV文件
    temp_dir = "temp_multi_data"
    os.makedirs(temp_dir, exist_ok=True)

    file_paths = {}
    for name, df in tables_data.items():
        file_path = os.path.join(temp_dir, f"{name}.csv")
        df.to_csv(file_path, index=False)
        file_paths[name] = file_path
        print(f"  ✅ 已保存: {file_path}")

    # # 使用 MultiTableStatisticalAnalyzer.from_files 加载
    # analyzer = MultiTableStatisticalAnalyzer.from_files(file_paths)
    # analyzer.analyze_all_tables()
    #
    # # 清理临时文件
    # import shutil
    # shutil.rmtree(temp_dir)
    # print(f"✅ 临时文件已清理")


def example_multi_table_from_json_strings():
    """多表分析示例 - 从JSON字符串加载"""
    print("=" * 60)
    print("示例6: 多表分析（从JSON字符串加载）")
    print("=" * 60)

    # 生成示例数据
    tables_data = generate_multi_table_sample_data()

    # 转换为JSON字符串
    json_strings = {}
    for name, df in tables_data.items():
        json_strings[name] = df.to_json(orient='records', force_ascii=False, date_format='iso')
        print(f"  ✅ 已转换: {name} -> JSON字符串 ({len(json_strings[name])} 字符)")

    # 使用 MultiTableStatisticalAnalyzer.from_json_strings 加载
    analyzer = MultiTableStatisticalAnalyzer.from_json_strings(json_strings)
    analyzer.analyze_all_tables()


def example_database():
    """数据库分析示例"""
    print("=" * 60)
    print("示例7: 数据库分析（SQL Server）")
    print("=" * 60)
    print("⚠️ 此示例需要配置数据库连接信息")
    print("请修改以下配置后取消注释运行:")
    print("""
    # 数据库配置示例
    DB_CONFIG = {
        'server': 'your_server',
        'database': 'your_database',
        'username': 'your_username',
        'password': 'your_password',
        'trusted_connection': False
    }
    
    TABLE_NAMES = ['table1', 'table2']
    RELATIONSHIPS = [
        {'from_table': 'table2', 'from_col': 'fk_id', 
         'to_table': 'table1', 'to_col': 'id'}
    ]
    
    tables = DataLoader.load_multiple_tables(
        server=DB_CONFIG['server'],
        database=DB_CONFIG['database'],
        table_names=TABLE_NAMES,
        username=DB_CONFIG['username'],
        password=DB_CONFIG['password'],
        trusted_connection=DB_CONFIG['trusted_connection'],
        relationships=RELATIONSHIPS,
        limit=10000
    )
    
    analyzer = MultiTableStatisticalAnalyzer(tables)
    analyzer.analyze_all_tables()
    """)

    # 实际使用示例（注释状态）

    DB_CONFIG = {
        'server': '10.17.207.163',
        'database': 'FCDB',
        'username': 'sa',
        'password': 'Finchina#2014',
        'trusted_connection': False
    }
    
    TABLE_NAMES = ['patients',
        'visits',
        'medications',
        'hospitals']
    RELATIONSHIPS = [
        {'from_table': 'visits', 'from_col': 'patient_id',
         'to_table': 'patients', 'to_col': 'patient_id', 'type': 'many_to_one'},
        {'from_table': 'medications', 'from_col': 'visit_id',
         'to_table': 'visits', 'to_col': 'visit_id', 'type': 'many_to_one'}
    ]
    
    tables = DataLoader.load_multiple_tables(
        server=DB_CONFIG['server'],
        database=DB_CONFIG['database'],
        table_names=TABLE_NAMES,
        username=DB_CONFIG['username'],
        password=DB_CONFIG['password'],
        trusted_connection=DB_CONFIG['trusted_connection'],
        relationships=RELATIONSHIPS,
        limit=5000
    )
    
    if tables:
        analyzer = MultiTableStatisticalAnalyzer(tables)
        analyzer.analyze_all_tables()


def example_database_single_table():
    """数据库分析示例"""
    print("=" * 60)
    print("示例7: 数据库分析（SQL Server）")
    print("=" * 60)
    print("⚠️ 此示例需要配置数据库连接信息")
    print("请修改以下配置后取消注释运行:")


    # 实际使用示例（注释状态）

    DB_CONFIG = {
        'server': '10.17.207.163',
        'database': 'FCDB',
        'username': 'sa',
        'password': 'Finchina#2014',
        'trusted_connection': False
    }

    TABLE_NAMES = ['Bondip'] #CompanyFixAsset']
    RELATIONSHIPS = [
    ]

    tables = DataLoader.load_multiple_tables(
        server=DB_CONFIG['server'],
        database=DB_CONFIG['database'],
        table_names=TABLE_NAMES,
        username=DB_CONFIG['username'],
        password=DB_CONFIG['password'],
        trusted_connection=DB_CONFIG['trusted_connection'],
        relationships=RELATIONSHIPS,
        limit=5000
    )

    if tables:
        analyzer = MultiTableStatisticalAnalyzer(tables)
        analyzer.analyze_all_tables()
        # reporter = Reporter(analyzer)
        # reporter.to_html("results\\database_single_table_report.html")


def example_single_table_from_file():
    """单表分析示例 - 从文件加载CSV"""
    print("=" * 60)
    print("示例8: 单表分析（从CSV文件加载）")
    print("=" * 60)

    # 生成示例数据并保存
    data = generate_sample_data2()
    csv_path = "sample_data.csv"
    data.to_csv(csv_path, index=False)
    print(f"✅ 数据已保存: {csv_path}")

    # 从文件加载并分析
    analyzer = AutoStatisticalAnalyzer(csv_path, auto_clean=False, quiet=False)
    analyzer.generate_full_report()

    # 清理临时文件
    os.remove(csv_path)
    print(f"✅ 临时文件已清理: {csv_path}")


def example_single_table_from_excel():
    """单表分析示例 - 从Excel文件加载"""
    print("=" * 60)
    print("示例9: 单表分析（从Excel文件加载）")
    print("=" * 60)

    data = generate_sample_data()
    excel_path = "sample_data.xlsx"
    data.to_excel(excel_path, index=False)
    print(f"✅ 数据已保存: {excel_path}")

    analyzer = AutoStatisticalAnalyzer(excel_path, auto_clean=False, quiet=False)
    analyzer.generate_full_report()

    os.remove(excel_path)
    print(f"✅ 临时文件已清理: {excel_path}")


def example_cli_equivalent():
    """命令行等效示例"""
    print("=" * 60)
    print("示例10: 命令行等效操作")
    print("=" * 60)

    data = generate_sample_data()
    data.to_csv("sample_data.csv", index=False)
    print("✅ 示例数据已保存: sample_data.csv")
    print("")
    print("命令行执行:")
    print("  autostat sample_data.csv -o report.html")
    print("  autostat sample_data.csv -f json -o result.json")
    print("  autostat sample_data.csv -f md -o report.md")
    print("  autostat sample_data.csv -f excel -o report.xlsx")
    print("  autostat sample_data.csv --quiet")
    print("  autostat sample_data.csv --auto-clean")


# ==================== 新增：日期关系发现测试 ====================

def example_date_rule_discovery():
    """测试日期关系发现模块（使用中国节假日库）"""
    print("=" * 60)
    print("示例11: 日期关系发现测试（使用中国节假日库）")
    print("=" * 60)

    from datetime import timedelta
    import pandas as pd
    import numpy as np

    try:
        from autostat.core.date_rules import discover_date_rules
    except ImportError:
        print("⚠️ 未找到 date_rules 模块")
        return

    # 更精确的工作日计算函数
    def is_workday(d):
        """判断是否为工作日（周一至周五）"""
        return d.weekday() < 5

    def add_workdays(start_date, days):
        """精确添加工作日（跳过周末）"""
        result = start_date
        added = 0
        while added < days:
            result += timedelta(days=1)
            if is_workday(result):
                added += 1
        return result

    # 生成连续的工作日序列
    start_date = pd.Timestamp('2024-01-01')
    workday_sequence = []
    current = start_date
    while len(workday_sequence) < 100:
        if is_workday(current):
            workday_sequence.append(current)
        current += timedelta(days=1)

    # 创建 DataFrame
    n = len(workday_sequence)

    # 使用精确的工作日间隔
    order_dates = workday_sequence
    payment_dates = [add_workdays(d, 2) for d in order_dates]  # 固定2个工作日
    ship_dates = [add_workdays(d, 1) for d in payment_dates]  # 固定1个工作日
    complete_dates = [add_workdays(d, 2) for d in ship_dates]  # 固定2个工作日

    # 验证没有跨周末问题：确保所有日期都是工作日
    for i, d in enumerate(payment_dates):
        if not is_workday(d):
            print(f"警告: payment_date[{i}] = {d} 不是工作日")

    categories = ['A'] * (n // 3) + ['B'] * (n // 3) + ['C'] * (n - 2 * (n // 3))

    df = pd.DataFrame({
        'order_date': order_dates,
        'payment_date': payment_dates,
        'ship_date': ship_dates,
        'complete_date': complete_dates,
        'category': categories
    })

    print(f"数据形状: {df.shape}")
    print("\n测试数据预览（前10行）:")
    print(df.head(10))

    # 验证工作日间隔
    print("\n验证工作日间隔计算:")
    for i in range(5):
        order = df.loc[i, 'order_date']
        payment = df.loc[i, 'payment_date']
        # 手动计算工作日间隔
        cnt = 0
        d = order
        while d < payment:
            d += timedelta(days=1)
            if is_workday(d):
                cnt += 1
        print(f"  {order.strftime('%Y-%m-%d')} → {payment.strftime('%Y-%m-%d')}: 工作日间隔={cnt}")

    date_columns = ['order_date', 'payment_date', 'ship_date', 'complete_date']

    print("\n开始发现日期关系...")

    rules = discover_date_rules(
        df,
        date_columns=date_columns,
        categorical_columns=['category'],
        debug=True,
        min_confidence=0.8,
        min_nonnull=5,
        use_chinese_calendar=False,  # 暂时不使用，避免干扰
        consider_workday=True,
        consider_shifted=False,  # 暂时关闭顺延关系
        consider_conditional=True
    )

    print("\n" + "=" * 60)
    print(f"发现 {len(rules)} 条日期关系规则:")
    print("=" * 60)

    # 分类输出
    workday_rules = [r for r in rules if '个工作日' in r['rule']]
    basic_rules = [r for r in rules if '个工作日' not in r['rule'] and '当' not in r['rule']]
    cond_rules = [r for r in rules if '当' in r['rule']]

    if workday_rules:
        print("\n✅ 工作日间隔规则:")
        for r in workday_rules:
            print(f"  {r['rule']} (置信度: {r['confidence']})")
    else:
        print("\n❌ 未发现工作日间隔规则")

    if basic_rules:
        print("\n基本时序规则:")
        for r in basic_rules[:10]:
            print(f"  {r['rule']}")

    # 验证已知业务关系
    print("\n" + "=" * 60)
    print("验证已知业务关系:")
    expected_relations = [
        ("payment_date = order_date + 2个工作日", "工作日间隔"),
        ("ship_date = payment_date + 1个工作日", "工作日间隔"),
        ("complete_date = ship_date + 2个工作日", "工作日间隔"),
    ]

    found_rules_str = [r['rule'] for r in rules]
    for expected, rel_type in expected_relations:
        if any(expected in s for s in found_rules_str):
            print(f"  ✅ {expected}")
        else:
            print(f"  ❌ {expected}")
            # 打印相似规则帮助调试
            similar = [s for s in found_rules_str if s.split('=')[0].strip() == expected.split('=')[0].strip()]
            if similar:
                print(f"     实际发现: {similar}")


def example_date_rule_with_precise_data():
    """使用精确数据测试日期关系发现（无跨周末问题）"""
    print("=" * 60)
    print("示例12: 精确数据测试（无跨周末）")
    print("=" * 60)

    from datetime import timedelta

    try:
        from autostat.core.date_rules import discover_date_rules
    except ImportError:
        print("⚠️ 未找到 date_rules 模块")
        return

    # 创建简单的测试数据，确保没有跨周末问题
    # 使用连续的日期，确保所有日期都是工作日

    def is_workday(d):
        return d.weekday() < 5

    # 生成工作日序列
    start = pd.Timestamp('2024-01-01')
    workdays = []
    current = start
    while len(workdays) < 50:
        if is_workday(current):
            workdays.append(current)
        current += timedelta(days=1)

    # 创建测试数据（所有日期都是工作日，没有跨周末）
    df = pd.DataFrame({
        'date1': workdays[:30],
        'date2': workdays[2:32],  # +2个工作日
        'date3': workdays[3:33],  # +3个工作日
        'category': ['X'] * 30
    })

    print("测试数据:")
    print(df.head(10))

    # 手动验证工作日间隔
    print("\n验证工作日间隔:")
    for i in range(5):
        d1 = df.loc[i, 'date1']
        d2 = df.loc[i, 'date2']
        # 计算工作日间隔
        cnt = 0
        d = d1
        while d < d2:
            d += timedelta(days=1)
            if is_workday(d):
                cnt += 1
        print(f"  {d1.strftime('%Y-%m-%d')} → {d2.strftime('%Y-%m-%d')}: 工作日间隔={cnt}")

    rules = discover_date_rules(
        df,
        date_columns=['date1', 'date2', 'date3'],
        categorical_columns=['category'],
        debug=True,
        min_confidence=0.8,
        min_nonnull=5,
        use_chinese_calendar=False,
        consider_workday=True,
        consider_shifted=False,
        consider_conditional=False
    )

    print("\n" + "=" * 60)
    print("发现规则:")
    for r in rules:
        print(f"  {r['rule']} (置信度: {r['confidence']})")


def example_date_rule_discovery_with_anomalies():
    """测试日期关系发现（不同分类不同规则 + 异常数据）"""
    print("=" * 60)
    print("示例13: 不同分类不同规则 + 异常数据测试")
    print("=" * 60)

    from datetime import timedelta
    import pandas as pd
    import random

    try:
        from autostat.core.date_rules import discover_date_rules
    except ImportError:
        print("⚠️ 未找到 date_rules 模块")
        return

    # 工作日判断函数
    def is_workday(d):
        return d.weekday() < 5

    def add_workdays(start_date, days):
        """添加工作日（正确处理顺序）"""
        if days <= 0:
            return start_date
        result = start_date
        added = 0
        while added < days:
            result += timedelta(days=1)
            if is_workday(result):
                added += 1
        return result

    # 生成连续的工作日序列
    start_date = pd.Timestamp('2024-01-01')
    workday_sequence = []
    current = start_date
    while len(workday_sequence) < 150:
        if is_workday(current):
            workday_sequence.append(current)
        current += timedelta(days=1)

    n = len(workday_sequence)

    # 设置不同分类的规则（不同的工作日间隔）
    # 分类A: 快速通道 - 付款1个工作日，发货1个工作日
    # 分类B: 标准通道 - 付款2个工作日，发货1个工作日
    # 分类C: 慢速通道 - 付款3个工作日，发货2个工作日

    categories = []
    order_dates = []
    payment_dates = []
    ship_dates = []
    complete_dates = []

    # 统计异常注入
    anomaly_count = {'A': 0, 'B': 0, 'C': 0}

    for i, d in enumerate(workday_sequence):
        # 分配分类（确保每个分类约50个样本）
        if i < 50:
            cat = 'A'  # 快速
            payment_gap = 1
            ship_gap = 1
            complete_gap = 2
        elif i < 100:
            cat = 'B'  # 标准
            payment_gap = 2
            ship_gap = 1
            complete_gap = 2
        else:
            cat = 'C'  # 慢速
            payment_gap = 3
            ship_gap = 2
            complete_gap = 2

        order = d

        # 决定是否注入异常（10%异常率）
        is_anomaly = random.random() < 0.1

        if is_anomaly and anomaly_count[cat] < 5:  # 每个分类最多5个异常
            # 随机选择不同的间隔
            wrong_gap = random.choice([g for g in [1, 2, 3, 4, 5] if g != payment_gap])
            payment = add_workdays(order, wrong_gap)
            anomaly_count[cat] += 1
            if anomaly_count[cat] == 1:
                print(f"  注入异常: 分类={cat}, 期望付款间隔={payment_gap}, 实际间隔={wrong_gap}")
        else:
            payment = add_workdays(order, payment_gap)

        ship = add_workdays(payment, ship_gap)
        complete = add_workdays(ship, complete_gap)

        categories.append(cat)
        order_dates.append(order)
        payment_dates.append(payment)
        ship_dates.append(ship)
        complete_dates.append(complete)

    # 创建DataFrame
    df = pd.DataFrame({
        'order_date': order_dates,
        'payment_date': payment_dates,
        'ship_date': ship_dates,
        'complete_date': complete_dates,
        'category': categories
    })

    print(f"\n数据形状: {df.shape}")
    print(f"分类分布:\n{df['category'].value_counts()}")
    print(f"异常注入统计: {anomaly_count}")

    print("\n测试数据预览（前15行）:")
    print(df.head(15))

    # 验证各分类的实际工作日间隔
    print("\n验证各分类的实际工作日间隔:")
    for cat in ['A', 'B', 'C']:
        subset = df[df['category'] == cat]
        print(f"\n【分类 {cat}】样本数: {len(subset)}")

        # 计算间隔分布
        gaps = []
        for _, row in subset.iterrows():
            order = row['order_date']
            payment = row['payment_date']
            cnt = 0
            d = order
            while d < payment:
                d += timedelta(days=1)
                if is_workday(d):
                    cnt += 1
            gaps.append(cnt)

        from collections import Counter
        gap_dist = Counter(gaps)
        print(f"  付款间隔分布: {dict(sorted(gap_dist.items()))}")

    # 开始发现日期关系
    date_columns = ['order_date', 'payment_date', 'ship_date', 'complete_date']

    print("\n" + "=" * 60)
    print("开始发现日期关系（不同分类不同规则 + 异常数据）")
    print("=" * 60)

    rules = discover_date_rules(
        df,
        date_columns=date_columns,
        categorical_columns=['category'],
        debug=True,
        min_confidence=0.65,  # 降低阈值以容忍异常数据
        min_nonnull=5,
        use_chinese_calendar=False,
        consider_workday=True,
        consider_shifted=False,
        consider_conditional=True
    )

    print("\n" + "=" * 60)
    print(f"发现 {len(rules)} 条日期关系规则:")
    print("=" * 60)

    # 分类输出
    unconditional_rules = [r for r in rules if '当' not in r['rule']]
    conditional_rules = [r for r in rules if '当' in r['rule']]

    print(f"\n无条件规则: {len(unconditional_rules)} 条")
    for r in unconditional_rules:
        if '个工作日' in r['rule']:
            print(f"  ✅ {r['rule']} (置信度: {r['confidence']})")
        else:
            print(f"  {r['rule']} (置信度: {r['confidence']})")

    # 按分类分组显示条件规则
    print(f"\n条件规则: {len(conditional_rules)} 条")
    for cat in ['A', 'B', 'C']:
        cat_rules = [r for r in conditional_rules if f"category = {cat}" in r['rule']]
        if cat_rules:
            print(f"\n  【分类 {cat}】:")
            for r in cat_rules:
                relation = r['rule'].split('时，')[1]
                if '个工作日' in relation:
                    print(f"    ✅ {relation} (置信度: {r['confidence']})")
                else:
                    print(f"    {relation} (置信度: {r['confidence']})")

    # 验证已知业务关系
    print("\n" + "=" * 60)
    print("验证已知业务关系:")

    expected_by_category = {
        'A': [
            ('payment_date = order_date + 1个工作日', '付款间隔1天'),
            ('ship_date = payment_date + 1个工作日', '发货间隔1天'),
        ],
        'B': [
            ('payment_date = order_date + 2个工作日', '付款间隔2天'),
            ('ship_date = payment_date + 1个工作日', '发货间隔1天'),
        ],
        'C': [
            ('payment_date = order_date + 3个工作日', '付款间隔3天'),
            ('ship_date = payment_date + 2个工作日', '发货间隔2天'),
        ]
    }

    found_rules_str = [r['rule'] for r in rules]

    for cat, expectations in expected_by_category.items():
        print(f"\n【分类 {cat}】:")
        for expected, desc in expectations:
            found = False
            for rule in found_rules_str:
                if f"category = {cat} 时，{expected}" in rule:
                    found = True
                    break
            if found:
                print(f"  ✅ {desc}: {expected}")
            else:
                print(f"  ❌ {desc}: {expected} 未发现")
                # 查找相似规则
                similar = [s for s in found_rules_str if f"category = {cat}" in s and '个工作日' in s]
                if similar:
                    print(f"     实际发现: {similar}")


def example_date_rule_robustness_test():
    """鲁棒性测试：高比例异常数据"""
    print("=" * 60)
    print("示例14: 鲁棒性测试（高比例异常数据）")
    print("=" * 60)

    from datetime import timedelta
    import pandas as pd
    import random

    try:
        from autostat.core.date_rules import discover_date_rules
    except ImportError:
        print("⚠️ 未找到 date_rules 模块")
        return

    def is_workday(d):
        return d.weekday() < 5

    def add_workdays(start_date, days):
        if days <= 0:
            return start_date
        result = start_date
        added = 0
        while added < days:
            result += timedelta(days=1)
            if is_workday(result):
                added += 1
        return result

    # 生成数据
    start_date = pd.Timestamp('2024-01-01')
    workday_sequence = []
    current = start_date
    while len(workday_sequence) < 200:
        if is_workday(current):
            workday_sequence.append(current)
        current += timedelta(days=1)

    n = len(workday_sequence)

    # 正常规则：payment = order + 2个工作日
    order_dates = workday_sequence
    payment_dates = []

    anomaly_count = 0
    for i, d in enumerate(workday_sequence):
        if i < n * 0.15:  # 15%异常数据
            # 随机间隔（排除2）
            wrong_gap = random.choice([1, 3, 4, 5])
            payment_dates.append(add_workdays(d, wrong_gap))
            anomaly_count += 1
        else:
            payment_dates.append(add_workdays(d, 2))

    categories = ['normal'] * n

    df = pd.DataFrame({
        'order_date': order_dates,
        'payment_date': payment_dates,
        'category': categories
    })

    print(f"数据形状: {df.shape}")
    print(f"异常数据比例: {anomaly_count}/{n} = {anomaly_count / n * 100:.1f}%")

    # 计算实际工作日间隔分布
    gaps = []
    for i in range(n):
        order = df.loc[i, 'order_date']
        payment = df.loc[i, 'payment_date']
        cnt = 0
        d = order
        while d < payment:
            d += timedelta(days=1)
            if is_workday(d):
                cnt += 1
        gaps.append(cnt)

    from collections import Counter
    gap_dist = Counter(gaps)
    print(f"\n工作日间隔分布: {dict(sorted(gap_dist.items()))}")

    # 主要规则的比例
    main_rule_ratio = gap_dist.get(2, 0) / n * 100
    print(f"主要规则(间隔=2)占比: {main_rule_ratio:.1f}%")

    rules = discover_date_rules(
        df,
        date_columns=['order_date', 'payment_date'],
        categorical_columns=None,  # 不使用分类
        debug=True,
        min_confidence=0.6,
        min_nonnull=5,
        use_chinese_calendar=False,
        consider_workday=True,
        consider_shifted=False,
        consider_conditional=False
    )

    print("\n" + "=" * 60)
    print("发现规则:")
    found = False
    for r in rules:
        if '个工作日' in r['rule']:
            print(f"  ✅ {r['rule']} (置信度: {r['confidence']})")
            found = True
        else:
            print(f"  {r['rule']} (置信度: {r['confidence']})")

    if found:
        print("\n✅ 成功发现核心规则: payment_date = order_date + 2个工作日")
    else:
        print("\n❌ 未发现核心规则")
        print("   提示: 主要规则占比需要高于 min_confidence 阈值")
        print(f"   当前主要规则占比: {main_rule_ratio:.1f}%, 阈值: 60%")





if __name__ == "__main__":
    # # 测试精确数据
    # example_date_rule_with_precise_data()
    #
    # print("\n" + "=" * 60 + "\n")
    #
    # # 测试原始数据
    # example_date_rule_discovery()

    # # 测试1: 不同分类不同规则 + 异常数据
    # example_date_rule_discovery_with_anomalies()
    #
    # print("\n" + "=" * 100 + "\n")
    #
    # # 测试2: 鲁棒性测试
    # example_date_rule_robustness_test()

    # 单表分析
    #example_single_table()

    # 生成HTML报告
    # example_html_report()

    # 生成JSON输出
    # example_json_output()

    # 多表分析（使用生成的数据）
    #example_multi_table()

    # 多表分析（从文件加载）
    example_multi_table_from_files()

    # 多表分析（从JSON字符串加载）
    # example_multi_table_from_json_strings()

    # 数据库分析（需要配置）
    #example_database()

    #example_database_single_table()

    # 单表分析（从CSV文件加载）
    # example_single_table_from_file()

    # 单表分析（从Excel文件加载）
    # example_single_table_from_excel()

    # 命令行等效操作
    # example_cli_equivalent()