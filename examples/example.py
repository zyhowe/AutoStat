"""
使用示例
"""

import pandas as pd
import numpy as np
import sys
import os

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

    # 使用 MultiTableStatisticalAnalyzer.from_files 加载
    analyzer = MultiTableStatisticalAnalyzer.from_files(file_paths)
    analyzer.analyze_all_tables()

    # 清理临时文件
    import shutil
    shutil.rmtree(temp_dir)
    print(f"✅ 临时文件已清理")


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


if __name__ == "__main__":
    # 运行示例（根据需要取消注释）

    # 单表分析
    example_single_table()

    # 生成HTML报告
    # example_html_report()

    # 生成JSON输出
    # example_json_output()

    # 多表分析（使用生成的数据）
    # example_multi_table()

    # 多表分析（从文件加载）
    # example_multi_table_from_files()

    # 多表分析（从JSON字符串加载）
    # example_multi_table_from_json_strings()

    # 数据库分析（需要配置）
    # example_database()

    # 单表分析（从CSV文件加载）
    # example_single_table_from_file()

    # 单表分析（从Excel文件加载）
    # example_single_table_from_excel()

    # 命令行等效操作
    # example_cli_equivalent()