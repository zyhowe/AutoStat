# AutoStat - 智能统计分析工具

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

AutoStat 是一款智能统计分析工具，能够自动识别数据类型、检测数据质量、选择合适的统计方法、生成专业的分析报告。无需统计学背景。

## 核心功能

- 自动数据类型识别 - 识别连续变量、分类变量、日期时间、标识符、有序分类变量
- 数据质量体检 - 检测缺失值、异常值、重复记录、类型不一致
- 智能统计方法选择 - 自动选择 t检验、ANOVA、Mann-Whitney、卡方检验、Fisher精确检验
- 多表关联分析 - 自动发现表间关系，进行联合分析
- 时间序列分析 - 平稳性检验(ADF)、自相关检验(Ljung-Box)、季节性检测
- 关系分析 - 相关性矩阵、Cramer's V、Eta-squared，带可视化热力图
- 智能采样 - 基于外键感知的分层采样，支持大数据集
- 多种输出格式 - HTML报告、JSON数据、MCP服务（供AI Agent调用）
- 多种数据源 - CSV、Excel、JSON、TXT、SQL Server

## 快速开始

### 安装

pip install autostat

从源码安装：

git clone https://github.com/zyhowe/AutoStat.git
cd autostat
pip install -e .

### 命令行使用

# 分析单个文件
autostat data.csv -o report.html

# 输出JSON格式
autostat data.csv -f json -o result.json

# 静默模式
autostat data.csv --quiet

### Python API使用

from autostat import AutoStatisticalAnalyzer

# 单表分析
analyzer = AutoStatisticalAnalyzer("sales_data.csv")
analyzer.generate_full_report()

# 生成HTML报告
from autostat import Reporter
reporter = Reporter(analyzer)
reporter.to_html("report.html")

# 生成JSON输出
reporter.to_json("result.json")

### 多表分析

from autostat import MultiTableStatisticalAnalyzer
import pandas as pd

# 加载表
tables = {
    "orders": pd.read_csv("orders.csv"),
    "users": pd.read_csv("users.csv"),
    "products": pd.read_csv("products.csv")
}

# 分析
analyzer = MultiTableStatisticalAnalyzer(tables)
analyzer.analyze_all_tables()

### Web界面

streamlit run web/app.py

### MCP服务（供AI Agent调用）

python -m autostat.mcp_server

## 输出示例

### HTML报告

HTML报告包含：
- 数据概览（行数、列数、缺失值、重复值）
- 变量类型分布
- 每个变量的统计摘要
- 相关性分析及热力图
- 时间序列分析结果
- 数据清洗建议

### JSON输出结构

{
  "analysis_time": "2024-01-01T12:00:00",
  "source_table": "sales_data",
  "data_shape": {"rows": 10000, "columns": 15},
  "variable_types": {
    "user_id": {"type": "identifier", "type_desc": "标识符列"},
    "amount": {"type": "continuous", "type_desc": "连续变量"},
    "city": {"type": "categorical", "type_desc": "分类变量"}
  },
  "quality_report": {
    "missing": [{"column": "city", "count": 150, "percent": 1.5}],
    "duplicates": {"count": 23, "percent": 0.23}
  },
  "cleaning_suggestions": ["建议处理缺失值"]
}

## 支持的数据源

| 格式 | 扩展名 | 方法 |
|------|--------|------|
| CSV | .csv | load_csv() |
| Excel | .xlsx, .xls | load_excel() |
| JSON | .json | load_json() |
| TXT | .txt | load_txt() |
| SQL Server | - | load_sql_server() |

## 环境要求

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scipy, statsmodels, networkx
- click, jinja2
- streamlit（可选，Web界面）
- fastmcp（可选，MCP服务）
- pyodbc（可选，SQL Server）

## 项目结构

autostat/
├── autostat/
│   ├── __init__.py
│   ├── analyzer.py      # 核心分析引擎
│   ├── loader.py        # 数据加载器
│   ├── multi_analyzer.py # 多表分析器
│   ├── checker.py       # 条件检查器
│   ├── reporter.py      # 报告生成器
│   ├── cli.py           # 命令行入口
│   └── mcp_server.py    # MCP服务
├── web/
│   └── app.py           # Streamlit Web界面
├── templates/
│   └── report.html      # HTML报告模板
├── tests/
│   └── test_analyzer.py # 单元测试
├── examples/
│   └── example.py       # 使用示例
├── setup.py
├── requirements.txt
├── README.md
├── README_cn.md
└── LICENSE

## 使用示例

查看 examples/ 目录获取完整示例：

from autostat import AutoStatisticalAnalyzer

# 基础分析
analyzer = AutoStatisticalAnalyzer("data.csv")
analyzer.auto_describe()

# 完整报告
analyzer.generate_full_report()

# 导出JSON
from autostat import Reporter
reporter = Reporter(analyzer)
reporter.to_json("output.json")

## 许可证

MIT License

## 贡献

欢迎贡献代码！请提交 Pull Request。

## 联系方式

- Issues: https://github.com/zyhowe/AutoStat/issues
- 邮箱: team@autostat.com