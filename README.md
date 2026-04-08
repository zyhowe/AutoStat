# AutoStat - Intelligent Statistical Analysis Tool

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

AutoStat is an intelligent statistical analysis tool that automatically identifies data types, detects data quality issues, selects appropriate statistical methods, and generates professional analysis reports. No statistical background required.

## Features

- Automatic Data Type Recognition - Identifies continuous, categorical, datetime, identifier, and ordinal variables
- Data Quality Inspection - Detects missing values, outliers, duplicates, and inconsistent types
- Smart Statistical Method Selection - Automatically chooses between t-test, ANOVA, Mann-Whitney, chi-square, Fisher's exact test
- Multi-table Association Analysis - Automatically discovers relationships between tables and performs joint analysis
- Time Series Analysis - Stationarity test (ADF), autocorrelation test (Ljung-Box), seasonality detection
- Relationship Analysis - Correlation matrix, Cramer's V, Eta-squared with visual heatmaps
- Intelligent Sampling - Foreign-key-aware stratified sampling for large datasets
- Multiple Output Formats - HTML reports, JSON data, MCP service for AI agents
- Multiple Data Sources - CSV, Excel, JSON, TXT, SQL Server

## Quick Start

### Installation

pip install autostat

Or install from source:

git clone https://github.com/zyhowe/AutoStat.git
cd autostat
pip install -e .

### Command Line Usage

autostat data.csv -o report.html
autostat data.csv -f json -o result.json
autostat data.csv --quiet

### Python API Usage

from autostat import AutoStatisticalAnalyzer

analyzer = AutoStatisticalAnalyzer("sales_data.csv")
analyzer.generate_full_report()

from autostat import Reporter
reporter = Reporter(analyzer)
reporter.to_html("report.html")
reporter.to_json("result.json")

### Multi-table Analysis

from autostat import MultiTableStatisticalAnalyzer
import pandas as pd

tables = {
    "orders": pd.read_csv("orders.csv"),
    "users": pd.read_csv("users.csv"),
    "products": pd.read_csv("products.csv")
}

analyzer = MultiTableStatisticalAnalyzer(tables)
analyzer.analyze_all_tables()

### Web Interface

streamlit run web/app.py

### MCP Service

python -m autostat.mcp_server

## Output Example

### JSON Output Structure

{
  "analysis_time": "2024-01-01T12:00:00",
  "source_table": "sales_data",
  "data_shape": {"rows": 10000, "columns": 15},
  "variable_types": {
    "user_id": {"type": "identifier", "type_desc": "identifier"},
    "amount": {"type": "continuous", "type_desc": "continuous"},
    "city": {"type": "categorical", "type_desc": "categorical"}
  },
  "quality_report": {
    "missing": [{"column": "city", "count": 150, "percent": 1.5}],
    "duplicates": {"count": 23, "percent": 0.23}
  }
}

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scipy, statsmodels, networkx
- click, jinja2
- streamlit (optional)
- fastmcp (optional)
- pyodbc (optional)

## License

MIT License