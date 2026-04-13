# AutoStat - Intelligent Statistical Analysis Tool

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/autostat-mcp.svg)](https://pypi.org/project/autostat-mcp/)

AutoStat is an intelligent statistical analysis tool that automatically identifies data types, detects data quality issues, selects appropriate statistical methods, and generates professional analysis reports. No statistical background required.

## Features

-   **Automatic Data Type Recognition** - Identifies continuous, categorical, datetime, identifier, and ordinal variables
-   **Data Quality Inspection** - Detects missing values, outliers, duplicates, and inconsistent types
-   **Smart Statistical Method Selection** - Automatically chooses between t-test, ANOVA, Mann-Whitney, chi-square, Fisher's exact test
-   **Multi-table Association Analysis** - Automatically discovers relationships between tables and performs joint analysis
-   **Time Series Analysis** - Stationarity test (ADF), autocorrelation test (Ljung-Box), seasonality detection
-   **Relationship Analysis** - Correlation matrix, Cramer's V, Eta-squared with visual heatmaps
-   **Intelligent Sampling** - Foreign-key-aware stratified sampling for large datasets
-   **Multiple Output Formats** - HTML reports, JSON, Markdown, Excel
-   **Multiple Data Sources** - CSV, Excel, JSON, TXT, SQL Server
-   **MCP Service** - Can be called by AI agents (Claude Desktop, etc.)
-   **Web Interface** - Streamlit-based UI

## Quick Start

### Installation

Install from source:

    git clone https://github.com/zyhowe/AutoStat.git
    cd AutoStat
    pip install -e .

Install MCP version from PyPI:

    pip install autostat-mcp

### Command Line Usage

    # Analyze a single file
    autostat data.csv -o report.html

    # Output as JSON
    autostat data.csv -f json -o result.json

    # Output as Markdown
    autostat data.csv -f md -o report.md

    # Output as Excel
    autostat data.csv -f excel -o report.xlsx

    # Quiet mode
    autostat data.csv --quiet

    # Auto clean
    autostat data.csv --auto-clean

### Python API Usage

    from autostat import AutoStatisticalAnalyzer

    # Single table analysis
    analyzer = AutoStatisticalAnalyzer("sales_data.csv")
    analyzer.generate_full_report()

    # Generate HTML report
    from autostat import Reporter
    reporter = Reporter(analyzer)
    reporter.to_html("report.html")

    # Generate JSON output
    reporter.to_json("result.json")

    # Generate Markdown output
    reporter.to_markdown("report.md")

    # Generate Excel output
    reporter.to_excel("report.xlsx")

### Multi-table Analysis

    from autostat import MultiTableStatisticalAnalyzer
    import pandas as pd

    # Load tables
    tables = {
        "orders": pd.read_csv("orders.csv"),
        "users": pd.read_csv("users.csv"),
        "products": pd.read_csv("products.csv")
    }

    # Analyze
    analyzer = MultiTableStatisticalAnalyzer(tables)
    analyzer.analyze_all_tables()

### Web Interface

    streamlit run web/app.py

### MCP Service (for AI Agents)

    # STDIO mode (default)
    autostat-mcp

    # HTTP mode
    autostat-mcp --transport http --port 6011

    # SSE mode
    autostat-mcp --transport sse --host 0.0.0.0 --port 6011

## Output Formats

### HTML Report

The HTML report includes:

-   Data overview (rows, columns, missing values, duplicates)
-   Variable type distribution
-   Statistical summaries for each variable
-   Data cleaning suggestions

### JSON Output Structure

    {
      "analysis_time": "2024-01-01T12:00:00",
      "source_table": "sales_data",
      "data_shape": {"rows": 10000, "columns": 15},
      "variable_types": {
        "user_id": {"type": "identifier", "type_desc": "identifier"},
        "amount": {"type": "continuous", "type_desc": "continuous"}
      },
      "quality_report": {
        "missing": [{"column": "city", "count": 150, "percent": 1.5}],
        "duplicates": {"count": 23, "percent": 0.23}
      },
      "cleaning_suggestions": ["建议处理缺失值"]
    }

## Supported Data Sources

| Format   | Extension          | Method          |
|----------|--------------------|-----------------|
| CSV      | .csv               | load_csv()      |
| Excel    | .xlsx, .xls        | load_excel()    |
| JSON     | .json              | load_json()     |
| TXT      | .txt               | load_txt()      |
| SQL Server| -                 | load_sql_server()|

## MCP Service Configuration

AutoStat provides MCP (Model Context Protocol) service for AI agent integration.

### STDIO Type (Recommended for Claude Desktop)

    {
      "mcpServers": {
        "autostat": {
          "command": "uvx",
          "args": ["autostat-mcp"]
        }
      }
    }

Or using pip installation:

    {
      "mcpServers": {
        "autostat": {
          "command": "python",
          "args": ["-m", "autostat.mcp_server"]
        }
      }
    }

### Streamable HTTP Type

    {
      "mcpServers": {
        "autostat": {
          "type": "streamable_http",
          "url": "https://your-server.com/mcp"
        }
      }
    }

### SSE Type

    {
      "mcpServers": {
        "autostat": {
          "type": "sse",
          "url": "https://your-server.com/sse"
        }
      }
    }

### Environment Variables

| Variable        | Description                    | Required | Default     |
|-----------------|--------------------------------|----------|-------------|
| AUTOSTAT_HOST   | HTTP/SSE service bind address  | No       | 0.0.0.0     |
| AUTOSTAT_PORT   | HTTP/SSE service port          | No       | 6011        |

### MCP Tools

| Tool Name                  | Description                                           |
|----------------------------|-------------------------------------------------------|
| analyze_from_file          | Analyze a single file (CSV, Excel, JSON, TXT)         |
| analyze_multiple_files     | Analyze multiple files, auto-discover relationships   |
| analyze_from_db            | Analyze data from SQL Server database                 |
| get_data_quality_report    | Quick data quality report                             |

### Using with Claude Desktop

1.  Install AutoStat MCP:

        pip install autostat-mcp

2.  Configure Claude Desktop's `claude_desktop_config.json`:

        {
          "mcpServers": {
            "autostat": {
              "command": "autostat-mcp"
            }
          }
        }

3.  Restart Claude Desktop to use AutoStat.

## Requirements

-   Python 3.8+
-   pandas, numpy, matplotlib, seaborn
-   scipy, statsmodels, networkx
-   click, jinja2

## Optional Dependencies

| Dependency | Purpose              |
|------------|----------------------|
| streamlit  | Web interface        |
| fastmcp    | MCP service          |
| pyodbc     | SQL Server support   |
| openpyxl   | Excel export         |

## Project Structure

    autostat/
    ├── autostat/
    │   ├── __init__.py
    │   ├── analyzer.py      # Core analysis engine
    │   ├── loader.py        # Data loader
    │   ├── multi_analyzer.py # Multi-table analysis
    │   ├── checker.py       # Condition checker
    │   ├── reporter.py      # Report generator
    │   ├── cli.py           # Command line interface
    │   ├── mcp_server.py    # MCP service
    │   ├── config_manager.py # Configuration management
    │   ├── llm_client.py    # LLM client
    │   ├── prompts.py       # Prompt templates
    │   └── core/
    │       ├── __init__.py
    │       ├── base.py
    │       ├── timeseries.py
    │       ├── relationship.py
    │       ├── recommendation.py
    │       ├── report_data.py
    │       └── plots.py
    ├── web/
    │   └── app.py           # Streamlit web UI
    ├── templates/
    │   └── report.html      # HTML report template
    ├── tests/
    │   └── test_analyzer.py # Unit tests
    ├── examples/
    │   └── example.py       # Usage examples
    ├── setup.py
    ├── pyproject.toml
    ├── requirements.txt
    ├── LICENSE
    ├── README.md
    └── README_cn.md

## License

MIT License

## Contributing

Contributions are welcome! Please submit a Pull Request.

## Contact

-   Issues: https://github.com/zyhowe/AutoStat/issues
-   Email: howe_min@163.com