# AutoStat - Intelligent Statistical Analysis Tool

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/autostat-mcp.svg)](https://pypi.org/project/autostat-mcp/)

AutoStat is an intelligent statistical analysis tool that automatically identifies data types, detects data quality issues, selects appropriate statistical methods, and generates professional analysis reports. No statistical background required.

## Features

- Automatic Data Type Recognition - Identifies continuous, categorical, datetime, identifier, and ordinal variables
- Data Quality Inspection - Detects missing values, outliers, duplicates, and inconsistent types
- Smart Statistical Method Selection - Automatically chooses between t-test, ANOVA, Mann-Whitney, chi-square, Fisher's exact test
- Multi-table Association Analysis - Automatically discovers relationships between tables and performs joint analysis
- Time Series Analysis - Stationarity test (ADF), autocorrelation test (Ljung-Box), seasonality detection
- Relationship Analysis - Correlation matrix, Cramer's V, Eta-squared with visual heatmaps
- Intelligent Sampling - Foreign-key-aware stratified sampling for large datasets
- Multiple Output Formats - HTML reports, JSON, Markdown, Excel
- Multiple Data Sources - CSV, Excel, JSON, TXT, SQL Server
- MCP Service - Can be called by AI agents (Claude Desktop, etc.)
- Web Interface - Streamlit-based UI with AI-powered data interpretation

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

Start the web interface:

    streamlit run web/app.py

The web interface features a 4-tab layout:

| Tab | Description |
|-----|-------------|
| Data Preparation | Upload files, select fields, adjust variable types, manage table relationships, start analysis |
| Preview Report | View the generated HTML analysis report |
| Analysis Log | View console output and execution logs |
| AI Interpretation | AI-powered chat with context selection (JSON results, HTML reports, raw data) |

#### AI Interpretation Features

- Context Selection - Choose from JSON results, HTML reports, or raw data as context
- Free Questions - Ask any questions about your data
- Scenario Recommendations - Auto-detect business scenarios and get analysis perspectives
- Natural Query - Query data using natural language
- SQL Generation - Generate SQL queries from natural language (database mode only)

## Output Formats

### HTML Report

The HTML report includes:

- Data overview (rows, columns, missing values, duplicates)
- Variable type distribution
- Statistical summaries for each variable
- Data cleaning suggestions
- Visualizations (histograms, box plots, correlation heatmaps)

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
      "cleaning_suggestions": ["Handle missing values"]
    }

## Supported Data Sources

| Format | Extension | Method |
|--------|-----------|--------|
| CSV | .csv | load_csv() |
| Excel | .xlsx, .xls | load_excel() |
| JSON | .json | load_json() |
| TXT | .txt | load_txt() |
| SQL Server | - | load_sql_server() |

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

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| AUTOSTAT_HOST | HTTP/SSE service bind address | No | 0.0.0.0 |
| AUTOSTAT_PORT | HTTP/SSE service port | No | 6011 |

### MCP Tools

| Tool Name | Description |
|-----------|-------------|
| analyze_from_file | Analyze a single file (CSV, Excel, JSON, TXT) |
| analyze_multiple_files | Analyze multiple files, auto-discover relationships |
| analyze_from_db | Analyze data from SQL Server database |
| get_data_quality_report | Quick data quality report |

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

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scipy, statsmodels, networkx
- click, jinja2

## Optional Dependencies

| Dependency | Purpose |
|------------|---------|
| streamlit | Web interface |
| fastmcp | MCP service |
| pyodbc | SQL Server support |
| openpyxl | Excel export |

## Project Structure

    autostat/
    ├── autostat/                          # Core analysis engine
    │   ├── __init__.py
    │   ├── __main__.py                    # Entry for python -m autostat
    │   ├── analyzer.py                    # Main analyzer (backward compatibility)
    │   ├── checker.py                     # Condition checker
    │   ├── cli.py                         # Command line interface
    │   ├── config_manager.py              # Configuration management
    │   ├── llm_client.py                  # LLM client
    │   ├── loader.py                      # Data loader
    │   ├── mcp_server.py                  # MCP service
    │   ├── multi_analyzer.py              # Multi-table analysis
    │   ├── prompts.py                     # Prompt templates
    │   ├── reporter.py                    # Report generator
    │   └── core/                          # Core modules
    │       ├── __init__.py
    │       ├── analyzer.py                # Main analyzer implementation
    │       ├── base.py                    # Base analyzer (type inference, quality check)
    │       ├── plots.py                   # Visualization
    │       ├── recommendation.py          # Model recommendations
    │       ├── relationship.py            # Relationship analysis
    │       ├── report_data.py             # Report data builder
    │       └── timeseries.py              # Time series analysis
    ├── web/                               # Web interface
    │   ├── app.py                         # Streamlit main entry
    │   ├── components/                    # UI components
    │   │   ├── __init__.py
    │   │   ├── sidebar.py                 # Sidebar with config
    │   │   ├── tabs.py                    # Tab navigation
    │   │   ├── data_preparation.py        # Data preparation UI
    │   │   ├── results.py                 # Results display
    │   │   ├── chat_interface.py          # AI chat interface
    │   │   ├── scenario_recommendation.py # Scenario recommendations
    │   │   ├── natural_query.py           # Natural language query
    │   │   └── sql_generator.py           # SQL generator
    │   ├── services/                      # Business logic layer
    │   │   ├── __init__.py
    │   │   ├── cache_service.py           # Cache management
    │   │   ├── file_service.py            # File operations
    │   │   └── analysis_service.py        # Analysis execution
    │   ├── config/                        # Client-side storage
    │   │   └── storage.py                 # Config management
    │   └── utils/                         # Utilities
    │       ├── __init__.py
    │       ├── helpers.py                 # Helper functions
    │       └── data_preprocessor.py       # Data preprocessing
    ├── templates/
    │   └── report.html                    # HTML report template
    ├── tests/
    │   └── test_analyzer.py               # Unit tests
    ├── examples/
    │   └── example.py                     # Usage examples
    ├── .gitignore
    ├── LICENSE
    ├── pyproject.toml
    ├── README.md
    ├── README_cn.md
    ├── requirements.txt
    └── setup.py

## License

MIT License

## Contributing

Contributions are welcome! Please submit a Pull Request.

## Contact

- Issues: https://github.com/zyhowe/AutoStat/issues
- Email: howe_min@163.com