# AutoStat - Intelligent Statistical Analysis Tool

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/autostat-mcp.svg)](https://pypi.org/project/autostat-mcp/)

AutoStat is an intelligent statistical analysis tool that automatically identifies data types, detects data quality issues, selects appropriate statistical methods, generates professional analysis reports, and provides LLM-powered insights. No statistical background required.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [AutoText - Text Analysis](#autotext---text-analysis)
- [Web Interface](#web-interface)
- [Model Training](#model-training)
- [MCP Service](#mcp-service)
- [Output Formats](#output-formats)
- [Supported Data Sources](#supported-data-sources)
- [Audit Rule Discovery](#audit-rule-discovery)
- [API Reference](#api-reference)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [License](#license)

---

## Features

### Data Analysis (AutoStat)

| Feature | Description |
|---------|-------------|
| **Automatic Data Type Recognition** | Identifies continuous, categorical, datetime, identifier, and ordinal variables |
| **Data Quality Inspection** | Detects missing values, outliers, duplicates, and inconsistent types |
| **Smart Statistical Method Selection** | Automatically chooses t-test, ANOVA, Mann-Whitney, chi-square, Fisher's exact test |
| **Multi-table Association Analysis** | Automatically discovers relationships between tables and performs joint analysis |
| **Time Series Analysis** | Stationarity test (ADF), autocorrelation test (Ljung-Box), seasonality detection |
| **Relationship Analysis** | Correlation matrix, Cramer's V, Eta-squared with visual heatmaps |
| **Audit Rule Discovery** | Automatically discovers arithmetic relationships, functional dependencies, and temporal constraints |
| **Date Rule Discovery** | Detects workday intervals, conditional temporal rules, and date constraints |
| **Intelligent Sampling** | Foreign-key-aware stratified sampling for large datasets |
| **Multiple Output Formats** | HTML reports, JSON, Markdown, Excel |
| **Multiple Data Sources** | CSV, Excel, JSON, TXT, SQL Server |
| **LLM-Powered Interpretation** | AI-driven data interpretation and question answering |

### Text Analysis (AutoText)

| Feature | Description |
|---------|-------------|
| **Text Preprocessing** | Automatic cleaning, tokenization, stopword removal, template detection |
| **Keyword Extraction** | Frequency-based and TF-IDF keyword extraction |
| **Sentiment Analysis** | Lexicon-based sentiment analysis with Chinese/English support |
| **Entity Recognition** | NER based on pre-trained models (BERT) |
| **Text Clustering** | K-Means, MiniBatch K-Means, Agglomerative clustering |
| **Topic Modeling** | LDA-based topic modeling with TextRank and LLM summaries |
| **LLM-Powered Extraction** | Entity, relationship, event, and theme extraction via LLM |
| **Knowledge Graph** | Entity-relationship-event graph with visualization |
| **Event Timeline** | Event chain discovery and timeline visualization |
| **Theme Hierarchy** | Hierarchical theme organization |

### Model Training

| Feature | Description |
|---------|-------------|
| **Classification Models** | Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost, SVM, KNN, MLP, CNN, RNN, LSTM, BERT |
| **Regression Models** | Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, XGBoost, LightGBM, SVR, KNN, MLP |
| **Clustering Models** | K-Means, DBSCAN, Agglomerative, BIRCH, Gaussian Mixture, OPTICS, Spectral, Mean Shift |
| **Time Series Models** | ARIMA, SARIMA, Prophet, LSTM, GRU, Transformer |
| **Model Management** | Save, load, delete models; inference with confidence scores |
| **Automatic Training** | Auto-train recommended models after analysis completes |

---

## Installation

### From Source

```bash
git clone https://github.com/zyhowe/AutoStat.git
cd AutoStat
pip install -e .
```

### From PyPI (MCP Version)

```bash
pip install autostat-mcp
```

### With Text Analysis Support

```bash
pip install -e ".[text]"
```

### Required Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
statsmodels>=0.13.0
networkx>=2.6.0
click>=8.0.0
jinja2>=3.0.0
requests>=2.25.0
```

---

## Quick Start

### Command Line Usage

```bash
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
```

### Python API Usage

```python
from autostat import AutoStatisticalAnalyzer
from autostat import Reporter

# Single table analysis
analyzer = AutoStatisticalAnalyzer("sales_data.csv")
analyzer.generate_full_report()

# Generate HTML report
reporter = Reporter(analyzer)
reporter.to_html("report.html")

# Generate JSON output
reporter.to_json("result.json")

# Generate Markdown
reporter.to_markdown("report.md")

# Generate Excel
reporter.to_excel("report.xlsx")
```

### Multi-table Analysis

```python
from autostat import MultiTableStatisticalAnalyzer
import pandas as pd

# Load tables
tables = {
    "orders": pd.read_csv("orders.csv"),
    "users": pd.read_csv("users.csv"),
    "products": pd.read_csv("products.csv")
}

# Analyze with automatic relationship discovery
analyzer = MultiTableStatisticalAnalyzer(tables)
analyzer.analyze_all_tables()

# Or define relationships manually
relationships = {
    "foreign_keys": [
        {"from_table": "orders", "from_col": "user_id", "to_table": "users", "to_col": "id"},
        {"from_table": "orders", "from_col": "product_id", "to_table": "products", "to_col": "id"}
    ]
}
analyzer = MultiTableStatisticalAnalyzer(tables, relationships=relationships)
analyzer.analyze_all_tables()

# Generate report from merged data
html = analyzer.to_html("multi_table_report.html")
json_data = analyzer.to_json("multi_table_result.json")
```

### Load Data from Various Sources

```python
from autostat.loader import DataLoader

# Load CSV
df = DataLoader.load_csv("data.csv")

# Load Excel
df = DataLoader.load_excel("data.xlsx")

# Load JSON
df = DataLoader.load_json("data.json")

# Load TXT
df = DataLoader.load_txt("data.txt")

# Load from SQL Server
df = DataLoader.load_sql_server(
    server="localhost",
    database="mydb",
    table_name="users",
    username="sa",
    password="password",
    limit=10000
)

# Load multiple SQL Server tables with relationships
tables = DataLoader.load_multiple_tables(
    server="localhost",
    database="mydb",
    table_names=["users", "orders", "products"],
    username="sa",
    password="password",
    relationships=[
        {"from_table": "orders", "from_col": "user_id", "to_table": "users", "to_col": "id"}
    ],
    limit=5000,
    max_text_length=100
)
```

---

## AutoText - Text Analysis

### Command Line Usage

```bash
# Analyze text file
autotext analyze comments.txt -o report.html

# Analyze folder of text files
autotext analyze ./texts/ -o report.json -f json

# Extract keywords
autotext keywords comments.txt --top 30

# Sentiment analysis
autotext sentiment reviews.txt

# Entity recognition
autotext entities news.txt
```

### Python API Usage

```python
from autotext import TextAnalyzer

# Analyze a text file
analyzer = TextAnalyzer("comments.txt")
analyzer.generate_full_report()
analyzer.to_html("text_report.html")

# Analyze from list
texts = ["Great product!", "Terrible service.", "Good value for money."]
analyzer = TextAnalyzer(texts, source_name="reviews")
analyzer.generate_full_report()

# Analyze from DataFrame
import pandas as pd
df = pd.DataFrame({
    "title": ["Review 1", "Review 2"],
    "content": ["Excellent quality!", "Very disappointed."],
    "date": ["2024-01-01", "2024-01-02"]
})
analyzer = TextAnalyzer(data=df, text_col="content", title_col="title", time_col="date")
analyzer.generate_full_report()

# With LLM-powered extraction
analyzer = TextAnalyzer(
    texts,
    use_bert=True,
    llm_config={
        "api_base": "https://api.deepseek.com/v1",
        "api_key": "your-api-key",
        "model": "deepseek-chat"
    }
)
analyzer.generate_full_report()

# Save results
analyzer.save_raw_texts("raw.txt")
analyzer.save_cleaned_texts("cleaned.txt")
analyzer.save_content_texts("content.txt")
```

---

## Web Interface

Start the web interface:

```bash
streamlit run web/app.py
```

### Data Analysis Interface (AutoStat)

The web interface features a **5-tab** layout:

| Tab | Description |
|-----|-------------|
| **Data Preparation** | Upload files, select fields, adjust variable types, manage table relationships, start analysis |
| **Preview Report** | View the generated HTML analysis report |
| **Model Training** | Train classification, regression, clustering, and time series models |
| **AI Interpretation** | LLM-powered chat with context selection (JSON results, HTML reports, raw data) |
| **Project Comparison** | Compare two analysis projects side by side |

#### AI Interpretation Features

| Feature | Description |
|---------|-------------|
| **Context Selection** | Choose from JSON results, HTML reports, or raw data as context |
| **Free Questions** | Ask any questions about your data |
| **Scenario Recommendations** | Auto-detect business scenarios and get analysis perspectives |
| **Natural Query** | Query data using natural language |
| **SQL Generation** | Generate SQL queries from natural language (database mode only) |
| **Agent Inference** | Natural language model prediction |
| **Audit Rule Validation** | Validate data consistency rules with business logic |

### Text Analysis Interface (AutoText)

The text analysis interface features a **4-tab** layout:

| Tab | Description |
|-----|-------------|
| **Data Preparation** | Input or paste text, configure analysis options |
| **Report Preview** | View the generated HTML text analysis report |
| **LLM Interpretation** | AI-powered chat for text insights |
| **Project Comparison** | Compare two text analysis projects side by side |

---

## Model Training

### Available Models

#### Classification Models
- Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost
- SVM, KNN, Naive Bayes, AdaBoost, Gradient Boosting, MLP
- CNN, RNN, LSTM, BERT (deep learning)

#### Regression Models
- Linear Regression, Ridge, Lasso, ElasticNet
- Decision Tree, Random Forest, XGBoost, LightGBM, SVR, KNN, MLP

#### Clustering Models
- K-Means, DBSCAN, Agglomerative, BIRCH
- Gaussian Mixture, OPTICS, Spectral, Mean Shift

#### Time Series Models
- ARIMA, SARIMA, Prophet
- LSTM, GRU, Transformer (deep learning)

### Training Example

```python
from autostat.models.trainer import ModelTrainer
from autostat.models.preprocessing import DataPreprocessor
from autostat.models.storage import ModelStorage
import pandas as pd

# Load data
data = pd.read_csv("data.csv")
features = ["age", "income", "score"]
target = "label"

# Preprocess
preprocessor = DataPreprocessor({
    "missing_strategy": "drop",
    "scaling": "standard",
    "encoding": "onehot"
})

# Train
trainer = ModelTrainer("classification", "random_forest", {"n_estimators": 100})
result = trainer.train(X_train, y_train, cv_folds=5)

# Save model
ModelStorage.save_model(
    session_id="my_session",
    model_key="rf_model_001",
    model=trainer.model,
    preprocessor=preprocessor,
    metrics=result["train_score"],
    config={"features": features, "target": target}
)

# Load and predict
model, preprocessor, metadata, metrics = ModelStorage.load_model("my_session", "rf_model_001")
predictor = ModelPredictor(model, preprocessor)
prediction = predictor.predict({"age": 35, "income": 50000, "score": 85})
print(f"Prediction: {prediction['prediction']}, Confidence: {prediction['confidence']}")
```

### Web Interface Training

1. Go to the **Model Training** tab
2. Select task type (classification/regression/clustering/time_series)
3. Choose target column (if supervised)
4. Select features
5. Pick a model
6. Configure parameters
7. Click **Start Training**

---

## MCP Service

AutoStat provides MCP (Model Context Protocol) service for AI agent integration.

### MCP Tools

| Tool Name | Description |
|-----------|-------------|
| `analyze_from_file` | Analyze a single file (CSV, Excel, JSON, TXT) |
| `analyze_multiple_files` | Analyze multiple files, auto-discover relationships |
| `analyze_from_db` | Analyze data from SQL Server database |
| `get_data_quality_report` | Quick data quality report |

### Starting the MCP Server

```bash
# STDIO mode (default, for Claude Desktop)
autostat-mcp

# HTTP mode
autostat-mcp --transport http --port 6011

# SSE mode
autostat-mcp --transport sse --host 0.0.0.0 --port 6011
```

### Claude Desktop Configuration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "autostat": {
      "command": "autostat-mcp"
    }
  }
}
```

Or using `uvx`:

```json
{
  "mcpServers": {
    "autostat": {
      "command": "uvx",
      "args": ["autostat-mcp"]
    }
  }
}
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AUTOSTAT_HOST` | HTTP/SSE service bind address | `0.0.0.0` |
| `AUTOSTAT_PORT` | HTTP/SSE service port | `6011` |

### AutoText MCP Server

```bash
# STDIO mode
autotext-mcp

# HTTP mode
autotext-mcp --transport http --port 6012

# SSE mode
autotext-mcp --transport sse --host 0.0.0.0 --port 6012
```

---

## Output Formats

### HTML Report

The HTML report includes:

- **Data Overview** - Rows, columns, missing values, duplicates, date range
- **Variable Type Distribution** - Categorical, continuous, datetime, identifier
- **Statistical Summaries** - Mean, median, std, quartiles for each variable
- **Visualizations** - Histograms, box plots, correlation heatmaps, time series plots
- **Data Quality Alerts** - Missing values, outliers, duplicates
- **Cleaning Suggestions** - Recommended data cleaning actions
- **Model Recommendations** - Suggested models based on data characteristics
- **Audit Rules** - Discovered arithmetic relationships, functional dependencies, temporal constraints
- **Core Insights** - Key findings and recommended actions

### JSON Output Structure

```json
{
  "analysis_time": "2024-01-01T12:00:00",
  "source_table": "sales_data",
  "data_shape": {"rows": 10000, "columns": 15},
  "variable_types": {
    "user_id": {"type": "identifier", "type_desc": "标识符列"},
    "amount": {"type": "continuous", "type_desc": "连续变量"}
  },
  "quality_report": {
    "missing": [{"column": "city", "count": 150, "percent": 1.5}],
    "outliers": {"amount": {"count": 45, "percent": 0.45}},
    "duplicates": {"count": 23, "percent": 0.23}
  },
  "cleaning_suggestions": ["建议处理缺失值"],
  "correlations": {
    "high_correlations": [{"var1": "income", "var2": "spending", "value": 0.85}]
  },
  "time_series_diagnostics": {
    "sales": {
      "is_stationary": false,
      "has_autocorrelation": true,
      "has_seasonality": true
    }
  },
  "model_recommendations": [
    {
      "priority": "高",
      "task_type": "回归预测",
      "target_column": "sales",
      "feature_columns": ["income", "advertising"],
      "ml": "随机森林 / XGBoost"
    }
  ]
}
```

### Multi-table JSON Output

```json
{
  "analysis_time": "2024-01-01T12:00:00",
  "analysis_type": "multi_table",
  "multi_table_info": {
    "tables": {
      "orders": {"rows": 5000, "columns": 10},
      "users": {"rows": 2000, "columns": 8}
    },
    "relationships": [
      {"from_table": "orders", "from_col": "user_id", "to_table": "users", "to_col": "id", "type": "many_to_one"}
    ],
    "table_groups": [
      {"type": "related", "tables": ["orders", "users"], "relationships": 1}
    ]
  }
}
```

---

## Supported Data Sources

| Format | Extension | Method | Notes |
|--------|-----------|--------|-------|
| CSV | `.csv` | `load_csv()` | Auto-detects encoding, handles special characters |
| Excel | `.xlsx`, `.xls` | `load_excel()` | Supports openpyxl and xlrd engines |
| JSON | `.json` | `load_json()` | Supports records, index, columns formats |
| TXT | `.txt` | `load_txt()` | Auto-detects delimiters |
| SQL Server | - | `load_sql_server()` | Supports Windows and SQL authentication |

---

## Audit Rule Discovery

AutoStat automatically discovers data consistency rules across fields.

### Types of Rules

| Rule Type | Description | Example |
|-----------|-------------|---------|
| **Arithmetic Rules** | Numerical relationships between fields | `revenue = price * quantity` |
| **Functional Dependencies** | Deterministic relationships | `user_id → user_name` |
| **Temporal Rules** | Date/time constraints | `payment_date ≥ order_date` |
| **Foreign Keys** | Cross-table referential integrity | `orders.user_id → users.id` |
| **Date Rules** | Workday intervals, conditional temporal rules | `payment_date = order_date + 2个工作日` |

### API Usage

```python
from autostat.core.audit import discover_audit_rules
from autostat.core.date_rules import discover_date_rules

# Discover audit rules
audit_rules = discover_audit_rules(
    data=df,
    variable_types=variable_types,
    foreign_keys=[],
    debug=True,
    min_confidence=0.5
)

# Discover date rules
date_rules = discover_date_rules(
    df,
    date_columns=["order_date", "payment_date", "ship_date"],
    categorical_columns=["category"],
    debug=True,
    consider_workday=True,
    consider_conditional=True
)
```

---

## API Reference

### AutoStatisticalAnalyzer

```python
class AutoStatisticalAnalyzer:
    def __init__(self, data, target_col=None, source_table_name=None,
                 predefined_types=None, auto_clean=False, quiet=False,
                 date_features_level="basic", skip_auto_inference=False):
        """
        Parameters:
        - data: DataFrame or file path string
        - target_col: Target column name (optional)
        - source_table_name: Source table name
        - predefined_types: Predefined variable types dict
        - auto_clean: Whether to auto-clean data
        - quiet: Quiet mode (no console output)
        - date_features_level: Date feature level (none/basic/full)
        - skip_auto_inference: Skip automatic type inference
        """

    def generate_full_report(self, show_outlier_details=False):
        """Generate complete analysis report"""

    def to_json(self, output_file=None, indent=2, ensure_ascii=False):
        """Export results as JSON"""

    def auto_time_series_analysis(self, max_numeric=10, group_by='auto'):
        """Auto-detect and analyze time series"""

    def auto_analyze_relationships(self):
        """Auto-analyze variable relationships"""

    def recommend_scenarios(self):
        """Recommend analysis scenarios and models"""
```

### MultiTableStatisticalAnalyzer

```python
class MultiTableStatisticalAnalyzer:
    def __init__(self, tables, relationships=None, date_features_level="basic",
                 predefined_types=None):
        """
        Parameters:
        - tables: Dict of {table_name: DataFrame}
        - relationships: Dict with 'foreign_keys' list
        - date_features_level: Date feature level
        - predefined_types: Dict of {table_name: {col: type}}
        """

    @classmethod
    def from_files(cls, table_files, relationships=None, **kwargs):
        """Load tables from files"""

    @classmethod
    def from_json_strings(cls, json_strings, relationships=None, **kwargs):
        """Load tables from JSON strings"""

    def analyze_all_tables(self):
        """Analyze all tables and relationships"""

    def get_merged_analyzer(self):
        """Get merged analyzer for report generation"""

    def to_html(self, output_file=None, title="多表分析报告"):
        """Generate HTML report"""

    def to_json(self, output_file=None, indent=2, ensure_ascii=False):
        """Export results as JSON"""
```

### DataLoader

```python
class DataLoader:
    @staticmethod
    def load_csv(file_path, encoding='utf-8-sig', parse_dates=True, date_columns=None, **kwargs):
        """Load CSV file"""

    @staticmethod
    def load_excel(file_path, sheet_name=0, **kwargs):
        """Load Excel file"""

    @staticmethod
    def load_json(file_path, encoding='utf-8-sig', orient='records', parse_dates=True, **kwargs):
        """Load JSON file"""

    @staticmethod
    def load_txt(file_path, delimiter='\t', encoding='utf-8-sig', **kwargs):
        """Load TXT file"""

    @staticmethod
    def load_sql_server(server, database, table_name=None, query=None,
                        username=None, password=None, trusted_connection=True,
                        limit=1000, smart_sampling=False, join_key=None,
                        main_table_df=None, match_key=None, **kwargs):
        """Load from SQL Server"""

    @staticmethod
    def load_multiple_tables(server, database, table_names, username=None,
                             password=None, trusted_connection=True,
                             relationships=None, limit=1000, **kwargs):
        """Batch load multiple SQL Server tables"""

    @staticmethod
    def load_from_file(file_path, parse_dates=True, date_columns=None, **kwargs):
        """Auto-detect and load file by extension"""
```

---

## Requirements

### Core Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scipy, statsmodels, networkx
- click, jinja2

### Optional Dependencies

| Dependency | Purpose |
|------------|---------|
| `streamlit` | Web interface |
| `fastmcp` | MCP service |
| `pyodbc` | SQL Server support |
| `openpyxl` | Excel export |
| `jieba` | Chinese text segmentation |
| `nltk` | English text processing |
| `snownlp` | Chinese sentiment analysis |
| `transformers` | BERT models |
| `tensorflow` | Deep learning models |
| `xgboost` | XGBoost models |
| `lightgbm` | LightGBM models |
| `catboost` | CatBoost models |
| `prophet` | Prophet time series |

---

## Project Structure

```
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
│   ├── core/                          # Core modules
│   │   ├── __init__.py
│   │   ├── analyzer.py                # Main analyzer implementation
│   │   ├── audit.py                   # Audit rule discovery
│   │   ├── base.py                    # Base analyzer (type inference, quality check)
│   │   ├── date_rules.py              # Date rule discovery
│   │   ├── plots.py                   # Visualization
│   │   ├── recommendation.py          # Model recommendations
│   │   ├── relationship.py            # Relationship analysis
│   │   ├── report_data.py             # Report data builder
│   │   └── timeseries.py              # Time series analysis
│   └── models/                        # ML models module
│       ├── __init__.py
│       ├── arima_wrapper.py           # ARIMA wrapper for sklearn interface
│       ├── deep_learning.py           # CNN, RNN, LSTM, BERT, Transformer
│       ├── metrics.py                 # Evaluation metrics
│       ├── predictor.py               # Model inference
│       ├── preprocessing.py           # Data preprocessing
│       ├── registry.py                # Model registry
│       ├── storage.py                 # Model storage
│       └── trainer.py                 # Model training
│
├── autotext/                          # Text analysis engine
│   ├── __init__.py
│   ├── analyzer.py                    # Main text analyzer
│   ├── checker.py                     # Text condition checker
│   ├── cli.py                         # Command line interface
│   ├── llm_extractor.py               # LLM-based information extraction
│   ├── llm_visualizer.py              # LLM extraction visualization
│   ├── loader.py                      # Text data loader
│   ├── mcp_server.py                  # MCP service
│   ├── reporter.py                    # Text report generator
│   └── core/                          # Core text modules
│       ├── __init__.py
│       ├── cache.py                   # Cache management
│       ├── cluster.py                 # Text clustering
│       ├── detector.py                # Field detector
│       ├── entity.py                  # Entity recognition
│       ├── entity_profile.py          # Entity profile builder
│       ├── event_extractor.py         # Event extraction
│       ├── event_timeline.py          # Event timeline analysis
│       ├── graph_analyzer.py          # Graph analysis
│       ├── graph_builder.py           # Graph builder
│       ├── info_extractor.py          # Information extraction
│       ├── insight.py                 # Insight discovery
│       ├── keyword_extractor.py       # Keyword extraction
│       ├── llm_enhance.py             # LLM enhancement
│       ├── llm_graph_extractor.py     # LLM graph extraction
│       ├── ner.py                     # Named entity recognition
│       ├── preprocessor.py            # Text preprocessing
│       ├── quality.py                 # Text quality check
│       ├── relation.py                # Relation discovery
│       ├── relation_mining.py         # Association rule mining
│       ├── sentiment.py               # Sentiment analysis
│       ├── stats.py                   # Text statistics
│       ├── summarizer.py              # Text summarization
│       ├── timeline_builder.py        # Timeline builder
│       ├── topic.py                   # Topic modeling
│       ├── topic_model.py             # LDA topic modeling
│       ├── trend.py                   # Trend analysis
│       ├── trend_detector.py          # Trend detection
│       └── vectorizer.py              # BERT vectorization
│
├── web/                               # Web interface
│   ├── __init__.py
│   ├── app.py                         # Streamlit main entry
│   ├── components/                    # UI components
│   │   ├── __init__.py
│   │   ├── agent_inference.py         # Agent inference
│   │   ├── audit_rule.py              # Audit rule UI
│   │   ├── chat_interface.py          # AI chat interface
│   │   ├── data_preparation.py        # Data preparation UI
│   │   ├── demo_data.py               # Demo data section
│   │   ├── empty_state.py             # Empty state guide
│   │   ├── model_training.py          # Model training UI
│   │   ├── natural_query.py           # Natural language query
│   │   ├── progress_stage.py          # Progress stages
│   │   ├── results.py                 # Results display
│   │   ├── scenario_recommendation.py # Scenario recommendations
│   │   ├── sidebar.py                 # Sidebar with config
│   │   ├── sql_generator.py           # SQL generator
│   │   ├── tabs.py                    # Tab navigation
│   │   ├── term_tooltip.py            # Term tooltips
│   │   ├── tips.py                    # Usage tips
│   │   └── value_preview.py           # Value preview
│   ├── config/                        # Client-side storage
│   │   └── storage.py                 # Config management
│   ├── services/                      # Business logic layer
│   │   ├── __init__.py
│   │   ├── agent_service.py           # Agent service
│   │   ├── analysis_service.py        # Analysis execution
│   │   ├── auto_interpret_service.py  # Auto interpretation
│   │   ├── auto_train_service.py      # Auto training
│   │   ├── cache_service.py           # Cache management
│   │   ├── error_handler.py           # Error handling
│   │   ├── feature_flags.py           # Feature flags
│   │   ├── file_service.py            # File operations
│   │   ├── insight_service.py         # Insight generation
│   │   ├── model_training_service.py  # Model training service
│   │   ├── recommendation_service.py  # Recommendation service
│   │   ├── session_service.py         # Session management
│   │   └── storage_service.py         # Unified storage
│   └── utils/                         # Utilities
│       ├── __init__.py
│       ├── data_preprocessor.py       # Data preprocessing utilities
│       └── helpers.py                 # Helper functions
│
├── webtext/                           # Text analysis web interface
│   ├── __init__.py
│   ├── app.py                         # Streamlit entry
│   ├── components/                    # UI components
│   │   ├── __init__.py
│   │   ├── chat_interface.py          # Text chat interface
│   │   ├── compare.py                 # Compare component
│   │   ├── data_preparation.py        # Data preparation
│   │   ├── results.py                 # Results display
│   │   ├── sidebar.py                 # Sidebar
│   │   └── tabs.py                    # Tab navigation
│   └── services/                      # Text services
│       ├── __init__.py
│       ├── analysis_service.py        # Analysis service
│       └── session_service.py         # Session management
│
├── pages/                             # Streamlit pages
│   ├── Home.py                        # Home page
│   ├── autostat.py                    # Data analysis page
│   └── autotext.py                    # Text analysis page
│
├── templates/                         # HTML templates
│   ├── report.html                    # Data analysis report
│   └── report_text.html               # Text analysis report
│
├── tests/                             # Unit tests
│   └── test_analyzer.py
│
├── examples/                          # Usage examples
│   ├── example.py                     # Data analysis examples
│   └── example_text.py                # Text analysis examples
│
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
├── README_cn.md
└── requirements.txt
```

---

## License

MIT License

Copyright (c) 2024 AutoStat Team

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Contributing

Contributions are welcome! Please submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Contact

- Issues: https://github.com/zyhowe/AutoStat/issues
- Email: howe_min@163.com
- Repository: https://github.com/zyhowe/AutoStat