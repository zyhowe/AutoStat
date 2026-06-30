# AutoStat - 智能统计分析工具

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/autostat-mcp.svg)](https://pypi.org/project/autostat-mcp/)

AutoStat 是一款智能统计分析工具，能够自动识别数据类型、检测数据质量、选择合适的统计方法、生成专业的分析报告，并提供大模型驱动的智能解读。无需统计学背景。

---

## 目录

- [核心功能](#核心功能)
- [安装](#安装)
- [快速开始](#快速开始)
- [AutoText - 文本分析](#autotext---文本分析)
- [Web界面](#web界面)
- [模型训练](#模型训练)
- [MCP服务](#mcp服务)
- [输出格式](#输出格式)
- [支持的数据源](#支持的数据源)
- [勾稽规则发现](#勾稽规则发现)
- [API参考](#api参考)
- [环境要求](#环境要求)
- [项目结构](#项目结构)
- [许可证](#许可证)

---

## 核心功能

### 数据分析 (AutoStat)

| 功能 | 说明 |
|------|------|
| **自动数据类型识别** | 识别连续变量、分类变量、日期时间、标识符、有序分类变量 |
| **数据质量体检** | 检测缺失值、异常值、重复记录、类型不一致 |
| **智能统计方法选择** | 自动选择 t检验、ANOVA、Mann-Whitney、卡方检验、Fisher精确检验 |
| **多表关联分析** | 自动发现表间关系，进行联合分析 |
| **时间序列分析** | 平稳性检验(ADF)、自相关检验(Ljung-Box)、季节性检测 |
| **关系分析** | 相关性矩阵、Cramer's V、Eta-squared，带可视化热力图 |
| **勾稽规则发现** | 自动发现数值关系、函数依赖、时序约束 |
| **日期规则发现** | 检测工作日间隔、条件时序规则、日期约束 |
| **智能采样** | 基于外键感知的分层采样，支持大数据集 |
| **多种输出格式** | HTML报告、JSON、Markdown、Excel |
| **多种数据源** | CSV、Excel、JSON、TXT、SQL Server |
| **大模型智能解读** | AI驱动的数据解读和问答 |

### 文本分析 (AutoText)

| 功能 | 说明 |
|------|------|
| **文本预处理** | 自动清洗、分词、去停用词、模板检测 |
| **关键词提取** | 基于词频和TF-IDF的关键词提取 |
| **情感分析** | 基于情感词典的中英文情感分析 |
| **实体识别** | 基于预训练模型(BERT)的命名实体识别 |
| **文本聚类** | K-Means、MiniBatch K-Means、层次聚类 |
| **主题建模** | 基于LDA的主题建模，支持TextRank和LLM摘要 |
| **大模型信息抽取** | 通过大模型抽取实体、关系、事件、主题 |
| **知识图谱** | 实体-关系-事件图谱及可视化 |
| **事件脉络** | 事件链发现和时间线可视化 |
| **主题层级** | 主题层级组织 |

### 模型训练

| 功能 | 说明 |
|------|------|
| **分类模型** | 逻辑回归、决策树、随机森林、XGBoost、LightGBM、CatBoost、SVM、KNN、MLP、CNN、RNN、LSTM、BERT |
| **回归模型** | 线性回归、岭回归、Lasso、弹性网络、决策树、随机森林、XGBoost、LightGBM、SVR、KNN、MLP |
| **聚类模型** | K-Means、DBSCAN、层次聚类、BIRCH、高斯混合、OPTICS、谱聚类、均值漂移 |
| **时序模型** | ARIMA、SARIMA、Prophet、LSTM、GRU、Transformer |
| **模型管理** | 保存、加载、删除模型，带置信度的推理预测 |
| **自动训练** | 分析完成后自动训练推荐模型 |

---

## 安装

### 从源码安装

```bash
git clone https://github.com/zyhowe/AutoStat.git
cd AutoStat
pip install -e .
```

### 从 PyPI 安装 (MCP 版本)

```bash
pip install autostat-mcp
```

### 安装文本分析支持

```bash
pip install -e ".[text]"
```

### 核心依赖

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

## 快速开始

### 命令行使用

```bash
# 分析单个文件
autostat data.csv -o report.html

# 输出JSON格式
autostat data.csv -f json -o result.json

# 输出Markdown格式
autostat data.csv -f md -o report.md

# 输出Excel格式
autostat data.csv -f excel -o report.xlsx

# 静默模式
autostat data.csv --quiet

# 自动清洗
autostat data.csv --auto-clean
```

### Python API 使用

```python
from autostat import AutoStatisticalAnalyzer
from autostat import Reporter

# 单表分析
analyzer = AutoStatisticalAnalyzer("sales_data.csv")
analyzer.generate_full_report()

# 生成HTML报告
reporter = Reporter(analyzer)
reporter.to_html("report.html")

# 生成JSON输出
reporter.to_json("result.json")

# 生成Markdown
reporter.to_markdown("report.md")

# 生成Excel
reporter.to_excel("report.xlsx")
```

### 多表分析

```python
from autostat import MultiTableStatisticalAnalyzer
import pandas as pd

# 加载表
tables = {
    "orders": pd.read_csv("orders.csv"),
    "users": pd.read_csv("users.csv"),
    "products": pd.read_csv("products.csv")
}

# 自动发现关系并分析
analyzer = MultiTableStatisticalAnalyzer(tables)
analyzer.analyze_all_tables()

# 或手动定义关系
relationships = {
    "foreign_keys": [
        {"from_table": "orders", "from_col": "user_id", "to_table": "users", "to_col": "id"},
        {"from_table": "orders", "from_col": "product_id", "to_table": "products", "to_col": "id"}
    ]
}
analyzer = MultiTableStatisticalAnalyzer(tables, relationships=relationships)
analyzer.analyze_all_tables()

# 生成报告
html = analyzer.to_html("multi_table_report.html")
json_data = analyzer.to_json("multi_table_result.json")
```

### 从多种数据源加载

```python
from autostat.loader import DataLoader

# 加载 CSV
df = DataLoader.load_csv("data.csv")

# 加载 Excel
df = DataLoader.load_excel("data.xlsx")

# 加载 JSON
df = DataLoader.load_json("data.json")

# 加载 TXT
df = DataLoader.load_txt("data.txt")

# 从 SQL Server 加载
df = DataLoader.load_sql_server(
    server="localhost",
    database="mydb",
    table_name="users",
    username="sa",
    password="password",
    limit=10000
)

# 批量加载 SQL Server 表
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

## AutoText - 文本分析

### 命令行使用

```bash
# 分析文本文件
autotext analyze comments.txt -o report.html

# 分析文件夹中的文本文件
autotext analyze ./texts/ -o report.json -f json

# 提取关键词
autotext keywords comments.txt --top 30

# 情感分析
autotext sentiment reviews.txt

# 实体识别
autotext entities news.txt
```

### Python API 使用

```python
from autotext import TextAnalyzer

# 分析文本文件
analyzer = TextAnalyzer("comments.txt")
analyzer.generate_full_report()
analyzer.to_html("text_report.html")

# 从列表分析
texts = ["产品很好！", "服务太差了。", "性价比高。"]
analyzer = TextAnalyzer(texts, source_name="评论分析")
analyzer.generate_full_report()

# 从 DataFrame 分析
import pandas as pd
df = pd.DataFrame({
    "标题": ["评价1", "评价2"],
    "内容": ["质量非常好！", "非常失望。"],
    "日期": ["2024-01-01", "2024-01-02"]
})
analyzer = TextAnalyzer(data=df, text_col="内容", title_col="标题", time_col="日期")
analyzer.generate_full_report()

# 使用大模型增强抽取
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

# 保存各阶段文本
analyzer.save_raw_texts("raw.txt")
analyzer.save_cleaned_texts("cleaned.txt")
analyzer.save_content_texts("content.txt")
```

---

## Web界面

启动 Web 界面：

```bash
streamlit run web/app.py
```

### 数据分析界面 (AutoStat)

Web 界面采用 **5标签页** 布局：

| 标签页 | 说明 |
|--------|------|
| **数据准备** | 上传文件、选择字段、调整变量类型、管理表间关系、开始分析 |
| **预览报告** | 查看生成的 HTML 分析报告 |
| **小模型训练** | 训练分类、回归、聚类、时序模型 |
| **大模型智能解读** | AI 智能对话，可选择上下文（JSON结果、HTML报告、源数据） |
| **项目对比** | 并排对比两个分析项目 |

#### AI 解读功能

| 功能 | 说明 |
|------|------|
| **上下文选择** | 可选择 JSON 结果、HTML 报告或源数据作为分析上下文 |
| **自由提问** | 随意提问关于数据的问题 |
| **场景推荐** | 自动识别业务场景，获取分析视角 |
| **自然查询** | 使用自然语言查询数据 |
| **SQL生成** | 从自然语言生成 SQL 查询语句（仅数据库模式） |
| **Agent推理** | 自然语言调用模型进行预测 |
| **勾稽校验** | 用业务逻辑验证数据一致性规则 |

### 文本分析界面 (AutoText)

文本分析界面采用 **4标签页** 布局：

| 标签页 | 说明 |
|--------|------|
| **数据准备** | 输入或粘贴文本，配置分析选项 |
| **报告预览** | 查看生成的 HTML 文本分析报告 |
| **大模型解读** | AI 智能对话，获取文本洞察 |
| **项目对比** | 并排对比两个文本分析项目 |

---

## 模型训练

### 可用模型

#### 分类模型
- 逻辑回归、决策树、随机森林、XGBoost、LightGBM、CatBoost
- SVM、KNN、朴素贝叶斯、AdaBoost、梯度提升、MLP
- CNN、RNN、LSTM、BERT（深度学习）

#### 回归模型
- 线性回归、岭回归、Lasso、弹性网络
- 决策树、随机森林、XGBoost、LightGBM、SVR、KNN、MLP

#### 聚类模型
- K-Means、DBSCAN、层次聚类、BIRCH
- 高斯混合、OPTICS、谱聚类、均值漂移

#### 时间序列模型
- ARIMA、SARIMA、Prophet
- LSTM、GRU、Transformer（深度学习）

### 训练示例

```python
from autostat.models.trainer import ModelTrainer
from autostat.models.preprocessing import DataPreprocessor
from autostat.models.storage import ModelStorage
import pandas as pd

# 加载数据
data = pd.read_csv("data.csv")
features = ["age", "income", "score"]
target = "label"

# 预处理
preprocessor = DataPreprocessor({
    "missing_strategy": "drop",
    "scaling": "standard",
    "encoding": "onehot"
})

# 训练
trainer = ModelTrainer("classification", "random_forest", {"n_estimators": 100})
result = trainer.train(X_train, y_train, cv_folds=5)

# 保存模型
ModelStorage.save_model(
    session_id="my_session",
    model_key="rf_model_001",
    model=trainer.model,
    preprocessor=preprocessor,
    metrics=result["train_score"],
    config={"features": features, "target": target}
)

# 加载并预测
model, preprocessor, metadata, metrics = ModelStorage.load_model("my_session", "rf_model_001")
predictor = ModelPredictor(model, preprocessor)
prediction = predictor.predict({"age": 35, "income": 50000, "score": 85})
print(f"预测值: {prediction['prediction']}, 置信度: {prediction['confidence']}")
```

### Web界面训练

1. 进入 **小模型训练** 标签页
2. 选择任务类型（分类/回归/聚类/时序）
3. 选择目标列（有监督学习）
4. 选择特征
5. 选择模型
6. 配置参数
7. 点击 **开始训练**

---

## MCP服务

AutoStat 提供 MCP (Model Context Protocol) 服务，可供 AI Agent 调用。

### MCP 工具列表

| 工具名称 | 说明 |
|----------|------|
| `analyze_from_file` | 分析单个文件（CSV、Excel、JSON、TXT） |
| `analyze_multiple_files` | 批量分析多个文件，自动发现表间关系 |
| `analyze_from_db` | 从 SQL Server 数据库分析数据 |
| `get_data_quality_report` | 快速获取数据质量报告 |

### 启动 MCP 服务

```bash
# STDIO 模式（默认，用于 Claude Desktop）
autostat-mcp

# HTTP 模式
autostat-mcp --transport http --port 6011

# SSE 模式
autostat-mcp --transport sse --host 0.0.0.0 --port 6011
```

### Claude Desktop 配置

在 `claude_desktop_config.json` 中添加：

```json
{
  "mcpServers": {
    "autostat": {
      "command": "autostat-mcp"
    }
  }
}
```

或使用 `uvx`：

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

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `AUTOSTAT_HOST` | HTTP/SSE 服务绑定地址 | `0.0.0.0` |
| `AUTOSTAT_PORT` | HTTP/SSE 服务端口 | `6011` |

### AutoText MCP 服务

```bash
# STDIO 模式
autotext-mcp

# HTTP 模式
autotext-mcp --transport http --port 6012

# SSE 模式
autotext-mcp --transport sse --host 0.0.0.0 --port 6012
```

---

## 输出格式

### HTML 报告

HTML 报告包含：

- **数据概览** - 行数、列数、缺失值、重复值、日期范围
- **变量类型分布** - 分类、连续、日期、标识符
- **统计摘要** - 每个变量的均值、中位数、标准差、四分位数
- **可视化图表** - 直方图、箱线图、相关性热力图、时间序列图
- **数据质量告警** - 缺失值、异常值、重复值
- **清洗建议** - 推荐的数据清洗操作
- **模型推荐** - 基于数据特征的推荐模型
- **勾稽规则** - 发现的数值关系、函数依赖、时序约束
- **核心洞察** - 关键发现和建议行动

### JSON 输出结构

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

### 多表 JSON 输出

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

## 支持的数据源

| 格式 | 扩展名 | 方法 | 说明 |
|------|--------|------|------|
| CSV | `.csv` | `load_csv()` | 自动检测编码，处理特殊字符 |
| Excel | `.xlsx`, `.xls` | `load_excel()` | 支持 openpyxl 和 xlrd 引擎 |
| JSON | `.json` | `load_json()` | 支持 records、index、columns 格式 |
| TXT | `.txt` | `load_txt()` | 自动检测分隔符 |
| SQL Server | - | `load_sql_server()` | 支持 Windows 和 SQL 认证 |

---

## 勾稽规则发现

AutoStat 自动发现数据一致性规则。

### 规则类型

| 规则类型 | 说明 | 示例 |
|----------|------|------|
| **数值关系** | 字段间的数值关系 | `revenue = price * quantity` |
| **函数依赖** | 确定性关系 | `user_id → user_name` |
| **时序约束** | 日期/时间约束 | `payment_date ≥ order_date` |
| **外键约束** | 跨表引用完整性 | `orders.user_id → users.id` |
| **日期规则** | 工作日间隔、条件时序 | `payment_date = order_date + 2个工作日` |

### API 使用

```python
from autostat.core.audit import discover_audit_rules
from autostat.core.date_rules import discover_date_rules

# 发现勾稽规则
audit_rules = discover_audit_rules(
    data=df,
    variable_types=variable_types,
    foreign_keys=[],
    debug=True,
    min_confidence=0.5
)

# 发现日期规则
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

## API 参考

### AutoStatisticalAnalyzer

```python
class AutoStatisticalAnalyzer:
    def __init__(self, data, target_col=None, source_table_name=None,
                 predefined_types=None, auto_clean=False, quiet=False,
                 date_features_level="basic", skip_auto_inference=False):
        """
        参数:
        - data: DataFrame 或文件路径字符串
        - target_col: 目标列名（可选）
        - source_table_name: 源表名
        - predefined_types: 预定义的变量类型字典
        - auto_clean: 是否自动清洗数据
        - quiet: 静默模式（无控制台输出）
        - date_features_level: 日期特征级别 (none/basic/full)
        - skip_auto_inference: 跳过自动类型推断
        """

    def generate_full_report(self, show_outlier_details=False):
        """生成完整分析报告"""

    def to_json(self, output_file=None, indent=2, ensure_ascii=False):
        """导出 JSON 结果"""

    def auto_time_series_analysis(self, max_numeric=10, group_by='auto'):
        """自动检测和分析时间序列"""

    def auto_analyze_relationships(self):
        """自动分析变量关系"""

    def recommend_scenarios(self):
        """推荐分析场景和模型"""
```

### MultiTableStatisticalAnalyzer

```python
class MultiTableStatisticalAnalyzer:
    def __init__(self, tables, relationships=None, date_features_level="basic",
                 predefined_types=None):
        """
        参数:
        - tables: {表名: DataFrame} 字典
        - relationships: 包含 'foreign_keys' 列表的字典
        - date_features_level: 日期特征级别
        - predefined_types: {表名: {列名: 类型}} 字典
        """

    @classmethod
    def from_files(cls, table_files, relationships=None, **kwargs):
        """从文件加载表"""

    @classmethod
    def from_json_strings(cls, json_strings, relationships=None, **kwargs):
        """从 JSON 字符串加载表"""

    def analyze_all_tables(self):
        """分析所有表和关系"""

    def get_merged_analyzer(self):
        """获取合并后的分析器用于报告生成"""

    def to_html(self, output_file=None, title="多表分析报告"):
        """生成 HTML 报告"""

    def to_json(self, output_file=None, indent=2, ensure_ascii=False):
        """导出 JSON 结果"""
```

---

## 环境要求

### 核心要求

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scipy, statsmodels, networkx
- click, jinja2

### 可选依赖

| 依赖 | 用途 |
|------|------|
| `streamlit` | Web 界面 |
| `fastmcp` | MCP 服务 |
| `pyodbc` | SQL Server 支持 |
| `openpyxl` | Excel 导出 |
| `jieba` | 中文分词 |
| `nltk` | 英文文本处理 |
| `snownlp` | 中文情感分析 |
| `transformers` | BERT 模型 |
| `tensorflow` | 深度学习模型 |
| `xgboost` | XGBoost 模型 |
| `lightgbm` | LightGBM 模型 |
| `catboost` | CatBoost 模型 |
| `prophet` | Prophet 时序模型 |

---

## 项目结构

```
autostat/
├── autostat/                          # 核心分析引擎
│   ├── __init__.py
│   ├── __main__.py                    # python -m autostat 入口
│   ├── analyzer.py                    # 主分析器（向后兼容）
│   ├── checker.py                     # 条件检查器
│   ├── cli.py                         # 命令行入口
│   ├── config_manager.py              # 配置管理
│   ├── llm_client.py                  # 大模型客户端
│   ├── loader.py                      # 数据加载器
│   ├── mcp_server.py                  # MCP 服务
│   ├── multi_analyzer.py              # 多表分析器
│   ├── prompts.py                     # 提示词模板
│   ├── reporter.py                    # 报告生成器
│   ├── core/                          # 核心模块
│   │   ├── __init__.py
│   │   ├── analyzer.py                # 主分析器实现
│   │   ├── audit.py                   # 勾稽规则发现
│   │   ├── base.py                    # 基础分析（类型识别、质量检查）
│   │   ├── date_rules.py              # 日期规则发现
│   │   ├── plots.py                   # 可视化
│   │   ├── recommendation.py          # 模型推荐
│   │   ├── relationship.py            # 关系分析
│   │   ├── report_data.py             # 报告数据构建
│   │   └── timeseries.py              # 时间序列分析
│   └── models/                        # 机器学习模型模块
│       ├── __init__.py
│       ├── arima_wrapper.py           # ARIMA 包装器
│       ├── deep_learning.py           # CNN、RNN、LSTM、BERT、Transformer
│       ├── metrics.py                 # 评估指标
│       ├── predictor.py               # 模型推理
│       ├── preprocessing.py           # 数据预处理
│       ├── registry.py                # 模型注册表
│       ├── storage.py                 # 模型存储
│       └── trainer.py                 # 模型训练
│
├── autotext/                          # 文本分析引擎
│   ├── __init__.py
│   ├── analyzer.py                    # 主分析器
│   ├── checker.py                     # 条件检查器
│   ├── cli.py                         # 命令行入口
│   ├── llm_extractor.py               # 大模型信息抽取
│   ├── llm_visualizer.py              # 抽取结果可视化
│   ├── loader.py                      # 文本数据加载器
│   ├── mcp_server.py                  # MCP 服务
│   ├── reporter.py                    # 报告生成器
│   └── core/                          # 核心文本模块
│       ├── __init__.py
│       ├── cache.py                   # 缓存管理
│       ├── cluster.py                 # 文本聚类
│       ├── detector.py                # 字段检测
│       ├── entity.py                  # 实体识别
│       ├── entity_profile.py          # 实体档案构建
│       ├── event_extractor.py         # 事件抽取
│       ├── event_timeline.py          # 事件脉络分析
│       ├── graph_analyzer.py          # 图分析
│       ├── graph_builder.py           # 图构建
│       ├── info_extractor.py          # 信息抽取
│       ├── insight.py                 # 洞察发现
│       ├── keyword_extractor.py       # 关键词提取
│       ├── llm_enhance.py             # 大模型增强
│       ├── llm_graph_extractor.py     # 大模型图谱抽取
│       ├── ner.py                     # 命名实体识别
│       ├── preprocessor.py            # 文本预处理
│       ├── quality.py                 # 文本质量检查
│       ├── relation.py                # 关系发现
│       ├── relation_mining.py         # 关联规则挖掘
│       ├── sentiment.py               # 情感分析
│       ├── stats.py                   # 文本统计
│       ├── summarizer.py              # 文本摘要
│       ├── timeline_builder.py        # 时间线构建
│       ├── topic.py                   # 主题建模
│       ├── topic_model.py             # LDA 主题建模
│       ├── trend.py                   # 趋势分析
│       ├── trend_detector.py          # 趋势检测
│       └── vectorizer.py              # BERT 向量化
│
├── web/                               # Web 界面
│   ├── __init__.py
│   ├── app.py                         # Streamlit 主入口
│   ├── components/                    # UI 组件
│   │   ├── __init__.py
│   │   ├── agent_inference.py         # Agent 推理
│   │   ├── audit_rule.py              # 勾稽规则 UI
│   │   ├── chat_interface.py          # AI 聊天界面
│   │   ├── data_preparation.py        # 数据准备 UI
│   │   ├── demo_data.py               # 演示数据
│   │   ├── empty_state.py             # 空状态引导
│   │   ├── model_training.py          # 模型训练 UI
│   │   ├── natural_query.py           # 自然语言查询
│   │   ├── progress_stage.py          # 分阶段进度
│   │   ├── results.py                 # 结果展示
│   │   ├── scenario_recommendation.py # 场景推荐
│   │   ├── sidebar.py                 # 侧边栏
│   │   ├── sql_generator.py           # SQL 生成器
│   │   ├── tabs.py                    # 标签页导航
│   │   ├── term_tooltip.py            # 术语解释
│   │   ├── tips.py                    # 使用技巧
│   │   └── value_preview.py           # 价值预览
│   ├── config/                        # 客户端存储
│   │   └── storage.py                 # 配置管理
│   ├── services/                      # 业务逻辑层
│   │   ├── __init__.py
│   │   ├── agent_service.py           # Agent 服务
│   │   ├── analysis_service.py        # 分析执行
│   │   ├── auto_interpret_service.py  # 自动解读
│   │   ├── auto_train_service.py      # 自动训练
│   │   ├── cache_service.py           # 缓存管理
│   │   ├── error_handler.py           # 错误处理
│   │   ├── feature_flags.py           # 功能开关
│   │   ├── file_service.py            # 文件操作
│   │   ├── insight_service.py         # 洞察生成
│   │   ├── model_training_service.py  # 模型训练服务
│   │   ├── recommendation_service.py  # 推荐服务
│   │   ├── session_service.py         # 会话管理
│   │   └── storage_service.py         # 统一存储
│   └── utils/                         # 工具函数
│       ├── __init__.py
│       ├── data_preprocessor.py       # 数据预处理工具
│       └── helpers.py                 # 辅助函数
│
├── webtext/                           # 文本分析 Web 界面
│   ├── __init__.py
│   ├── app.py                         # Streamlit 入口
│   ├── components/                    # UI 组件
│   │   ├── __init__.py
│   │   ├── chat_interface.py          # 聊天界面
│   │   ├── compare.py                 # 对比组件
│   │   ├── data_preparation.py        # 数据准备
│   │   ├── results.py                 # 结果展示
│   │   ├── sidebar.py                 # 侧边栏
│   │   └── tabs.py                    # 标签页导航
│   └── services/                      # 文本服务
│       ├── __init__.py
│       ├── analysis_service.py        # 分析服务
│       └── session_service.py         # 会话管理
│
├── pages/                             # Streamlit 页面
│   ├── Home.py                        # 首页
│   ├── autostat.py                    # 数据分析页面
│   └── autotext.py                    # 文本分析页面
│
├── templates/                         # HTML 模板
│   ├── report.html                    # 数据分析报告
│   └── report_text.html               # 文本分析报告
│
├── tests/                             # 单元测试
│   └── test_analyzer.py
│
├── examples/                          # 使用示例
│   ├── example.py                     # 数据分析示例
│   └── example_text.py                # 文本分析示例
│
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
├── README_cn.md
└── requirements.txt
```

---

## 许可证

MIT License

版权所有 (c) 2024 AutoStat Team

特此授权，免费向任何获得本软件及相关文档文件（"软件"）副本的人授予使用本软件的权限，包括但不限于使用、复制、修改、合并、发布、分发、再许可和/或销售本软件副本的权利，并允许获得本软件的人这样做，但须遵守以下条件：

上述版权声明和本许可声明应包含在本软件的所有副本或实质性部分中。

本软件按"原样"提供，不作任何明示或暗示的保证，包括但不限于适销性、特定用途适用性和非侵权性的保证。在任何情况下，作者或版权持有人均不对因本软件或本软件的使用或其他交易引起的任何索赔、损害或其他责任负责，无论是在合同、侵权或其他方面。

---

## 贡献

欢迎贡献代码！请提交 Pull Request。

1. Fork 本仓库
2. 创建你的功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交你的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开 Pull Request

---

## 联系方式

- Issues: https://github.com/zyhowe/AutoStat/issues
- 邮箱: howe_min@163.com
- 仓库: https://github.com/zyhowe/AutoStat