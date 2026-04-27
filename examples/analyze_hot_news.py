"""
分析热点新闻数据示例
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from autotext.analyzer import TextAnalyzer
from autotext.reporter import TextReporter


def analyze_hot_news():
    """分析热点新闻数据"""
    print("=" * 60)
    print("热点新闻分析示例")
    print("=" * 60)

    # 读取数据
    data_path = os.path.join(os.path.dirname(__file__), "hot_news_1000.csv")

    if not os.path.exists(data_path):
        print(f"❌ 文件不存在: {data_path}")
        print("请先运行 generate_hot_news.py 生成数据")
        return

    df = pd.read_csv(data_path)
    print(f"✅ 加载数据: {len(df)} 条新闻")

    # 创建分析器，指定字段
    analyzer = TextAnalyzer(
        data=df,
        text_col="content",
        title_col="title",
        time_col="date",
        source_name="热点新闻",
        quiet=False
    )

    # 执行分析
    analyzer.generate_full_report()

    # 生成报告
    reporter = TextReporter(analyzer)
    reporter.to_html("hot_news_report.html")
    reporter.to_json("hot_news_result.json")

    print("\n✅ 报告已生成: hot_news_report.html, hot_news_result.json")


def analyze_sample_comments():
    """分析评论数据"""
    print("=" * 60)
    print("评论数据分析示例")
    print("=" * 60)

    # 读取数据
    data_path = os.path.join(os.path.dirname(__file__), "sample_comments_1000.csv")

    if not os.path.exists(data_path):
        print(f"❌ 文件不存在: {data_path}")
        print("请先运行 generate_sample_comments.py 生成数据")
        return

    df = pd.read_csv(data_path)
    print(f"✅ 加载数据: {len(df)} 条评论")

    # 创建分析器
    analyzer = TextAnalyzer(
        data=df,
        text_col="content",
        title_col="title",
        time_col="date",
        source_name="商品评论",
        quiet=False
    )

    # 执行分析
    analyzer.generate_full_report()

    # 生成报告
    reporter = TextReporter(analyzer)
    reporter.to_html("comments_report.html")
    reporter.to_json("comments_result.json")

    print("\n✅ 报告已生成: comments_report.html, comments_result.json")


if __name__ == "__main__":
    # 先运行数据生成（如果还没有生成）
    print("1. 生成评论数据...")
    os.system("python generate_sample_comments.py")

    print("\n2. 生成热点新闻数据...")
    os.system("python generate_hot_news.py")

    print("\n3. 分析评论数据...")
    analyze_sample_comments()

    print("\n4. 分析热点新闻数据...")
    analyze_hot_news()