"""
文本分析使用示例
"""

import sys
import os
import pandas as pd
import random
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autotext.analyzer import TextAnalyzer
from autotext.reporter import TextReporter


def generate_sample_texts():
    """生成示例文本"""
    texts = [
        "这款手机电池续航很差，屏幕显示效果不错，但价格太贵了。",
        "客服态度很好，解决问题很迅速，非常满意！",
        "物流太慢了，等了一周才收到，包装也有破损。",
        "产品质量很好，做工精细，用起来很顺手，推荐购买。",
        "售后太差了，打了三次电话都没人接，再也不买这家的了。",
        "价格实惠，性价比很高，已经推荐给朋友了。",
        "不太满意，和描述不符，申请退货还被拒绝了。",
        "东西很不错，快递也给力，五星好评！",
    ]
    print(f"生成文本数量: {len(texts)}")  # 添加调试
    return texts


def example_basic_analysis():
    """基础分析示例"""
    print("=" * 60)
    print("示例1: 基础文本分析")
    print("=" * 60)

    texts = generate_sample_texts()
    analyzer = TextAnalyzer(texts, source_name="示例评论数据", quiet=False)
    analyzer.generate_full_report()

    print(f"\n📊 分析结果:")
    print(f"  总文本数: {analyzer.stats_result['total_count']}")
    print(f"  平均长度: {analyzer.stats_result['char_length']['mean']:.1f} 字符")
    print(f"  积极文本: {analyzer.sentiment_distribution['positive_rate']:.1%}")
    print(f"  消极文本: {analyzer.sentiment_distribution['negative_rate']:.1%}")


def example_html_report():
    """生成 HTML 报告示例"""
    print("\n" + "=" * 60)
    print("示例2: 生成 HTML 报告")
    print("=" * 60)

    texts = generate_sample_texts()
    analyzer = TextAnalyzer(texts, quiet=True)
    analyzer.generate_full_report()

    reporter = TextReporter(analyzer)
    reporter.to_html("results\\text_report_example.html")
    print("✅ 报告已生成: text_report_example.html")


def example_json_output():
    """生成 JSON 输出示例"""
    print("\n" + "=" * 60)
    print("示例3: 生成 JSON 输出")
    print("=" * 60)

    texts = generate_sample_texts()
    analyzer = TextAnalyzer(texts, quiet=True)
    analyzer.generate_full_report()

    reporter = TextReporter(analyzer)
    reporter.to_json("results\\results\\text_result_example.json")
    print("✅ JSON 已生成: text_result_example.json")


def example_from_file():
    """从文件分析示例"""
    print("\n" + "=" * 60)
    print("示例4: 从文件分析")
    print("=" * 60)

    # 创建示例文件
    texts = generate_sample_texts()
    with open("results\\sample_texts.txt", "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")

    analyzer = TextAnalyzer("results\\sample_texts.txt", quiet=True)
    analyzer.generate_full_report()

    print(f"📊 分析完成，共 {analyzer.stats_result['total_count']} 条文本")

    # 清理
    os.remove("results\\sample_texts.txt")
    print("✅ 临时文件已清理")


def example_keyword_extraction():
    """关键词提取示例"""
    print("\n" + "=" * 60)
    print("示例5: 关键词提取")
    print("=" * 60)

    texts = generate_sample_texts()
    analyzer = TextAnalyzer(texts, quiet=True)
    analyzer._preprocess()
    analyzer._extract_keywords()

    print("高频关键词 TOP 10:")
    for i, (word, count) in enumerate(analyzer.keywords.get("frequency", [])[:10], 1):
        print(f"  {i:2d}. {word}: {count}")


def example_sentiment_analysis():
    """情感分析示例"""
    print("\n" + "=" * 60)
    print("示例6: 情感分析")
    print("=" * 60)

    texts = generate_sample_texts()
    analyzer = TextAnalyzer(texts, quiet=True)
    analyzer._preprocess()
    analyzer._analyze_sentiment()

    print(f"情感分布:")
    print(f"  积极: {analyzer.sentiment_distribution['positive_rate']:.1%}")
    print(f"  消极: {analyzer.sentiment_distribution['negative_rate']:.1%}")
    print(f"  中性: {analyzer.sentiment_distribution['neutral_rate']:.1%}")


def example_entity_recognition():
    """实体识别示例"""
    print("\n" + "=" * 60)
    print("示例7: 实体识别")
    print("=" * 60)

    texts = generate_sample_texts()
    analyzer = TextAnalyzer(texts, quiet=True)
    analyzer._preprocess()
    analyzer._recognize_entities()

    print("实体统计:")
    for entity_type in ["person", "location", "organization"]:
        stats = analyzer.entity_stats.get(entity_type, {})
        print(f"  {entity_type}: {stats.get('unique', 0)} 个")


def analyze_sina_finance_news():
    """分析热点新闻数据"""
    print("=" * 60)
    print("热点新闻分析示例")
    print("=" * 60)

    # 读取数据
    data_path = os.path.join(os.path.dirname(__file__), "results\\sina_finance_news.csv")

    if not os.path.exists(data_path):
        print(f"❌ 文件不存在: {data_path}")
        print("请先运行 fetch_sina_finance.py 生成数据")
        return

    df = pd.read_csv(data_path)
    print(f"✅ 加载数据: {len(df)} 条新闻")

    # 创建分析器，指定字段
    analyzer = TextAnalyzer(
        data=df,
        text_col="content",
        title_col="title",
        time_col="date",
        source_name="财经新闻",
        quiet=False
    )

    # 执行分析
    analyzer.generate_full_report()

    # 保存各阶段数据，便于检查
    analyzer.save_raw_texts("results\\0_raw_texts.txt")
    analyzer.save_cleaned_texts("results\\1_cleaned_texts.txt")
    analyzer.save_content_texts("results\\2_content_texts.txt")
    analyzer.save_filtered_texts("results\\3_filtered_texts.txt")
    analyzer.save_templates("results\\templates.txt")

    # 生成报告
    reporter = TextReporter(analyzer)
    reporter.to_html("results\\sina_finance_news.html")
    reporter.to_json("results\\sina_finance_news.json")

    print("\n✅ 报告已生成: sina_finance_news.html, sina_finance_news.json")

if __name__ == "__main__":
    # 运行示例
    #example_basic_analysis()
    #example_html_report()
    analyze_sina_finance_news()
    #example_json_output()
    #example_from_file()
    # example_keyword_extraction()
    # example_sentiment_analysis()
    # example_entity_recognition()