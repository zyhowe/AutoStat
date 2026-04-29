"""
文本分析使用示例
"""

import sys
import os
import pandas as pd
from datetime import datetime,timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autotext.analyzer import TextAnalyzer, analyze_texts, analyze_file, analyze_folder
from autotext.reporter import TextReporter


def generate_sample_texts():
    """生成示例文本（短文本评论）"""
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
    return texts


def generate_long_texts():
    """生成示例文本（长文本新闻）"""
    return [
        "新华社北京1月15日电 国家统计局今日发布数据显示，2023年全年国内生产总值1260582亿元，按不变价格计算，比上年增长5.2%。其中，第一产业增加值89755亿元，比上年增长4.1%；第二产业增加值482589亿元，增长4.7%；第三产业增加值688238亿元，增长5.8%。",
        "中国人民银行决定下调金融机构存款准备金率0.25个百分点。本次下调后，金融机构加权平均存款准备金率约为7.6%。预计此次降准将释放长期资金约5000亿元，有助于降低实体经济融资成本。",
        "中国证监会召开2024年系统工作会议，会议强调要全力维护资本市场稳定运行，把资本市场稳定运行放在更加突出的位置，着力稳市场、稳信心、稳预期。",
        "宁德时代发布2023年度业绩预告，预计全年净利润425亿元至455亿元，同比增长38%至48%。报告期内，公司新技术、新产品陆续落地，海外市场拓展加速，产销量较快增长。",
        "华为技术有限公司与长安汽车签署投资合作备忘录，拟成立一家新公司，聚焦智能网联汽车的智能驾驶系统及增量部件的研发、生产、销售和服务。",
        "比亚迪股份有限公司2023年新能源汽车销量302.44万辆，同比增长62.3%，超额完成300万辆年度目标，夺得全球新能源汽车销量冠军。",
        "腾讯控股宣布斥资10亿港元回购股份，这是今年以来第10次回购。公司表示对未来发展充满信心，将继续加大在人工智能领域的投入。",
        "阿里巴巴集团旗下淘天集团调整组织架构，成立三个行业发展部，分别负责各个垂直行业的运营和发展，以应对日益激烈的市场竞争。",
    ]


def generate_texts_with_dates():
    """生成带时间戳的文本（用于事件脉络分析）"""
    base_date = datetime(2024, 1, 1)
    texts = []
    dates = []

    events = [
        ("华为发布Mate 60系列手机，搭载自研麒麟芯片", 1),
        ("华为Mate 60销量突破100万台，市场反响热烈", 5),
        ("华为宣布鸿蒙系统升级至4.0版本", 10),
        ("华为与赛力斯合作推出问界M9", 15),
        ("华为2023年营收超7000亿元，同比增长9%", 20),
        ("华为云发布盘古大模型3.0", 25),
        ("华为Mate 60 Pro降价促销", 30),
        ("宁德时代与华为签署战略合作协议", 35),
        ("华为花瓣支付正式上线", 40),
        ("华为秋季全场景新品发布会召开", 45),
    ]

    for event, days in events:
        texts.append(event)
        dates.append(base_date + timedelta(days=days))

    return texts, dates


def print_separator(title):
    """打印分隔符"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def example_single_text():
    """示例1: 分析单条文本"""
    print_separator("示例1: 单条文本分析")

    text = "腾讯控股2024年第一季度营收1595亿元，同比增长6%，净利润502亿元，同比增长54%。"

    print(f"📝 输入文本: {text}\n")

    analyzer = analyze_texts(
        texts=[text],
        output_file="output/single_text_report.html",
        format="html",
        quiet=False,
        use_bert=True
    )

    print(f"\n📊 分析结果:")
    print(f"  文本长度: {len(text)} 字符")
    print(f"  情感倾向: {analyzer.sentiment_distribution}")

    if analyzer.entity_stats:
        print(f"  实体识别:")
        for entity_type in ['per', 'org', 'loc']:
            count = analyzer.entity_stats.get(entity_type, {}).get('unique', 0)
            if count > 0:
                print(f"    - {entity_type}: {count} 个")


def example_multiple_texts():
    """示例2: 分析多条文本（短文本评论）"""
    print_separator("示例2: 多条文本分析（评论数据）")

    texts = generate_sample_texts()

    print(f"📝 共 {len(texts)} 条评论:\n")
    for i, t in enumerate(texts, 1):
        print(f"  {i}. {t[:50]}...")

    analyzer = analyze_texts(
        texts=texts,
        output_file="output/review_analysis_report.html",
        format="html",
        quiet=False,
        use_bert=True
    )

    print(f"\n📊 分析结果:")
    print(f"  总文本数: {analyzer.stats_result['total_count']}")
    print(f"  平均长度: {analyzer.stats_result['char_length']['mean']:.1f} 字符")
    print(f"  积极文本: {analyzer.sentiment_distribution['positive_rate']:.1%}")
    print(f"  消极文本: {analyzer.sentiment_distribution['negative_rate']:.1%}")

    if analyzer.keywords:
        print(f"\n  高频关键词:")
        for word, count in analyzer.keywords.get('frequency', [])[:5]:
            print(f"    - {word}: {count}次")

    if analyzer.insights:
        print(f"\n  💡 核心洞察:")
        for insight in analyzer.insights[:3]:
            print(f"    - {insight['title']}")


def example_long_texts():
    """示例3: 分析长文本（财经新闻）"""
    print_separator("示例3: 长文本分析（财经新闻）")

    texts = generate_long_texts()

    print(f"📝 共 {len(texts)} 条财经新闻\n")
    for i, t in enumerate(texts, 1):
        print(f"  {i}. {t[:60]}...")

    analyzer = analyze_texts(
        texts=texts,
        output_file="output/finance_news_report.html",
        format="html",
        quiet=False,
        use_bert=True
    )

    print(f"\n📊 分析结果:")
    print(f"  总文本数: {analyzer.stats_result['total_count']}")
    print(f"  平均长度: {analyzer.stats_result['char_length']['mean']:.1f} 字符")

    if analyzer.entity_stats:
        print(f"\n  🏷️ 实体识别统计:")
        for entity_type in ['org', 'per', 'loc']:
            stats = analyzer.entity_stats.get(entity_type, {})
            if stats.get('unique', 0) > 0:
                top_entities = stats.get('top', [])[:5]
                top_names = [name for name, _ in top_entities]
                print(f"    - {entity_type}: {stats['unique']} 个实体, 高频: {', '.join(top_names)}")

    if analyzer.topics:
        print(f"\n  📚 主题识别:")
        for topic in analyzer.topics[:3]:
            keywords = ', '.join(topic.get('keywords', [])[:5])
            print(f"    - 主题{topic['topic_id']}: {keywords}")


def example_text_with_dates():
    """示例4: 带时间戳的文本分析（事件脉络）"""
    print_separator("示例4: 带时间戳文本分析（事件脉络）")

    texts, dates = generate_texts_with_dates()

    print(f"📝 共 {len(texts)} 条带时间戳的文本:\n")
    for i, (text, date) in enumerate(zip(texts, dates), 1):
        print(f"  {i}. [{date.strftime('%Y-%m-%d')}] {text[:50]}...")

    analyzer = TextAnalyzer(
        texts,
        source_name="华为事件追踪",
        quiet=False,
        use_bert=True
    )
    analyzer.dates = dates  # 设置时间戳
    analyzer.generate_full_report()

    # 生成报告
    os.makedirs("output", exist_ok=True)
    analyzer.to_html("output/event_timeline_report.html", title="华为事件脉络分析报告")
    analyzer.to_json("output/event_timeline_result.json")

    print(f"\n📊 分析结果:")
    print(f"  时间范围: {min(dates).strftime('%Y-%m-%d')} ~ {max(dates).strftime('%Y-%m-%d')}")

    if analyzer.event_timeline and 'error' not in analyzer.event_timeline:
        print(f"  事件脉络: 已识别 {len(analyzer.event_timeline.get('hot_topics', []))} 个热点")
        sentiment = analyzer.event_timeline.get('sentiment_evolution', {})
        if sentiment.get('overall_sentiment'):
            print(f"  整体情感: {sentiment['overall_sentiment']:.2f}")


def example_from_csv_file():
    """示例5: 从CSV文件读取文本分析"""
    print_separator("示例5: 从CSV文件读取文本分析")

    # CSV 文件路径
    csv_file = "results/sina_finance_news.csv"

    # 检查文件是否存在
    if not os.path.exists(csv_file):
        print(f"⚠️ 文件不存在: {csv_file}")
        print("   请先运行 fetch_sina_finance.py 生成数据")
        return

    from autostat.llm_client import LLMClient

    # 1. 配置大模型
    config = {
        "api_base": "https://api.deepseek.com/v1",  # 或其他兼容 API
        "api_key": "sk-c0e1f1ad1a3b41429a92f29251775ecf",
        "model": "deepseek-chat"
    }
    llm_client = LLMClient(config)


    # 从CSV读取并分析
    # 方式1: 直接传入文件路径和列名
    analyzer = TextAnalyzer(
        data=csv_file,  # CSV 文件路径
        text_col='content',  # 文本列名
        title_col='title',  # 标题列名（可选）
        time_col='date',  # 时间列名（可选）
        source_name="财经新闻分析",
        quiet=False,
        use_bert=True
    )
    analyzer.set_llm_client(llm_client)
    analyzer.generate_full_report()

    # 生成报告
    os.makedirs("output", exist_ok=True)
    analyzer.to_html("output/csv_analysis_report.html")
    analyzer.to_json("output/csv_analysis_result.json")

    print(f"\n📊 分析完成: {analyzer.stats_result['total_count']} 条文本")
    print(f"  平均长度: {analyzer.stats_result['char_length']['mean']:.1f} 字符")
    print(f"  情感分布: 积极 {analyzer.sentiment_distribution['positive_rate']:.1%}, "
          f"消极 {analyzer.sentiment_distribution['negative_rate']:.1%}")

    if analyzer.entity_stats:
        orgs = analyzer.entity_stats.get('org', {}).get('unique', 0)
        print(f"  识别组织: {orgs} 个")


def example_from_txt_file():
    """示例6: 从TXT文件读取文本分析"""
    print_separator("示例6: 从TXT文件读取文本分析")

    # 创建示例TXT文件
    txt_file = "output/sample_texts.txt"
    os.makedirs("output", exist_ok=True)

    texts = generate_long_texts()
    with open(txt_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + "\n\n")
    print(f"✅ 已创建示例TXT文件: {txt_file}")
    print(f"   包含 {len(texts)} 条文本")

    # 从TXT读取并分析
    analyzer = analyze_file(
        file_path=txt_file,
        output_file="output/txt_analysis_report.html",
        format="html",
        quiet=False,
        use_bert=True
    )

    print(f"\n📊 分析完成: {analyzer.stats_result['total_count']} 条文本")


def example_from_folder():
    """示例7: 分析文件夹中的所有文本文件"""
    print_separator("示例7: 分析文件夹中的文本文件")

    # 创建示例文件夹和文件
    folder_path = "output/sample_texts_folder"
    os.makedirs(folder_path, exist_ok=True)

    texts_by_file = {
        "news_1.txt": generate_long_texts()[:3],
        "news_2.txt": generate_long_texts()[3:6],
        "comments.txt": generate_sample_texts()
    }

    for filename, file_texts in texts_by_file.items():
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            for text in file_texts:
                f.write(text + "\n\n")
    print(f"✅ 已创建示例文件夹: {folder_path}")
    print(f"   包含 {len(texts_by_file)} 个文件")

    # 分析整个文件夹
    analyzer = analyze_folder(
        folder_path=folder_path,
        output_file="output/folder_analysis_report.html",
        format="html",
        quiet=False,
        use_bert=True
    )

    print(f"\n📊 分析完成: {analyzer.stats_result['total_count']} 条文本")


def example_dataframe_analysis():
    """示例8: 从DataFrame分析文本"""
    print_separator("示例8: 从DataFrame分析文本")

    # 创建DataFrame
    df = pd.DataFrame({
        'title': ['产品A评价', '产品B评价', '产品C评价', '产品D评价', '产品E评价'],
        'content': [
            "非常好用，值得购买，客服态度也很好。",
            "质量太差了，用了一天就坏了，差评。",
            "性价比很高，物流也很快，五星好评。",
            "一般般吧，没有想象中好，但也能接受。",
            "包装破损，产品也有划痕，非常不满意。"
        ],
        'date': pd.date_range('2024-01-01', periods=5),
        'rating': [5, 1, 5, 3, 2]
    })

    print(f"📊 DataFrame 信息:")
    print(f"  形状: {df.shape}")
    print(f"  列名: {list(df.columns)}")
    print(f"\n数据预览:\n{df.to_string()}\n")

    # 从DataFrame分析
    analyzer = TextAnalyzer(
        data=df,
        text_col='content',
        title_col='title',
        time_col='date',
        source_name="产品评价数据",
        quiet=False,
        use_bert=True
    )
    analyzer.generate_full_report()

    # 生成多种格式报告
    analyzer.to_html("output/dataframe_report.html")
    analyzer.to_json("output/dataframe_result.json")
    analyzer.to_markdown("output/dataframe_report.md")

    print(f"\n📊 分析完成:")
    print(f"  总文本数: {analyzer.stats_result['total_count']}")
    print(f"  平均长度: {analyzer.stats_result['char_length']['mean']:.1f} 字符")
    print(f"  积极文本: {analyzer.sentiment_distribution['positive_rate']:.1%}")


def example_no_bert():
    """示例9: 不使用BERT增强（仅基础分析）"""
    print_separator("示例9: 基础分析模式（不使用BERT）")

    texts = generate_long_texts()

    print(f"📝 共 {len(texts)} 条文本")
    print("  模式: 仅基础分析（关键词、情感、质量检查）\n")

    analyzer = analyze_texts(
        texts=texts,
        output_file="output/basic_analysis_report.html",
        format="html",
        quiet=False,
        use_bert=False  # 关闭BERT增强
    )

    print(f"\n📊 分析结果:")
    print(f"  总文本数: {analyzer.stats_result['total_count']}")
    print(f"  平均长度: {analyzer.stats_result['char_length']['mean']:.1f} 字符")
    print(f"  积极文本: {analyzer.sentiment_distribution['positive_rate']:.1%}")
    print(f"  消极文本: {analyzer.sentiment_distribution['negative_rate']:.1%}")

    if analyzer.keywords:
        print(f"\n  高频关键词:")
        for word, count in analyzer.keywords.get('frequency', [])[:10]:
            print(f"    - {word}: {count}次")


def example_export_all_formats():
    """示例10: 导出所有格式报告"""
    print_separator("示例10: 导出多种格式报告")

    texts = generate_long_texts()

    analyzer = analyze_texts(
        texts=texts,
        quiet=True,
        use_bert=True
    )

    # 导出多种格式
    base_name = "output/full_report"
    os.makedirs("output", exist_ok=True)

    analyzer.to_html(f"{base_name}.html", title="完整分析报告")
    analyzer.to_json(f"{base_name}.json")
    analyzer.to_markdown(f"{base_name}.md")

    # 保存各阶段数据
    analyzer.save_raw_texts(f"{base_name}_raw.txt")
    analyzer.save_cleaned_texts(f"{base_name}_cleaned.txt")
    analyzer.save_content_texts(f"{base_name}_content.txt")
    analyzer.save_filtered_texts(f"{base_name}_filtered.txt")
    analyzer.save_templates(f"{base_name}_templates.txt")

    print(f"✅ 报告已导出到 output/ 目录:")
    print(f"   - full_report.html (HTML报告)")
    print(f"   - full_report.json (JSON数据)")
    print(f"   - full_report.md (Markdown报告)")
    print(f"   - full_report_raw.txt (原始文本)")
    print(f"   - full_report_cleaned.txt (清洗后文本)")
    print(f"   - full_report_content.txt (正文文本)")
    print(f"   - full_report_filtered.txt (过滤后文本)")
    print(f"   - full_report_templates.txt (模板词)")


def main_menu():
    """主菜单"""
    print("\n" + "=" * 70)
    print("  AutoText 文本分析工具 - 示例程序")
    print("=" * 70)
    print("""
请选择要运行的示例:

  1. 单条文本分析
  2. 多条文本分析（短文本评论）
  3. 长文本分析（财经新闻）
  4. 带时间戳文本分析（事件脉络）
  5. 从CSV文件读取分析
  6. 从TXT文件读取分析
  7. 分析文件夹中的文本文件
  8. 从DataFrame分析文本
  9. 基础分析模式（不使用BERT）
 10. 导出所有格式报告
 11. 运行所有示例
 0. 退出

注意: 首次运行会下载BERT模型（约400MB），请耐心等待
    """)


if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)

    # 可选：直接运行特定示例（取消注释即可）
    #example_single_text()
    #example_multiple_texts()
    #example_long_texts()
    #example_text_with_dates()
    example_from_csv_file()
    # example_from_txt_file()
    # example_from_folder()
    # example_dataframe_analysis()
    # example_no_bert()
    # example_export_all_formats()

