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

    #text = "腾讯控股2024年第一季度营收1595亿元，同比增长6%，净利润502亿元，同比增长54%。"
    text = """
       2024年3月15日，宁德时代新能源科技股份有限公司（以下简称"宁德时代"）发布2023年年度报告。
       报告显示，公司2023年全年营业收入达到4009亿元，同比增长22.01%；归母净利润为441亿元，同比增长43.58%。
       同时，宁德时代宣布拟向全体股东每10股派发现金红利50元。

       此外，宁德时代与特斯拉于3月10日签署了战略合作协议，双方将在电池供应和技术研发方面展开深度合作。
       根据协议，宁德时代将为特斯拉上海工厂提供磷酸铁锂电池，协议期为2024年至2026年。

       受此消息影响，宁德时代股价次日上涨5.6%。
       """

    text="""
       4月份，受国际原油价格变动和假期出行需求增加影响，居民消费价格指数（CPI）环比上涨0.3%，同比上涨1.2%，扣除食品和能源价格的核心CPI同比上涨1.2%，保持温和回升。受国际大宗商品价格快速上涨，国内部分行业需求增加、市场竞争秩序不断优化等因素影响，工业生产者出厂价格指数（PPI）环比上涨1.7%，同比上涨2.8%，涨幅比上月均有所扩大。

一、CPI温和回升

从环比看，全国CPI环比由上月下降0.7%转为上涨0.3%，高于季节性水平0.4个百分点，主要受能源和出行服务价格上涨影响。受国际原油价格波动影响，国内能源价格上涨5.7%，涨幅比上月扩大0.9个百分点，影响CPI环比上涨约0.39个百分点。其中，汽油价格上涨12.6%。服务价格由上月下降1.1%转为上涨0.5%，高于季节性水平0.2个百分点，影响CPI环比上涨约0.22个百分点。其中，受清明节假期、“五一”假期及部分地区春假影响，出行服务需求明显增加，飞机票、交通工具租赁、旅行社收费和宾馆住宿价格分别上涨29.2%、8.6%、4.5%和3.9%，涨幅均高于季节性水平，四项合计影响CPI环比上涨约0.17个百分点；医疗服务价格上涨0.6%，影响CPI环比上涨约0.04个百分点。食品价格下降1.6%，降幅比上月收窄1.1个百分点，影响CPI环比下降约0.28个百分点。食品中，随着天气转暖，鲜菜和鲜果大量上市，价格分别下降6.4%和2.3%，猪肉和水产品供应充足，价格分别下降5.7%和1.2%，四项合计影响CPI环比下降约0.28个百分点；鸡蛋价格上涨3.4%，影响CPI环比上涨约0.01个百分点。扣除能源的工业消费品价格下降0.2%，基本保持稳定。

从同比看，全国CPI同比上涨1.2%，涨幅比上月扩大0.2个百分点。本月工业消费品价格上涨3.5%，涨幅比上月扩大1.3个百分点，影响CPI同比上涨约1.06个百分点。工业消费品中，受国际大宗商品价格波动影响，国内汽油和黄金饰品价格变动较大，其中汽油价格涨幅扩大至19.3%，影响CPI同比上涨约0.56个百分点；黄金饰品价格涨幅回落至46.9%，影响CPI同比上涨约0.20个百分点；家用器具和服装价格分别上涨2.6%和1.6%，合计影响CPI同比上涨约0.11个百分点。服务价格上涨0.9%，涨幅比上月扩大0.1个百分点，影响CPI同比上涨约0.44个百分点。服务中，基本公共服务项目价格涨幅总体稳定，其中医疗服务和教育服务价格分别上涨3.4%和0.5%，合计影响CPI同比上涨约0.25个百分点；出行服务价格涨幅有所扩大，本月上涨3.7%，影响CPI同比上涨约0.13个百分点；人工服务类价格小幅上涨，其中宠物服务、在外餐饮、家政服务、车辆修理与保养价格涨幅在1.1%—1.4%之间，四项合计影响CPI同比上涨约0.10个百分点。食品价格由上月上涨0.3%转为下降1.6%。食品中，猪肉价格下降15.2%，降幅比上月扩大3.7个百分点，影响CPI同比下降约0.29个百分点；鲜菜和鲜果价格分别下降0.5%和1.0%；牛肉、羊肉、水产品和鸡蛋价格均上涨，涨幅在1.0%—6.2%之间。

二、PPI涨幅扩大

从环比看，全国PPI环比上涨1.7%，涨幅比上月扩大0.7个百分点。本月PPI环比运行的主要特点：一是国际输入性因素影响国内石油相关行业价格上涨。国际原油价格上行带动国内石油相关行业价格上涨。其中，石油和天然气开采业价格环比上涨18.5%，石油煤炭及其他燃料加工业价格上涨16.4%，化学原料和化学制品制造业价格上涨8.3%，化学纤维制造业价格上涨5.6%，橡胶和塑料制品业价格上涨1.7%。二是国内部分行业需求增加带动价格上行。算力需求快速增长，电气化进程加快，光纤制造价格环比上涨22.5%，外存储设备及部件价格上涨3.2%，有色金属冶炼和压延加工业价格上涨0.2%；电煤补库存需求有所释放，叠加化工、冶金等非电用煤需求增加，煤炭开采和洗选业价格上涨1.9%；制造业设备更新持续推进带动用钢需求增加，黑色金属冶炼和压延加工业价格上涨0.6%。三是国内市场竞争秩序不断优化，相关行业价格上涨或降幅收窄。深入整治“内卷式”竞争成效继续显现，锂离子电池制造价格环比上涨1.6%，新能源车整车制造价格下降0.1%，降幅比上月收窄0.7个百分点。

从同比看，全国PPI同比上涨2.8%，涨幅比上月扩大2.3个百分点。价格上涨的主要行业中，有色金属矿采选业上涨38.9%，有色金属冶炼和压延加工业上涨22.5%，合计影响PPI同比上涨约1.58个百分点；石油和天然气开采业上涨28.6%，石油煤炭及其他燃料加工业上涨14.2%，化学原料和化学制品制造业上涨8.9%，合计影响PPI同比上涨约1.50个百分点；电气机械和器材制造业上涨3.6%，计算机通信和其他电子设备制造业上涨1.5%，合计影响PPI同比上涨约0.46个百分点。价格下降的主要行业中，非金属矿物制品业下降5.5%，电力热力生产和供应业下降4.2%，汽车制造业下降2.0%，黑色金属冶炼和压延加工业下降1.1%，合计影响PPI同比下降约0.75个百分点。
    """


    print(f"📝 输入文本: {text}\n")

    analyzer = analyze_texts(
        texts=[text],
        #output_file="output/single_text_report.html",
        format="html",
        quiet=False,
        use_bert=True
    )

    analyzer.to_html("output/single_analysis_report.html")
    analyzer.to_json("output/single_analysis_result.json")

    print(f"\n📊 分析结果:")
    print(f"  文本长度: {len(text)} 字符")
    print(f"  情感倾向: {analyzer.sentiment_distribution}")

    if analyzer.entity_stats:
        print(f"  实体识别:")
        for entity_type in ['per', 'org', 'loc']:
            count = analyzer.entity_stats.get(entity_type, {}).get('unique', 0)
            if count > 0:
                print(f"    - {entity_type}: {count} 个")


def example_from_json_file():
    """示例: 从已有的 JSON 结果文件生成 HTML 报告"""
    print_separator("示例: 从 JSON 文件生成 HTML 报告")

    import json
    from autotext.reporter import TextReporter

    json_file = "output/single_analysis_result.json"
    html_file = "output/single_analysis_from_json.html"

    if not os.path.exists(json_file):
        print(f"⚠️ JSON 文件不存在: {json_file}")
        print("   请先运行 example_single_text() 生成 JSON 文件")
        return

    # 读取 JSON 数据
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 创建模拟 analyzer 对象
    class DummyAnalyzer:
        pass

    dummy = DummyAnalyzer()

    # ========== 基础属性 ==========
    dummy.stats_result = data.get('stats', {})
    dummy.sentiment_distribution = data.get('sentiment', {}).get('distribution', {})
    dummy.keywords = data.get('keywords', {'frequency': []})
    dummy.content_texts = data.get('sample_texts', [])
    dummy.raw_texts = data.get('sample_texts', [])
    dummy.quality_report = data.get('quality', {})
    dummy.cleaning_suggestions = data.get('cleaning_suggestions', [])
    dummy.start_templates = set(data.get('templates', {}).get('start', []))
    dummy.end_templates = set(data.get('templates', {}).get('end', []))
    dummy.source_name = data.get('source', '未知')
    dummy.insights = data.get('insights', [])
    dummy.entity_stats = data.get('entity_stats', {})
    dummy.events_by_type = data.get('events_by_type', {})
    dummy.topics = data.get('topics', [])
    dummy.quality_details = data.get('quality_details', {})
    dummy.language_distribution = data.get('language_distribution', {})

    # ========== 大模型相关属性（公有） ==========
    dummy.llm_extraction = data.get('llm_extraction', {})
    dummy.entity_stats_by_type = data.get('entity_stats_by_type', {})
    dummy.relation_network = data.get('relation_network', {})
    dummy.event_timeline = data.get('event_timeline', {})
    dummy.theme_hierarchy = data.get('theme_hierarchy', {})
    dummy.entity_profiles = data.get('entity_profiles', [])
    dummy.event_chains = data.get('event_chains', [])
    dummy.global_graph = data.get('global_graph', {})
    dummy.static_graph = data.get('static_graph', {})
    dummy.llm_statistics = data.get('llm_statistics', {})
    dummy.has_llm_data = data.get('has_llm_data', False)
    dummy.relation_tree_data = data.get('relation_tree_data', {})
    dummy.entity_table_data = data.get('entity_table_data', [])

    # ========== 大模型相关私有属性（has_llm_data 判断依赖这些） ==========
    llm_extraction = data.get('llm_extraction', {})
    dummy._extracted_entities = llm_extraction.get('entities', [])
    dummy._extracted_relationships = llm_extraction.get('relationships', [])
    dummy._extracted_events = llm_extraction.get('events', [])
    dummy._extracted_themes = llm_extraction.get('themes', [])
    dummy._categorization = llm_extraction.get('categorization', {})

    # 确保 has_llm_data 正确设置
    if dummy._extracted_entities:
        dummy.has_llm_data = True

    # 创建 reporter 并生成 HTML
    reporter = TextReporter(dummy)
    reporter.to_html(html_file, title="文本分析报告")

    print(f"✅ HTML 报告已生成: {html_file}")

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
    example_single_text()

    #example_from_json_file() #直接读JSON生成HTML
    #example_multiple_texts()
    #example_long_texts()
    #example_text_with_dates()
    #example_from_csv_file()
    # example_from_txt_file()
    # example_from_folder()
    # example_dataframe_analysis()
    # example_no_bert()
    # example_export_all_formats()

