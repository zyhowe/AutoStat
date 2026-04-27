"""
生成热点新闻模拟数据
基于真实热点事件（2024年），构造多个新闻事件，每个事件有多条相关报道
"""

import pandas as pd
import random
from datetime import datetime, timedelta

# ==================== 热点事件定义 ====================
# 每个事件包含: 事件名称、开始日期、持续天数、相关关键词、事件描述模板

HOT_EVENTS = [
    {
        "name": "某手机品牌新品发布会",
        "start_date": "2024-09-10",
        "duration_days": 30,
        "keywords": ["新品发布", "旗舰手机", "技术创新", "价格", "预售", "评测"],
        "sources": ["科技日报", "新浪科技", "腾讯科技", "中关村在线", "IT之家"],
        "authors": ["Tech记者", "科技观察员", "数码博主", "编辑小张", "行业分析师"],
        "title_templates": [
            "{}发布{}，{}引热议",
            "{}史上最强{}，{}",
            "{}发布会回顾：{}惊艳亮相",
            "深度评测：{}的{}到底怎么样？",
        ],
        "content_templates": [
            "{}今日正式发布新一代{}，售价{}元起，{}。",
            "备受期待的{}终于亮相，搭载{}，性能提升{}%。",
            "{}发布会上，{}成为最大亮点，{}。",
            "{}首销{}，用户{}。"
        ]
    },
    {
        "name": "电商平台双11购物节",
        "start_date": "2024-10-20",
        "duration_days": 25,
        "keywords": ["双11", "购物节", "促销", "优惠券", "销售额", "快递", "包裹"],
        "sources": ["电商报", "天下网商", "亿邦动力", "联商网", "新浪财经"],
        "authors": ["电商观察员", "零售分析师", "财经记者", "行业评论员", "编辑老王"],
        "title_templates": [
            "{}正式启动，{}再创新高",
            "{}战报：{}突破{}亿",
            "{}攻略：{}省钱秘籍",
            "{}物流高峰，{}",
        ],
        "content_templates": [
            "{}今晚8点正式开启，{}分钟销售额破{}亿。",
            "{}期间，平台推出{}，预计带动{}。",
            "{}快递量激增，各快递公司{}。",
            "{}消费者最爱买{}，同比{}%。"
        ]
    },
    {
        "name": "冬季寒潮来袭",
        "start_date": "2024-12-10",
        "duration_days": 15,
        "keywords": ["寒潮", "降温", "暴雪", "供暖", "交通", "预警", "保暖"],
        "sources": ["中国天气网", "央视新闻", "人民日报", "新华网", "澎湃新闻"],
        "authors": ["气象分析师", "记者小周", "天气君", "民生记者", "编辑小刘"],
        "title_templates": [
            "{}发布寒潮预警，{}",
            "{}来袭，多地{}",
            "{}持续降温，{}",
            "{}应对寒潮，{}",
        ],
        "content_templates": [
            "受{}影响，北方地区将迎来{}℃降温。",
            "气象部门发布{}预警，提醒市民{}。",
            "{}导致多地交通受阻，{}。",
            "为应对{}，相关部门{}。"
        ]
    },
    {
        "name": "新能源汽车补贴政策",
        "start_date": "2024-06-01",
        "duration_days": 20,
        "keywords": ["新能源", "汽车", "补贴", "政策", "购置税", "销量", "充电桩"],
        "sources": ["第一财经", "经济观察报", "证券时报", "财联社", "汽车之家"],
        "authors": ["财经记者", "汽车分析师", "政策研究员", "行业观察员", "编辑小李"],
        "title_templates": [
            "{}发布新政，{}",
            "{}刺激汽车消费，{}",
            "{}解读：{}",
            "{}受益，{}",
        ],
        "content_templates": [
            "三部委联合发布{}，将{}延续至{}年。",
            "新政明确{}减免，预计带动{}。",
            "业内专家认为，{}将利好{}。",
            "{}企业积极响应，推出{}。"
        ]
    },
    {
        "name": "热门游戏上线",
        "start_date": "2024-08-20",
        "duration_days": 25,
        "keywords": ["游戏", "上线", "玩家", "评分", "下载", "服务器", "更新"],
        "sources": ["游戏葡萄", "游民星空", "3DM", "游侠网", "B站游戏"],
        "authors": ["游戏编辑", "评测员小王", "游戏博主", "玩家代表", "运营小陈"],
        "title_templates": [
            "{}正式上线，{}",
            "{}评测：{}分",
            "{}火爆，服务器{}",
            "{}攻略：{}",
        ],
        "content_templates": [
            "备受期待的{}今日全球上线，首日下载量突破{}万。",
            "专业媒体给{}打出{}分，称{}。",
            "由于玩家{}，服务器出现{}。",
            "官方发布{}，新增{}内容。"
        ]
    }
]

# 通用变量池
CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京", "西安", "重庆"]
NUMBERS = ["10", "50", "100", "500", "1000", "5000", "10000", "50000"]
PERCENTAGES = ["10%", "20%", "30%", "50%", "80%", "100%"]
COMPANIES = ["腾讯", "阿里巴巴", "华为", "小米", "字节跳动", "百度", "京东", "美团", "滴滴", "拼多多"]


def random_value(pool):
    """从池中随机取值"""
    return random.choice(pool)


def generate_article(event: dict, event_date: datetime, article_index: int) -> dict:
    """生成单篇新闻文章"""
    # 随机选择模板
    title_template = random.choice(event["title_templates"])
    content_template = random.choice(event["content_templates"])

    # 随机选择来源和作者
    source = random.choice(event["sources"])
    author = random.choice(event["authors"])

    # 随机填充变量
    keyword = random.choice(event["keywords"])
    num = random_value(NUMBERS)
    pct = random_value(PERCENTAGES)
    city = random_value(CITIES)
    company = random_value(COMPANIES)

    # 格式化标题和内容
    title = title_template.format(
        event["name"],
        keyword,
        random.choice(["引发热议", "备受关注", "登上热搜", "反响热烈"])
    )
    content = content_template.format(
        event["name"],
        keyword,
        num,
        pct,
        city,
        company
    )

    # 生成不同报道角度
    angles = ["深度分析", "独家采访", "数据解读", "专家观点", "用户反馈", "行业影响"]
    angle = random.choice(angles)

    return {
        "id": None,  # 稍后填充
        "title": title[:50] + ("..." if len(title) > 50 else ""),
        "content": content,
        "author": author,
        "source": source,
        "date": event_date.strftime("%Y-%m-%d %H:%M:%S"),
        "event_name": event["name"],
        "angle": angle
    }


def generate_hot_news_data(target_count: int = 1000) -> pd.DataFrame:
    """生成热点新闻数据"""
    articles = []
    article_id = 1

    for event in HOT_EVENTS:
        start = datetime.strptime(event["start_date"], "%Y-%m-%d")
        # 计算该事件应生成的新闻数量（按热度分配）
        event_ratio = random.uniform(0.15, 0.35)
        event_count = int(target_count * event_ratio)

        # 确保总数不超过目标
        if sum(len([a for a in articles if a["event_name"] == event["name"]]) for event in
               HOT_EVENTS) + event_count > target_count:
            event_count = target_count - len(articles)

        print(f"  事件: {event['name']} - 生成 {event_count} 条")

        for i in range(event_count):
            # 在事件持续时间内随机分布日期
            offset_days = random.randint(0, event["duration_days"])
            article_date = start + timedelta(days=offset_days)

            article = generate_article(event, article_date, i)
            article["id"] = article_id
            articles.append(article)
            article_id += 1

    # 如果数量不足，随机补充
    while len(articles) < target_count:
        event = random.choice(HOT_EVENTS)
        start = datetime.strptime(event["start_date"], "%Y-%m-%d")
        offset_days = random.randint(0, event["duration_days"])
        article_date = start + timedelta(days=offset_days)
        article = generate_article(event, article_date, len(articles))
        article["id"] = len(articles) + 1
        articles.append(article)

    # 转换为 DataFrame 并按时间排序
    df = pd.DataFrame(articles)
    df = df.sort_values("date").reset_index(drop=True)

    return df


def main():
    """生成数据并保存"""
    print("正在生成热点新闻模拟数据...")
    df = generate_hot_news_data(1000)

    # 保存为 CSV
    output_path = os.path.join(os.path.dirname(__file__), "hot_news_1000.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ 已保存到 {output_path}")
    print(f"   总条数: {len(df)}")
    print(f"   列: {list(df.columns)}")

    print(f"\n事件分布:")
    print(df["event_name"].value_counts())

    print(f"\n来源分布 (Top 10):")
    print(df["source"].value_counts().head(10))

    print(f"\n时间范围: {df['date'].min()} ~ {df['date'].max()}")

    print("\n样例数据:")
    print(df.head(5).to_string())

    # 显示热点事件时间线
    print("\n事件时间线:")
    for event in HOT_EVENTS:
        event_df = df[df["event_name"] == event["name"]]
        if len(event_df) > 0:
            print(f"  {event['name']}: {event_df['date'].min()} ~ {event_df['date'].max()} ({len(event_df)}条)")


if __name__ == "__main__":
    import os

    main()