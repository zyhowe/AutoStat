"""
生成1000条模拟评论数据
"""

import pandas as pd
import random
import os
from datetime import datetime, timedelta

# 模板数据
PRODUCTS = ["手机", "电脑", "耳机", "充电宝", "鼠标", "键盘", "显示器", "路由器", "智能手表", "平板电脑"]
BRANDS = ["华为", "小米", "苹果", "OPPO", "vivo", "荣耀", "三星", "一加", "魅族", "联想"]

# 积极评论模板
POSITIVE_COMMENTS = [
    "{}的{}真的很好，{}很出色，{}。",
    "强烈推荐{}，{}效果超出预期，{}。",
    "{}性价比很高，{}很满意，{}。",
    "{}做工精细，{}体验很好，{}。",
    "{}售后服务很好，{}解决问题很快，{}。",
]

POSITIVE_PHRASES = [
    ["质量", "非常满意", "会推荐给朋友"],
    ["性能", "完全达到预期", "五星好评"],
    ["外观", "很喜欢", "下次还会买"],
    ["续航", "很给力", "物超所值"],
    ["屏幕", "显示效果很棒", "快递也很快"],
    ["音质", "声音很清晰", "包装很精美"],
    ["手感", "握着很舒服", "做工精细"],
    ["系统", "运行很流畅", "没有卡顿"],
    ["拍照", "照片很清晰", "夜景模式很棒"],
    ["充电", "充电速度很快", "一天一充足够"],
]

# 消极评论模板
NEGATIVE_COMMENTS = [
    "{}的{}太差了，{}，{}。",
    "千万别买{}，{}非常糟糕，{}。",
    "{}质量有问题，{}让人失望，{}。",
    "{}售后太差，{}根本没解决，{}。",
]

NEGATIVE_PHRASES = [
    ["质量", "用了三天就坏了", "申请退货还被拒绝"],
    ["续航", "半天就没电", "和宣传严重不符"],
    ["屏幕", "出现闪屏", "售后不管"],
    ["系统", "经常卡顿死机", "更新后更差"],
    ["充电", "充电很慢", "充电口松动"],
    ["音质", "有杂音", "还不如几十块的"],
    ["手感", "做工粗糙", "边缘割手"],
    ["拍照", "照片模糊", "对焦很慢"],
    ["售后", "客服态度差", "打了多次电话没人接"],
    ["物流", "等了一周才到", "包装破损"],
]

# 中性评论模板
NEUTRAL_COMMENTS = [
    "{}的{}一般般，{}，{}。",
    "{}还行吧，{}，{}。",
    "{}中规中矩，{}，{}。",
]
NEUTRAL_PHRASES = [
    ["质量", "没有想象中好", "能用"],
    ["性能", "够用", "没什么惊喜"],
    ["外观", "一般", "不算难看"],
    ["价格", "有点贵", "等降价再买"],
    ["物流", "正常速度", "包装完好"],
]

# 作者
AUTHORS = ["小明", "小红", "小刚", "小丽", "小华", "小强", "小美", "小军", "小芳", "小伟",
           "用户123", "数码爱好者", "购物狂", "理性消费", "评测达人", "路人甲", "老王", "小李", "小张", "小陈"]

# 来源
SOURCES = ["京东", "淘宝", "天猫", "拼多多", "官网", "线下门店"]


def generate_comment(sentiment: str) -> str:
    """生成单条评论"""
    product = random.choice(PRODUCTS)
    brand = random.choice(BRANDS)

    if sentiment == "positive":
        template = random.choice(POSITIVE_COMMENTS)
        phrases = random.choice(POSITIVE_PHRASES)
        return template.format(brand, product, phrases[0], phrases[1])
    elif sentiment == "negative":
        template = random.choice(NEGATIVE_COMMENTS)
        phrases = random.choice(NEGATIVE_PHRASES)
        return template.format(brand, product, phrases[0], phrases[1])
    else:
        template = random.choice(NEUTRAL_COMMENTS)
        phrases = random.choice(NEUTRAL_PHRASES)
        return template.format(brand, product, phrases[0], phrases[1])


def generate_title(comment: str) -> str:
    """从评论生成标题"""
    words = comment[:30]
    return words + ("..." if len(comment) > 30 else "")


def generate_date(start_date: datetime, end_date: datetime) -> str:
    """生成随机日期"""
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    return (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d %H:%M:%S")


def generate_sample_data(n: int = 1000) -> pd.DataFrame:
    """生成n条样本数据"""
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)

    # 情感分布：60%积极，25%消极，15%中性
    sentiments = ["positive"] * 600 + ["negative"] * 250 + ["neutral"] * 150

    data = []
    for i in range(n):
        sentiment = random.choice(sentiments)
        comment = generate_comment(sentiment)
        title = generate_title(comment)
        author = random.choice(AUTHORS)
        source = random.choice(SOURCES)
        date = generate_date(start_date, end_date)

        data.append({
            "id": i + 1,
            "title": title,
            "content": comment,
            "author": author,
            "source": source,
            "date": date,
            "sentiment_label": sentiment
        })

    # 按时间排序
    df = pd.DataFrame(data)
    df = df.sort_values("date").reset_index(drop=True)

    return df


def main():
    """生成数据并保存"""
    print("正在生成1000条模拟评论数据...")
    df = generate_sample_data(1000)

    # 保存为 CSV
    output_path = os.path.join(os.path.dirname(__file__), "sample_comments_1000.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ 已保存到 {output_path}")
    print(f"   总条数: {len(df)}")
    print(f"   列: {list(df.columns)}")
    print(f"\n情感分布:")
    print(df["sentiment_label"].value_counts())
    print(f"\n来源分布:")
    print(df["source"].value_counts())
    print(f"\n时间范围: {df['date'].min()} ~ {df['date'].max()}")

    # 显示几条样例
    print("\n样例数据:")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()