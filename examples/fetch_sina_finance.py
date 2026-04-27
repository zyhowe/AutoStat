"""
新浪财经新闻爬虫 - 抓取财经热点新闻
"""

import requests
import re
import time
import random
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from typing import List, Dict, Optional

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
}


class SinaFinanceCrawler:
    """新浪财经新闻爬虫"""

    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _request(self, url: str) -> Optional[str]:
        """发送请求"""
        try:
            time.sleep(self.delay + random.random() * 0.5)
            resp = self.session.get(url, timeout=15)
            resp.encoding = 'utf-8'
            if resp.status_code == 200:
                return resp.text
            else:
                print(f"请求失败: {url} - {resp.status_code}")
                return None
        except Exception as e:
            print(f"请求异常: {url} - {e}")
            return None

    def get_finance_links(self, category: str = "roll", page: int = 1) -> List[str]:
        """
        获取财经新闻链接

        参数:
        - category: 分类 (roll/stock/money/fund/hk/us)
        - page: 页码

        返回: 文章URL列表
        """
        url_map = {
            'roll': 'https://finance.sina.com.cn/roll/',           # 滚动
            'stock': 'https://finance.sina.com.cn/stock/',         # 股票
            'money': 'https://finance.sina.com.cn/money/',         # 理财
            'fund': 'https://finance.sina.com.cn/fund/',           # 基金
            'hk': 'https://finance.sina.com.cn/stock/hkstock/',    # 港股
            'us': 'https://finance.sina.com.cn/stock/usstock/',    # 美股
        }

        base_url = url_map.get(category, url_map['roll'])

        if page == 1:
            url = base_url
        else:
            # 不同分类的分页规则不同
            if 'roll' in base_url:
                url = f"{base_url}index_{page}.shtml"
            else:
                url = f"{base_url}index_{page}.shtml"

        html = self._request(url)
        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        links = []

        # 查找财经新闻链接
        for a in soup.find_all('a', href=True):
            href = a['href']
            # 新浪财经新闻链接特征
            if 'finance.sina.com.cn' in href:
                if '/doc-' in href or '/detail-' in href or '/i/' in href:
                    if href not in links:
                        links.append(href)

        # 过滤非文章链接
        links = [l for l in links if 'shtml' in l and ('doc' in l or 'detail' in l)]
        links = list(set(links))

        print(f"分类 {category} 第 {page} 页获取到 {len(links)} 个链接")
        return links

    def get_hot_news_links(self) -> List[str]:
        """
        获取热门财经新闻链接（从首页）
        """
        url = "https://finance.sina.com.cn/"
        html = self._request(url)

        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        links = []

        # 查找热门新闻区域
        hot_areas = soup.find_all('div', class_='news-item')
        if not hot_areas:
            hot_areas = soup.find_all('li', class_='news-item')

        for area in hot_areas:
            a = area.find('a', href=True)
            if a:
                href = a['href']
                if 'finance.sina.com.cn' in href and 'shtml' in href:
                    links.append(href)

        # 如果没找到，搜索所有链接
        if not links:
            for a in soup.find_all('a', href=True):
                href = a['href']
                if 'finance.sina.com.cn' in href and 'shtml' in href:
                    if href not in links:
                        links.append(href)

        links = list(set(links))
        print(f"热门新闻获取到 {len(links)} 个链接")
        return links

    def parse_article(self, url: str) -> Optional[Dict]:
        """
        解析文章详情

        返回:
        {
            'title': str,
            'content': str,
            'author': str,
            'publish_time': str,
            'source': str,
            'keywords': str,
            'url': str
        }
        """
        html = self._request(url)
        if not html:
            return None

        soup = BeautifulSoup(html, 'html.parser')

        # 提取标题
        title = ''
        title_tag = soup.find('h1', class_='main-title')
        if not title_tag:
            title_tag = soup.find('h1', id='artibodyTitle')
        if not title_tag:
            title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
            # 去除后缀
            title = re.sub(r'_\S+$', '', title)
            title = re.sub(r'\|新浪财经\|新浪网$', '', title)

        # 提取时间
        publish_time = ''
        time_tag = soup.find('span', class_='date')
        if not time_tag:
            time_tag = soup.find('span', class_='time')
        if not time_tag:
            time_tag = soup.find('div', class_='date')
        if time_tag:
            publish_time = time_tag.get_text().strip()
        else:
            # 从 meta 标签获取
            meta_time = soup.find('meta', {'name': 'publishdate'})
            if meta_time:
                publish_time = meta_time.get('content', '')

        # 提取来源
        source = '新浪财经'
        source_tag = soup.find('span', class_='source')
        if not source_tag:
            source_tag = soup.find('a', class_='source')
        if source_tag:
            source = source_tag.get_text().strip()
        else:
            meta_source = soup.find('meta', {'name': 'source'})
            if meta_source:
                source = meta_source.get('content', '')

        # 提取作者
        author = source
        author_tag = soup.find('span', class_='author')
        if author_tag:
            author = author_tag.get_text().strip()

        # 提取关键词
        keywords = ''
        meta_keywords = soup.find('meta', {'name': 'keywords'})
        if meta_keywords:
            keywords = meta_keywords.get('content', '')

        # 提取正文
        content = ''
        article_tag = soup.find('div', id='artibody')
        if not article_tag:
            article_tag = soup.find('div', class_='article')
        if not article_tag:
            article_tag = soup.find('div', class_='main-content')

        if article_tag:
            # 移除脚本和样式
            for tag in article_tag(['script', 'style', 'iframe', 'aside', 'div', 'p']):
                if tag.get('class') and ('recommend' in str(tag.get('class')) or 'ad' in str(tag.get('class'))):
                    tag.decompose()
            content = article_tag.get_text().strip()
            # 清理空白
            content = re.sub(r'\n+', '\n', content)
            content = re.sub(r'\s+', ' ', content)

        # 如果正文太短，尝试其他方式
        if len(content) < 100:
            p_tags = soup.find_all('p')
            if p_tags:
                content = ' '.join([p.get_text().strip() for p in p_tags if len(p.get_text().strip()) > 30])

        # 清理无效字符
        if content:
            content = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffefa-zA-Z0-9\.,;:!?\'"()\[\]{}，。；：！？""''《》【】、\s]', '', content)

        return {
            'title': title[:100] if title else '',
            'content': content[:5000] if content else '',
            'author': author,
            'publish_time': publish_time,
            'source': source,
            'keywords': keywords[:200] if keywords else '',
            'url': url
        }

    def crawl(self, max_articles: int = 500, categories: List[str] = None) -> List[Dict]:
        """
        批量抓取财经新闻

        参数:
        - max_articles: 最大文章数
        - categories: 分类列表，默认 ['roll', 'stock', 'money']

        返回: 文章列表
        """
        if categories is None:
            categories = ['roll', 'stock', 'money']

        all_links = []
        articles = []

        print(f"开始抓取新浪财经新闻，目标 {max_articles} 条...")

        # 1. 获取热门新闻链接
        hot_links = self.get_hot_news_links()
        all_links.extend(hot_links)

        # 2. 获取各分类链接
        for category in categories:
            if len(all_links) >= max_articles * 2:
                break

            for page in range(1, 6):  # 每个分类最多5页
                if len(all_links) >= max_articles * 2:
                    break
                links = self.get_finance_links(category, page)
                all_links.extend(links)

        # 去重
        all_links = list(set(all_links))
        print(f"去重后共 {len(all_links)} 个链接")

        # 3. 抓取文章详情
        for i, url in enumerate(all_links):
            if len(articles) >= max_articles:
                break

            print(f"抓取 [{i+1}/{len(all_links)}]: {url[:80]}...")
            article = self.parse_article(url)

            if article and article['content'] and len(article['content']) > 50:
                articles.append(article)
                print(f"  ✅ 成功: {article['title'][:50]}...")

        print(f"抓取完成，共 {len(articles)} 条有效新闻")
        return articles


def generate_finance_dataset(n: int = 500) -> pd.DataFrame:
    """
    生成财经新闻数据集
    """
    crawler = SinaFinanceCrawler(delay=0.8)
    articles = crawler.crawl(max_articles=n)

    if not articles:
        print("抓取失败")
        return pd.DataFrame()

    df = pd.DataFrame(articles)

    # 转换时间格式
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')

    # 添加分类列（根据关键词简单分类）
    def categorize(row):
        content = (row.get('title', '') + row.get('content', '')).lower()
        if any(k in content for k in ['股', '涨停', '跌停', '大盘', 'a股', '港股', '美股']):
            return '股票'
        elif any(k in content for k in ['基金', '净值', '基金经理', '基民']):
            return '基金'
        elif any(k in content for k in ['理财', '收益', '存款', '利率', '银行']):
            return '理财'
        elif any(k in content for k in ['房地产', '房价', '楼市', '地产']):
            return '房地产'
        elif any(k in content for k in ['央行', '降息', '加息', '货币政策']):
            return '宏观政策'
        else:
            return '财经综合'

    df['category'] = df.apply(categorize, axis=1)

    return df


def main():
    """主函数"""
    print("=" * 60)
    print("新浪财经新闻爬虫")
    print("=" * 60)

    # 抓取 300 条财经新闻（避免被封）
    df = generate_finance_dataset(300)

    if df.empty:
        print("抓取失败，请检查网络或稍后重试")
        return

    # 保存
    output_file = 'sina_finance_news.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 已保存到 {output_file}")
    print(f"   总条数: {len(df)}")
    print(f"   列: {list(df.columns)}")

    print(f"\n分类分布:")
    print(df['category'].value_counts())

    print(f"\n来源分布 (Top 5):")
    print(df['source'].value_counts().head(5))

    print(f"\n时间范围: {df['publish_time'].min()} ~ {df['publish_time'].max()}")

    print("\n样例数据:")
    for i, row in df.head(3).iterrows():
        print(f"\n--- 样例 {i+1} ---")
        print(f"标题: {row['title'][:60]}...")
        print(f"正文: {row['content'][:100]}...")
        print(f"时间: {row['publish_time']}")
        print(f"来源: {row['source']}")
        print(f"分类: {row['category']}")


if __name__ == "__main__":
    main()