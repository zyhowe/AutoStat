"""
新浪新闻爬虫 - 抓取热点新闻
"""

import requests
import re
import time
import random
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from typing import List, Dict, Optional

# 请求头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
}


class SinaNewsCrawler:
    """新浪新闻爬虫"""

    def __init__(self, delay: float = 1.0):
        """
        初始化爬虫

        参数:
        - delay: 请求间隔（秒）
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _request(self, url: str) -> Optional[str]:
        """发送请求"""
        try:
            time.sleep(self.delay + random.random() * 0.5)
            resp = self.session.get(url, timeout=10)
            resp.encoding = 'utf-8'
            if resp.status_code == 200:
                return resp.text
            else:
                print(f"请求失败: {url} - {resp.status_code}")
                return None
        except Exception as e:
            print(f"请求异常: {url} - {e}")
            return None

    def get_roll_news_links(self, page: int = 1) -> List[str]:
        """
        获取滚动新闻列表页中的文章链接

        参数:
        - page: 页码

        返回: 文章URL列表
        """
        if page == 1:
            url = "https://news.sina.com.cn/roll/"
        else:
            url = f"https://news.sina.com.cn/roll/index_{page}.shtml"

        html = self._request(url)
        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        links = []

        # 查找新闻链接
        for a in soup.find_all('a', href=True):
            href = a['href']
            # 新浪新闻链接格式
            if 'news.sina.com.cn' in href and '/doc-' in href:
                if href not in links:
                    links.append(href)

        # 去重并限制数量
        links = list(set(links))
        print(f"第 {page} 页获取到 {len(links)} 个链接")
        return links

    def get_cate_news_links(self, cate: str, page: int = 1) -> List[str]:
        """
        获取分类新闻链接

        参数:
        - cate: 分类 (china/world/tech/finance/sports)
        - page: 页码
        """
        url_map = {
            'china': 'https://news.sina.com.cn/china/',
            'world': 'https://news.sina.com.cn/world/',
            'tech': 'https://tech.sina.com.cn/',
            'finance': 'https://finance.sina.com.cn/',
            'sports': 'https://sports.sina.com.cn/'
        }

        base_url = url_map.get(cate, url_map['china'])
        if page > 1:
            url = f"{base_url}index_{page}.shtml"
        else:
            url = base_url

        html = self._request(url)
        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        links = []

        for a in soup.find_all('a', href=True):
            href = a['href']
            if 'news.sina.com.cn' in href and '/doc-' in href:
                if href not in links:
                    links.append(href)

        links = list(set(links))
        print(f"分类 {cate} 第 {page} 页获取到 {len(links)} 个链接")
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
            'url': str
        }
        """
        html = self._request(url)
        if not html:
            return None

        soup = BeautifulSoup(html, 'html.parser')

        # 提取标题
        title_tag = soup.find('h1', class_='main-title')
        if not title_tag:
            title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else ''

        # 提取时间、来源、作者
        date_tag = soup.find('span', class_='date')
        if not date_tag:
            date_tag = soup.find('span', class_='time')
        publish_time = date_tag.get_text().strip() if date_tag else ''

        source_tag = soup.find('span', class_='source')
        if not source_tag:
            source_tag = soup.find('a', class_='source')
        source = source_tag.get_text().strip() if source_tag else '新浪新闻'

        author_tag = soup.find('span', class_='author')
        author = author_tag.get_text().strip() if author_tag else source

        # 提取正文
        article_tag = soup.find('div', class_='article')
        if not article_tag:
            article_tag = soup.find('div', id='article')
        if not article_tag:
            article_tag = soup.find('div', class_='main-content')

        content = ''
        if article_tag:
            # 移除脚本和样式
            for script in article_tag(['script', 'style', 'iframe', 'aside']):
                script.decompose()
            content = article_tag.get_text().strip()
            # 清理空白
            content = re.sub(r'\n+', '\n', content)
            content = re.sub(r'\s+', ' ', content)

        # 如果正文太短，可能解析失败
        if len(content) < 50:
            # 尝试其他方式
            p_tags = soup.find_all('p')
            if p_tags:
                content = ' '.join([p.get_text().strip() for p in p_tags if len(p.get_text().strip()) > 20])

        return {
            'title': title,
            'content': content,
            'author': author,
            'publish_time': publish_time,
            'source': source,
            'url': url
        }

    def crawl(self, pages: int = 5, max_articles: int = 1000) -> List[Dict]:
        """
        批量抓取新闻

        参数:
        - pages: 抓取页数
        - max_articles: 最大文章数

        返回: 文章列表
        """
        all_links = []
        articles = []

        print(f"开始抓取新浪新闻，目标 {max_articles} 条...")

        # 1. 获取所有链接
        for page in range(1, pages + 1):
            if len(all_links) >= max_articles * 2:
                break
            links = self.get_roll_news_links(page)
            all_links.extend(links)
            print(f"已收集 {len(all_links)} 个链接")

        # 去除重复
        all_links = list(set(all_links))
        print(f"去重后共 {len(all_links)} 个链接")

        # 2. 抓取文章详情
        for i, url in enumerate(all_links):
            if len(articles) >= max_articles:
                break

            print(f"抓取 [{i+1}/{len(all_links)}]: {url}")
            article = self.parse_article(url)

            if article and article['content'] and len(article['content']) > 50:
                articles.append(article)
                print(f"  ✅ 成功: {article['title'][:50]}...")

        print(f"抓取完成，共 {len(articles)} 条有效新闻")
        return articles


def generate_news_dataset(n: int = 1000) -> pd.DataFrame:
    """
    生成新闻数据集

    参数:
    - n: 目标数量

    返回: DataFrame
    """
    crawler = SinaNewsCrawler(delay=0.5)
    pages = max(3, n // 100 + 1)
    articles = crawler.crawl(pages=pages, max_articles=n)

    if not articles:
        print("抓取失败，使用模拟数据")
        return _generate_mock_news(n)

    df = pd.DataFrame(articles)
    return df


def _generate_mock_news(n: int) -> pd.DataFrame:
    """生成模拟新闻数据（备用）"""
    import random
    from datetime import datetime, timedelta

    categories = ['时事', '科技', '财经', '体育', '娱乐', '国际']
    sources = ['新华社', '人民日报', '央视新闻', '新浪财经', '科技日报']

    data = []
    start_date = datetime(2024, 1, 1)

    for i in range(n):
        date = start_date + timedelta(days=random.randint(0, 365))
        category = random.choice(categories)

        data.append({
            'title': f'新闻标题{i+1}：这是关于{category}的报道',
            'content': f'这是一条关于{category}的新闻内容，事件发生在{date.strftime("%Y年%m月%d日")}。' * random.randint(5, 20),
            'author': random.choice(sources),
            'publish_time': date.strftime('%Y-%m-%d %H:%M:%S'),
            'source': random.choice(sources),
            'url': f'https://news.sina.com.cn/doc-{i+1}.shtml',
            'category': category
        })

    return pd.DataFrame(data)


def main():
    """主函数"""
    print("=" * 60)
    print("新浪新闻爬虫")
    print("=" * 60)

    # 抓取 500 条新闻（避免被封）
    df = generate_news_dataset(500)

    # 保存
    output_file = 'sina_news_500.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 已保存到 {output_file}")
    print(f"   总条数: {len(df)}")
    print(f"   列: {list(df.columns)}")

    # 显示样例
    print("\n样例数据:")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()