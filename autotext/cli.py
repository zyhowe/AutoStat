"""
命令行入口模块 - 文本分析命令行工具
"""

import click
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autotext.analyzer import TextAnalyzer
from autotext.reporter import TextReporter


@click.group()
def text():
    """文本分析命令行工具"""
    pass


@text.command()
@click.argument('input_path')
@click.option('--output', '-o', default='text_report.html', help='输出文件路径')
@click.option('--format', '-f', default='html', type=click.Choice(['html', 'json', 'md']), help='输出格式')
@click.option('--title', default='文本分析报告', help='报告标题')
@click.option('--quiet', is_flag=True, help='静默模式')
def analyze(input_path, output, format, title, quiet):
    """分析文本文件或文件夹

    示例:
        autotext analyze comments.txt -o report.html
        autotext analyze ./texts/ -o report.json -f json
    """
    if not os.path.exists(input_path):
        click.echo(f"❌ 路径不存在: {input_path}")
        sys.exit(1)

    click.echo(f"📝 分析路径: {input_path}")

    try:
        analyzer = TextAnalyzer(input_path, quiet=quiet)
        analyzer.generate_full_report()

        reporter = TextReporter(analyzer)

        if format == 'html':
            reporter.to_html(output, title)
            click.echo(f"✅ HTML报告已保存到 {output}")
        elif format == 'json':
            reporter.to_json(output)
            click.echo(f"✅ JSON结果已保存到 {output}")
        elif format == 'md':
            reporter.to_markdown(output)
            click.echo(f"✅ Markdown报告已保存到 {output}")

    except Exception as e:
        click.echo(f"❌ 分析失败: {e}")
        if not quiet:
            import traceback
            click.echo(traceback.format_exc())
        sys.exit(1)


@text.command()
@click.argument('input_path')
@click.option('--top', '-n', default=20, help='关键词数量')
def keywords(input_path, top):
    """提取文本关键词"""
    analyzer = TextAnalyzer(input_path, quiet=True)
    analyzer._preprocess()
    analyzer._extract_keywords()

    click.echo(f"\n📊 高频关键词 TOP {top}:")
    for i, (word, count) in enumerate(analyzer.keywords.get("frequency", [])[:top], 1):
        click.echo(f"  {i:2d}. {word}: {count}")


@text.command()
@click.argument('input_path')
def sentiment(input_path):
    """情感分析"""
    analyzer = TextAnalyzer(input_path, quiet=True)
    analyzer._preprocess()
    analyzer._analyze_sentiment()

    click.echo(f"\n😊 情感分布:")
    dist = analyzer.sentiment_distribution
    click.echo(f"  积极: {dist['positive_rate']:.1%}")
    click.echo(f"  消极: {dist['negative_rate']:.1%}")
    click.echo(f"  中性: {dist['neutral_rate']:.1%}")


@text.command()
@click.argument('input_path')
def entities(input_path):
    """实体识别"""
    analyzer = TextAnalyzer(input_path, quiet=True)
    analyzer._preprocess()
    analyzer._recognize_entities()

    click.echo(f"\n🏷️ 实体统计:")
    for entity_type in ["person", "location", "organization"]:
        stats = analyzer.entity_stats.get(entity_type, {})
        click.echo(f"  {entity_type}: {stats.get('unique', 0)} 个")
        top_entities = stats.get("top", [])[:5]
        if top_entities:
            names = ", ".join([name for name, _ in top_entities])
            click.echo(f"    示例: {names}")


@text.command()
@click.argument('input_path')
def summary(input_path):
    """分析摘要"""
    analyzer = TextAnalyzer(input_path, quiet=True)
    analyzer.generate_full_report()

    click.echo(f"\n📋 分析摘要:")
    click.echo(f"  总文本数: {analyzer.stats_result['total_count']}")
    click.echo(f"  平均长度: {analyzer.stats_result['char_length']['mean']:.1f} 字符")
    click.echo(f"  空文本率: {analyzer.stats_result['empty_rate']:.1%}")

    dist = analyzer.sentiment_distribution
    click.echo(f"  积极文本: {dist['positive_rate']:.1%}")
    click.echo(f"  消极文本: {dist['negative_rate']:.1%}")


def main():
    text()


if __name__ == '__main__':
    main()