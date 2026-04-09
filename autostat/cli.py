"""
命令行入口模块
"""

import click
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autostat.analyzer import AutoStatisticalAnalyzer
from autostat.reporter import Reporter


@click.command()
@click.argument('file_path', required=False)
@click.option('--multi', is_flag=True, help='多表模式')
@click.option('--output', '-o', default='report.html', help='输出文件路径')
@click.option('--format', '-f', default='html', type=click.Choice(['html', 'json', 'md', 'excel']), help='输出格式')
@click.option('--sample-rate', default=0.1, help='采样率（默认0.1）')
@click.option('--date-features', default='basic', type=click.Choice(['none', 'basic', 'full']), help='日期特征级别')
@click.option('--quiet', is_flag=True, help='静默模式')
@click.option('--auto-clean', is_flag=True, help='自动清洗')
def analyze(file_path, multi, output, format, sample_rate, date_features, quiet, auto_clean):
    """
    智能数据分析工具

    示例:
        autostat data.csv
        autostat data.csv -o report.html
        autostat data.csv -f json -o result.json
        autostat --multi orders.csv users.csv --relation "orders.user_id=users.id"
    """
    if not file_path and not multi:
        click.echo("请指定文件路径或使用 --multi 模式")
        click.echo("用法: autostat data.csv")
        click.echo("      autostat --multi file1.csv file2.csv")
        sys.exit(1)

    if multi:
        click.echo("📊 多表分析模式")
        click.echo("提示: 多表分析请使用Python API")
        click.echo("示例: from autostat import MultiTableStatisticalAnalyzer")
        return

    if not os.path.exists(file_path):
        click.echo(f"❌ 文件不存在: {file_path}")
        sys.exit(1)

    click.echo(f"📁 分析文件: {file_path}")

    try:
        analyzer = AutoStatisticalAnalyzer(
            file_path,
            auto_clean=auto_clean,
            quiet=quiet,
            date_features_level=date_features
        )
        reporter = Reporter(analyzer)

        if format == 'html':
            reporter.to_html(output)
            click.echo(f"✅ HTML报告已保存到 {output}")
        elif format == 'json':
            reporter.to_json(output)
            click.echo(f"✅ JSON结果已保存到 {output}")
        elif format == 'md':
            reporter.to_markdown(output)
            click.echo(f"✅ Markdown报告已保存到 {output}")
        elif format == 'excel':
            reporter.to_excel(output)
            click.echo(f"✅ Excel报告已保存到 {output}")

    except Exception as e:
        click.echo(f"❌ 分析失败: {e}")
        if not quiet:
            import traceback
            click.echo(traceback.format_exc())
        sys.exit(1)


def main():
    analyze()


if __name__ == '__main__':
    main()