"""
命令行入口模块
"""

import click
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autostat.analyzer import AutoStatisticalAnalyzer
from autostat.multi_analyzer import MultiTableStatisticalAnalyzer
from autostat.reporter import Reporter


@click.command()
@click.argument('file_path', required=False)
@click.option('--multi', is_flag=True, help='多表模式')
@click.option('--output', '-o', default='report.html', help='输出文件路径')
@click.option('--format', '-f', default='html', type=click.Choice(['html', 'json']), help='输出格式')
@click.option('--sample-rate', default=0.1, help='采样率（默认0.1）')
@click.option('--quiet', is_flag=True, help='静默模式')
def analyze(file_path, multi, output, format, sample_rate, quiet):
    """智能数据分析工具"""

    if not file_path and not multi:
        click.echo("请指定文件路径或使用 --multi 模式")
        click.echo("用法: auto-analyzer data.csv")
        click.echo("      auto-analyzer --multi file1.csv file2.csv")
        sys.exit(1)

    if multi:
        # 多表模式
        click.echo("📊 多表分析模式")
        # 简化实现，实际需要解析多个文件
        click.echo("提示: 多表分析请使用Python API")
        return

    # 单表模式
    if not os.path.exists(file_path):
        click.echo(f"❌ 文件不存在: {file_path}")
        sys.exit(1)

    click.echo(f"📁 分析文件: {file_path}")

    try:
        analyzer = AutoStatisticalAnalyzer(file_path, auto_clean=False, quiet=quiet)
        reporter = Reporter(analyzer)

        if format == 'html':
            reporter.to_html(output)
            click.echo(f"✅ 报告已保存到 {output}")
        else:
            json_str = reporter.to_json(output)
            if not output:
                click.echo(json_str)
            else:
                click.echo(f"✅ JSON已保存到 {output}")

    except Exception as e:
        click.echo(f"❌ 分析失败: {e}")
        sys.exit(1)


def main():
    analyze()


if __name__ == '__main__':
    main()