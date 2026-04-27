"""
MCP服务模块 - 供AI Agent调用文本分析功能
"""

import json
import os
import time
import argparse
import sys
from typing import Dict, Any, List

from fastmcp import FastMCP

from autotext.analyzer import TextAnalyzer
from autotext.reporter import TextReporter

# 创建 MCP 服务实例
mcp = FastMCP("AutoText")


@mcp.tool
def analyze_text_file(
    file_path: str,
    output_level: str = "compact",
    top_n_keywords: int = 30
) -> str:
    """
    分析文本文件

    参数:
    - file_path: 文件路径（.txt）
    - output_level: 输出级别，可选 minimal/compact/full
    - top_n_keywords: 关键词数量

    返回:
    - JSON格式的分析结果
    """
    try:
        start_time = time.time()

        if not os.path.exists(file_path):
            return json.dumps({"success": False, "error": f"文件不存在: {file_path}"})

        analyzer = TextAnalyzer(file_path, quiet=True)
        analyzer.generate_full_report()

        # 构建结果
        result = {
            'success': True,
            'analysis_time': time.time() - start_time,
            'file_name': os.path.basename(file_path),
            'stats': {
                'total_texts': analyzer.stats_result.get('total_count', 0),
                'avg_length': analyzer.stats_result.get('char_length', {}).get('mean', 0),
                'empty_rate': analyzer.stats_result.get('empty_rate', 0)
            },
            'sentiment': {
                'positive': analyzer.sentiment_distribution.get('positive_rate', 0),
                'negative': analyzer.sentiment_distribution.get('negative_rate', 0),
                'neutral': analyzer.sentiment_distribution.get('neutral_rate', 0)
            },
            'top_keywords': analyzer.keywords.get('frequency', [])[:top_n_keywords],
            'entities': {
                'person': analyzer.entity_stats.get('person', {}).get('top', [])[:10],
                'location': analyzer.entity_stats.get('location', {}).get('top', [])[:10]
            } if hasattr(analyzer, 'entity_stats') else {}
        }

        return json.dumps(result, ensure_ascii=False, indent=2, default=str)

    except Exception as e:
        import traceback
        return json.dumps({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }, ensure_ascii=False, indent=2)


@mcp.tool
def analyze_text_folder(
    folder_path: str,
    output_level: str = "compact"
) -> str:
    """
    分析文件夹中的所有文本文件

    参数:
    - folder_path: 文件夹路径
    - output_level: 输出级别

    返回:
    - JSON格式的分析结果
    """
    try:
        start_time = time.time()

        if not os.path.isdir(folder_path):
            return json.dumps({"success": False, "error": f"文件夹不存在: {folder_path}"})

        analyzer = TextAnalyzer(folder_path, quiet=True)
        analyzer.generate_full_report()

        result = {
            'success': True,
            'analysis_time': time.time() - start_time,
            'folder_name': os.path.basename(folder_path),
            'stats': {
                'total_texts': analyzer.stats_result.get('total_count', 0),
                'avg_length': analyzer.stats_result.get('char_length', {}).get('mean', 0),
                'empty_rate': analyzer.stats_result.get('empty_rate', 0)
            },
            'sentiment': {
                'positive': analyzer.sentiment_distribution.get('positive_rate', 0),
                'negative': analyzer.sentiment_distribution.get('negative_rate', 0),
                'neutral': analyzer.sentiment_distribution.get('neutral_rate', 0)
            },
            'top_keywords': analyzer.keywords.get('frequency', [])[:30]
        }

        return json.dumps(result, ensure_ascii=False, indent=2, default=str)

    except Exception as e:
        return json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False, indent=2)


@mcp.tool
def extract_keywords(text: str, top_n: int = 20) -> str:
    """
    从文本中提取关键词

    参数:
    - text: 文本内容
    - top_n: 关键词数量

    返回:
    - JSON格式的关键词列表
    """
    try:
        from autotext.core.preprocessor import TextPreprocessor
        from autotext.core.keyword import KeywordExtractor

        preprocessor = TextPreprocessor()
        processed = preprocessor.process(text)
        tokens = processed.get('tokens_cleaned', [])

        extractor = KeywordExtractor()
        # 使用词频统计
        from collections import Counter
        counter = Counter(tokens)
        keywords = counter.most_common(top_n)

        return json.dumps({
            'success': True,
            'keywords': [{'word': w, 'count': c} for w, c in keywords]
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False, indent=2)


@mcp.tool
def analyze_sentiment(text: str) -> str:
    """
    分析文本情感

    参数:
    - text: 文本内容

    返回:
    - JSON格式的情感分析结果
    """
    try:
        from autotext.core.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(text)

        return json.dumps({
            'success': True,
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'positive_words': result['positive_words'],
            'negative_words': result['negative_words']
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False, indent=2)


@mcp.tool
def extract_entities(text: str) -> str:
    """
    提取文本中的实体（人名、地名、组织名）

    参数:
    - text: 文本内容

    返回:
    - JSON格式的实体列表
    """
    try:
        from autotext.core.entity import EntityRecognizer

        recognizer = EntityRecognizer()
        entities = recognizer.recognize(text)

        return json.dumps({
            'success': True,
            'person': entities.get('person', []),
            'location': entities.get('location', []),
            'organization': entities.get('organization', [])
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False, indent=2)


def main():
    """MCP 服务主入口"""
    parser = argparse.ArgumentParser(
        description="AutoText MCP Server - 智能文本分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio", "http", "sse"],
        help="传输协议 (默认: stdio)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("AUTOTEXT_HOST", "0.0.0.0"),
        help="HTTP/SSE 服务绑定地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("AUTOTEXT_PORT", "6012")),
        help="HTTP/SSE 服务端口 (默认: 6012)"
    )

    if len(sys.argv) == 1:
        mcp.run(transport="stdio")
        return

    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)


if __name__ == "__main__":
    main()