"""
文本分析 MCP 服务
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autotext.mcp_server import mcp as text_mcp

if __name__ == "__main__":
    text_mcp.run()