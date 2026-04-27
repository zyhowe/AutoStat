"""
数据分析 MCP 服务
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autostat.mcp_server import mcp as data_mcp

if __name__ == "__main__":
    data_mcp.run()