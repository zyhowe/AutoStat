"""
错误处理服务 - 统一错误处理和日志记录
"""

import traceback
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


class ErrorHandler:
    """统一错误处理器"""

    # 日志路径
    LOG_DIR = Path.home() / ".autostat" / "logs"

    @classmethod
    def _ensure_log_dir(cls):
        """确保日志目录存在"""
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _get_log_file(cls) -> Path:
        """获取当前日期的日志文件路径"""
        cls._ensure_log_dir()
        date_str = datetime.now().strftime("%Y%m%d")
        return cls.LOG_DIR / f"error_{date_str}.log"

    @classmethod
    def log_error(cls, error: Exception, context: str = ""):
        """
        记录错误到日志文件

        参数:
        - error: 异常对象
        - context: 错误发生的上下文（如函数名、操作描述）
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_type = type(error).__name__
        error_msg = str(error)
        stack_trace = traceback.format_exc()

        log_entry = f"""
{'=' * 80}
时间: {timestamp}
上下文: {context}
错误类型: {error_type}
错误信息: {error_msg}
堆栈跟踪:
{stack_trace}
{'=' * 80}
"""

        log_file = cls._get_log_file()
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)

    @classmethod
    def format_user_message(cls, error: Exception) -> Tuple[str, str]:
        """
        格式化用户友好的错误消息

        返回: (用户消息, 是否需要显示详情按钮)
        """
        error_type = type(error).__name__
        error_msg = str(error)

        # MemoryError
        if error_type == "MemoryError" or "memory" in error_msg.lower():
            return "文件过大，建议使用采样分析或拆分文件后重试。", True

        # 编码错误
        if error_type in ["UnicodeDecodeError", "UnicodeError"]:
            return "文件编码不支持，请保存为 UTF-8 格式后重试。", False

        # 文件不存在
        if error_type == "FileNotFoundError":
            return "文件不存在，请检查文件路径。", False

        # 列名/数据格式错误
        if error_type in ["KeyError", "ValueError"]:
            if "column" in error_msg.lower() or "列" in error_msg:
                return "数据格式不符合预期，请检查列名是否包含特殊字符或重复。", True
            return f"数据格式错误：{error_msg[:100]}", True

        # 超时
        if error_type == "TimeoutError" or "timeout" in error_msg.lower():
            return "连接超时，请检查网络后重试。", True

        # 导入错误
        if error_type == "ImportError" or "module" in error_msg.lower():
            return f"缺少依赖包，请运行 pip install 安装所需依赖。", True

        # OSError (threadpoolctl 等)
        if error_type == "OSError":
            return "系统资源不足，请重启应用后重试。", False

        # 默认
        return f"分析失败：{error_msg[:100]}", True

    @classmethod
    def handle_error(cls, error: Exception, context: str = "", fallback_message: str = None) -> str:
        """
        处理错误：记录日志 + 返回用户友好消息

        参数:
        - error: 异常对象
        - context: 错误上下文
        - fallback_message: 备选消息（当无法从错误中提取时使用）

        返回: 用户友好消息
        """
        # 记录日志
        cls.log_error(error, context)

        # 获取用户友好消息
        user_msg, has_detail = cls.format_user_message(error)

        if fallback_message:
            user_msg = f"{user_msg} {fallback_message}"

        return user_msg

    @classmethod
    def get_logs(cls, days: int = 7) -> list:
        """
        获取最近N天的日志摘要

        参数:
        - days: 天数

        返回: 日志文件列表 [(文件名, 大小, 修改时间)]
        """
        cls._ensure_log_dir()
        logs = []
        for log_file in cls.LOG_DIR.glob("error_*.log"):
            stat = log_file.stat()
            logs.append({
                "name": log_file.name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })
        return sorted(logs, key=lambda x: x["modified"], reverse=True)[:days]

    @classmethod
    def read_log(cls, filename: str) -> str:
        """读取日志文件内容"""
        log_file = cls.LOG_DIR / filename
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                return f.read()
        return ""