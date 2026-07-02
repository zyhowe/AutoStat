"""文件工具"""
import os
from pathlib import Path
import re
from typing import Optional

from api_server.config import settings


def get_file_extension(filename: str) -> str:
    """获取文件扩展名"""
    ext = Path(filename).suffix.lower()
    return ext


def is_allowed_extension(filename: str) -> bool:
    """检查扩展名是否允许"""
    ext = get_file_extension(filename)
    return ext in settings.ALLOWED_EXTENSIONS


def get_file_size(file_path: str) -> int:
    """获取文件大小（字节）"""
    return os.path.getsize(file_path)


def safe_filename(filename: str) -> str:
    """生成安全的文件名"""
    # 移除路径
    filename = os.path.basename(filename)
    # 移除特殊字符
    filename = re.sub(r'[^\w\-_.]', '_', filename)
    return filename


def ensure_directory(path: Path) -> Path:
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)
    return path