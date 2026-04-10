"""
配置管理模块 - 管理数据库和大模型配置
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# 配置文件目录
CONFIG_DIR = Path.home() / ".autostat"
DB_CONFIG_FILE = CONFIG_DIR / "db_config.json"
LLM_CONFIG_FILE = CONFIG_DIR / "llm_config.json"


def ensure_config_dir():
    """确保配置目录存在"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_db_configs() -> List[Dict[str, Any]]:
    """加载数据库配置列表"""
    ensure_config_dir()
    if DB_CONFIG_FILE.exists():
        try:
            with open(DB_CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []


def save_db_configs(configs: List[Dict[str, Any]]):
    """保存数据库配置列表"""
    ensure_config_dir()
    with open(DB_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(configs, f, ensure_ascii=False, indent=2)


def add_db_config(config: Dict[str, Any]) -> bool:
    """添加数据库配置"""
    configs = load_db_configs()
    # 检查名称是否已存在
    if any(c.get('name') == config.get('name') for c in configs):
        return False
    configs.append(config)
    save_db_configs(configs)
    return True


def update_db_config(name: str, config: Dict[str, Any]) -> bool:
    """更新数据库配置"""
    configs = load_db_configs()
    for i, c in enumerate(configs):
        if c.get('name') == name:
            configs[i] = config
            save_db_configs(configs)
            return True
    return False


def delete_db_config(name: str) -> bool:
    """删除数据库配置"""
    configs = load_db_configs()
    new_configs = [c for c in configs if c.get('name') != name]
    if len(new_configs) == len(configs):
        return False
    save_db_configs(new_configs)
    return True


def load_llm_configs() -> List[Dict[str, Any]]:
    """加载大模型配置列表"""
    ensure_config_dir()
    if LLM_CONFIG_FILE.exists():
        try:
            with open(LLM_CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []


def save_llm_configs(configs: List[Dict[str, Any]]):
    """保存大模型配置列表"""
    ensure_config_dir()
    with open(LLM_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(configs, f, ensure_ascii=False, indent=2)


def add_llm_config(config: Dict[str, Any]) -> bool:
    """添加大模型配置"""
    configs = load_llm_configs()
    if any(c.get('name') == config.get('name') for c in configs):
        return False
    configs.append(config)
    save_llm_configs(configs)
    return True


def update_llm_config(name: str, config: Dict[str, Any]) -> bool:
    """更新大模型配置"""
    configs = load_llm_configs()
    for i, c in enumerate(configs):
        if c.get('name') == name:
            configs[i] = config
            save_llm_configs(configs)
            return True
    return False


def delete_llm_config(name: str) -> bool:
    """删除大模型配置"""
    configs = load_llm_configs()
    new_configs = [c for c in configs if c.get('name') != name]
    if len(new_configs) == len(configs):
        return False
    save_llm_configs(new_configs)
    return True


def test_llm_connection(config: Dict[str, Any]) -> tuple:
    """测试大模型连接

    返回: (success, message)
    """
    import requests

    api_base = config.get('api_base', '').rstrip('/')
    api_key = config.get('api_key', '')
    model = config.get('model', '')

    if not api_base or not model:
        return False, "API地址和模型名称不能为空"

    try:
        response = requests.post(
            f"{api_base}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}" if api_key else ""
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": "你好"}],
                "max_tokens": 10,
                "stream": False
            },
            timeout=10
        )

        if response.status_code == 200:
            return True, "连接成功"
        else:
            return False, f"连接失败: HTTP {response.status_code} - {response.text[:100]}"
    except requests.exceptions.Timeout:
        return False, "连接超时"
    except requests.exceptions.ConnectionError:
        return False, "连接失败: 无法连接到API地址"
    except Exception as e:
        return False, f"连接失败: {str(e)}"