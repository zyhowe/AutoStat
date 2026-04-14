"""
客户端配置存储 - 使用浏览器的 localStorage
"""

import streamlit as st
import json
from typing import List, Dict, Any, Optional

# localStorage 的 key 前缀
STORAGE_PREFIX = "autostat_"
DB_CONFIGS_KEY = f"{STORAGE_PREFIX}db_configs"
LLM_CONFIGS_KEY = f"{STORAGE_PREFIX}llm_configs"


def _get_storage() -> dict:
    """获取 session_state 中的存储对象"""
    if "local_storage" not in st.session_state:
        st.session_state.local_storage = {}
    return st.session_state.local_storage


def _save_to_session(key: str, data: Any):
    """保存到 session_state"""
    storage = _get_storage()
    storage[key] = data


def _load_from_session(key: str) -> Any:
    """从 session_state 加载"""
    storage = _get_storage()
    return storage.get(key, [])


def load_db_configs() -> List[Dict[str, Any]]:
    """加载数据库配置列表"""
    return _load_from_session(DB_CONFIGS_KEY)


def save_db_configs(configs: List[Dict[str, Any]]):
    """保存数据库配置列表"""
    _save_to_session(DB_CONFIGS_KEY, configs)


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
    return _load_from_session(LLM_CONFIGS_KEY)


def save_llm_configs(configs: List[Dict[str, Any]]):
    """保存大模型配置列表"""
    _save_to_session(LLM_CONFIGS_KEY, configs)


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
            return False, f"连接失败: HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "连接超时"
    except requests.exceptions.ConnectionError:
        return False, "连接失败: 无法连接到API地址"
    except Exception as e:
        return False, f"连接失败: {str(e)}"


def export_configs() -> str:
    """导出所有配置为 JSON 字符串"""
    configs = {
        "db_configs": load_db_configs(),
        "llm_configs": load_llm_configs()
    }
    return json.dumps(configs, ensure_ascii=False, indent=2)


def import_configs(json_str: str) -> bool:
    """从 JSON 字符串导入配置"""
    try:
        configs = json.loads(json_str)
        if "db_configs" in configs:
            save_db_configs(configs["db_configs"])
        if "llm_configs" in configs:
            save_llm_configs(configs["llm_configs"])
        return True
    except Exception as e:
        print(f"导入失败: {e}")
        return False