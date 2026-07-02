"""配置管理服务 - 复用 autostat/config_manager.py """
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from autostat.config_manager import (
    load_db_configs, save_db_configs, add_db_config,
    update_db_config, delete_db_config,
    load_llm_configs, save_llm_configs, add_llm_config,
    update_llm_config, delete_llm_config, test_llm_connection
)


class ConfigService:
    """配置管理服务"""

    @staticmethod
    def get_db_configs() -> List[Dict]:
        """获取数据库配置列表"""
        return load_db_configs()

    @staticmethod
    def save_db_config(config: Dict) -> bool:
        """保存数据库配置"""
        # 检查是否已存在
        configs = load_db_configs()
        if any(c.get('name') == config.get('name') for c in configs):
            return False
        configs.append(config)
        save_db_configs(configs)
        return True

    @staticmethod
    def delete_db_config(name: str) -> bool:
        """删除数据库配置"""
        return delete_db_config(name)

    @staticmethod
    def get_llm_configs() -> List[Dict]:
        """获取大模型配置列表"""
        return load_llm_configs()

    @staticmethod
    def save_llm_config(config: Dict) -> bool:
        """保存大模型配置"""
        configs = load_llm_configs()
        if any(c.get('name') == config.get('name') for c in configs):
            return False
        configs.append(config)
        save_llm_configs(configs)
        return True

    @staticmethod
    def delete_llm_config(name: str) -> bool:
        """删除大模型配置"""
        return delete_llm_config(name)

    @staticmethod
    def test_llm_connection(config: Dict) -> Tuple[bool, str]:
        """测试大模型连接"""
        return test_llm_connection(config)