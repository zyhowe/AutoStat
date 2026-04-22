"""
功能开关 - 按IP隔离配置
"""

import os
import json
from typing import Dict, Any
from pathlib import Path
import streamlit as st


class FeatureFlags:
    """功能开关管理器 - 按IP隔离"""

    # 基础路径
    BASE_PATH = Path.home() / ".autostat" / "feature_flags"

    # 默认配置
    DEFAULT_FLAGS = {
        # 用户体验优化
        "demo_data": True,
        "auto_analysis": False,  # 默认关闭
        "value_preview": True,
        "empty_state_guide": True,
        "smart_summary": True,

        # 报告优化
        "conclusion_first": True,
        "natural_language_insight": True,
        "term_tooltip": True,
        "one_click_export": True,

        # 智能推荐
        "smart_params": True,
        "smart_target": True,
        "rule_based_insight": True,

        # 其他
        "usage_tips": True
    }

    @classmethod
    def _get_client_ip(cls) -> str:
        """获取客户端IP"""
        try:
            # Streamlit Cloud 或其他部署环境
            if hasattr(st, 'context') and hasattr(st.context, 'headers'):
                headers = st.context.headers
                # 尝试获取真实IP
                if 'X-Forwarded-For' in headers:
                    ip = headers['X-Forwarded-For'].split(',')[0].strip()
                    if ip:
                        return ip
                if 'X-Real-IP' in headers:
                    ip = headers['X-Real-IP']
                    if ip:
                        return ip
        except Exception:
            pass

        # 本地开发环境
        return "localhost"

    @classmethod
    def _get_config_path(cls) -> Path:
        """获取当前用户的配置文件路径"""
        client_ip = cls._get_client_ip()
        # IP中的点替换为下划线，避免路径问题
        safe_ip = client_ip.replace(':', '_').replace('.', '_')
        cls.BASE_PATH.mkdir(parents=True, exist_ok=True)
        return cls.BASE_PATH / f"{safe_ip}.json"

    @classmethod
    def _load(cls) -> Dict[str, Any]:
        """加载当前用户的配置"""
        config_path = cls._get_config_path()

        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                    # 合并默认配置
                    return {**cls.DEFAULT_FLAGS, **saved}
            except:
                pass

        return cls.DEFAULT_FLAGS.copy()

    @classmethod
    def _save(cls, flags: Dict[str, Any]):
        """保存当前用户的配置"""
        config_path = cls._get_config_path()
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(flags, f, ensure_ascii=False, indent=2)

    @classmethod
    def is_enabled(cls, flag_name: str) -> bool:
        """检查功能是否启用"""
        flags = cls._load()
        return flags.get(flag_name, False)

    @classmethod
    def enable(cls, flag_name: str):
        """启用功能"""
        flags = cls._load()
        flags[flag_name] = True
        cls._save(flags)

    @classmethod
    def disable(cls, flag_name: str):
        """禁用功能"""
        flags = cls._load()
        flags[flag_name] = False
        cls._save(flags)

    @classmethod
    def set_flag(cls, flag_name: str, enabled: bool):
        """设置功能开关状态"""
        if enabled:
            cls.enable(flag_name)
        else:
            cls.disable(flag_name)

    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """获取所有配置"""
        return cls._load()

    @classmethod
    def reset_to_default(cls):
        """重置为默认配置"""
        cls._save(cls.DEFAULT_FLAGS.copy())

    # ========== 自动分析便捷方法 ==========
    @classmethod
    def is_auto_analysis_enabled(cls) -> bool:
        """检查自动分析是否开启"""
        return cls.is_enabled("auto_analysis")

    @classmethod
    def set_auto_analysis(cls, enabled: bool):
        """设置自动分析开关"""
        cls.set_flag("auto_analysis", enabled)