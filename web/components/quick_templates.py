"""快速模板组件 - 保存和加载分析模板"""

import streamlit as st
import json
from pathlib import Path
from typing import Dict, Any


class QuickTemplates:
    TEMPLATE_PATH = Path.home() / ".autostat" / "templates"

    @classmethod
    def save_template(cls, name: str, config: Dict):
        """保存模板"""
        cls.TEMPLATE_PATH.mkdir(parents=True, exist_ok=True)
        with open(cls.TEMPLATE_PATH / f"{name}.json", 'w') as f:
            json.dump(config, f)

    @classmethod
    def load_template(cls, name: str) -> Dict:
        """加载模板"""
        with open(cls.TEMPLATE_PATH / f"{name}.json", 'r') as f:
            return json.load(f)

    @classmethod
    def list_templates(cls) -> list:
        """列出所有模板"""
        if not cls.TEMPLATE_PATH.exists():
            return []
        return [f.stem for f in cls.TEMPLATE_PATH.glob("*.json")]