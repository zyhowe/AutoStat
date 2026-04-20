# web/services/session_service.py

"""会话管理服务 - 管理分析会话和文件存储"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import streamlit as st
import pandas as pd


class SessionService:
    """会话管理服务"""

    # 基础存储路径
    BASE_PATH = Path.home() / ".autostat" / "data"
    PROJECTS_PATH = Path.home() / ".autostat" / "projects"

    @classmethod
    def _ensure_base_dir(cls):
        """确保基础目录存在"""
        cls.BASE_PATH.mkdir(parents=True, exist_ok=True)
        cls.PROJECTS_PATH.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _get_client_ip(cls) -> str:
        """获取客户端IP地址"""
        try:
            # Streamlit 1.39.0+ 官方方法
            if hasattr(st, 'context') and hasattr(st.context, 'ip_address'):
                ip = st.context.ip_address
                if ip:
                    return ip
        except Exception:
            pass

        # 兼容旧版本
        try:
            if hasattr(st, 'context') and hasattr(st.context, 'headers'):
                headers = st.context.headers
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

        return "localhost"

    @classmethod
    def _get_projects_file(cls, client_ip: str = None) -> Path:
        """获取指定IP的项目文件路径"""
        if client_ip is None:
            client_ip = cls._get_client_ip()
        # 替换IP中的特殊字符（冒号、点等）为下划线
        safe_ip = client_ip.replace(':', '_').replace('.', '_')
        return cls.PROJECTS_PATH / f"{safe_ip}.json"

    @classmethod
    def _load_projects(cls, client_ip: str = None) -> List[dict]:
        """加载指定IP的项目列表"""
        cls._ensure_base_dir()
        projects_file = cls._get_projects_file(client_ip)
        if projects_file.exists():
            with open(projects_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    @classmethod
    def _save_projects(cls, projects: List[dict], client_ip: str = None):
        """保存指定IP的项目列表"""
        cls._ensure_base_dir()
        projects_file = cls._get_projects_file(client_ip)
        with open(projects_file, "w", encoding="utf-8") as f:
            json.dump(projects, f, ensure_ascii=False, indent=2)

    @classmethod
    def _generate_session_name(cls, source_name: str, analysis_type: str, tables_info: dict = None) -> str:
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        safe_name = "".join(c for c in source_name if c.isalnum() or c in "._-")

        if analysis_type == "single":
            return f"{safe_name}_{timestamp}"
        elif analysis_type == "multi":
            if tables_info:
                first_name = list(tables_info.keys())[0] if tables_info else "unknown"
                safe_first = "".join(c for c in first_name if c.isalnum() or c in "._-")
                count = len(tables_info)
                return f"{safe_first}_等{count}个文件_{timestamp}"
            return f"多文件_{timestamp}"
        elif analysis_type == "database":
            if tables_info:
                first_name = list(tables_info.keys())[0] if tables_info else "unknown"
                safe_first = "".join(c for c in first_name if c.isalnum() or c in "._-")
                count = len(tables_info)
                return f"{safe_first}_等{count}个表_{timestamp}"
            return f"数据库_{timestamp}"
        else:
            return f"{safe_name}_{timestamp}"

    @classmethod
    def create_session(cls, source_name: str, analysis_type: str, tables_info: dict = None) -> str:
        """创建新会话，返回session_id"""
        cls._ensure_base_dir()

        session_id = cls._generate_session_name(source_name, analysis_type, tables_info)

        session_path = cls.BASE_PATH / session_id
        session_path.mkdir(parents=True, exist_ok=True)

        models_path = session_path / "models"
        models_path.mkdir(parents=True, exist_ok=True)

        client_ip = cls._get_client_ip()

        metadata = {
            "session_id": session_id,
            "source_name": source_name,
            "analysis_type": analysis_type,
            "client_ip": client_ip,
            "created_at": datetime.now().isoformat(),
            "last_accessed_at": datetime.now().isoformat(),
            "data_shape": {},
            "variable_types": {},
            "files": {},
            "models": []
        }

        cls.save_metadata(metadata, session_id)

        # 更新项目列表
        projects = cls._load_projects(client_ip)
        existing = [p for p in projects if p.get("session_id") == session_id]
        if not existing:
            projects.append({
                "session_id": session_id,
                "source_name": source_name,
                "analysis_type": analysis_type,
                "created_at": metadata["created_at"],
                "last_accessed_at": metadata["last_accessed_at"],
                "data_shape": {}
            })
            cls._save_projects(projects, client_ip)

        return session_id

    @classmethod
    def get_current_session(cls) -> Optional[str]:
        return st.session_state.get("current_session_id")

    @classmethod
    def set_current_session(cls, session_id: str):
        st.session_state.current_session_id = session_id
        cls.update_last_accessed(session_id)

    @classmethod
    def get_session_path(cls, session_id: str = None) -> Path:
        if session_id is None:
            session_id = cls.get_current_session()
        if session_id is None:
            raise ValueError("没有当前会话")
        return cls.BASE_PATH / session_id

    @classmethod
    def save_metadata(cls, metadata: dict, session_id: str = None):
        session_path = cls.get_session_path(session_id)
        metadata["updated_at"] = datetime.now().isoformat()
        with open(session_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_metadata(cls, session_id: str = None) -> dict:
        session_path = cls.get_session_path(session_id)
        metadata_path = session_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    @classmethod
    def update_last_accessed(cls, session_id: str = None):
        metadata = cls.load_metadata(session_id)
        metadata["last_accessed_at"] = datetime.now().isoformat()
        cls.save_metadata(metadata, session_id)

        # 更新项目列表中的最后访问时间
        client_ip = metadata.get("client_ip", cls._get_client_ip())
        projects = cls._load_projects(client_ip)
        for project in projects:
            if project.get("session_id") == session_id:
                project["last_accessed_at"] = metadata["last_accessed_at"]
                break
        cls._save_projects(projects, client_ip)

    @classmethod
    def update_data_shape(cls, rows: int, columns: int, session_id: str = None):
        metadata = cls.load_metadata(session_id)
        metadata["data_shape"] = {"rows": rows, "columns": columns}
        cls.save_metadata(metadata, session_id)

        # 更新项目列表中的数据形状
        client_ip = metadata.get("client_ip", cls._get_client_ip())
        projects = cls._load_projects(client_ip)
        for project in projects:
            if project.get("session_id") == session_id:
                project["data_shape"] = {"rows": rows, "columns": columns}
                break
        cls._save_projects(projects, client_ip)

    @classmethod
    def update_variable_types(cls, variable_types: dict, session_id: str = None):
        metadata = cls.load_metadata(session_id)
        metadata["variable_types"] = variable_types
        cls.save_metadata(metadata, session_id)

    @classmethod
    def add_file_record(cls, file_name: str, file_path: str, session_id: str = None):
        metadata = cls.load_metadata(session_id)
        metadata["files"][file_name] = file_path
        cls.save_metadata(metadata, session_id)

    @classmethod
    def add_model_to_session(cls, model_info: dict, session_id: str = None):
        metadata = cls.load_metadata(session_id)
        models = metadata.get("models", [])
        existing = [m for m in models if m.get("model_key") == model_info.get("model_key")]
        if not existing:
            models.append(model_info)
            metadata["models"] = models
            cls.save_metadata(metadata, session_id)

    @classmethod
    def remove_model_from_session(cls, model_key: str, session_id: str = None):
        metadata = cls.load_metadata(session_id)
        models = metadata.get("models", [])
        models = [m for m in models if m.get("model_key") != model_key]
        metadata["models"] = models
        cls.save_metadata(metadata, session_id)

    @classmethod
    def list_user_projects(cls) -> List[dict]:
        """获取当前用户的所有项目（按IP过滤）"""
        client_ip = cls._get_client_ip()
        projects = cls._load_projects(client_ip)
        return sorted(projects, key=lambda x: x.get("last_accessed_at", ""), reverse=True)

    @classmethod
    def get_current_user_ip(cls) -> str:
        """获取当前用户IP（用于显示）"""
        return cls._get_client_ip()

    @classmethod
    def load_project(cls, session_id: str):
        """加载指定项目"""
        metadata = cls.load_metadata(session_id)
        if not metadata:
            return False

        # 验证IP权限
        client_ip = cls._get_client_ip()
        if metadata.get("client_ip") != client_ip:
            return False

        cls.set_current_session(session_id)
        cls.update_last_accessed(session_id)

        return True

    @classmethod
    def project_exists(cls, session_id: str) -> bool:
        session_path = cls.BASE_PATH / session_id
        return session_path.exists()