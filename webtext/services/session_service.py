# webtext/services/session_service.py
"""文本分析会话管理服务 - 按IP隔离"""

import json
import shutil
import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List


class TextSessionService:
    """文本分析会话管理 - 按IP隔离"""

    BASE_PATH = Path.home() / ".autotext" / "data"
    PROJECTS_PATH = Path.home() / ".autotext" / "projects"

    @classmethod
    def _ensure_base_dir(cls):
        """确保基础目录存在"""
        cls.BASE_PATH.mkdir(parents=True, exist_ok=True)
        cls.PROJECTS_PATH.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _get_client_ip(cls) -> str:
        """获取客户端IP"""
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
        """获取当前用户的项目文件路径"""
        if client_ip is None:
            client_ip = cls._get_client_ip()
        # IP中的点替换为下划线，避免路径问题
        safe_ip = client_ip.replace(':', '_').replace('.', '_')
        return cls.PROJECTS_PATH / f"{safe_ip}.json"

    @classmethod
    def _load_projects(cls, client_ip: str = None) -> List[Dict]:
        """加载当前用户的项目列表"""
        cls._ensure_base_dir()
        projects_file = cls._get_projects_file(client_ip)
        if projects_file.exists():
            try:
                with open(projects_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return []
        return []

    @classmethod
    def _save_projects(cls, projects: List[Dict], client_ip: str = None):
        """保存当前用户的项目列表"""
        cls._ensure_base_dir()
        projects_file = cls._get_projects_file(client_ip)
        with open(projects_file, "w", encoding="utf-8") as f:
            json.dump(projects, f, ensure_ascii=False, indent=2)

    @classmethod
    def _generate_session_name(cls, text_preview: str) -> str:
        """生成会话名称"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        preview = text_preview[:30].replace("\n", " ").strip()
        if not preview:
            preview = "未命名"
        return f"{preview}_{timestamp}"

    @classmethod
    def create_session(cls, text_preview: str, client_ip: str = None) -> str:
        """创建新会话"""
        cls._ensure_base_dir()

        if client_ip is None:
            client_ip = cls._get_client_ip()

        session_id = cls._generate_session_name(text_preview)
        session_path = cls.BASE_PATH / session_id
        session_path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "session_id": session_id,
            "text_preview": text_preview[:50],
            "client_ip": client_ip,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "has_analysis": False
        }

        cls.save_metadata(metadata, session_id)
        cls._add_to_projects(metadata, client_ip)
        return session_id

    @classmethod
    def _add_to_projects(cls, metadata: Dict, client_ip: str = None):
        """添加到项目列表"""
        projects = cls._load_projects(client_ip)

        # 检查是否已存在
        existing = [p for p in projects if p["session_id"] == metadata["session_id"]]
        if not existing:
            projects.insert(0, {
                "session_id": metadata["session_id"],
                "text_preview": metadata["text_preview"],
                "created_at": metadata["created_at"],
                "last_accessed": metadata["last_accessed"],
                "has_analysis": metadata["has_analysis"]
            })
            cls._save_projects(projects, client_ip)

    @classmethod
    def list_projects(cls, client_ip: str = None) -> List[Dict]:
        """获取当前用户的所有项目"""
        projects = cls._load_projects(client_ip)
        return sorted(projects, key=lambda x: x.get("last_accessed", ""), reverse=True)

    @classmethod
    def get_session_path(cls, session_id: str) -> Path:
        """获取会话路径"""
        return cls.BASE_PATH / session_id

    @classmethod
    def save_metadata(cls, metadata: Dict, session_id: str = None):
        """保存元数据"""
        if session_id is None:
            return
        session_path = cls.get_session_path(session_id)
        with open(session_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_metadata(cls, session_id: str) -> Dict:
        """加载元数据"""
        session_path = cls.get_session_path(session_id)
        metadata_path = session_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    @classmethod
    def save_analysis_results(cls, session_id: str, html_content: str, json_data: Dict):
        """保存分析结果"""
        session_path = cls.get_session_path(session_id)

        # 保存 HTML
        with open(session_path / "report.html", "w", encoding="utf-8") as f:
            f.write(html_content)

        # 保存 JSON
        with open(session_path / "result.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        # 更新元数据
        metadata = cls.load_metadata(session_id)
        metadata["has_analysis"] = True
        metadata["last_accessed"] = datetime.now().isoformat()
        cls.save_metadata(metadata, session_id)

        # 更新项目列表
        client_ip = metadata.get("client_ip")
        cls._update_project_metadata(session_id, metadata, client_ip)

    @classmethod
    def _update_project_metadata(cls, session_id: str, metadata: Dict, client_ip: str = None):
        """更新项目列表中的元数据"""
        projects = cls._load_projects(client_ip)
        for p in projects:
            if p["session_id"] == session_id:
                p["has_analysis"] = metadata["has_analysis"]
                p["last_accessed"] = metadata["last_accessed"]
                break
        cls._save_projects(projects, client_ip)

    @classmethod
    def load_html(cls, session_id: str) -> Optional[str]:
        """加载 HTML 报告"""
        session_path = cls.get_session_path(session_id)
        html_path = session_path / "report.html"
        if html_path.exists():
            with open(html_path, "r", encoding="utf-8") as f:
                return f.read()
        return None

    @classmethod
    def load_json(cls, session_id: str) -> Optional[Dict]:
        """加载 JSON 结果"""
        session_path = cls.get_session_path(session_id)
        json_path = session_path / "result.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    @classmethod
    def delete_session(cls, session_id: str) -> bool:
        """删除会话"""
        try:
            # 获取元数据以获取 client_ip
            metadata = cls.load_metadata(session_id)
            client_ip = metadata.get("client_ip")

            # 删除会话目录
            session_path = cls.get_session_path(session_id)
            if session_path.exists():
                shutil.rmtree(session_path)

            # 从项目列表中移除
            projects = cls._load_projects(client_ip)
            projects = [p for p in projects if p["session_id"] != session_id]
            cls._save_projects(projects, client_ip)
            return True
        except Exception as e:
            print(f"删除失败: {e}")
            return False

    @classmethod
    def update_last_accessed(cls, session_id: str):
        """更新最后访问时间"""
        metadata = cls.load_metadata(session_id)
        if metadata:
            metadata["last_accessed"] = datetime.now().isoformat()
            cls.save_metadata(metadata, session_id)

            client_ip = metadata.get("client_ip")
            cls._update_project_metadata(session_id, metadata, client_ip)

    @classmethod
    def verify_session_ownership(cls, session_id: str) -> bool:
        """验证会话是否属于当前用户"""
        metadata = cls.load_metadata(session_id)
        if not metadata:
            return False
        client_ip = metadata.get("client_ip")
        current_ip = cls._get_client_ip()
        return client_ip == current_ip