# webtext/services/session_service.py
"""文本分析会话管理服务 - 独立存储"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List


class TextSessionService:
    """文本分析会话管理 - 独立于数据分析"""

    BASE_PATH = Path.home() / ".autotext" / "data"
    PROJECTS_FILE = Path.home() / ".autotext" / "projects.json"

    @classmethod
    def _ensure_base_dir(cls):
        cls.BASE_PATH.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _generate_session_name(cls, text_preview: str) -> str:
        """生成会话名称"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        preview = text_preview[:20].replace("\n", " ").strip()
        if not preview:
            preview = "未命名"
        return f"{preview}_{timestamp}"

    @classmethod
    def create_session(cls, text_preview: str) -> str:
        """创建新会话"""
        cls._ensure_base_dir()
        session_id = cls._generate_session_name(text_preview)
        session_path = cls.BASE_PATH / session_id
        session_path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "session_id": session_id,
            "text_preview": text_preview[:50],
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "has_analysis": False
        }

        cls.save_metadata(metadata, session_id)
        cls._add_to_projects(metadata)
        return session_id

    @classmethod
    def _add_to_projects(cls, metadata: Dict):
        """添加到项目列表"""
        projects = cls.list_projects()
        existing = [p for p in projects if p["session_id"] == metadata["session_id"]]
        if not existing:
            projects.insert(0, {
                "session_id": metadata["session_id"],
                "text_preview": metadata["text_preview"],
                "created_at": metadata["created_at"],
                "last_accessed": metadata["last_accessed"],
                "has_analysis": metadata["has_analysis"]
            })
            cls._save_projects(projects)

    @classmethod
    def _save_projects(cls, projects: List[Dict]):
        """保存项目列表"""
        cls._ensure_base_dir()
        with open(cls.PROJECTS_FILE, "w", encoding="utf-8") as f:
            json.dump(projects, f, ensure_ascii=False, indent=2)

    @classmethod
    def list_projects(cls) -> List[Dict]:
        """获取所有项目"""
        cls._ensure_base_dir()
        if cls.PROJECTS_FILE.exists():
            with open(cls.PROJECTS_FILE, "r", encoding="utf-8") as f:
                projects = json.load(f)
                return sorted(projects, key=lambda x: x.get("last_accessed", ""), reverse=True)
        return []

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
        cls._update_project_metadata(session_id, metadata)

    @classmethod
    def _update_project_metadata(cls, session_id: str, metadata: Dict):
        """更新项目列表中的元数据"""
        projects = cls.list_projects()
        for p in projects:
            if p["session_id"] == session_id:
                p["has_analysis"] = metadata["has_analysis"]
                p["last_accessed"] = metadata["last_accessed"]
                break
        cls._save_projects(projects)

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
            session_path = cls.get_session_path(session_id)
            if session_path.exists():
                shutil.rmtree(session_path)

            # 从项目列表中移除
            projects = cls.list_projects()
            projects = [p for p in projects if p["session_id"] != session_id]
            cls._save_projects(projects)
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
            cls._update_project_metadata(session_id, metadata)