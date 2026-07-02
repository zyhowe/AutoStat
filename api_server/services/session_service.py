"""会话管理服务 - 完全独立，不依赖 web/"""
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from api_server.config import settings


class SessionService:
    """会话管理服务 - 独立实现"""

    def __init__(self):
        self.data_dir = settings.DATA_DIR
        self.projects_dir = settings.PROJECTS_DIR
        self._ensure_dirs()

    def _ensure_dirs(self):
        """确保目录存在"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_id: str) -> Path:
        """获取会话路径"""
        return self.data_dir / session_id

    def _get_metadata_path(self, session_id: str) -> Path:
        """获取元数据文件路径"""
        return self._get_session_path(session_id) / "metadata.json"

    def _get_projects_file(self) -> Path:
        """获取项目列表文件路径"""
        return self.projects_dir / "projects.json"

    def _load_metadata(self, session_id: str) -> Optional[Dict]:
        """加载会话元数据"""
        metadata_path = self._get_metadata_path(session_id)
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_metadata(self, session_id: str, metadata: Dict):
        """保存会话元数据"""
        session_path = self._get_session_path(session_id)
        session_path.mkdir(parents=True, exist_ok=True)
        with open(self._get_metadata_path(session_id), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _load_projects(self) -> List[Dict]:
        """加载项目列表"""
        projects_file = self._get_projects_file()
        if projects_file.exists():
            with open(projects_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_projects(self, projects: List[Dict]):
        """保存项目列表"""
        with open(self._get_projects_file(), "w", encoding="utf-8") as f:
            json.dump(projects, f, ensure_ascii=False, indent=2)

    def create_session(self, source_name: str, analysis_type: str = "single", tables_info: dict = None) -> str:
        """创建会话"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{source_name}_{timestamp}"

        metadata = {
            "session_id": session_id,
            "source_name": source_name,
            "analysis_type": analysis_type,
            "created_at": datetime.now().isoformat(),
            "last_accessed_at": datetime.now().isoformat(),
            "tables_info": tables_info or {},
            "files": {},
            "variable_types": {},
            "analysis_result": None,
            "data_shape": {}
        }

        self._save_metadata(session_id, metadata)

        # 添加到项目列表
        projects = self._load_projects()
        projects.append({
            "session_id": session_id,
            "source_name": source_name,
            "analysis_type": analysis_type,
            "created_at": metadata["created_at"],
            "last_accessed_at": metadata["last_accessed_at"],
            "data_shape": {}
        })
        self._save_projects(projects)

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """获取会话信息"""
        metadata = self._load_metadata(session_id)
        if metadata:
            # 更新最后访问时间
            metadata["last_accessed_at"] = datetime.now().isoformat()
            self._save_metadata(session_id, metadata)

            # 更新项目列表
            projects = self._load_projects()
            for p in projects:
                if p.get("session_id") == session_id:
                    p["last_accessed_at"] = metadata["last_accessed_at"]
                    break
            self._save_projects(projects)
        return metadata

    def list_projects(self) -> List[Dict]:
        """列出最近项目"""
        projects = self._load_projects()
        return sorted(projects, key=lambda x: x.get("last_accessed_at", ""), reverse=True)

    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        session_path = self._get_session_path(session_id)
        if session_path.exists():
            shutil.rmtree(session_path)

        # 从项目中移除
        projects = self._load_projects()
        projects = [p for p in projects if p.get("session_id") != session_id]
        self._save_projects(projects)
        return True

    def add_file(self, session_id: str, file_name: str, file_path: str):
        """添加文件到会话"""
        metadata = self._load_metadata(session_id)
        if metadata:
            metadata["files"][file_name] = file_path
            self._save_metadata(session_id, metadata)

    def get_file(self, session_id: str) -> Optional[Dict]:
        """获取会话关联的文件"""
        metadata = self._load_metadata(session_id)
        if metadata and metadata.get("files"):
            for name, path in metadata["files"].items():
                return {"name": name, "path": path}
        return None

    def save_analysis_result(self, session_id: str, result: Dict):
        """保存分析结果"""
        metadata = self._load_metadata(session_id)
        if metadata:
            metadata["analysis_result"] = result
            metadata["data_shape"] = result.get("data_shape", {})
            self._save_metadata(session_id, metadata)

    def save_variable_types(self, session_id: str, variable_types: Dict):
        """保存变量类型"""
        metadata = self._load_metadata(session_id)
        if metadata:
            metadata["variable_types"] = variable_types
            self._save_metadata(session_id, metadata)

    def save_analyzer(self, session_id: str, analyzer):
        """保存分析器对象"""
        session_path = self._get_session_path(session_id)
        with open(session_path / "analyzer.pkl", "wb") as f:
            pickle.dump(analyzer, f)

    def get_analyzer(self, session_id: str):
        """获取分析器对象"""
        session_path = self._get_session_path(session_id)
        analyzer_path = session_path / "analyzer.pkl"
        if analyzer_path.exists():
            with open(analyzer_path, "rb") as f:
                return pickle.load(f)
        return None