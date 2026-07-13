"""会话管理服务 - 完全独立，不依赖 web/"""
import json
import pickle
import shutil
import socket
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
        # ✅ 存储客户端IP，一次设置，全局使用
        self._client_ip = None

    def _ensure_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    # ✅ 设置客户端IP
    def set_client_ip(self, client_ip: str):
        self._client_ip = client_ip

    # ✅ 获取客户端IP
    def get_client_ip(self) -> str:
        return self._client_ip or "localhost"

    def _get_session_path(self, session_id: str) -> Path:
        return self.data_dir / session_id

    def _get_metadata_path(self, session_id: str) -> Path:
        return self._get_session_path(session_id) / "metadata.json"

    def _get_projects_file(self) -> Path:
        """获取项目列表文件路径（使用存储的client_ip）"""
        ip = self._client_ip or "localhost"
        ip_key = ip.replace('.', '_')
        return self.projects_dir / f"{ip_key}.json"

    # ==================== ✅ 新增：获取 Parquet 缓存路径 ====================
    def get_data_parquet_path(self, session_id: str) -> Path:
        """获取当前会话的 Parquet 缓存文件路径"""
        return self._get_session_path(session_id) / "data.parquet"

    def _load_metadata(self, session_id: str) -> Optional[Dict]:
        metadata_path = self._get_metadata_path(session_id)
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_metadata(self, session_id: str, metadata: Dict):
        session_path = self._get_session_path(session_id)
        session_path.mkdir(parents=True, exist_ok=True)
        with open(self._get_metadata_path(session_id), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _load_projects(self) -> List[Dict]:
        projects_file = self._get_projects_file()
        if projects_file.exists():
            with open(projects_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_projects(self, projects: List[Dict]):
        with open(self._get_projects_file(), "w", encoding="utf-8") as f:
            json.dump(projects, f, ensure_ascii=False, indent=2)

    def create_session(self, source_name: str, analysis_type: str = "single", tables_info: dict = None) -> str:
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
        metadata = self._load_metadata(session_id)
        if metadata:
            metadata["last_accessed_at"] = datetime.now().isoformat()
            self._save_metadata(session_id, metadata)

            projects = self._load_projects()
            for p in projects:
                if p.get("session_id") == session_id:
                    p["last_accessed_at"] = metadata["last_accessed_at"]
                    break
            self._save_projects(projects)
        return metadata

    def list_projects(self) -> List[Dict]:
        projects = self._load_projects()
        return sorted(projects, key=lambda x: x.get("last_accessed_at", ""), reverse=True)

    def delete_session(self, session_id: str) -> bool:
        session_path = self._get_session_path(session_id)
        if session_path.exists():
            shutil.rmtree(session_path)

        projects = self._load_projects()
        projects = [p for p in projects if p.get("session_id") != session_id]
        self._save_projects(projects)
        return True

    def add_file(self, session_id: str, file_name: str, file_path: str):
        metadata = self._load_metadata(session_id)
        if metadata:
            metadata["files"][file_name] = file_path
            self._save_metadata(session_id, metadata)

    def get_file(self, session_id: str) -> Optional[Dict]:
        metadata = self._load_metadata(session_id)
        if metadata and metadata.get("files"):
            for name, path in metadata["files"].items():
                return {"name": name, "path": path}
        return None

    def save_analysis_result(self, session_id: str, result: Dict):
        metadata = self._load_metadata(session_id)
        if metadata:
            metadata["analysis_result"] = result
            metadata["data_shape"] = result.get("data_shape", {})
            self._save_metadata(session_id, metadata)
            print(f"✅ 已保存 analysis_result 到 metadata: {session_id}")

            # ✅ 使用存储的client_ip
            projects = self._load_projects()
            for p in projects:
                if p.get("session_id") == session_id:
                    p["data_shape"] = result.get("data_shape", {})
                    p["last_accessed_at"] = datetime.now().isoformat()
                    break
            self._save_projects(projects)
        else:
            print(f"❌ metadata 为空，无法保存 analysis_result: {session_id}")

    def save_variable_types(self, session_id: str, variable_types: Dict):
        metadata = self._load_metadata(session_id)
        if metadata:
            metadata["variable_types"] = variable_types
            self._save_metadata(session_id, metadata)

    def save_analyzer(self, session_id: str, analyzer):
        session_path = self._get_session_path(session_id)
        with open(session_path / "analyzer.pkl", "wb") as f:
            pickle.dump(analyzer, f)

    def get_analyzer(self, session_id: str):
        session_path = self._get_session_path(session_id)
        analyzer_path = session_path / "analyzer.pkl"
        if analyzer_path.exists():
            with open(analyzer_path, "rb") as f:
                return pickle.load(f)
        return None

    # ==================== 日志管理 ====================
    def save_log(self, session_id: str, content: str):
        session_path = self._get_session_path(session_id)
        with open(session_path / "analysis.log", "w", encoding="utf-8") as f:
            f.write(content)

    def get_log(self, session_id: str) -> Optional[str]:
        session_path = self._get_session_path(session_id)
        log_path = session_path / "analysis.log"
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                return f.read()
        return None

    # ==================== HTML 报告管理 ====================
    def save_html(self, session_id: str, html_content: str):
        session_path = self._get_session_path(session_id)
        with open(session_path / "report.html", "w", encoding="utf-8") as f:
            f.write(html_content)

    def get_html(self, session_id: str) -> Optional[str]:
        session_path = self._get_session_path(session_id)
        html_path = session_path / "report.html"
        if html_path.exists():
            with open(html_path, "r", encoding="utf-8") as f:
                return f.read()
        return None

    # ==================== 推荐问题管理 ====================
    def save_recommended_questions(self, session_id: str, questions: Dict[str, List[Dict]]):
        session_path = self._get_session_path(session_id)
        questions_file = session_path / "recommended_questions.json"

        data = {
            "session_id": session_id,
            "generated_at": datetime.now().isoformat(),
            "questions": questions
        }

        with open(questions_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if hasattr(self, '_recommended_questions_cache'):
            self._recommended_questions_cache[session_id] = questions
        else:
            self._recommended_questions_cache = {session_id: questions}

    def get_recommended_questions(self, session_id: str) -> Optional[Dict[str, List[Dict]]]:
        if hasattr(self, '_recommended_questions_cache'):
            if session_id in self._recommended_questions_cache:
                return self._recommended_questions_cache[session_id]

        session_path = self._get_session_path(session_id)
        questions_file = session_path / "recommended_questions.json"

        if questions_file.exists():
            with open(questions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                questions = data.get('questions', {})

                if not hasattr(self, '_recommended_questions_cache'):
                    self._recommended_questions_cache = {}
                self._recommended_questions_cache[session_id] = questions

                return questions

        return None

    def get_recommended_questions_by_scene(self, session_id: str, scene: str) -> List[Dict]:
        questions = self.get_recommended_questions(session_id)
        if questions:
            return questions.get(scene, [])
        return []

    def get_data_path(self, session_id: str) -> Optional[str]:
        """
        获取会话的数据文件路径，优先返回 Parquet 缓存

        返回:
        - 如果 Parquet 存在，返回 Parquet 路径
        - 否则返回原始文件路径（如果有）
        - 否则返回 None
        """
        # 1. 优先 Parquet
        parquet_path = self.get_data_parquet_path(session_id)
        if parquet_path.exists():
            print(f"📂 使用 Parquet 缓存: {parquet_path}")
            return str(parquet_path)

        # 2. 回退到原始文件
        file_info = self.get_file(session_id)
        if file_info:
            print(f"📂 Parquet 不存在，使用原始文件: {file_info['path']}")
            return file_info['path']

        return None