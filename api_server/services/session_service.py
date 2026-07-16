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
        self._client_ip = None

    def _ensure_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def set_client_ip(self, client_ip: str):
        self._client_ip = client_ip

    def get_client_ip(self) -> str:
        return self._client_ip or "localhost"

    def _get_session_path(self, session_id: str) -> Path:
        return self.data_dir / session_id

    def _get_metadata_path(self, session_id: str) -> Path:
        return self._get_session_path(session_id) / "metadata.json"

    def _get_projects_file(self) -> Path:
        ip = self._client_ip or "localhost"
        ip_key = ip.replace('.', '_')
        return self.projects_dir / f"{ip_key}.json"

    # ==================== ✅ 统一 Parquet 路径 ====================
    def get_table_parquet_path(self, session_id: str, table_name: str) -> Path:
        """
        获取指定表的 Parquet 路径
        统一格式: ~/.autostat/data/{session_id}/{table_name}.parquet
        """
        return self._get_session_path(session_id) / f"{table_name}.parquet"

    def get_data_parquet_path(self, session_id: str) -> Path:
        """
        兼容旧代码，默认表名为 'data'
        实际调用 get_table_parquet_path(session_id, "data")
        """
        return self.get_table_parquet_path(session_id, "data")

    def get_merged_parquet_path(self, session_id: str) -> Path:
        """获取合并表的 Parquet 路径（表名为 'merged'）"""
        return self.get_table_parquet_path(session_id, "merged")

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

    # ==================== 多表支持 ====================
    def get_tables_dir(self, session_id: str) -> Path:
        """获取会话的表存储目录（保留用于兼容，但实际已不使用子目录）"""
        session_path = self._get_session_path(session_id)
        tables_dir = session_path / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        return tables_dir

    def get_all_table_names(self, session_id: str) -> List[str]:
        """获取会话的所有表名（从 tables_meta 读取）"""
        metadata = self._load_metadata(session_id)
        if metadata:
            tables_meta = metadata.get("tables_meta", {})
            return list(tables_meta.keys())
        return []

    def save_relationships(self, session_id: str, relationships: List[Dict]):
        """保存表间关系"""
        metadata = self._load_metadata(session_id)
        if not metadata:
            metadata = {}
        metadata["relationships"] = relationships
        self._save_metadata(session_id, metadata)

    def get_relationships(self, session_id: str) -> List[Dict]:
        """获取表间关系"""
        metadata = self._load_metadata(session_id)
        if metadata:
            return metadata.get("relationships", [])
        return []

    def get_table_count(self, session_id: str) -> int:
        """获取会话的表数量"""
        return len(self.get_all_table_names(session_id))

    def is_multi_table(self, session_id: str) -> bool:
        """判断是否为多表会话"""
        return self.get_table_count(session_id) > 1

    def save_table_info(self, session_id: str, table_name: str, info: Dict):
        """保存单个表的元信息"""
        metadata = self._load_metadata(session_id)
        if not metadata:
            metadata = {}
        if "tables_meta" not in metadata:
            metadata["tables_meta"] = {}
        metadata["tables_meta"][table_name] = info
        self._save_metadata(session_id, metadata)

    def get_tables_meta(self, session_id: str) -> Dict:
        """获取所有表的元信息"""
        metadata = self._load_metadata(session_id)
        if metadata:
            return metadata.get("tables_meta", {})
        return {}

    # ==================== 兼容旧方法 ====================
    def get_data_path(self, session_id: str) -> Optional[str]:
        """
        获取会话的数据文件路径（兼容旧版）
        优先返回 Parquet 路径，如果不存在则返回原始文件路径
        """
        # 1. 检查是否有表（从 Parquet 加载）
        table_names = self.get_all_table_names(session_id)
        if table_names:
            parquet_path = self.get_table_parquet_path(session_id, table_names[0])
            if parquet_path.exists():
                return str(parquet_path)

        # 2. 检查根目录 data.parquet
        parquet_path = self.get_data_parquet_path(session_id)
        if parquet_path.exists():
            return str(parquet_path)

        # 3. 回退到原始文件
        file_info = self.get_file(session_id)
        if file_info:
            return file_info['path']

        return None