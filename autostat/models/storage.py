"""模型存储模块"""
import os
import json
import pickle
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


class ModelStorage:
    """模型存储管理器"""

    BASE_PATH = Path.home() / ".autostat" / "data"

    @classmethod
    def _get_model_path(cls, session_id: str, model_key: str) -> Path:
        """获取模型存储路径: data/{session_id}/models/{model_key}/"""
        path = cls.BASE_PATH / session_id / "models" / model_key
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def save_model(cls, session_id: str, model_key: str, model, preprocessor,
                   metrics: Dict[str, Any], config: Dict[str, Any]):
        """保存模型"""
        model_path = cls._get_model_path(session_id, model_key)

        with open(model_path / "model.pkl", 'wb') as f:
            pickle.dump(model, f)

        if preprocessor:
            with open(model_path / "preprocessor.pkl", 'wb') as f:
                pickle.dump(preprocessor, f)

        metadata = {
            "model_key": model_key,
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics,
            "config": config
        }

        with open(model_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return str(model_path)

    @classmethod
    def load_model(cls, session_id: str, model_key: str):
        """加载模型，返回 (model, preprocessor, metadata)"""
        model_path = cls._get_model_path(session_id, model_key)

        with open(model_path / "model.pkl", 'rb') as f:
            model = pickle.load(f)

        preprocessor_path = model_path / "preprocessor.pkl"
        preprocessor = None
        if preprocessor_path.exists():
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)

        with open(model_path / "metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        return model, preprocessor, metadata

    @classmethod
    def list_models(cls, session_id: str) -> List[Dict]:
        """列出指定会话的所有模型"""
        session_path = cls.BASE_PATH / session_id / "models"
        if not session_path.exists():
            return []

        models = []
        for model_dir in session_path.iterdir():
            if model_dir.is_dir():
                metadata_path = model_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    models.append(metadata)

        models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return models

    @classmethod
    def get_best_model(cls, session_id: str, metric: str = "accuracy") -> Optional[Dict[str, Any]]:
        """获取最佳模型（按指定指标）"""
        models = cls.list_models(session_id)
        if not models:
            return None

        best = None
        best_score = -1

        for model_info in models:
            metrics = model_info.get("metrics", {})
            score = metrics.get(metric, -1)
            if score > best_score:
                best_score = score
                best = model_info

        return best

    @classmethod
    def delete_model(cls, session_id: str, model_key: str) -> bool:
        """删除模型"""
        model_path = cls._get_model_path(session_id, model_key)
        if model_path.exists():
            shutil.rmtree(model_path)
            return True
        return False

    @classmethod
    def delete_session(cls, session_id: str) -> bool:
        """删除会话的所有模型"""
        session_path = cls.BASE_PATH / session_id / "models"
        if session_path.exists():
            shutil.rmtree(session_path)
            return True
        return False