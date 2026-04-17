"""模型存储模块 - 管理模型的保存和加载"""

import os
import json
import pickle
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


class ModelStorage:
    """模型存储管理器"""

    # 基础存储路径
    BASE_PATH = Path.home() / ".autostat" / "models"

    @classmethod
    def _get_model_path(cls, session_id: str, model_key: str) -> Path:
        """获取模型存储路径"""
        path = cls.BASE_PATH / session_id / model_key
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def save_model(cls, session_id: str, model_key: str, model, preprocessor,
                   metrics: Dict[str, Any], config: Dict[str, Any]):
        """
        保存模型及元信息

        参数:
        - session_id: 会话ID
        - model_key: 模型标识
        - model: 训练好的模型
        - preprocessor: 预处理器
        - metrics: 评估指标
        - config: 训练配置
        """
        model_path = cls._get_model_path(session_id, model_key)

        # 保存模型
        with open(model_path / "model.pkl", 'wb') as f:
            pickle.dump(model, f)

        # 保存预处理器
        if preprocessor:
            with open(model_path / "preprocessor.pkl", 'wb') as f:
                pickle.dump(preprocessor, f)

        # 保存元信息
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
        """加载模型"""
        model_path = cls._get_model_path(session_id, model_key)

        # 加载模型
        with open(model_path / "model.pkl", 'rb') as f:
            model = pickle.load(f)

        # 加载预处理器
        preprocessor_path = model_path / "preprocessor.pkl"
        preprocessor = None
        if preprocessor_path.exists():
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)

        # 加载元信息
        with open(model_path / "metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        return model, preprocessor, metadata

    @classmethod
    def list_models(cls, session_id: str) -> List[Dict[str, Any]]:
        """列出指定会话的所有模型"""
        session_path = cls.BASE_PATH / session_id
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

        # 按创建时间排序
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
        session_path = cls.BASE_PATH / session_id
        if session_path.exists():
            shutil.rmtree(session_path)
            return True
        return False