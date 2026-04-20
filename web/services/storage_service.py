# web/services/storage_service.py

"""统一存储服务 - 管理文件存储"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

import pandas as pd
import streamlit as st

from web.services.session_service import SessionService


class StorageService:
    """统一存储服务"""

    @classmethod
    def _get_session_path(cls, session_id: str = None) -> Path:
        """获取会话路径"""
        return SessionService.get_session_path(session_id)

    @classmethod
    def _get_models_path(cls, session_id: str = None) -> Path:
        """获取模型目录路径"""
        session_path = cls._get_session_path(session_id)
        models_path = session_path / "models"
        models_path.mkdir(parents=True, exist_ok=True)
        return models_path

    # ==================== DataFrame 存储 ====================

    @classmethod
    def save_dataframe(cls, name: str, df: pd.DataFrame, session_id: str = None):
        """保存DataFrame到pkl文件"""
        session_path = cls._get_session_path(session_id)
        file_path = session_path / f"{name}.pkl"
        df.to_pickle(file_path)
        SessionService.add_file_record(name, f"{name}.pkl", session_id)

    @classmethod
    def load_dataframe(cls, name: str, session_id: str = None) -> Optional[pd.DataFrame]:
        """加载DataFrame"""
        session_path = cls._get_session_path(session_id)
        file_path = session_path / f"{name}.pkl"
        if file_path.exists():
            return pd.read_pickle(file_path)
        return None

    # ==================== JSON 存储 ====================

    @classmethod
    def save_json(cls, name: str, data: dict, session_id: str = None):
        """保存JSON文件"""
        session_path = cls._get_session_path(session_id)
        file_path = session_path / f"{name}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        SessionService.add_file_record(name, f"{name}.json", session_id)

    @classmethod
    def load_json(cls, name: str, session_id: str = None) -> Optional[dict]:
        """加载JSON文件"""
        session_path = cls._get_session_path(session_id)
        file_path = session_path / f"{name}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    # ==================== 文本存储 ====================

    @classmethod
    def save_text(cls, name: str, content: str, session_id: str = None):
        """保存文本文件"""
        session_path = cls._get_session_path(session_id)
        file_path = session_path / f"{name}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        SessionService.add_file_record(name, f"{name}.txt", session_id)

    @classmethod
    def load_text(cls, name: str, session_id: str = None) -> Optional[str]:
        """加载文本文件"""
        session_path = cls._get_session_path(session_id)
        file_path = session_path / f"{name}.txt"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        return None

    # ==================== 模型存储 ====================

    @classmethod
    def save_model(cls, model_key: str, model, preprocessor,
                   metadata: dict, metrics: dict, session_id: str = None):
        """保存模型到会话目录"""
        models_path = cls._get_models_path(session_id)
        model_path = models_path / model_key
        model_path.mkdir(parents=True, exist_ok=True)

        # 保存模型
        with open(model_path / "model.pkl", "wb") as f:
            pickle.dump(model, f)

        # 保存预处理器
        if preprocessor:
            with open(model_path / "preprocessor.pkl", "wb") as f:
                pickle.dump(preprocessor, f)

        # 保存元数据
        with open(model_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # 保存指标
        with open(model_path / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        # 更新会话元数据中的模型列表
        model_info = {
            "model_key": model_key,
            "user_model_name": metadata.get("user_model_name", model_key),
            "task_type": metadata.get("task_type"),
            "model_type": metadata.get("model_type"),
            "target_column": metadata.get("target_column"),
            "features": metadata.get("features", []),
            "created_at": metadata.get("created_at"),
            "metrics": metrics.get("train_score", {})
        }
        SessionService.add_model_to_session(model_info, session_id)

    @classmethod
    def load_model(cls, model_key: str, session_id: str = None) -> Tuple[Any, Any, dict, dict]:
        """加载模型，返回 (model, preprocessor, metadata, metrics)"""
        models_path = cls._get_models_path(session_id)
        model_path = models_path / model_key

        # 检查路径是否存在
        if not model_path.exists():
            raise FileNotFoundError(f"模型目录不存在: {model_path}")

        # 加载模型
        model_file = model_path / "model.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file}")

        with open(model_file, "rb") as f:
            model = pickle.load(f)

        # 加载预处理器
        preprocessor_path = model_path / "preprocessor.pkl"
        preprocessor = None
        if preprocessor_path.exists():
            with open(preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)

        # 加载元数据
        metadata_path = model_path / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        # 加载指标
        metrics = {}
        metrics_path = model_path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

        return model, preprocessor, metadata, metrics

    @classmethod
    def list_models(cls, session_id: str = None) -> List[dict]:
        """列出会话中的所有模型"""
        metadata = SessionService.load_metadata(session_id)
        return metadata.get("models", [])

    @classmethod
    def delete_model(cls, model_key: str, session_id: str = None) -> bool:
        """删除模型"""
        import shutil
        models_path = cls._get_models_path(session_id)
        model_path = models_path / model_key
        if model_path.exists():
            shutil.rmtree(model_path)
            SessionService.remove_model_from_session(model_key, session_id)
            return True
        return False

    @classmethod
    def get_model_path(cls, model_key: str, session_id: str = None) -> Path:
        """获取模型路径"""
        models_path = cls._get_models_path(session_id)
        return models_path / model_key

    # ==================== 分析结果存储 ====================

    @classmethod
    def save_analysis_results(cls, session_id: str, analyzer, html_content: str,
                              json_data: dict, log_content: str, filtered_df: pd.DataFrame):
        """保存分析结果"""
        # 保存原始数据
        cls.save_dataframe("raw_data", filtered_df, session_id)

        # 保存处理后的数据（含日期派生列）
        cls.save_dataframe("processed_data", analyzer.data, session_id)

        # 保存变量类型
        cls.save_json("variable_types", analyzer.variable_types, session_id)

        # 保存HTML报告
        cls.save_text("analysis_report", html_content, session_id)

        # 保存JSON结果
        cls.save_json("analysis_result", json_data, session_id)

        # 保存分析日志
        cls.save_text("analysis_log", log_content, session_id)

        # 更新会话元数据
        SessionService.update_data_shape(len(analyzer.data), len(analyzer.data.columns), session_id)
        SessionService.update_variable_types(analyzer.variable_types, session_id)