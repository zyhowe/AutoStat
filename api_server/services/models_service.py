"""模型服务"""
import os
import json
import pickle
import time
import traceback
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from autostat.models.registry import ModelRegistry
from autostat.models.trainer import ModelTrainer
from autostat.models.preprocessing import DataPreprocessor
from autostat.models.storage import ModelStorage
from autostat.models.predictor import ModelPredictor
from autostat.loader import DataLoader
from autostat.core.base import BaseAnalyzer

from api_server.config import settings


class ModelsService:
    """模型服务"""

    def __init__(self):
        self.train_status = {}

    def train_model(
        self,
        session_id: str,
        file_path: str,
        task_type: str,
        model_key: str,
        target_column: Optional[str],
        features: List[str],
        params: Dict[str, Any],
        user_model_name: Optional[str],
        task_id: str
    ):
        """训练模型（后台任务）"""
        try:
            self.train_status[task_id] = {
                "status": "running",
                "progress": 10,
                "message": "加载数据中..."
            }

            df = DataLoader.load_from_file(file_path)

            self.train_status[task_id] = {
                "status": "running",
                "progress": 30,
                "message": "准备数据..."
            }

            X = df[features].copy()

            if task_type in ["classification", "regression"] and target_column:
                y = df[target_column].copy()
            else:
                y = None

            mask = ~X.isna().any(axis=1)
            if y is not None:
                mask = mask & ~y.isna()
            X = X[mask]
            if y is not None:
                y = y[mask]

            if len(X) < 10:
                raise ValueError(f"有效样本不足: {len(X)} < 10")

            self.train_status[task_id] = {
                "status": "running",
                "progress": 50,
                "message": "训练模型中..."
            }

            categorical_features = [col for col in features if col in df.select_dtypes(include=['object']).columns]
            numerical_features = [col for col in features if col not in categorical_features]

            preprocess_config = {
                "missing_strategy": "drop",
                "scaling": "standard",
                "encoding": "onehot",
                "categorical_features": categorical_features,
                "numerical_features": numerical_features,
                "target_column": target_column
            }

            preprocessor = DataPreprocessor(preprocess_config)

            if task_type in ["classification", "regression"]:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
                X_train_processed = preprocessor.fit_transform(X_train, y_train)
                X_test_processed = preprocessor.transform(X_test)
            else:
                X_train_processed = preprocessor.fit_transform(X)
                X_test_processed = None
                y_train = None
                y_test = None

            trainer = ModelTrainer(task_type, model_key, params)
            train_result = trainer.train(
                X_train_processed, y_train,
                X_test_processed, y_test,
                cv_folds=5
            )

            self.train_status[task_id] = {
                "status": "running",
                "progress": 80,
                "message": "保存模型中..."
            }

            if user_model_name is None:
                user_model_name = f"{task_type}_{model_key}_{int(time.time())}"

            model_key_full = f"{task_type}_{model_key}_{int(time.time())}"

            # 🆕 调用修改后的 ModelStorage，参数顺序: session_id, model_key
            ModelStorage.save_model(
                session_id=session_id,
                model_key=model_key_full,
                model=trainer.model,
                preprocessor=preprocessor,
                metrics=train_result.get("train_score", {}),
                config={
                    "task_type": task_type,
                    "model_type": model_key,
                    "target_column": target_column,
                    "features": features,
                    "params": params,
                    "user_model_name": user_model_name
                }
            )

            self.train_status[task_id] = {
                "status": "completed",
                "progress": 100,
                "message": "训练完成",
                "model_key": model_key_full,
                "user_model_name": user_model_name
            }

        except Exception as e:
            self.train_status[task_id] = {
                "status": "failed",
                "progress": 0,
                "message": f"训练失败: {str(e)}",
                "error": traceback.format_exc()
            }

    def list_models(self, session_id: str) -> List[Dict]:
        """列出已训练模型"""
        # 🆕 调用修改后的 ModelStorage，参数: session_id
        return ModelStorage.list_models(session_id)

    def predict(self, model_key: str, input_values: Dict[str, Any], session_id: str) -> Optional[Dict]:
        """执行预测"""
        try:
            # 🆕 调用修改后的 ModelStorage，参数: session_id, model_key
            model, preprocessor, metadata = ModelStorage.load_model(session_id, model_key)

            if model is None:
                return None

            features = metadata.get("features", [])
            if not features:
                config = metadata.get("config", {})
                features = config.get("features", [])

            if not features:
                return {"error": "模型没有特征信息"}

            df = pd.DataFrame([input_values])
            for feature in features:
                if feature not in df.columns:
                    df[feature] = None

            df = df[features]

            predictor = ModelPredictor(model, preprocessor)
            result = predictor.predict_with_confidence(df)

            prediction = result.get("prediction")
            if prediction is None:
                predictions = result.get("predictions")
                if predictions and isinstance(predictions, list) and len(predictions) == 1:
                    prediction = predictions[0]

            confidence = result.get("confidence")
            if confidence is None and result.get("probabilities"):
                probs = result.get("probabilities")
                if probs and isinstance(probs, list) and len(probs) > 0:
                    if isinstance(probs[0], list):
                        confidence = max(probs[0])
                    else:
                        confidence = max(probs)

            return {
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": result.get("probabilities"),
                "model_name": metadata.get("user_model_name", model_key)
            }

        except Exception as e:
            print(f"预测失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def delete_model(self, session_id: str, model_key: str) -> bool:
        """删除模型"""
        # 🆕 调用修改后的 ModelStorage，参数: session_id, model_key
        return ModelStorage.delete_model(session_id, model_key)

    def get_train_status(self, task_id: str) -> Dict:
        """获取训练状态"""
        return self.train_status.get(task_id, {"status": "not_found", "progress": 0})

    def get_alert_rules(self) -> List[Dict]:
        """获取预警规则"""
        return [
            {"id": "value_below", "name": "数值低于阈值", "level": "warning", "enabled": True},
            {"id": "value_above", "name": "数值高于阈值", "level": "warning", "enabled": True},
            {"id": "continuous_decline", "name": "连续下降（3期）", "level": "error", "enabled": True},
            {"id": "continuous_increase", "name": "连续上升（5期）", "level": "info", "enabled": True},
            {"id": "out_of_range", "name": "超出预测范围", "level": "error", "enabled": True},
            {"id": "high_anomaly_rate", "name": "异常率过高", "level": "warning", "enabled": True}
        ]

    def check_alert(self, data: Dict[str, Any]) -> List[Dict]:
        """检查预警"""
        alerts = []
        value = data.get("value", 0)
        threshold = data.get("threshold", 0)
        target = data.get("target", "未知")

        if value < threshold * 0.8:
            alerts.append({
                "level": "error",
                "title": f"{target} 严重低于阈值",
                "message": f"当前值 {value} 低于阈值 {threshold} 的80%"
            })
        elif value < threshold:
            alerts.append({
                "level": "warning",
                "title": f"{target} 低于阈值",
                "message": f"当前值 {value} 低于阈值 {threshold}"
            })

        return alerts

    def _list_sessions(self) -> List[str]:
        """列出所有会话"""
        data_dir = settings.DATA_DIR
        if data_dir.exists():
            return [d.name for d in data_dir.iterdir() if d.is_dir()]
        return []