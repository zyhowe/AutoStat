"""推理器模块 - 模型推理预测"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
from autostat.models.preprocessing import DataPreprocessor
from autostat.models.registry import ModelRegistry


class ModelPredictor:
    """模型推理器 - 支持单样本和批量预测"""

    def __init__(self, model, preprocessor: DataPreprocessor = None,
                 task_type: str = None, model_key: str = None):
        """
        初始化推理器

        参数:
        - model: 训练好的模型
        - preprocessor: 预处理器
        - task_type: 任务类型
        - model_key: 模型标识
        """
        self.model = model
        self.preprocessor = preprocessor
        self.task_type = task_type
        self.model_key = model_key

        if model_key:
            self.model_info = ModelRegistry.get_model(task_type, model_key)
        else:
            self.model_info = None

    def predict(self, X: Union[np.ndarray, pd.DataFrame, List, Dict]) -> Dict[str, Any]:
        """
        执行预测

        参数:
        - X: 输入数据（数组、DataFrame、列表或字典）

        返回:
        - 预测结果字典
        """
        # 数据预处理
        X_processed = self._preprocess_input(X)

        # 执行预测
        if self.task_type == "clustering":
            predictions = self.model.predict(X_processed)
            result = self._format_clustering_result(predictions, X_processed)
        elif self.task_type == "classification":
            predictions = self.model.predict(X_processed)
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_processed)
            result = self._format_classification_result(predictions, probabilities)
        elif self.task_type == "regression":
            predictions = self.model.predict(X_processed)
            result = self._format_regression_result(predictions)
        elif self.task_type == "time_series":
            predictions = self.model.predict(X_processed)
            result = self._format_timeseries_result(predictions)
        else:
            predictions = self.model.predict(X_processed)
            result = {"predictions": predictions.tolist() if hasattr(predictions, 'tolist') else predictions}

        # 添加置信度/不确定性信息
        result = self._add_confidence_info(result)

        return result

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame, List, Dict]) -> np.ndarray:
        """获取分类概率"""
        if self.task_type != "classification":
            raise ValueError("predict_proba 仅支持分类任务")

        X_processed = self._preprocess_input(X)

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_processed)
        else:
            # 对于不支持概率的模型，返回 one-hot 编码的预测
            predictions = self.model.predict(X_processed)
            # 处理字符串标签
            unique_labels = np.unique(predictions)
            label_to_idx = {label: i for i, label in enumerate(unique_labels)}
            n_classes = len(unique_labels)
            proba = np.zeros((len(predictions), n_classes))
            for i, pred in enumerate(predictions):
                idx = label_to_idx[pred]
                proba[i, idx] = 1
            return proba

    def predict_with_confidence(self, X: Union[np.ndarray, pd.DataFrame, List, Dict]) -> Dict[str, Any]:
        """
        带置信度的预测

        返回:
        - 包含预测值和置信度的字典
        """
        result = self.predict(X)

        if self.task_type == "classification":
            try:
                proba = self.predict_proba(X)
                confidence = np.max(proba, axis=1)
                result["confidence"] = confidence.tolist()
                result["confidence_mean"] = float(np.mean(confidence))
            except Exception as e:
                result["confidence"] = None
                result["confidence_mean"] = None

        return result

    def _preprocess_input(self, X: Union[np.ndarray, pd.DataFrame, List, Dict]) -> np.ndarray:
        """预处理输入数据"""
        # 转换为 DataFrame
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, list):
            if len(X) > 0 and isinstance(X[0], dict):
                X = pd.DataFrame(X)
            else:
                X = np.array(X).reshape(1, -1) if len(np.array(X).shape) == 1 else np.array(X)

        # 应用预处理器
        if self.preprocessor is not None:
            if isinstance(X, pd.DataFrame):
                X_processed = self.preprocessor.transform(X)
            else:
                # 需要重建列名
                feature_names = self.preprocessor.get_feature_names()
                if len(feature_names) == X.shape[1]:
                    X_df = pd.DataFrame(X, columns=feature_names)
                    X_processed = self.preprocessor.transform(X_df)
                else:
                    X_processed = X
        else:
            X_processed = X if isinstance(X, np.ndarray) else np.array(X)

        return X_processed

    def _format_classification_result(self, predictions: np.ndarray,
                                        probabilities: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """格式化分类结果"""
        # 处理预测值：可能是数字、字符串或数组
        if len(predictions) == 1:
            pred_value = predictions[0]
            # 如果是 numpy 类型，转换为 Python 原生类型
            if hasattr(pred_value, 'item'):
                pred_value = pred_value.item()
            # 如果是字节类型，解码为字符串
            if isinstance(pred_value, bytes):
                pred_value = pred_value.decode('utf-8')
            prediction = pred_value
        else:
            prediction = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)

        result = {
            "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
            "prediction": prediction
        }

        if probabilities is not None:
            result["probabilities"] = probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities
            if len(probabilities) == 1:
                result["confidence"] = float(np.max(probabilities[0]))

        return result

    def _format_regression_result(self, predictions: np.ndarray) -> Dict[str, Any]:
        """格式化回归结果"""
        if len(predictions) == 1:
            pred_value = predictions[0]
            if hasattr(pred_value, 'item'):
                pred_value = pred_value.item()
            prediction = float(pred_value)
        else:
            prediction = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)

        return {
            "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
            "prediction": prediction
        }

    def _format_clustering_result(self, predictions: np.ndarray, X: np.ndarray) -> Dict[str, Any]:
        """格式化聚类结果"""
        if len(predictions) == 1:
            pred_value = predictions[0]
            if hasattr(pred_value, 'item'):
                pred_value = pred_value.item()
            cluster_id = int(pred_value) if isinstance(pred_value, (int, float)) else pred_value
        else:
            cluster_id = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)

        return {
            "cluster_ids": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
            "cluster_id": cluster_id,
            "n_clusters": len(np.unique(predictions))
        }

    def _format_timeseries_result(self, predictions: np.ndarray) -> Dict[str, Any]:
        """格式化时间序列结果"""
        return {
            "forecast": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
            "forecast_length": len(predictions)
        }

    def _add_confidence_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """添加置信度信息"""
        if self.model_info:
            result["model_name"] = self.model_info.get("name")
            result["model_description"] = self.model_info.get("description")

        return result

    def explain_prediction(self, X: Union[np.ndarray, pd.DataFrame, List, Dict]) -> Dict[str, Any]:
        """
        解释预测结果（特征重要性）

        仅支持具有 feature_importances_ 属性的模型
        """
        result = self.predict(X)

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            result["feature_importances"] = importances.tolist() if hasattr(importances, 'tolist') else list(importances)

            # 获取特征名
            if self.preprocessor:
                feature_names = self.preprocessor.get_feature_names()
                if len(feature_names) == len(importances):
                    # 按重要性排序
                    sorted_idx = np.argsort(importances)[::-1]
                    result["top_features"] = [
                        {"feature": feature_names[i], "importance": float(importances[i])}
                        for i in sorted_idx[:10]
                    ]

        return result