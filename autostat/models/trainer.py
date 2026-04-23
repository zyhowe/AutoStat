"""训练器模块 - 模型训练、验证、测试"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Callable
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer
import time
import warnings
import logging

from autostat.models.preprocessing import DataPreprocessor
from autostat.models.metrics import MetricsCalculator
from autostat.models.registry import ModelRegistry

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型训练器 - 支持分类、回归、聚类、时间序列"""

    def __init__(self, task_type: str, model_key: str, params: Dict[str, Any] = None):
        """
        初始化训练器

        参数:
        - task_type: 任务类型 (classification/regression/clustering/time_series)
        - model_key: 模型标识
        - params: 模型参数（覆盖默认值）
        """
        self.task_type = task_type
        self.model_key = model_key
        self.model_info = ModelRegistry.get_model(task_type, model_key)

        if not self.model_info:
            raise ValueError(f"模型不存在: {task_type}/{model_key}")

        # 合并参数
        default_params = self.model_info.get("params", {})
        self.params = {}
        for key, param_info in default_params.items():
            self.params[key] = param_info.get("default")

        if params:
            self.params.update(params)

        self.model = None
        self.preprocessor = None
        self.history = []
        self.best_score = None

        # 动态导入模型类
        self._load_model_class()

    def _load_model_class(self):
        """动态加载模型类"""
        module_name = self.model_info["module"]
        class_name = self.model_info["class"]

        try:
            # ARIMA 包装器
            if module_name == "autostat.models.arima_wrapper":
                from autostat.models.arima_wrapper import ARIMAWrapper
                self.model_class = ARIMAWrapper
            elif module_name == "autostat.models.deep_learning":
                from autostat.models.deep_learning import (
                    CNNClassifier, RNNClassifier, LSTMClassifier, BertClassifier,
                    LSTMTimeSeries, GRUTimeSeries, TransformerTimeSeries
                )
                self.model_class = locals().get(class_name)
            else:
                module = __import__(module_name, fromlist=[class_name])
                self.model_class = getattr(module, class_name)
        except ImportError as e:
            logger.error(f"导入模型失败: {e}")
            self.model_class = None

    def _create_model(self):
        """创建模型实例"""
        if self.model_class is None:
            raise ImportError(f"无法导入模型: {self.model_info['name']}")

        # 过滤参数（只保留模型支持的参数）
        import inspect
        sig = inspect.signature(self.model_class.__init__)
        valid_params = {}

        for key, value in self.params.items():
            if key in sig.parameters:
                valid_params[key] = value

        self.model = self.model_class(**valid_params)

    def _prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple:
        """准备数据（根据任务类型）"""
        if self.task_type == "clustering":
            return X, None
        return X, y

    def train(self, X_train, y_train: Optional[np.ndarray] = None,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              cv_folds: int = 5, early_stopping: bool = False,
              callbacks: List[Callable] = None) -> Dict[str, Any]:
        """
        训练模型

        参数:
        - X_train: 训练特征
        - y_train: 训练标签（聚类任务可为None）
        - X_val: 验证特征
        - y_val: 验证标签
        - cv_folds: 交叉验证折数
        - early_stopping: 是否早停
        - callbacks: 回调函数列表

        返回:
        - 训练结果字典
        """
        start_time = time.time()

        # 准备数据
        X_train, y_train = self._prepare_data(X_train, y_train)
        if X_val is not None:
            X_val, y_val = self._prepare_data(X_val, y_val)

        # ========== 时间序列任务特殊处理 ==========
        if self.task_type == "time_series" and self.model_key == "arima":
            # ARIMA 不需要提前创建模型，直接在 fit 中创建
            if self.model_class is None:
                self._load_model_class()

            # 创建模型实例
            self.model = self.model_class(**self.params)

            # 训练
            self.model.fit(X_train)
            train_score = {
                "aic": getattr(self.model, 'aic', None),
                "bic": getattr(self.model, 'bic', None),
                "message": "ARIMA 训练完成"
            }

            # 验证集评估
            val_score = None
            if X_val is not None:
                from autostat.models.metrics import MetricsCalculator
                y_pred = self.model.predict(X_val)
                val_score = MetricsCalculator.calculate_time_series_metrics(
                    y_val if y_val is not None else X_val,
                    y_pred
                )

            training_time = time.time() - start_time

            result = {
                "model_key": self.model_key,
                "model_name": self.model_info["name"],
                "task_type": self.task_type,
                "params": self.params,
                "train_score": train_score,
                "val_score": val_score,
                "cv_scores": None,
                "training_time": training_time,
                "history": self.history
            }
            return result

        # ========== 聚类任务 ==========
        if self.task_type == "clustering":
            # 创建模型
            self._create_model()

            # 训练
            self.model.fit(X_train)

            # 评估训练结果
            if hasattr(self.model, 'labels_'):
                train_score = self._evaluate_clustering(X_train, self.model.labels_)
            else:
                train_score = {"message": "聚类训练完成"}

            # 验证集评估
            val_score = None
            if X_val is not None and hasattr(self.model, 'predict'):
                try:
                    y_pred = self.model.predict(X_val)
                    val_score = self._evaluate_clustering(X_val, y_pred)
                except Exception as e:
                    logger.warning(f"聚类验证失败: {e}")

            training_time = time.time() - start_time

            result = {
                "model_key": self.model_key,
                "model_name": self.model_info["name"],
                "task_type": self.task_type,
                "params": self.params,
                "train_score": train_score,
                "val_score": val_score,
                "cv_scores": None,
                "training_time": training_time,
                "history": self.history
            }
            return result

        # ========== 监督学习（分类/回归） ==========
        # 创建模型
        self._create_model()

        # 训练
        if hasattr(self.model, 'fit'):
            if X_val is not None and y_val is not None and early_stopping:
                # 支持验证集和早停
                self._fit_with_validation(X_train, y_train, X_val, y_val)
            else:
                self.model.fit(X_train, y_train)

        # 计算训练分数
        train_score = self._evaluate_supervised(X_train, y_train)

        # 验证集评估
        val_score = None
        if X_val is not None:
            val_score = self._evaluate_supervised(X_val, y_val)

        # 交叉验证
        cv_scores = None
        if cv_folds > 1:
            cv_scores = self._cross_validate(X_train, y_train, cv_folds)

        training_time = time.time() - start_time

        result = {
            "model_key": self.model_key,
            "model_name": self.model_info["name"],
            "task_type": self.task_type,
            "params": self.params,
            "train_score": train_score,
            "val_score": val_score,
            "cv_scores": cv_scores,
            "training_time": training_time,
            "history": self.history
        }

        return result

    def _fit_with_validation(self, X_train, y_train, X_val, y_val):
        """带验证集的训练（支持早停）"""
        if hasattr(self.model, 'fit'):
            # 尝试传递验证集参数
            try:
                if self.task_type == "classification" and "xgboost" in self.model_key:
                    self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
                elif "lightgbm" in self.model_key:
                    self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                                   callbacks=[self._early_stopping_callback()])
                elif "catboost" in self.model_key:
                    self.model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10, verbose=False)
                else:
                    self.model.fit(X_train, y_train)
            except:
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)

    def _early_stopping_callback(self):
        """早停回调"""
        from lightgbm import early_stopping
        return early_stopping(10)

    def _evaluate_supervised(self, X, y) -> Dict[str, float]:
        """评估监督学习模型"""
        y_pred = self.model.predict(X)

        if self.task_type == "classification":
            # 获取概率（如果有）
            y_proba = None
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(X)
            return MetricsCalculator.calculate_classification_metrics(y, y_pred, y_proba)
        else:
            return MetricsCalculator.calculate_regression_metrics(y, y_pred)

    def _evaluate_clustering(self, X, labels) -> Dict[str, float]:
        """评估聚类模型"""
        return MetricsCalculator.calculate_clustering_metrics(X, labels)

    def _cross_validate(self, X, y, cv_folds: int) -> Dict[str, Any]:
        """交叉验证"""
        if self.task_type == "classification":
            scoring = 'accuracy'
        elif self.task_type == "regression":
            scoring = 'r2'
        else:
            return None

        try:
            scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring=scoring)
            return {
                "scores": scores.tolist(),
                "mean": float(scores.mean()),
                "std": float(scores.std())
            }
        except Exception as e:
            logger.warning(f"交叉验证失败: {e}")
            return None

    def evaluate(self, X_test: np.ndarray, y_test: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """测试集评估"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train()")

        if self.task_type == "clustering":
            y_pred = self.model.predict(X_test)
            metrics = self._evaluate_clustering(X_test, y_pred)
        elif self.task_type == "time_series":
            y_pred = self.model.predict(X_test)
            if y_test is not None:
                metrics = MetricsCalculator.calculate_time_series_metrics(y_test, y_pred)
            else:
                metrics = {"message": "无真实标签，仅返回预测结果"}
                metrics["predictions"] = y_pred.tolist()
        else:
            y_pred = self.model.predict(X_test)
            if y_test is not None:
                metrics = self._evaluate_supervised(X_test, y_test)
            else:
                metrics = {"message": "无真实标签，仅返回预测结果"}
                metrics["predictions"] = y_pred.tolist()

        return metrics

    def grid_search(self, X_train: np.ndarray, y_train: np.ndarray,
                    param_grid: Dict[str, List], cv_folds: int = 5,
                    scoring: str = None) -> Dict[str, Any]:
        """
        网格搜索超参数

        参数:
        - X_train: 训练特征
        - y_train: 训练标签
        - param_grid: 参数网格
        - cv_folds: 交叉验证折数
        - scoring: 评估指标

        返回:
        - 最佳参数和分数
        """
        if self.task_type == "classification":
            scoring = scoring or 'accuracy'
        elif self.task_type == "regression":
            scoring = scoring or 'r2'
        else:
            raise ValueError(f"网格搜索不支持任务类型: {self.task_type}")

        self._create_model()

        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv_folds, scoring=scoring,
            n_jobs=-1, verbose=0
        )

        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        self.params = grid_search.best_params_

        return {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_
        }

    def save(self, path: str):
        """保存模型"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'params': self.params,
                'task_type': self.task_type,
                'model_key': self.model_key,
                'history': self.history
            }, f)

    def load(self, path: str):
        """加载模型"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.params = data['params']
        self.task_type = data['task_type']
        self.model_key = data['model_key']
        self.history = data.get('history', [])