"""
通用预测模块

调用 models/ 中的模型进行预测
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class Predictor:
    """
    通用预测器

    使用方式:
        predictor = Predictor()
        result = predictor.predict(df, target, features, model_key)
    """

    def __init__(self, session_id: Optional[str] = None):
        """
        初始化

        参数:
        - session_id: 会话ID（用于加载已训练的模型）
        """
        self.session_id = session_id

    def predict(
        self,
        df: pd.DataFrame,
        target: str,
        features: List[str],
        model_key: Optional[str] = None,
        task_type: str = "regression",
        train_ratio: float = 0.7
    ) -> Dict[str, Any]:
        """
        执行预测

        参数:
        - df: 数据框
        - target: 目标列
        - features: 特征列
        - model_key: 指定模型（默认自动选择）
        - task_type: 任务类型 (regression/classification)
        - train_ratio: 训练集比例

        返回: {
            "model": str,
            "predictions": np.ndarray,
            "metrics": dict,
            "feature_importance": dict
        }
        """
        from autostat.models.trainer import ModelTrainer
        from autostat.models.preprocessing import DataPreprocessor
        from autostat.models.registry import ModelRegistry

        # 准备数据
        X = df[features].copy()
        y = df[target].copy()

        # 删除缺失值
        mask = ~X.isna().any(axis=1)
        if y is not None:
            mask = mask & ~y.isna()

        X = X[mask]
        y = y[mask]

        if len(X) < 10:
            return {"error": "数据不足，需要至少10条有效样本"}

        # 自动选择模型
        if model_key is None:
            if task_type == "regression":
                model_key = "random_forest_regressor"
            else:
                model_key = "random_forest"

        # 预处理
        categorical_features = [col for col in features if col in df.select_dtypes(include=['object']).columns]
        numerical_features = [col for col in features if col not in categorical_features]

        preprocessor = DataPreprocessor({
            "missing_strategy": "drop",
            "scaling": "standard",
            "encoding": "onehot",
            "categorical_features": categorical_features,
            "numerical_features": numerical_features,
            "target_column": target
        })

        # 切分数据
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 - train_ratio, random_state=42
        )

        # 训练模型
        trainer = ModelTrainer(task_type, model_key)
        X_train_processed = preprocessor.fit_transform(X_train, y_train)
        X_test_processed = preprocessor.transform(X_test)

        train_result = trainer.train(
            X_train_processed, y_train,
            X_test_processed, y_test,
            cv_folds=5
        )

        # 预测
        predictions = trainer.model.predict(X_test_processed)

        # 特征重要性（如果有）
        feature_importance = {}
        if hasattr(trainer.model, 'feature_importances_'):
            importances = trainer.model.feature_importances_
            feature_names = preprocessor.get_feature_names()
            if len(feature_names) == len(importances):
                feature_importance = {
                    feature_names[i]: float(importances[i])
                    for i in range(len(importances))
                }
                # 按重要性排序
                feature_importance = dict(
                    sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                )

        return {
            "model": model_key,
            "model_display": ModelRegistry.get_model(task_type, model_key).get("name", model_key),
            "predictions": predictions.tolist(),
            "actual": y_test.tolist(),
            "metrics": train_result.get("val_score", {}),
            "feature_importance": feature_importance,
            "n_samples": len(X),
            "n_features": len(features)
        }

    def predict_from_result(
        self,
        analysis_result: Dict[str, Any],
        df: pd.DataFrame,
        target: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        从分析结果中获取推荐并进行预测

        参数:
        - analysis_result: 分析结果
        - df: 数据框
        - target: 目标列（可选，默认使用推荐）

        返回: 预测结果
        """
        recommendations = analysis_result.get("model_recommendations", [])

        # 找到回归或分类推荐
        for rec in recommendations:
            task_type = rec.get("task_type", "")
            if "回归" in task_type or "分类" in task_type:
                target = target or rec.get("target_column")
                features = rec.get("feature_columns", [])
                model_key = rec.get("model_key")

                if target and features:
                    if "回归" in task_type:
                        task_type = "regression"
                    else:
                        task_type = "classification"

                    return self.predict(df, target, features, model_key, task_type)

        return {"error": "未找到合适的预测目标"}


def quick_predict(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    **kwargs
) -> Dict[str, Any]:
    """便捷函数：快速预测"""
    predictor = Predictor(**kwargs)
    return predictor.predict(df, target, features)