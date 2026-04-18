# web/services/model_training_service.py

"""模型训练服务 - 业务逻辑层"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from typing import Dict, Any, Optional, List, Tuple
from sklearn.model_selection import train_test_split

from autostat.models.registry import ModelRegistry
from autostat.models.trainer import ModelTrainer
from autostat.models.preprocessing import DataPreprocessor
from autostat.models.storage import ModelStorage
from autostat.models.predictor import ModelPredictor


def get_model_display_name(model_key: str) -> str:
    """获取模型显示名称"""
    model_names = {
        "logistic_regression": "逻辑回归",
        "decision_tree": "决策树",
        "random_forest": "随机森林",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "svm": "SVM",
        "knn": "KNN",
        "naive_bayes": "朴素贝叶斯",
        "adaboost": "AdaBoost",
        "gradient_boosting": "梯度提升",
        "mlp_classifier": "MLP神经网络",
        "linear_regression": "线性回归",
        "ridge": "岭回归",
        "lasso": "Lasso回归",
        "elastic_net": "弹性网络",
        "decision_tree_regressor": "决策树回归",
        "random_forest_regressor": "随机森林回归",
        "xgboost_regressor": "XGBoost回归",
        "lightgbm_regressor": "LightGBM回归",
        "svr": "SVR",
        "knn_regressor": "KNN回归",
        "gradient_boosting_regressor": "梯度提升回归",
        "mlp_regressor": "MLP回归",
        "kmeans": "K-Means",
        "dbscan": "DBSCAN",
        "agglomerative": "层次聚类",
        "birch": "BIRCH",
        "gaussian_mixture": "高斯混合模型",
        "optics": "OPTICS",
        "spectral": "谱聚类",
        "mean_shift": "均值漂移",
        "arima": "ARIMA",
        "sarima": "SARIMA",
        "prophet": "Prophet",
        "lstm_ts": "LSTM时序",
        "gru_ts": "GRU时序",
        "transformer_ts": "Transformer时序"
    }
    return model_names.get(model_key, model_key)


def generate_model_name(task_type: str, target_col: str, model_key: str) -> str:
    """生成默认模型名称"""
    task_names = {
        "classification": "分类",
        "regression": "回归",
        "clustering": "聚类",
        "time_series": "时序"
    }

    task_name = task_names.get(task_type, task_type)
    model_name = get_model_display_name(model_key)
    target = target_col if target_col else "无目标"
    target_short = target[:15] if len(target) > 15 else target

    return f"{task_name}_{target_short}_{model_name}"


def get_available_features(data: pd.DataFrame, variable_types: Dict) -> List[str]:
    """获取可用的特征列"""
    numeric_cols = [col for col, info in variable_types.items() if info.get('type') == 'continuous']
    categorical_cols = [col for col, info in variable_types.items()
                        if info.get('type') in ['categorical', 'categorical_numeric', 'ordinal']]
    identifier_cols = [col for col, info in variable_types.items() if info.get('type') == 'identifier']

    available_cols = numeric_cols + categorical_cols
    available_cols = [col for col in available_cols if col not in identifier_cols]

    return available_cols


def get_model_recommendations_from_json(json_data: Dict) -> List[Dict]:
    """从分析结果中获取模型推荐"""
    return json_data.get('model_recommendations', [])


def get_models_by_task_type(task_type: str) -> Dict:
    """获取指定任务类型的模型列表"""
    return ModelRegistry.get_models(task_type)


def get_model_params(task_type: str, model_key: str) -> Dict:
    """获取模型参数配置"""
    return ModelRegistry.get_model_params(task_type, model_key)


def execute_training(
    data: pd.DataFrame,
    features: List[str],
    target_col: Optional[str],
    task_type: str,
    model_key: str,
    user_params: Dict[str, Any],
    train_ratio: float,
    val_ratio: float,
    missing_strategy: str,
    scaling: str,
    encoding: str,
    cv_folds: int,
    random_seed: int,
    session_id: str
) -> Tuple[bool, Optional[Dict]]:
    """
    执行模型训练

    返回: (success, result_dict)
    """
    try:
        # 确保所有特征列都在数据中（处理日期派生列）
        available_features = []
        missing_features = []

        for f in features:
            if f in data.columns:
                available_features.append(f)
            else:
                missing_features.append(f)

        if missing_features:
            st.warning(f"以下特征列不存在，已跳过: {missing_features}")

        if not available_features:
            return False, {"error": "没有有效的特征列"}

        X = data[available_features].copy()

        if task_type in ["classification", "regression"] and target_col:
            if target_col not in data.columns:
                return False, {"error": f"目标列 '{target_col}' 不存在于数据中"}
            y = data[target_col].copy()
        else:
            y = None

        # 预处理配置
        categorical_features = [col for col in available_features if col in data.select_dtypes(include=['object']).columns]
        numerical_features = [col for col in available_features if col not in categorical_features]

        preprocess_config = {
            "missing_strategy": missing_strategy,
            "scaling": scaling if scaling != "none" else None,
            "encoding": encoding if encoding != "none" else None,
            "categorical_features": categorical_features,
            "numerical_features": numerical_features,
            "target_column": target_col
        }

        preprocessor = DataPreprocessor(preprocess_config)

        np.random.seed(random_seed)

        test_ratio = 1 - train_ratio - val_ratio

        if test_ratio > 0:
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=(val_ratio + test_ratio), random_state=random_seed
            )
            if val_ratio > 0:
                val_size = val_ratio / (val_ratio + test_ratio)
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=val_size, random_state=random_seed
                )
            else:
                X_val, X_test = None, X_temp
                y_val, y_test = None, y_temp
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_ratio, random_state=random_seed
            )
            X_test, y_test = None, None

        X_train_processed = preprocessor.fit_transform(X_train, y_train)

        if X_val is not None:
            X_val_processed = preprocessor.transform(X_val)
        else:
            X_val_processed = None

        trainer = ModelTrainer(task_type, model_key, user_params)

        train_result = trainer.train(
            X_train_processed, y_train,
            X_val_processed, y_val,
            cv_folds=cv_folds if cv_folds > 0 else None,
            early_stopping=False
        )

        test_metrics = None
        if X_test is not None and y_test is not None:
            X_test_processed = preprocessor.transform(X_test)
            test_metrics = trainer.evaluate(X_test_processed, y_test)

        # 保存训练结果
        result = {
            "model": trainer.model,
            "preprocessor": preprocessor,
            "config": {
                "task_type": task_type,
                "model_key": model_key,
                "model_name": get_model_display_name(model_key),
                "params": user_params,
                "features": available_features,
                "target_col": target_col,
                "preprocess_config": preprocess_config
            },
            "metrics": train_result.get("train_score", {}),
            "test_metrics": test_metrics,
            "train_result": train_result
        }

        if test_metrics:
            result["metrics"]["test"] = test_metrics

        return True, result

    except Exception as e:
        import traceback
        return False, {"error": str(e), "traceback": traceback.format_exc()}


def save_trained_model(session_id: str, model, preprocessor, config: Dict,
                       metrics: Dict, user_model_name: str) -> bool:
    """保存训练好的模型"""
    try:
        config["user_model_name"] = user_model_name
        model_key = f"{config['task_type']}_{config['model_key']}_{int(time.time())}"

        ModelStorage.save_model(
            session_id=session_id,
            model_key=model_key,
            model=model,
            preprocessor=preprocessor,
            metrics=metrics,
            config=config
        )
        return True
    except Exception as e:
        print(f"保存模型失败: {e}")
        return False


def list_saved_models(session_id: str) -> List[Dict]:
    """列出已保存的模型"""
    return ModelStorage.list_models(session_id)


def delete_model(session_id: str, model_key: str) -> bool:
    """删除模型"""
    return ModelStorage.delete_model(session_id, model_key)


def load_model_for_inference(session_id: str, model_key: str) -> Tuple[Any, Any, Dict]:
    """加载模型用于推理"""
    return ModelStorage.load_model(session_id, model_key)


def execute_inference(model, preprocessor, input_data: Dict, features: List[str],
                      categorical_features: List[str]) -> Dict:
    """执行推理"""
    try:
        # 构建输入DataFrame
        df = pd.DataFrame([input_data])

        # 确保所有特征列都存在
        for feature in features:
            if feature not in df.columns:
                df[feature] = None

        # 确保数值列转换为数字
        for col in df.columns:
            if col not in categorical_features:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass

        # 只保留特征列并按顺序排列
        df = df[features]

        # 创建预测器
        predictor = ModelPredictor(model, preprocessor)
        result = predictor.predict_with_confidence(df)

        return result

    except Exception as e:
        import traceback
        print(f"推理失败: {e}")
        print(traceback.format_exc())
        return {"error": str(e), "traceback": traceback.format_exc()}