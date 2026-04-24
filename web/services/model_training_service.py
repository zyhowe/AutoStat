# web/services/model_training_service.py

"""模型训练服务 - 业务逻辑层"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from typing import Dict, Any, Optional, List, Tuple
from sklearn.model_selection import train_test_split

from autostat.models.registry import ModelRegistry
from autostat.models.trainer import ModelTrainer
from autostat.models.preprocessing import DataPreprocessor
from autostat.models.storage import ModelStorage
from autostat.models.predictor import ModelPredictor
from web.services.storage_service import StorageService
from web.services.session_service import SessionService

# 解决 Windows 下 threadpoolctl 的 OSError 问题
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


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


def generate_model_name(task_type: str, target_col: str, model_key: str, features: List[str] = None) -> str:
    """生成默认模型名称，包含特征信息（前3个）"""
    task_names = {
        "classification": "分类",
        "regression": "回归",
        "clustering": "聚类",
        "time_series": "时序"
    }

    task_name = task_names.get(task_type, task_type)
    model_name = get_model_display_name(model_key)

    # 目标列处理
    target = target_col if target_col else "无目标"
    target_short = target[:15] if len(target) > 15 else target

    # 特征处理
    if features and len(features) > 0:
        # 取前3个特征，每个特征取前10个字符
        feature_list = [f[:10] for f in features[:3]]
        feature_str = "_" + "_".join(feature_list)
        if len(features) > 3:
            feature_str += "_等"
    else:
        feature_str = ""

    return f"{task_name}_{target_short}_{model_name}{feature_str}"


def get_available_features(data: pd.DataFrame, variable_types: Dict) -> List[str]:
    """获取可用的特征列（不限制数量）"""
    numeric_cols = [col for col, info in variable_types.items() if info.get('type') == 'continuous']
    categorical_cols = [col for col, info in variable_types.items()
                        if info.get('type') in ['categorical', 'categorical_numeric', 'ordinal']]
    identifier_cols = [col for col, info in variable_types.items() if info.get('type') == 'identifier']

    available_cols = numeric_cols + categorical_cols
    available_cols = [col for col in available_cols if col not in identifier_cols]

    return available_cols


def get_model_recommendations_from_json(json_data: Dict) -> List[Dict]:
    """从分析结果中获取模型推荐（全部返回，不限制）"""
    return json_data.get('model_recommendations', [])


def get_best_recommendation(json_data: Dict) -> Optional[Dict]:
    """获取最佳模型推荐"""
    recommendations = get_model_recommendations_from_json(json_data)
    if not recommendations:
        return None
    high_priority = [r for r in recommendations if r.get('priority') == '高']
    if high_priority:
        return high_priority[0]
    return recommendations[0]


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
        session_id: str,
        user_model_name: str = None
) -> Tuple[bool, Optional[Dict]]:
    """
    执行模型训练
    """
    try:
        # 确保所有特征列都在数据中
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

        # 获取特征数据
        X = data[available_features].copy()

        # 获取目标数据（分类/回归需要）
        if task_type in ["classification", "regression"] and target_col:
            if target_col not in data.columns:
                return False, {"error": f"目标列 '{target_col}' 不存在于数据中"}
            y = data[target_col].copy()
        else:
            y = None

        np.random.seed(random_seed)

        test_ratio = 1 - train_ratio - val_ratio

        # ========== 根据任务类型处理数据切分 ==========
        if task_type == "time_series":
            # 时间序列任务：按时间顺序划分，不随机打乱
            X_clean = X.dropna()

            n = len(X_clean)
            train_size = int(n * train_ratio)
            val_size = int(n * val_ratio) if val_ratio > 0 else 0

            if len(available_features) > 1:
                target_series = X_clean.iloc[:, 0]
            else:
                target_series = X_clean.iloc[:, 0] if len(X_clean.columns) > 0 else X_clean

            X_train = target_series.iloc[:train_size]
            X_val = target_series.iloc[train_size:train_size + val_size] if val_size > 0 else None
            X_test = target_series.iloc[train_size + val_size:] if test_ratio > 0 else None
            y_train = y_val = y_test = None

            preprocessor = None

            st.info(f"时间序列数据: 总样本 {n}, 训练集 {len(X_train)}")

        elif task_type == "clustering":
            # 聚类任务：无监督，不需要y，随机切分
            X_clean = X.dropna()

            if test_ratio > 0:
                X_train, X_temp = train_test_split(
                    X_clean, test_size=(val_ratio + test_ratio), random_state=random_seed
                )
                if val_ratio > 0 and test_ratio > 0:
                    val_size = val_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0
                    X_val, X_test = train_test_split(X_temp, test_size=val_size, random_state=random_seed)
                elif val_ratio > 0:
                    X_val, X_test = X_temp, None
                else:
                    X_val, X_test = None, X_temp
            else:
                if val_ratio > 0:
                    X_train, X_val = train_test_split(
                        X_clean, test_size=val_ratio, random_state=random_seed
                    )
                    X_test = None
                else:
                    X_train, X_val, X_test = X_clean, None, None

            y_train = y_val = y_test = None

            categorical_features = [col for col in available_features if
                                    col in data.select_dtypes(include=['object']).columns]
            numerical_features = [col for col in available_features if col not in categorical_features]

            preprocess_config = {
                "missing_strategy": missing_strategy,
                "scaling": scaling if scaling != "none" else None,
                "encoding": encoding if encoding != "none" else None,
                "categorical_features": categorical_features,
                "numerical_features": numerical_features,
                "target_column": None
            }

            preprocessor = DataPreprocessor(preprocess_config)

        else:
            # 分类和回归任务：监督学习，正常切分
            # 创建布尔掩码：X 和 y 都没有缺失值
            mask = ~X.isna().any(axis=1)
            if y is not None:
                mask = mask & ~y.isna()

            X_clean = X[mask]
            if y is not None:
                y_clean = y[mask]
            else:
                y_clean = None

            st.info(f"数据清洗: 原始样本 {len(X)} -> 清洗后 {len(X_clean)}")

            if len(X_clean) == 0:
                return False, {"error": "清洗后没有有效样本"}

            if test_ratio > 0:
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X_clean, y_clean, test_size=(val_ratio + test_ratio), random_state=random_seed
                )
                if val_ratio > 0:
                    val_size = val_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0
                    X_val, X_test, y_val, y_test = train_test_split(
                        X_temp, y_temp, test_size=val_size, random_state=random_seed
                    )
                else:
                    X_val, X_test = None, X_temp
                    y_val, y_test = None, y_temp
            else:
                if val_ratio > 0:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_clean, y_clean, test_size=val_ratio, random_state=random_seed
                    )
                    X_test, y_test = None, None
                else:
                    X_train, X_val, y_train, y_val = X_clean, None, y_clean, None
                    X_test, y_test = None, None

            # 预处理配置
            categorical_features = [col for col in available_features if
                                    col in data.select_dtypes(include=['object']).columns]
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

        # ========== 训练模型 ==========
        trainer = ModelTrainer(task_type, model_key, user_params)

        if task_type == "time_series" and model_key == "arima":
            train_result = trainer.train(
                X_train, y_train, X_val, y_val,
                cv_folds=cv_folds if cv_folds > 0 else None,
                early_stopping=False
            )
            preprocessor = None
        else:
            if X_train is not None:
                X_train_processed = preprocessor.fit_transform(X_train, y_train) if preprocessor else X_train
            else:
                X_train_processed = None

            if X_val is not None:
                X_val_processed = preprocessor.transform(X_val) if preprocessor else X_val
            else:
                X_val_processed = None

            train_result = trainer.train(
                X_train_processed, y_train,
                X_val_processed, y_val,
                cv_folds=cv_folds if cv_folds > 0 else None,
                early_stopping=False
            )

        # 测试集评估
        test_metrics = None
        if X_test is not None:
            if task_type == "time_series":
                test_metrics = trainer.evaluate(X_test, y_test)
            elif task_type in ["classification", "regression"] and preprocessor:
                X_test_processed = preprocessor.transform(X_test)
                test_metrics = trainer.evaluate(X_test_processed, y_test)

        timestamp = int(time.time())
        model_key_full = f"{task_type}_{model_key}_{timestamp}"

        if user_model_name is None:
            user_model_name = generate_model_name(task_type, target_col, model_key, available_features)

        model_metadata = {
            "model_key": model_key_full,
            "session_id": session_id,
            "user_model_name": user_model_name,
            "task_type": task_type,
            "model_type": model_key,
            "target_column": target_col,
            "features": available_features,
            "params": user_params,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "training_time": train_result.get("training_time", 0),
            "preprocess_config": preprocess_config if preprocessor else {}
        }

        StorageService.save_model(
            model_key_full,
            trainer.model,
            preprocessor,
            model_metadata,
            {"train_score": train_result.get("train_score", {}), "test_metrics": test_metrics}
        )

        result = {
            "success": True,
            "model_key": model_key_full,
            "model": trainer.model,
            "preprocessor": preprocessor,
            "metadata": model_metadata,
            "metrics": train_result.get("train_score", {}),
            "test_metrics": test_metrics,
            "train_result": train_result
        }

        return True, result

    except Exception as e:
        import traceback
        return False, {"error": str(e), "traceback": traceback.format_exc()}


def list_saved_models(session_id: str = None) -> List[Dict]:
    """列出已保存的模型"""
    return StorageService.list_models(session_id)


def delete_model(model_key: str, session_id: str = None) -> bool:
    """删除模型"""
    return StorageService.delete_model(model_key, session_id)


def load_model_for_inference(model_key: str, session_id: str = None) -> Tuple[Any, Any, Dict, Dict]:
    """加载模型用于推理"""
    return StorageService.load_model(model_key, session_id)


def execute_inference(model, preprocessor, input_data: Dict, features: List[str],
                      categorical_features: List[str]) -> Dict:
    """执行推理"""
    try:
        df = pd.DataFrame([input_data])

        for feature in features:
            if feature not in df.columns:
                df[feature] = None

        for col in df.columns:
            if col not in categorical_features:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    pass

        df = df[features]

        predictor = ModelPredictor(model, preprocessor)
        result = predictor.predict_with_confidence(df)

        return result

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}