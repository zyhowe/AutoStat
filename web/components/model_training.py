"""模型训练标签页组件 - 选择模型、配置参数、训练、评估"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional, List
from sklearn.model_selection import train_test_split

from autostat.models.registry import ModelRegistry
from autostat.models.trainer import ModelTrainer
from autostat.models.preprocessing import DataPreprocessor
from autostat.models.metrics import MetricsCalculator
from autostat.models.storage import ModelStorage
from autostat.models.predictor import ModelPredictor


def render_model_training():
    """渲染模型训练标签页"""
    st.markdown("### 🧠 模型训练")

    # 检查是否有分析完成的数据
    if not st.session_state.analysis_completed or st.session_state.current_json_data is None:
        st.info("📌 请先在「数据准备」中完成数据分析")
        st.caption("分析完成后，可基于标准化数据进行模型训练")
        return

    # 获取标准化数据
    raw_data_preview = st.session_state.raw_data_preview
    if raw_data_preview is None:
        st.info("📌 请先在「数据准备」中完成数据分析")
        return

    full_data = raw_data_preview.get('full_data')
    if full_data is None or full_data.empty:
        st.info("📌 无有效数据，请检查数据文件")
        return

    # 获取变量类型信息
    json_data = st.session_state.current_json_data
    variable_types = json_data.get('variable_types', {})

    # 提取可用的特征列和目标列
    numeric_cols = [col for col, info in variable_types.items() if info.get('type') == 'continuous']
    categorical_cols = [col for col, info in variable_types.items()
                        if info.get('type') in ['categorical', 'categorical_numeric', 'ordinal']]
    datetime_cols = [col for col, info in variable_types.items() if info.get('type') == 'datetime']
    identifier_cols = [col for col, info in variable_types.items() if info.get('type') == 'identifier']

    # 所有可用列（排除标识符）
    available_cols = numeric_cols + categorical_cols
    available_cols = [col for col in available_cols if col not in identifier_cols]

    if not available_cols:
        st.warning("⚠️ 没有可用的特征列，请检查数据")
        return

    # ==================== 任务配置 ====================
    st.markdown("#### 📋 任务配置")

    col1, col2 = st.columns(2)

    with col1:
        # 任务类型选择
        task_type = st.selectbox(
            "任务类型",
            options=["classification", "regression", "clustering", "time_series"],
            format_func=lambda x: {
                "classification": "📊 分类 (Classification)",
                "regression": "📈 回归 (Regression)",
                "clustering": "🔘 聚类 (Clustering)",
                "time_series": "📅 时间序列 (Time Series)"
            }.get(x, x),
            key="train_task_type"
        )

    with col2:
        # 目标列选择（分类/回归任务需要）
        if task_type in ["classification", "regression"]:
            target_col = st.selectbox(
                "目标列",
                options=available_cols,
                key="train_target_col"
            )
        else:
            target_col = None
            st.selectbox("目标列", options=["无（无监督学习）"], disabled=True, key="train_target_col_disabled")

    # ==================== 特征选择 ====================
    st.markdown("#### 🔧 特征选择")

    # 自动推荐特征（排除目标列）
    default_features = [col for col in available_cols if col != target_col]

    selected_features = st.multiselect(
        "选择特征列",
        options=available_cols,
        default=default_features[:20] if len(default_features) > 20 else default_features,
        key="train_features"
    )

    if not selected_features:
        st.warning("⚠️ 请至少选择一个特征列")
        return

    # ==================== 数据划分 ====================
    st.markdown("#### 📊 数据划分")

    col1, col2, col3 = st.columns(3)

    with col1:
        train_ratio = st.slider("训练集比例", 0.5, 0.9, 0.7, 0.05, key="train_ratio")

    with col2:
        val_ratio = st.slider("验证集比例", 0.0, 0.3, 0.15, 0.05, key="val_ratio")

    with col3:
        test_ratio = 1 - train_ratio - val_ratio
        st.metric("测试集比例", f"{test_ratio:.0%}")

    if test_ratio < 0:
        st.error("训练集+验证集比例不能超过100%")
        return

    # ==================== 模型选择 ====================
    st.markdown("#### 🤖 模型选择")

    # 获取可用模型
    models = ModelRegistry.get_models(task_type)
    model_options = list(models.keys())

    if not model_options:
        st.warning(f"⚠️ 暂无{task_type}类型的模型可用")
        return

    # 模型选择
    selected_model_key = st.selectbox(
        "选择模型",
        options=model_options,
        format_func=lambda x: f"{models[x]['name']} - {models[x]['description'][:50]}",
        key="train_model_key"
    )

    if selected_model_key:
        model_info = models[selected_model_key]

        # 显示模型信息
        with st.expander(f"📖 {model_info['name']} 详细信息"):
            st.markdown(f"**描述:** {model_info['description']}")
            st.markdown(f"**依赖:** {', '.join(model_info.get('requirements', []))}")

            # 显示参数配置
            st.markdown("**参数配置:**")
            params = model_info.get('params', {})
            for param_name, param_info in params.items():
                default_val = param_info.get('default')
                param_type = param_info.get('type', 'unknown')
                param_range = param_info.get('range', [])
                st.caption(f"- **{param_name}**: {default_val} (类型: {param_type}, 范围: {param_range})")

    # ==================== 参数配置 ====================
    st.markdown("#### ⚙️ 参数配置")

    # 获取模型参数
    model_params = ModelRegistry.get_model_params(task_type, selected_model_key)
    user_params = {}

    # 动态生成参数输入控件
    cols = st.columns(2)
    param_items = list(model_params.items())

    for i, (param_name, param_info) in enumerate(param_items):
        with cols[i % 2]:
            param_type = param_info.get('type', 'unknown')
            default_val = param_info.get('default')
            description = param_info.get('description', '')

            if param_type == 'int':
                param_range = param_info.get('range', [1, 100])
                user_params[param_name] = st.number_input(
                    f"{param_name}",
                    min_value=param_range[0],
                    max_value=param_range[1] if len(param_range) > 1 else 1000,
                    value=default_val if default_val else param_range[0],
                    step=1,
                    help=description,
                    key=f"param_{param_name}"
                )
            elif param_type == 'float':
                param_range = param_info.get('range', [0.0, 1.0])
                user_params[param_name] = st.number_input(
                    f"{param_name}",
                    min_value=float(param_range[0]),
                    max_value=float(param_range[1]) if len(param_range) > 1 else 1.0,
                    value=float(default_val) if default_val else param_range[0],
                    step=0.01,
                    help=description,
                    key=f"param_{param_name}"
                )
            elif param_type == 'choice':
                options = param_info.get('options', [])
                user_params[param_name] = st.selectbox(
                    f"{param_name}",
                    options=options,
                    index=options.index(default_val) if default_val in options else 0,
                    help=description,
                    key=f"param_{param_name}"
                )
            elif param_type == 'bool':
                user_params[param_name] = st.checkbox(
                    f"{param_name}",
                    value=default_val if default_val else False,
                    help=description,
                    key=f"param_{param_name}"
                )
            else:
                user_params[param_name] = st.text_input(
                    f"{param_name}",
                    value=str(default_val) if default_val else "",
                    help=description,
                    key=f"param_{param_name}"
                )

    # ==================== 预处理配置 ====================
    st.markdown("#### 🔧 预处理配置")

    col1, col2, col3 = st.columns(3)

    with col1:
        missing_strategy = st.selectbox(
            "缺失值处理",
            options=["drop", "fill_mean", "fill_median", "fill_mode", "fill_constant"],
            format_func=lambda x: {
                "drop": "删除缺失行",
                "fill_mean": "用均值填充",
                "fill_median": "用中位数填充",
                "fill_mode": "用众数填充",
                "fill_constant": "用常数填充"
            }.get(x, x),
            key="preprocess_missing"
        )

    with col2:
        scaling = st.selectbox(
            "特征缩放",
            options=["standard", "minmax", "robust", "none"],
            format_func=lambda x: {
                "standard": "标准化 (StandardScaler)",
                "minmax": "归一化 (MinMaxScaler)",
                "robust": "鲁棒缩放 (RobustScaler)",
                "none": "不缩放"
            }.get(x, x),
            key="preprocess_scaling"
        )

    with col3:
        encoding = st.selectbox(
            "分类编码",
            options=["onehot", "label", "none"],
            format_func=lambda x: {
                "onehot": "独热编码 (OneHot)",
                "label": "标签编码 (Label)",
                "none": "不编码"
            }.get(x, x),
            key="preprocess_encoding"
        )

    # ==================== 训练控制 ====================
    st.markdown("#### 🎮 训练控制")

    col1, col2, col3 = st.columns(3)

    with col1:
        cv_folds = st.number_input("交叉验证折数", min_value=0, max_value=10, value=5, key="cv_folds")

    with col2:
        early_stopping = st.checkbox("启用早停", value=False, key="early_stopping")

    with col3:
        random_seed = st.number_input("随机种子", min_value=0, max_value=9999, value=42, key="random_seed")

    # ==================== 开始训练按钮 ====================
    st.markdown("---")

    if st.button("▶️ 开始训练", type="primary", use_container_width=True):
        train_model(
            full_data, selected_features, target_col, task_type,
            selected_model_key, user_params,
            train_ratio, val_ratio, test_ratio,
            missing_strategy, scaling, encoding,
            cv_folds, early_stopping, random_seed
        )

    # ==================== 显示已保存的模型 ====================
    st.markdown("---")
    st.markdown("#### 💾 已保存的模型")

    session_id = st.session_state.current_source_name or "default"
    saved_models = ModelStorage.list_models(session_id)

    if saved_models:
        for model_info in saved_models:
            with st.expander(f"📦 {model_info.get('model_name', '未知模型')}"):
                st.markdown(f"**模型标识:** {model_info.get('model_key', 'N/A')}")
                st.markdown(f"**创建时间:** {model_info.get('created_at', 'N/A')}")

                metrics = model_info.get('metrics', {})
                if metrics:
                    st.markdown("**评估指标:**")
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            st.caption(f"- {metric_name}: {metric_value:.4f}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("加载此模型", key=f"load_{model_info.get('model_key')}"):
                        load_model_for_inference(session_id, model_info.get('model_key'))
                with col2:
                    if st.button("删除", key=f"delete_{model_info.get('model_key')}"):
                        ModelStorage.delete_model(session_id, model_info.get('model_key'))
                        st.rerun()
    else:
        st.info("暂无已保存的模型，请先训练模型")


def train_model(
        data: pd.DataFrame,
        features: List[str],
        target_col: Optional[str],
        task_type: str,
        model_key: str,
        user_params: Dict[str, Any],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        missing_strategy: str,
        scaling: str,
        encoding: str,
        cv_folds: int,
        early_stopping: bool,
        random_seed: int
):
    """执行模型训练"""

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # 1. 准备数据
        status_text.info("📊 正在准备数据...")
        progress_bar.progress(10)

        X = data[features].copy()

        if task_type in ["classification", "regression"]:
            y = data[target_col].copy()
        else:
            y = None

        # 2. 数据预处理配置
        status_text.info("🔧 正在配置预处理器...")
        progress_bar.progress(20)

        categorical_features = [col for col in features if col in data.select_dtypes(include=['object']).columns]
        numerical_features = [col for col in features if col not in categorical_features]

        preprocess_config = {
            "missing_strategy": missing_strategy,
            "scaling": scaling if scaling != "none" else None,
            "encoding": encoding if encoding != "none" else None,
            "categorical_features": categorical_features,
            "numerical_features": numerical_features,
            "target_column": target_col
        }

        preprocessor = DataPreprocessor(preprocess_config)

        # 3. 数据划分
        status_text.info("📊 正在划分数据集...")
        progress_bar.progress(30)

        np.random.seed(random_seed)

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

        # 4. 预处理训练数据
        status_text.info("🔧 正在预处理数据...")
        progress_bar.progress(40)

        X_train_processed = preprocessor.fit_transform(X_train, y_train)

        if X_val is not None:
            X_val_processed = preprocessor.transform(X_val)
        else:
            X_val_processed = None

        if X_test is not None:
            X_test_processed = preprocessor.transform(X_test)
        else:
            X_test_processed = None

        # 5. 训练模型
        status_text.info(f"🤖 正在训练模型: {model_key}...")
        progress_bar.progress(50)

        trainer = ModelTrainer(task_type, model_key, user_params)

        train_result = trainer.train(
            X_train_processed, y_train,
            X_val_processed, y_val,
            cv_folds=cv_folds if cv_folds > 0 else None,
            early_stopping=early_stopping
        )

        progress_bar.progress(70)

        # 6. 测试集评估
        status_text.info("📊 正在评估模型...")
        progress_bar.progress(80)

        test_metrics = None
        if X_test_processed is not None and y_test is not None:
            test_metrics = trainer.evaluate(X_test_processed, y_test)

        # 7. 保存模型
        status_text.info("💾 正在保存模型...")
        progress_bar.progress(90)

        session_id = st.session_state.current_source_name or "default"

        # 准备保存的指标
        save_metrics = train_result.get("train_score", {})
        if test_metrics:
            save_metrics["test"] = test_metrics

        ModelStorage.save_model(
            session_id=session_id,
            model_key=f"{task_type}_{model_key}_{int(time.time())}",
            model=trainer.model,
            preprocessor=preprocessor,
            metrics=save_metrics,
            config={
                "task_type": task_type,
                "model_key": model_key,
                "model_name": ModelRegistry.get_model(task_type, model_key).get("name"),
                "params": user_params,
                "features": features,
                "target_col": target_col,
                "preprocess_config": preprocess_config
            }
        )

        progress_bar.progress(100)
        status_text.success("✅ 训练完成！模型已保存")

        # 显示训练结果
        display_training_results(train_result, test_metrics)

        time.sleep(2)
        status_text.empty()
        progress_bar.empty()

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"训练失败: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language='python')


def display_training_results(train_result: Dict[str, Any], test_metrics: Optional[Dict[str, Any]]):
    """显示训练结果"""
    st.markdown("### 📊 训练结果")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**训练指标**")
        train_score = train_result.get("train_score", {})
        if isinstance(train_score, dict):
            for metric_name, metric_value in train_score.items():
                if isinstance(metric_value, (int, float)):
                    st.metric(metric_name, f"{metric_value:.4f}")
        else:
            st.metric("训练分数", f"{train_score:.4f}" if isinstance(train_score, float) else str(train_score))

    with col2:
        if test_metrics:
            st.markdown("**测试指标**")
            for metric_name, metric_value in test_metrics.items():
                if isinstance(metric_value, (int, float)):
                    st.metric(metric_name, f"{metric_value:.4f}")

    # 交叉验证结果
    cv_scores = train_result.get("cv_scores")
    if cv_scores and isinstance(cv_scores, dict):
        st.markdown("**交叉验证结果**")
        st.metric("平均分数", f"{cv_scores.get('mean', 0):.4f}")
        st.metric("标准差", f"{cv_scores.get('std', 0):.4f}")

    # 训练时间
    training_time = train_result.get("training_time", 0)
    st.caption(f"训练耗时: {training_time:.2f} 秒")


def load_model_for_inference(session_id: str, model_key: str):
    """加载模型用于推理"""
    try:
        model, preprocessor, metadata = ModelStorage.load_model(session_id, model_key)

        st.session_state.inference_model = model
        st.session_state.inference_preprocessor = preprocessor
        st.session_state.inference_metadata = metadata

        st.success(f"✅ 已加载模型: {metadata.get('model_name', model_key)}")
        st.rerun()

    except Exception as e:
        st.error(f"加载模型失败: {str(e)}")