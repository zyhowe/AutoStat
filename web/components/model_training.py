# web/components/model_training.py

"""模型训练UI组件 - 界面渲染"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from typing import Dict, Any, Optional, List

from web.services.model_training_service import (
    get_model_display_name,
    generate_model_name,
    get_available_features,
    get_model_recommendations_from_json,
    get_models_by_task_type,
    get_model_params,
    execute_training,
    list_saved_models,
    delete_model,
    load_model_for_inference,
    execute_inference
)
from web.services.storage_service import StorageService
from web.services.session_service import SessionService


def render_model_training():
    """渲染模型训练标签页"""
    st.markdown("### 🤖 小模型训练")
    st.caption("基于分析结果，训练传统机器学习/深度学习模型")

    session_id = SessionService.get_current_session()
    if session_id is None:
        st.info("📌 请先在「数据准备」中完成数据分析")
        return

    # 从存储加载数据
    processed_data = StorageService.load_dataframe("processed_data", session_id)
    if processed_data is None:
        st.info("📌 无有效数据，请检查数据文件")
        return

    json_data = StorageService.load_json("analysis_result", session_id)
    if json_data is None:
        st.info("📌 请先在「数据准备」中完成数据分析")
        return

    variable_types = json_data.get('variable_types', {})
    available_cols = get_available_features(processed_data, variable_types)

    if not available_cols:
        st.warning("⚠️ 没有可用的特征列，请检查数据")
        return

    # 初始化状态
    if 'auto_fill_task_type' not in st.session_state:
        st.session_state.auto_fill_task_type = None
    if 'auto_fill_target_col' not in st.session_state:
        st.session_state.auto_fill_target_col = None
    if 'auto_fill_features' not in st.session_state:
        st.session_state.auto_fill_features = None
    if 'auto_fill_model_key' not in st.session_state:
        st.session_state.auto_fill_model_key = None
    if 'auto_fill_model_params' not in st.session_state:
        st.session_state.auto_fill_model_params = None
    if 'auto_fill_model_name' not in st.session_state:
        st.session_state.auto_fill_model_name = None
    if 'infer_model_key' not in st.session_state:
        st.session_state.infer_model_key = None
    if 'model_training_tab' not in st.session_state:
        st.session_state.model_training_tab = 0

    # 标签页
    tab_options = ["📊 模型推荐", "🆕 模型创建", "🔮 模型推理"]
    selected_tab = st.radio(
        "选择标签页",
        options=tab_options,
        index=st.session_state.model_training_tab,
        horizontal=True,
        label_visibility="collapsed"
    )
    st.divider()

    current_tab = tab_options.index(selected_tab)

    if current_tab == 0:
        render_model_recommendations(json_data, processed_data, available_cols, variable_types)
    elif current_tab == 1:
        render_model_creation(processed_data, available_cols, variable_types, session_id)
    else:
        render_model_inference(session_id)


def render_model_recommendations(json_data: Dict, data: pd.DataFrame,
                                  available_cols: List[str], variable_types: Dict):
    """模型推荐标签页"""
    st.markdown("基于数据分析结果，自动推荐适合的模型。点击「使用」快速填充到下方表单")

    n_samples = len(data)
    n_features = len(available_cols)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("样本数", f"{n_samples:,}")
    with col2:
        st.metric("特征数", n_features)
    with col3:
        st.metric("数据类型", f"{len(variable_types)} 种")

    st.markdown("---")

    model_recommendations = get_model_recommendations_from_json(json_data)

    if model_recommendations:
        for i, rec in enumerate(model_recommendations[:5]):
            priority = rec.get('priority', '中')
            priority_icon = "🔴" if priority == "高" else "🟠" if priority == "中" else "🟢"

            with st.container():
                cols = st.columns([1.5, 1.5, 2, 1.5, 1])
                with cols[0]:
                    st.markdown(f"**{priority_icon} {rec.get('task_type', '')}**")
                    st.caption(f"目标: {rec.get('target_column', '')}")
                with cols[1]:
                    ml_text = rec.get('ml', '')
                    if ml_text:
                        ml_short = ml_text.split(' / ')[0] if ' / ' in ml_text else ml_text
                        st.caption(f"推荐模型: {ml_short}")
                    else:
                        st.caption("推荐模型: 随机森林")
                with cols[2]:
                    feature_cols = rec.get('feature_columns', [])
                    if feature_cols:
                        valid_features = [f for f in feature_cols if f in available_cols]
                        if valid_features:
                            feature_str = ', '.join(valid_features[:5])
                            if len(valid_features) > 5:
                                feature_str += f'... 等{len(valid_features)}个'
                            st.caption(f"推荐特征: {feature_str}")
                        else:
                            st.caption("推荐特征: 自动选择所有可用特征")
                    else:
                        st.caption("推荐特征: 自动选择所有可用特征")
                with cols[3]:
                    reason = rec.get('reason', '')
                    if len(reason) > 40:
                        st.caption(f"原因: {reason[:40]}...")
                    else:
                        st.caption(f"原因: {reason}")
                with cols[4]:
                    if st.button("📋 使用", key=f"use_rec_{i}"):
                        task_type_str = rec.get('task_type', '')
                        if "分类" in task_type_str:
                            task_type = "classification"
                        elif "数值预测" in task_type_str or "回归" in task_type_str:
                            task_type = "regression"
                        else:
                            task_type = "classification"

                        ml_text = rec.get('ml', '随机森林')
                        first_model = ml_text.split(' / ')[0] if ' / ' in ml_text else ml_text

                        if 'XGBoost' in first_model:
                            model_key = 'xgboost' if task_type == 'classification' else 'xgboost_regressor'
                        elif 'LightGBM' in first_model:
                            model_key = 'lightgbm' if task_type == 'classification' else 'lightgbm_regressor'
                        elif 'CatBoost' in first_model:
                            model_key = 'catboost' if task_type == 'classification' else 'catboost'
                        elif '逻辑回归' in first_model:
                            model_key = 'logistic_regression'
                        elif '线性回归' in first_model:
                            model_key = 'linear_regression'
                        elif '决策树' in first_model:
                            model_key = 'decision_tree' if task_type == 'classification' else 'decision_tree_regressor'
                        elif '随机森林' in first_model:
                            model_key = 'random_forest' if task_type == 'classification' else 'random_forest_regressor'
                        else:
                            model_key = 'random_forest' if task_type == 'classification' else 'random_forest_regressor'

                        recommended_features = rec.get('feature_columns', [])
                        valid_features = [f for f in recommended_features if f in available_cols]
                        if not valid_features:
                            target = rec.get('target_column', '')
                            valid_features = [col for col in available_cols if col != target][:20]

                        model_params = get_model_params(task_type, model_key)
                        default_params = {}
                        for param_name, param_info in model_params.items():
                            default_params[param_name] = param_info.get('default')

                        auto_model_name = generate_model_name(task_type, rec.get('target_column', ''), model_key)

                        st.session_state.auto_fill_task_type = task_type
                        st.session_state.auto_fill_target_col = rec.get('target_column', '')
                        st.session_state.auto_fill_features = valid_features
                        st.session_state.auto_fill_model_key = model_key
                        st.session_state.auto_fill_model_params = default_params
                        st.session_state.auto_fill_model_name = auto_model_name
                        st.session_state.model_training_tab = 1
                        st.rerun()
            st.divider()
    else:
        st.info("暂无模型推荐")


def render_model_creation(data: pd.DataFrame, available_cols: List[str],
                           variable_types: Dict, session_id: str):
    """模型创建标签页"""
    st.markdown("配置模型参数并开始训练")

    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("➕ 创建新模型", use_container_width=True):
            # 清空自动填充标志
            st.session_state.auto_fill_task_type = None
            st.session_state.auto_fill_target_col = None
            st.session_state.auto_fill_features = None
            st.session_state.auto_fill_model_key = None
            st.session_state.auto_fill_model_params = None
            st.session_state.auto_fill_model_name = None

            # 清空表单控件的值
            if 'train_task_type_selected' in st.session_state:
                del st.session_state.train_task_type_selected
            if 'train_target_col_selected' in st.session_state:
                del st.session_state.train_target_col_selected
            if 'train_features_selected' in st.session_state:
                del st.session_state.train_features_selected
            if 'train_model_key_selected' in st.session_state:
                del st.session_state.train_model_key_selected

            # 清空参数值
            param_keys = [k for k in st.session_state.keys() if k.startswith('param_')]
            for key in param_keys:
                del st.session_state[key]

            # 清空模型名称
            if 'model_name_input' in st.session_state:
                del st.session_state.model_name_input

            st.rerun()

    st.markdown("---")

    # 初始化默认值
    if 'train_task_type_selected' not in st.session_state:
        st.session_state.train_task_type_selected = "classification"
    if 'train_target_col_selected' not in st.session_state:
        st.session_state.train_target_col_selected = available_cols[0] if available_cols else None
    if 'train_features_selected' not in st.session_state:
        st.session_state.train_features_selected = [col for col in available_cols if col != st.session_state.train_target_col_selected][:20]
    if 'train_model_key_selected' not in st.session_state:
        st.session_state.train_model_key_selected = "random_forest"

    # 任务配置
    st.markdown("**📋 任务配置**")

    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.auto_fill_task_type:
            st.session_state.train_task_type_selected = st.session_state.auto_fill_task_type
            st.session_state.auto_fill_task_type = None

        task_type = st.selectbox(
            "任务类型",
            options=["classification", "regression", "clustering", "time_series"],
            format_func=lambda x: {
                "classification": "📊 分类",
                "regression": "📈 回归",
                "clustering": "🔘 聚类",
                "time_series": "📅 时间序列"
            }.get(x, x),
            key="train_task_type_selected"
        )

    with col2:
        if task_type in ["classification", "regression"]:
            if st.session_state.auto_fill_target_col:
                st.session_state.train_target_col_selected = st.session_state.auto_fill_target_col
                st.session_state.auto_fill_target_col = None

            target_col = st.selectbox(
                "目标列",
                options=available_cols,
                key="train_target_col_selected"
            )
        else:
            target_col = None
            st.selectbox("目标列", options=["无（无监督学习）"], disabled=True)

    st.markdown("---")

    # 特征选择
    st.markdown("**🔧 特征选择**")

    all_columns = list(data.columns)

    if st.session_state.auto_fill_features:
        st.session_state.train_features_selected = st.session_state.auto_fill_features
        st.session_state.auto_fill_features = None

    selected_features = st.multiselect(
        "选择特征列",
        options=all_columns,
        key="train_features_selected"
    )

    if not selected_features:
        st.warning("⚠️ 请至少选择一个特征列")
        return

    st.caption(f"已选择 {len(selected_features)} 个特征")
    st.markdown("---")

    # 模型选择
    st.markdown("**🤖 模型选择**")

    models = get_models_by_task_type(task_type)
    model_options = list(models.keys())

    if not model_options:
        st.warning(f"⚠️ 暂无{task_type}类型的模型可用")
        return

    if st.session_state.auto_fill_model_key:
        if st.session_state.auto_fill_model_key in model_options:
            st.session_state.train_model_key_selected = st.session_state.auto_fill_model_key
        st.session_state.auto_fill_model_key = None

    selected_model_key = st.selectbox(
        "选择模型",
        options=model_options,
        format_func=lambda x: f"{get_model_display_name(x)} - {models[x]['description'][:50]}",
        key="train_model_key_selected"
    )

    if selected_model_key:
        model_info = models[selected_model_key]
        st.markdown(f"**{get_model_display_name(selected_model_key)}** - {model_info['description']}")

    st.markdown("---")

    # 高级配置
    with st.expander("⚙️ 高级配置", expanded=False):
        if selected_model_key:
            st.markdown("**模型参数配置**")
            model_params = get_model_params(task_type, selected_model_key)
            user_params = {}

            param_items = list(model_params.items())
            auto_params = st.session_state.auto_fill_model_params or {}

            if param_items:
                param_cols = st.columns(3)
                for i, (param_name, param_info) in enumerate(param_items):
                    with param_cols[i % 3]:
                        param_type = param_info.get('type', 'unknown')
                        default_val = param_info.get('default')

                        if param_name in auto_params:
                            default_val = auto_params[param_name]

                        description = param_info.get('description', '')
                        param_key = f"param_{param_name}"

                        if param_type == 'int':
                            param_range = param_info.get('range', [1, 100])
                            value = st.number_input(
                                f"{param_name}",
                                min_value=param_range[0],
                                max_value=param_range[1] if len(param_range) > 1 else 1000,
                                value=default_val if default_val else param_range[0],
                                step=1,
                                help=description,
                                key=param_key
                            )
                            user_params[param_name] = value
                        elif param_type == 'float':
                            param_range = param_info.get('range', [0.0, 1.0])
                            value = st.number_input(
                                f"{param_name}",
                                min_value=float(param_range[0]),
                                max_value=float(param_range[1]) if len(param_range) > 1 else 1.0,
                                value=float(default_val) if default_val else param_range[0],
                                step=0.01,
                                help=description,
                                key=param_key
                            )
                            user_params[param_name] = value
                        elif param_type == 'choice':
                            options = param_info.get('options', [])
                            default_idx = 0
                            if default_val in options:
                                default_idx = options.index(default_val)
                            value = st.selectbox(
                                f"{param_name}",
                                options=options,
                                index=default_idx,
                                help=description,
                                key=param_key
                            )
                            user_params[param_name] = value
                        elif param_type == 'bool':
                            value = st.checkbox(
                                f"{param_name}",
                                value=default_val if default_val else False,
                                help=description,
                                key=param_key
                            )
                            user_params[param_name] = value
                        else:
                            value = st.text_input(
                                f"{param_name}",
                                value=str(default_val) if default_val else "",
                                help=description,
                                key=param_key
                            )
                            user_params[param_name] = value

                st.session_state.auto_fill_model_params = None
            else:
                st.info("该模型无额外参数配置")
                user_params = {}

        st.markdown("---")
        st.markdown("**数据划分**")

        col1, col2, col3 = st.columns(3)
        with col1:
            train_ratio = st.slider("训练集比例", 0.5, 0.9, 0.7, 0.05, key="train_ratio")
        with col2:
            val_ratio = st.slider("验证集比例", 0.0, 0.3, 0.15, 0.05, key="val_ratio")
        with col3:
            test_ratio = 1 - train_ratio - val_ratio
            st.metric("测试集比例", f"{test_ratio:.0%}")

        st.markdown("---")
        st.markdown("**预处理配置**")

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
                    "standard": "标准化",
                    "minmax": "归一化",
                    "robust": "鲁棒缩放",
                    "none": "不缩放"
                }.get(x, x),
                key="preprocess_scaling"
            )
        with col3:
            encoding = st.selectbox(
                "分类编码",
                options=["onehot", "label", "none"],
                format_func=lambda x: {
                    "onehot": "独热编码",
                    "label": "标签编码",
                    "none": "不编码"
                }.get(x, x),
                key="preprocess_encoding"
            )

        st.markdown("---")
        st.markdown("**训练控制**")

        col1, col2 = st.columns(2)
        with col1:
            cv_folds = st.number_input("交叉验证折数", min_value=0, max_value=10, value=5, key="cv_folds")
        with col2:
            random_seed = st.number_input("随机种子", min_value=0, max_value=9999, value=42, key="random_seed")

    # 模型名称
    st.markdown("**📝 模型名称**")

    if st.session_state.auto_fill_model_name:
        default_model_name = st.session_state.auto_fill_model_name
        st.session_state.auto_fill_model_name = None
    else:
        default_model_name = generate_model_name(task_type, target_col if target_col else "", selected_model_key)

    user_model_name = st.text_input(
        "模型名称",
        value=default_model_name,
        key="model_name_input"
    )

    st.markdown("---")

    if st.button("▶️ 开始训练", type="primary", use_container_width=True):
        user_params = {}
        if selected_model_key:
            model_params = get_model_params(task_type, selected_model_key)
            for param_name in model_params.keys():
                stored_value = st.session_state.get(f"param_{param_name}")
                if stored_value is not None:
                    user_params[param_name] = stored_value
                else:
                    default_info = model_params[param_name]
                    user_params[param_name] = default_info.get('default')

        train_ratio = st.session_state.get("train_ratio", 0.7)
        val_ratio = st.session_state.get("val_ratio", 0.15)
        missing_strategy = st.session_state.get("preprocess_missing", "drop")
        scaling = st.session_state.get("preprocess_scaling", "standard")
        encoding = st.session_state.get("preprocess_encoding", "onehot")
        cv_folds = st.session_state.get("cv_folds", 5)
        random_seed = st.session_state.get("random_seed", 42)

        with st.spinner("训练中..."):
            success, result = execute_training(
                data, selected_features, target_col, task_type,
                selected_model_key, user_params,
                train_ratio, val_ratio,
                missing_strategy, scaling, encoding,
                cv_folds, random_seed, session_id,
                user_model_name
            )

        if success:
            st.success(f"✅ 训练完成！模型已保存: {user_model_name}")
            st.session_state.model_training_tab = 2
            time.sleep(1)
            st.rerun()
        else:
            st.error(f"训练失败: {result.get('error', '未知错误')}")
            if 'traceback' in result:
                with st.expander("详细错误信息"):
                    st.code(result['traceback'], language='python')


def render_model_inference(session_id: str):
    """模型推理标签页"""
    st.markdown("选择已训练的模型，输入特征值进行预测")

    saved_models = list_saved_models(session_id)

    if not saved_models:
        st.info("暂无已保存的模型，请先在「模型创建」中训练模型")
        return

    # 模型列表
    st.markdown("#### 📦 已保存的模型")
    for item in saved_models:
        with st.container():
            cols = st.columns([2, 1, 1.5, 1.5, 1])
            with cols[0]:
                st.markdown(f"**{item.get('user_model_name', item.get('model_key'))}**")
            with cols[1]:
                task_display = {
                    "classification": "分类",
                    "regression": "回归",
                    "clustering": "聚类",
                    "time_series": "时序"
                }.get(item.get('task_type'), item.get('task_type'))
                st.caption(task_display)
            with cols[2]:
                metrics = item.get('metrics', {})
                if metrics.get('accuracy'):
                    st.caption(f"准确率: {metrics['accuracy']:.2%}")
                elif metrics.get('r2'):
                    st.caption(f"R²: {metrics['r2']:.3f}")
                else:
                    st.caption("-")
            with cols[3]:
                st.caption(f"特征数: {len(item.get('features', []))}")
            with cols[4]:
                if st.button("🗑️ 删除", key=f"del_infer_{item.get('model_key')}", use_container_width=True):
                    if delete_model(item.get('model_key'), session_id):
                        st.success(f"已删除模型: {item.get('user_model_name')}")
                        st.rerun()
                    else:
                        st.error("删除失败")
        st.divider()

    st.markdown("---")
    st.markdown("#### 🔮 模型预测")

    model_options = {m.get('model_key'): m.get('user_model_name', m.get('model_key')) for m in saved_models}
    selected_model_key = st.selectbox(
        "选择模型",
        options=list(model_options.keys()),
        format_func=lambda x: model_options.get(x, x),
        key="inference_model_select"
    )

    if not selected_model_key:
        return

    try:
        model, preprocessor, metadata, metrics = load_model_for_inference(selected_model_key, session_id)

        config = metadata
        features = config.get('features', [])
        preprocess_config = config.get('preprocess_config', {})
        categorical_features = preprocess_config.get('categorical_features', [])

        if not features:
            st.warning("模型没有特征列，无法进行推理")
            return

        st.info(f"特征列 ({len(features)}个): {', '.join(features[:8])}{'...' if len(features) > 8 else ''}")

        with st.expander("📊 模型评估指标", expanded=False):
            train_score = metrics.get("train_score", {})
            if train_score:
                cols = st.columns(3)
                for i, (metric_name, metric_value) in enumerate(train_score.items()):
                    if isinstance(metric_value, (int, float)):
                        with cols[i % 3]:
                            st.metric(metric_name, f"{metric_value:.4f}")

        st.markdown("**📝 输入特征值**")

        input_data = {}
        cols = st.columns(2)
        for i, feature in enumerate(features[:16]):
            with cols[i % 2]:
                if feature in categorical_features:
                    input_data[feature] = st.text_input(
                        f"{feature}",
                        placeholder=f"输入 {feature} 的值",
                        key=f"infer_{feature}_{selected_model_key[:10]}"
                    )
                else:
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        value=0.0,
                        step=0.01,
                        format="%.4f",
                        key=f"infer_{feature}_{selected_model_key[:10]}"
                    )

        if len(features) > 16:
            st.caption(f"... 还有 {len(features) - 16} 个特征未显示")

        if st.button("🔍 执行推理", type="primary", use_container_width=True):
            valid_data = {}
            for feature in features:
                value = input_data.get(feature)
                if value is not None and value != "":
                    valid_data[feature] = value
                else:
                    valid_data[feature] = None

            result = execute_inference(model, preprocessor, valid_data, features, categorical_features)

            if result and "error" not in result:
                st.markdown("### 📊 推理结果")

                pred = result.get("prediction")
                if pred is None:
                    predictions = result.get("predictions")
                    if predictions and isinstance(predictions, list) and len(predictions) == 1:
                        pred = predictions[0]

                if pred is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        if isinstance(pred, (int, float)):
                            st.metric("预测值", f"{pred:.4f}")
                        else:
                            st.metric("预测结果", str(pred))

                    if result.get("confidence"):
                        conf = result["confidence"]
                        with col2:
                            if isinstance(conf, list) and len(conf) > 0:
                                st.metric("置信度", f"{conf[0]:.2%}")
                            elif isinstance(conf, (int, float)):
                                st.metric("置信度", f"{conf:.2%}")

                if result.get("probabilities"):
                    probs = result["probabilities"]
                    if probs and len(probs) > 0:
                        st.markdown("**类别概率分布:**")
                        prob_list = probs[0] if isinstance(probs, list) and len(probs) > 0 else probs
                        if isinstance(prob_list, (list, np.ndarray)):
                            for i, prob in enumerate(prob_list[:5]):
                                if prob is not None:
                                    st.progress(float(prob), text=f"类别 {i}: {float(prob):.2%}")
            else:
                st.error(f"推理失败: {result.get('error', '未知错误')}")

    except Exception as e:
        st.error(f"加载模型失败: {str(e)}")