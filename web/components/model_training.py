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
    save_trained_model,
    list_saved_models,
    delete_model,
    load_model_for_inference,
    execute_inference
)


def render_model_training():
    """渲染模型训练标签页 - 三大区块"""
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

    # 提取可用的特征列
    available_cols = get_available_features(full_data, variable_types)

    if not available_cols:
        st.warning("⚠️ 没有可用的特征列，请检查数据")
        return

    session_id = st.session_state.current_source_name or "default"

    # 检查是否需要刷新模型列表
    if st.session_state.get('refresh_model_list', False):
        st.session_state.refresh_model_list = False
        st.rerun()

    # ==================== 区块1：模型推荐 ====================
    st.markdown("---")
    render_model_recommendations(json_data, full_data, available_cols, variable_types)

    # ==================== 区块2：模型列表 ====================
    st.markdown("---")
    render_model_list_with_test(session_id)

    # ==================== 区块3：模型创建 ====================
    st.markdown("---")
    render_model_creation(full_data, available_cols, variable_types, session_id)

    # ==================== 训练完成后显示保存表单 ====================
    if st.session_state.get('training_completed', False):
        render_save_form(session_id)


def render_model_recommendations(json_data: Dict, data: pd.DataFrame,
                                  available_cols: List[str], variable_types: Dict):
    """区块1：模型推荐"""
    st.markdown("#### 📊 模型推荐")
    st.caption("基于数据分析结果，自动推荐适合的模型")

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
        html = """
        <style>
            .recommend-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }
            .recommend-table th {
                background-color: #1f77b4;
                color: white;
                padding: 10px 8px;
                text-align: center;
                border: 1px solid #ddd;
            }
            .recommend-table td {
                padding: 8px;
                text-align: left;
                border: 1px solid #ddd;
                vertical-align: top;
            }
            .recommend-table tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .recommend-table tr:hover {
                background-color: #f0f0f0;
            }
            .priority-high {
                color: #d32f2f;
                font-weight: bold;
            }
            .priority-mid {
                color: #f57c00;
                font-weight: bold;
            }
            .target-col {
                font-family: monospace;
                background-color: #f5f5f5;
                padding: 2px 4px;
                border-radius: 3px;
            }
        </style>
        <table class="recommend-table">
            <thead>
                <tr>
                    <th>优先级</th>
                    <th>任务类型</th>
                    <th>目标字段</th>
                    <th>推荐特征</th>
                    <th>传统模型</th>
                    <th>机器学习</th>
                    <th>深度学习</th>
                    <th>大模型</th>
                    <th>建议原因</th>
                </tr>
            </thead>
            <tbody>
        """

        for rec in model_recommendations:
            priority = rec.get('priority', '中')
            priority_class = "priority-high" if priority == "高" else "priority-mid" if priority == "中" else ""

            task_type = rec.get('task_type', '')
            target_column = rec.get('target_column', '')
            feature_columns = rec.get('feature_columns', [])
            feature_str = ', '.join(feature_columns[:5]) if feature_columns else '-'
            traditional = rec.get('traditional', '-')
            ml = rec.get('ml', '-')
            dl = rec.get('dl', '-')
            llm = rec.get('llm', '-')
            reason = rec.get('reason', '')

            html += f"""
                <tr>
                    <td class="{priority_class}">{priority}</td>
                    <td>{task_type}</td>
                    <td><code class="target-col">{target_column}</code></td>
                    <td>{feature_str}</td>
                    <td>{traditional}</td>
                    <td>{ml}</td>
                    <td>{dl}</td>
                    <td>{llm}</td>
                    <td>{reason[:60]}...</td>
                </tr>
            """

        html += """
            </tbody>
        </table>
        """

        st.html(html)
    else:
        st.info("暂无模型推荐，请先完成数据分析")


def render_model_list_with_test(session_id: str):
    """区块2：模型列表 + 结构化测试表单"""
    st.markdown("#### 📦 模型列表")

    saved_models = list_saved_models(session_id)

    if not saved_models:
        st.info("暂无已保存的模型，请先在下方「模型创建」中训练模型")
        return

    if 'test_model_key' not in st.session_state:
        st.session_state.test_model_key = None
    if 'test_model_display_name' not in st.session_state:
        st.session_state.test_model_display_name = None

    table_data = []
    for model_info in saved_models:
        config = model_info.get('config', {})
        metrics = model_info.get('metrics', {})

        main_metric = ""
        if config.get('task_type') == 'classification':
            train_metrics = metrics.get('train_score', {})
            if isinstance(train_metrics, dict):
                acc = train_metrics.get('accuracy')
                if acc:
                    main_metric = f"准确率: {acc:.4f}"
                else:
                    f1 = train_metrics.get('f1_score')
                    if f1:
                        main_metric = f"F1: {f1:.4f}"
        elif config.get('task_type') == 'regression':
            train_metrics = metrics.get('train_score', {})
            if isinstance(train_metrics, dict):
                r2 = train_metrics.get('r2')
                if r2:
                    main_metric = f"R²: {r2:.4f}"
        elif config.get('task_type') == 'clustering':
            train_metrics = metrics.get('train_score', {})
            if isinstance(train_metrics, dict):
                sil = train_metrics.get('silhouette_score')
                if sil:
                    main_metric = f"轮廓系数: {sil:.4f}"

        model_key = config.get('model_key', '')
        model_display_name = get_model_display_name(model_key)
        user_model_name = model_info.get('user_model_name', '')
        display_name = user_model_name if user_model_name else f"{model_display_name}_{model_info.get('created_at', '')[:10]}"

        # 获取特征列表
        features = config.get('features', [])
        features_str = ', '.join(features[:3]) + ('...' if len(features) > 3 else '') if features else '-'

        table_data.append({
            "model_key": model_info.get('model_key'),
            "display_name": display_name,
            "model_name": model_display_name,
            "task_type": config.get('task_type', '未知'),
            "metric": main_metric,
            "features": features_str,
            "feature_count": len(features),
            "created_at": model_info.get('created_at', '')[:19],
            "config": config
        })

    # 显示表格
    for item in table_data:
        with st.container():
            col1, col2, col3, col4, col5, col6 = st.columns([1.5, 1, 1, 1.5, 1, 1])

            with col1:
                st.markdown(f"**{item['display_name']}**")
                st.caption(item['model_name'])

            with col2:
                task_display = {
                    "classification": "分类",
                    "regression": "回归",
                    "clustering": "聚类",
                    "time_series": "时序"
                }.get(item['task_type'], item['task_type'])
                st.markdown(f"`{task_display}`")

            with col3:
                st.caption(item['metric'] if item['metric'] else "-")

            with col4:
                st.caption(f"{item['feature_count']}个")
                st.caption(item['features'])

            with col5:
                st.caption(item['created_at'][:10] if item['created_at'] else "-")

            with col6:
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("推理", key=f"test_{item['model_key']}", use_container_width=True):
                        st.session_state.test_model_key = item['model_key']
                        st.session_state.test_model_display_name = item['display_name']
                        st.session_state.test_model_config = item['config']
                        st.rerun()
                with btn_col2:
                    if st.button("删除", key=f"del_{item['model_key']}", use_container_width=True):
                        if delete_model(session_id, item['model_key']):
                            st.success(f"已删除模型: {item['display_name']}")
                            if st.session_state.test_model_key == item['model_key']:
                                st.session_state.test_model_key = None
                            st.rerun()
                        else:
                            st.error("删除失败")

            st.markdown("---")

    if st.session_state.test_model_key is not None:
        render_test_form(session_id)


def render_test_form(session_id: str):
    """渲染结构化测试表单"""
    st.markdown(f"#### 📝 结构化测试 - 模型: {st.session_state.test_model_display_name}")

    try:
        model, preprocessor, metadata = load_model_for_inference(session_id, st.session_state.test_model_key)
        config = metadata.get('config', {})
        features = config.get('features', [])

        # 显示特征数量
        st.info(f"模型特征数: {len(features)}，特征列表: {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")

        preprocess_config = config.get('preprocess_config', {})
        categorical_features = preprocess_config.get('categorical_features', [])

        if not features:
            st.warning("模型没有特征列，无法进行推理")
            return

        input_data = {}
        cols = st.columns(2)
        for i, feature in enumerate(features[:20]):
            with cols[i % 2]:
                if feature in categorical_features:
                    input_data[feature] = st.text_input(
                        f"{feature}",
                        placeholder=f"输入 {feature} 的值",
                        key=f"test_inp_{feature}"
                    )
                else:
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        value=0.0,
                        step=0.01,
                        format="%.4f",
                        key=f"test_inp_{feature}"
                    )

        if len(features) > 20:
            st.caption(f"... 还有 {len(features) - 20} 个特征未显示")

        if st.button("🔍 执行推理", key="execute_test_inference", use_container_width=True):
            valid_data = {}
            for feature in features:
                value = input_data.get(feature)
                if value is not None and value != "":
                    valid_data[feature] = value
                else:
                    valid_data[feature] = None

            with st.spinner("推理中..."):
                result = execute_inference(model, preprocessor, valid_data, features, categorical_features)

            if result:
                st.markdown("**📊 推理结果**")

                # 尝试多种可能的键名获取预测值
                pred = None
                if "prediction" in result:
                    pred = result["prediction"]
                elif "predictions" in result:
                    pred = result["predictions"]
                    # 如果是列表且只有一个元素，取出第一个
                    if isinstance(pred, list) and len(pred) == 1:
                        pred = pred[0]
                elif "forecast" in result:
                    pred = result["forecast"]
                    if isinstance(pred, list) and len(pred) == 1:
                        pred = pred[0]

                if pred is not None:
                    if isinstance(pred, (int, float)):
                        st.metric("预测值", f"{pred:.4f}" if isinstance(pred, float) else str(pred))
                    else:
                        st.metric("预测结果", str(pred))
                else:
                    # 如果都没有，显示原始结果
                    st.json(result)

                # 置信度（分类任务可能有）
                if "confidence" in result:
                    conf = result["confidence"]
                    if isinstance(conf, list) and len(conf) > 0:
                        st.metric("置信度", f"{conf[0]:.2%}")
                    elif isinstance(conf, (int, float)):
                        st.metric("置信度", f"{conf:.2%}")

                # 概率分布（分类任务可能有）
                if "probabilities" in result:
                    probs = result["probabilities"]
                    if probs and len(probs) > 0:
                        st.markdown("**类别概率分布:**")
                        prob_list = probs[0] if isinstance(probs, list) and len(probs) > 0 else probs
                        if isinstance(prob_list, (list, np.ndarray)):
                            for i, prob in enumerate(prob_list[:5]):
                                if prob is not None:
                                    st.progress(float(prob), text=f"类别 {i}: {float(prob):.2%}")

                # 聚类结果
                if "cluster_id" in result:
                    st.metric("所属簇", result["cluster_id"])
                if "cluster_ids" in result:
                    cluster_ids = result["cluster_ids"]
                    if isinstance(cluster_ids, list) and len(cluster_ids) == 1:
                        st.metric("所属簇", cluster_ids[0])

                # 如果是错误
                if "error" in result:
                    st.error(f"推理错误: {result['error']}")
                    if "traceback" in result:
                        with st.expander("详细错误信息"):
                            st.code(result['traceback'], language='python')
            else:
                st.error("推理失败，未返回结果")
    except Exception as e:
        import traceback
        st.error(f"加载模型失败: {str(e)}")
        with st.expander("详细错误信息"):
            st.code(traceback.format_exc(), language='python')


def render_model_creation(data: pd.DataFrame, available_cols: List[str],
                           variable_types: Dict, session_id: str):
    """区块3：模型创建"""
    st.markdown("#### 🆕 模型创建")

    # 任务配置
    st.markdown("**📋 任务配置**")

    col1, col2 = st.columns(2)

    with col1:
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
        if task_type in ["classification", "regression"]:
            target_col = st.selectbox(
                "目标列",
                options=available_cols,
                key="train_target_col"
            )
        else:
            target_col = None
            st.selectbox("目标列", options=["无（无监督学习）"], disabled=True)

    st.markdown("---")

    # 特征选择 - 使用所有列（包括日期派生列）
    st.markdown("**🔧 特征选择**")

    all_columns = list(data.columns)
    default_features = [col for col in all_columns if col != target_col]

    selected_features = st.multiselect(
        "选择特征列",
        options=all_columns,
        default=default_features[:20] if len(default_features) > 20 else default_features,
        key="train_features"
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

    selected_model_key = st.selectbox(
        "选择模型",
        options=model_options,
        format_func=lambda x: f"{get_model_display_name(x)} - {models[x]['description'][:50]}",
        key="train_model_key"
    )

    if selected_model_key:
        model_info = models[selected_model_key]
        st.markdown(f"**{get_model_display_name(selected_model_key)}** - {model_info['description']}")
        st.caption(f"依赖: {', '.join(model_info.get('requirements', []))}")

    st.markdown("---")

    # 高级配置（折叠）
    with st.expander("⚙️ 高级配置", expanded=False):
        # 模型参数配置
        if selected_model_key:
            st.markdown("**模型参数配置**")
            model_params = get_model_params(task_type, selected_model_key)
            user_params = {}

            param_items = list(model_params.items())
            if param_items:
                param_cols = st.columns(3)
                for i, (param_name, param_info) in enumerate(param_items):
                    with param_cols[i % 3]:
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

        st.markdown("---")
        st.markdown("**训练控制**")

        col1, col2 = st.columns(2)
        with col1:
            cv_folds = st.number_input("交叉验证折数", min_value=0, max_value=10, value=5, key="cv_folds")
        with col2:
            random_seed = st.number_input("随机种子", min_value=0, max_value=9999, value=42, key="random_seed")

    # 开始训练按钮
    st.markdown("---")

    if st.button("▶️ 开始训练", type="primary", use_container_width=True):
        # 获取用户参数
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

        # 获取高级配置参数
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
                cv_folds, random_seed, session_id
            )

        if success:
            st.session_state.trained_model = result["model"]
            st.session_state.trained_preprocessor = result["preprocessor"]
            st.session_state.trained_config = result["config"]
            st.session_state.trained_metrics = result["metrics"]
            st.session_state.trained_train_result = result.get("train_result", {})
            st.session_state.training_completed = True
            st.rerun()
        else:
            st.error(f"训练失败: {result.get('error', '未知错误')}")


def render_save_form(session_id: str):
    """渲染保存表单"""
    st.markdown("---")
    st.markdown("### 💾 保存模型")

    config = st.session_state.trained_config
    task_type = config.get('task_type', 'classification')
    target_col = config.get('target_col', '')
    model_key = config.get('model_key', '')

    default_name = generate_model_name(task_type, target_col, model_key)

    user_model_name = st.text_input(
        "模型名称",
        value=default_name,
        key="model_name_input",
        help="可为模型设置一个易于识别的名称"
    )

    # 显示训练结果
    train_result = st.session_state.get('trained_train_result', {})
    if train_result:
        st.markdown("**训练结果**")
        train_score = train_result.get("train_score", {})
        if isinstance(train_score, dict):
            for metric_name, metric_value in train_score.items():
                if isinstance(metric_value, (int, float)):
                    st.metric(metric_name, f"{metric_value:.4f}")

        training_time = train_result.get("training_time", 0)
        st.caption(f"训练耗时: {training_time:.2f} 秒")

    if st.button("确认保存", type="primary", use_container_width=True):
        success = save_trained_model(
            session_id=session_id,
            model=st.session_state.trained_model,
            preprocessor=st.session_state.trained_preprocessor,
            config=config,
            metrics=st.session_state.trained_metrics,
            user_model_name=user_model_name
        )

        if success:
            # 清除训练状态
            del st.session_state.training_completed
            del st.session_state.trained_model
            del st.session_state.trained_preprocessor
            del st.session_state.trained_config
            del st.session_state.trained_metrics
            if 'trained_train_result' in st.session_state:
                del st.session_state.trained_train_result

            st.session_state.refresh_model_list = True
            st.success("✅ 模型已保存！")
            time.sleep(1)
            st.rerun()
        else:
            st.error("保存失败")