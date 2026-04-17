"""推理组件 - 在AI解读中集成模型推理"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from typing import Dict, Any, Optional

from autostat.models.predictor import ModelPredictor
from autostat.models.storage import ModelStorage
from autostat.models.registry import ModelRegistry


def render_inference_interface():
    """渲染推理界面"""
    st.markdown("#### 🔮 模型推理预测")
    st.caption("输入结构化数据，调用已训练的模型进行预测")

    # 检查是否有已加载的模型
    if not hasattr(st.session_state, 'inference_model') or st.session_state.inference_model is None:
        # 尝试加载最佳模型
        session_id = st.session_state.current_source_name or "default"
        best_model = ModelStorage.get_best_model(session_id)

        if best_model:
            st.info(f"发现已保存的模型: {best_model.get('model_name')}")
            if st.button("加载此模型"):
                load_model_for_inference(session_id, best_model.get('model_key'))
        else:
            st.warning("⚠️ 暂无可用模型，请先在「模型训练」标签页训练模型")
        return

    # 显示当前加载的模型信息
    metadata = st.session_state.inference_metadata
    st.info(f"**当前模型:** {metadata.get('model_name', '未知')}")
    st.caption(f"模型标识: {metadata.get('model_key', 'N/A')}")

    # 显示评估指标
    metrics = metadata.get('metrics', {})
    if metrics:
        st.caption("**评估指标:**")
        for metric_name, metric_value in list(metrics.items())[:3]:
            if isinstance(metric_value, (int, float)):
                st.caption(f"- {metric_name}: {metric_value:.4f}")

    st.divider()

    # 只保留结构化数据输入（移除自然语言，避免解析问题）
    render_structured_data_inference()


def render_structured_data_inference():
    """结构化数据推理"""
    st.markdown("**结构化数据输入**")
    st.caption("请输入预测所需的特征值")

    # 获取模型特征列表
    metadata = st.session_state.inference_metadata
    config = metadata.get('config', {})
    features = config.get('features', [])

    if not features:
        st.warning("无法获取模型特征列表")
        return

    # 获取特征类型信息（从训练配置中获取）
    preprocess_config = config.get('preprocess_config', {})
    categorical_features = preprocess_config.get('categorical_features', [])
    numerical_features = preprocess_config.get('numerical_features', [])

    # 动态生成输入表单
    input_data = {}

    # 使用多列布局
    cols = st.columns(2)
    for i, feature in enumerate(features[:20]):  # 最多显示20个特征
        with cols[i % 2]:
            # 根据特征类型显示不同的输入控件
            if feature in categorical_features:
                # 分类特征：使用下拉选择
                # 获取训练数据中的唯一值（简化处理）
                input_data[feature] = st.text_input(
                    f"{feature} (分类)",
                    placeholder=f"输入 {feature} 的值",
                    key=f"inp_{feature}"
                )
            else:
                # 数值特征：使用数字输入
                input_data[feature] = st.number_input(
                    f"{feature}",
                    value=0.0,
                    step=0.01,
                    format="%.4f",
                    key=f"inp_{feature}"
                )

    if len(features) > 20:
        st.caption(f"... 还有 {len(features) - 20} 个特征未显示")

    # 批量预测选项
    if st.button("🔍 执行推理", type="primary", use_container_width=True):
        # 构建输入数据
        valid_data = {}
        for feature in features:
            value = input_data.get(feature)
            if value is not None and value != "":
                valid_data[feature] = value
            else:
                valid_data[feature] = None

        if not valid_data:
            st.warning("请至少输入一个特征值")
            return

        result = execute_inference(valid_data)
        if result:
            display_inference_result(result, {"input_data": valid_data})


def execute_inference(input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """执行推理"""
    try:
        model = st.session_state.inference_model
        preprocessor = st.session_state.inference_preprocessor
        metadata = st.session_state.inference_metadata

        config = metadata.get('config', {})
        task_type = config.get('task_type', 'classification')
        features = config.get('features', [])

        # 构建输入DataFrame
        if isinstance(input_data, dict):
            # 单样本预测
            df = pd.DataFrame([input_data])

            # 确保所有特征都存在
            for feature in features:
                if feature not in df.columns:
                    df[feature] = None

            # 确保数据类型正确（数值列转换为float）
            for col in df.columns:
                if col not in config.get('preprocess_config', {}).get('categorical_features', []):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
        else:
            df = input_data

        # 创建预测器
        predictor = ModelPredictor(model, preprocessor, task_type)

        # 执行预测
        result = predictor.predict_with_confidence(df)

        return result

    except Exception as e:
        st.error(f"推理失败: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language='python')
        return None


def display_inference_result(result: Dict[str, Any], query_info: Dict[str, Any]):
    """显示推理结果"""
    st.markdown("### 📊 推理结果")

    # 显示输入数据
    with st.expander("📥 输入数据", expanded=False):
        input_data = query_info.get("input_data", {})
        # 过滤空值
        filtered_data = {k: v for k, v in input_data.items() if v is not None and v != ""}
        st.json(filtered_data)

    # 显示预测结果
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**预测结果**")

        if "prediction" in result:
            prediction = result["prediction"]
            if isinstance(prediction, (int, float)):
                st.metric("预测值", f"{prediction:.4f}" if isinstance(prediction, float) else str(prediction))
            else:
                st.metric("预测类别", str(prediction))

        if "cluster_id" in result:
            st.metric("所属簇", result["cluster_id"])

        if "forecast" in result:
            forecast = result["forecast"]
            st.metric("预测值", f"{forecast[0]:.4f}" if forecast else "N/A")
            st.caption(f"预测长度: {len(forecast)}")

    with col2:
        st.markdown("**置信度**")

        if "confidence" in result:
            confidence = result["confidence"]
            if isinstance(confidence, list):
                st.metric("平均置信度", f"{np.mean(confidence):.2%}")
            else:
                st.metric("置信度", f"{confidence:.2%}")

        if "probabilities" in result:
            probs = result["probabilities"]
            if probs and len(probs) > 0:
                st.caption("类别概率分布:")
                prob_list = probs[0] if isinstance(probs, list) and len(probs) > 0 else probs
                if isinstance(prob_list, (list, np.ndarray)):
                    for i, prob in enumerate(prob_list[:5]):
                        st.progress(float(prob), text=f"类别 {i}: {float(prob):.2%}")

    # 显示特征重要性（如果有）
    if "feature_importances" in result:
        with st.expander("📊 特征重要性", expanded=False):
            importances = result["feature_importances"]
            top_features = result.get("top_features", [])

            if top_features:
                for feat in top_features[:10]:
                    st.progress(feat["importance"], text=f"{feat['feature']}: {feat['importance']:.4f}")

    # 生成自然语言解读
    st.markdown("### 💬 结果解读")

    interpretation = generate_interpretation(result, query_info)
    st.markdown(interpretation)


def generate_interpretation(result: Dict[str, Any], query_info: Dict[str, Any]) -> str:
    """生成自然语言解读"""
    lines = []

    if "prediction" in result:
        pred = result["prediction"]
        if isinstance(pred, (int, float)):
            lines.append(f"根据模型预测，结果为 **{pred:.4f}**")
        else:
            lines.append(f"根据模型预测，该样本属于类别 **{pred}**")

    if "confidence" in result:
        conf = result["confidence"]
        if isinstance(conf, list):
            conf_mean = np.mean(conf)
            lines.append(f"模型对此预测的置信度为 **{conf_mean:.2%}**")
        else:
            lines.append(f"模型对此预测的置信度为 **{conf:.2%}**")

    if "cluster_id" in result:
        lines.append(f"该样本被分配到簇 **{result['cluster_id']}**")

    if "probabilities" in result:
        probs = result["probabilities"]
        if probs and len(probs) > 0:
            prob_list = probs[0] if isinstance(probs, list) and len(probs) > 0 else probs
            if isinstance(prob_list, (list, np.ndarray)) and len(prob_list) > 0:
                max_idx = np.argmax(prob_list)
                lines.append(f"最可能的类别是类别 {max_idx}，概率为 **{float(prob_list[max_idx]):.2%}**")

    if not lines:
        lines.append("推理完成，请查看上方详细结果")

    return "\n\n".join(lines)


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