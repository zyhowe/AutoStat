"""推理组件 - 在AI解读中集成模型推理"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from autostat.models.predictor import ModelPredictor


def render_inference_interface():
    """渲染推理界面"""
    st.markdown("#### 🔮 推理预测")
    st.caption("使用已训练的模型进行预测")

    # 检查是否有已加载的模型
    if not hasattr(st.session_state, 'inference_model_loaded') or not st.session_state.inference_model_loaded:
        st.info("💡 请在「模型训练」标签页的模型列表中点击 [推理] 按钮加载模型")
        st.caption("加载模型后，此处会显示模型信息和输入表单")
        return

    # 显示当前加载的模型信息
    model_name = st.session_state.get('inference_model_name', '未知模型')
    metadata = st.session_state.inference_metadata

    st.success(f"**当前模型:** {model_name}")

    config = metadata.get('config', {})
    task_type = config.get('task_type', 'classification')
    features = config.get('features', [])

    # 显示评估指标
    metrics = metadata.get('metrics', {})
    if metrics:
        with st.expander("📊 模型评估指标", expanded=False):
            train_metrics = metrics.get('train_score', {})
            if isinstance(train_metrics, dict):
                for metric_name, metric_value in train_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        st.caption(f"- {metric_name}: {metric_value:.4f}")

    st.divider()

    # 获取特征类型信息
    preprocess_config = config.get('preprocess_config', {})
    categorical_features = preprocess_config.get('categorical_features', [])

    # 动态生成输入表单
    st.markdown("**请输入特征值**")

    input_data = {}

    # 使用多列布局
    cols = st.columns(2)
    for i, feature in enumerate(features[:20]):
        with cols[i % 2]:
            if feature in categorical_features:
                input_data[feature] = st.text_input(
                    f"{feature}",
                    placeholder=f"输入 {feature} 的值",
                    key=f"inp_{feature}"
                )
            else:
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
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔍 执行推理", type="primary", use_container_width=True):
            # 构建输入数据
            valid_data = {}
            for feature in features:
                value = input_data.get(feature)
                if value is not None and value != "":
                    valid_data[feature] = value
                else:
                    valid_data[feature] = None

            result = execute_inference(valid_data, task_type)
            if result:
                display_inference_result(result, valid_data)


def execute_inference(input_data: Dict[str, Any], task_type: str) -> Optional[Dict[str, Any]]:
    """执行推理"""
    try:
        model = st.session_state.inference_model
        preprocessor = st.session_state.inference_preprocessor
        metadata = st.session_state.inference_metadata

        config = metadata.get('config', {})
        features = config.get('features', [])

        # 构建输入DataFrame
        df = pd.DataFrame([input_data])

        # 确保所有特征都存在
        for feature in features:
            if feature not in df.columns:
                df[feature] = None

        # 确保数值列转换为float
        for col in df.columns:
            if col not in config.get('preprocess_config', {}).get('categorical_features', []):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass

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


def display_inference_result(result: Dict[str, Any], input_data: Dict[str, Any]):
    """显示推理结果"""
    st.markdown("### 📊 推理结果")

    # 显示输入数据
    with st.expander("📥 输入数据", expanded=False):
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

    # 生成自然语言解读
    st.markdown("### 💬 结果解读")

    interpretation = generate_interpretation(result)
    st.markdown(interpretation)


def generate_interpretation(result: Dict[str, Any]) -> str:
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