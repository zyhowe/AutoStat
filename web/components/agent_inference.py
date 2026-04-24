# web/components/agent_inference.py

"""Agent推理预测组件 - 自然语言调用模型"""

import streamlit as st
import json
import re
from typing import Dict, Any, List

from web.services.session_service import SessionService
from web.services.model_training_service import list_saved_models, load_model_for_inference, execute_inference


class ModelInferenceTool:
    """模型推理工具 - 供Agent调用"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._models_cache = None

    def get_available_models(self) -> List[Dict]:
        if self._models_cache is None:
            self._models_cache = list_saved_models(self.session_id)
        return self._models_cache

    def get_models_by_type(self, task_type: str) -> List[Dict]:
        """按任务类型获取模型"""
        models = self.get_available_models()
        return [m for m in models if m.get('task_type') == task_type]

    def predict(self, model_key: str, input_values: Dict[str, Any]) -> Dict[str, Any]:
        try:
            model, preprocessor, metadata, metrics = load_model_for_inference(model_key, self.session_id)

            config = metadata
            features = config.get('features', [])
            preprocess_config = config.get('preprocess_config', {})
            categorical_features = preprocess_config.get('categorical_features', [])

            valid_data = {}
            for feature in features:
                if feature in input_values:
                    valid_data[feature] = input_values[feature]
                else:
                    valid_data[feature] = None

            result = execute_inference(model, preprocessor, valid_data, features, categorical_features)

            prediction = result.get("prediction")
            if prediction is None:
                predictions = result.get("predictions")
                if predictions and isinstance(predictions, list) and len(predictions) == 1:
                    prediction = predictions[0]

            confidence = result.get("confidence")
            if confidence is None and result.get("probabilities"):
                probs = result.get("probabilities")
                if probs and isinstance(probs, list) and len(probs) > 0:
                    if isinstance(probs[0], list):
                        confidence = max(probs[0])
                    else:
                        confidence = max(probs)

            return {
                "success": True,
                "model_name": metadata.get('user_model_name', model_key),
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": result.get("probabilities")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


def parse_and_execute_tool(response: str, session_id: str) -> str:
    """解析并执行工具调用，返回结果或None"""
    patterns = [
        r'\{[^{}]*"tool"\s*:\s*"predict"[^{}]*"input_values"\s*:\s*\{[^{}]*\}[^{}]*\}',
        r'\{[^{}]*"tool"\s*:\s*"predict"[^{}]*\}',
        r'```json\s*(\{[^{}]*"tool"\s*:\s*"predict"[^{}]*\})\s*```',
        r'```\s*(\{[^{}]*"tool"\s*:\s*"predict"[^{}]*\})\s*```',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            json_str = match.group(1) if '```' in pattern and match.group(1) else match.group(0)
            try:
                tool_call = json.loads(json_str)
                if tool_call.get('tool') == 'predict':
                    tool = ModelInferenceTool(session_id)
                    result = tool.predict(
                        tool_call.get('model_key', ''),
                        tool_call.get('input_values', {})
                    )
                    if result.get('success'):
                        pred = result.get('prediction')
                        conf = result.get('confidence')
                        output = f"**预测结果**\n\n- 使用模型: {result.get('model_name')}\n"
                        if isinstance(pred, (int, float)):
                            output += f"- 预测值: {pred:.4f}\n"
                        else:
                            output += f"- 预测结果: {pred}\n"
                        if conf:
                            if isinstance(conf, list):
                                output += f"- 置信度: {conf[0]:.2%}\n"
                            else:
                                output += f"- 置信度: {conf:.2%}\n"
                        return output
                    else:
                        return f"预测失败: {result.get('error')}"
            except json.JSONDecodeError:
                continue
    return None


def render_agent_inference():
    """渲染推理预测界面 - 显示模型列表和示例按钮"""
    st.markdown("#### 🔮 推理预测")
    st.caption("点击下方示例，AI会自动调用对应模型进行预测")

    session_id = SessionService.get_current_session()
    if session_id is None:
        st.warning("请先完成数据分析")
        return

    tool = ModelInferenceTool(session_id)
    available_models = tool.get_available_models()

    if not available_models:
        st.info("暂无已训练的模型，请先在「小模型训练」中训练模型")
    else:
        # 按任务类型分组
        classification_models = tool.get_models_by_type("classification")
        regression_models = tool.get_models_by_type("regression")
        time_series_models = tool.get_models_by_type("time_series")
        clustering_models = tool.get_models_by_type("clustering")

        all_models = classification_models + regression_models + time_series_models + clustering_models

        st.markdown("**📋 可用模型及示例：**")

        for model in all_models:
            model_name = model.get('user_model_name', model.get('model_key'))
            model_key = model.get('model_key')
            task_type = model.get('task_type', 'unknown')
            target = model.get('target_column', '未知')
            features = model.get('features', [])

            with st.expander(f"📊 {model_name}", expanded=False):
                task_display = {
                    "classification": "分类",
                    "regression": "回归",
                    "time_series": "时序",
                    "clustering": "聚类"
                }.get(task_type, task_type)
                st.caption(f"类型: {task_display}, 预测目标: {target}")
                st.caption(f"特征: {', '.join(features[:8])}{'...' if len(features) > 8 else ''}")

                if features:
                    sample_values = {}
                    for f in features[:3]:
                        if '年龄' in f or 'age' in f.lower():
                            sample_values[f] = 35
                        elif '收入' in f or 'income' in f.lower():
                            sample_values[f] = 50000
                        elif '销售额' in f or 'sales' in f.lower():
                            sample_values[f] = 8000
                        elif '收缩压' in f or 'systolic' in f.lower():
                            sample_values[f] = 120
                        elif '舒张压' in f or 'diastolic' in f.lower():
                            sample_values[f] = 80
                        elif '心率' in f or 'heart' in f.lower():
                            sample_values[f] = 75
                        elif '血糖' in f:
                            sample_values[f] = 5.2
                        elif '胆固醇' in f:
                            sample_values[f] = 4.8
                        else:
                            sample_values[f] = 100

                    # 根据任务类型生成不同的示例文本
                    if task_type == "time_series":
                        sample_text = f"用「{model_name}」预测未来7天的{target}"
                    elif task_type == "clustering":
                        sample_values_str = "、".join([f"{k}{v}" for k, v in sample_values.items()])
                        sample_text = f"用「{model_name}」对 [{sample_values_str}] 进行聚类"
                    else:
                        sample_text = f"用「{model_name}」预测"
                        for f, v in sample_values.items():
                            sample_text += f"、{f}{v}"
                        sample_text += f"的{target}"

                    if st.button(f"🔍 {sample_text}", key=f"example_{model_key}", use_container_width=True):
                        st.session_state.pending_question = sample_text
                        st.rerun()

    # ========== 内置分析示例（无需训练） ==========
    st.markdown("---")
    st.markdown("**📌 内置分析示例（无需训练）：**")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔗 挖掘分类变量间的关联规则", use_container_width=True):
            st.session_state.pending_question = "请对分类变量进行关联规则挖掘，找出频繁项集和强关联规则，并解释业务含义。"
            st.rerun()

    with col2:
        if st.button("🚨 检测异常值并分析", use_container_width=True):
            st.session_state.pending_question = "请分析数据中的异常值，判断哪些是数据错误，哪些是有意义的异常，并给出处理建议。"
            st.rerun()