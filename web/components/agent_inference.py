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
        return

    st.markdown("**📋 可用模型及示例：**")

    for model in available_models:
        model_name = model.get('user_model_name', model.get('model_key'))
        model_key = model.get('model_key')
        task_type = model.get('task_type', 'unknown')
        target = model.get('target_column', '未知')
        features = model.get('features', [])

        with st.expander(f"📊 {model_name}", expanded=False):
            st.caption(f"类型: {task_type}, 预测目标: {target}")
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
                    else:
                        sample_values[f] = 100

                sample_text = f"用「{model_name}」预测"
                for f, v in sample_values.items():
                    sample_text += f"、{f}{v}"
                sample_text += f"的{target}"

                if st.button(f"🔍 {sample_text}", key=f"example_{model_key}", use_container_width=True):
                    st.session_state.pending_question = sample_text
                    st.rerun()


def parse_and_execute_tool(response: str, session_id: str) -> str:
    """解析并执行工具调用，返回结果或None"""
    # print("=== parse_and_execute_tool 被调用 ===")
    # print(f"响应内容: {response[:500]}")

    # 修改后的正则：允许嵌套一层大括号
    # 匹配 {"tool": "predict", "model_key": "...", "input_values": {...}}
    patterns = [
        r'\{[^{}]*"tool"\s*:\s*"predict"[^{}]*"input_values"\s*:\s*\{[^{}]*\}[^{}]*\}',  # 精确匹配predict格式
        r'\{[^{}]*"tool"\s*:\s*"predict"[^{}]*\}',  # 简单匹配（后备）
        r'```json\s*(\{[^{}]*"tool"\s*:\s*"predict"[^{}]*\})\s*```',
        r'```\s*(\{[^{}]*"tool"\s*:\s*"predict"[^{}]*\})\s*```',
    ]

    for idx, pattern in enumerate(patterns):
        # print(f"尝试模式 {idx}: {pattern[:100]}...")
        match = re.search(pattern, response, re.DOTALL)
        if match:
            # print(f"模式 {idx} 匹配成功")
            json_str = match.group(1) if '```' in pattern and match.group(1) else match.group(0)
            # print(f"提取的JSON: {json_str}")
            try:
                tool_call = json.loads(json_str)
                # print(f"解析成功: {tool_call}")
                if tool_call.get('tool') == 'predict':
                    # print("tool是predict，开始执行")
                    tool = ModelInferenceTool(session_id)
                    result = tool.predict(
                        tool_call.get('model_key', ''),
                        tool_call.get('input_values', {})
                    )
                    # print(f"预测结果: {result}")
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
                        # print(f"返回输出: {output}")
                        return output
                    else:
                        error_msg = f"预测失败: {result.get('error')}"
                        # print(error_msg)
                        return error_msg
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
                continue
        # else:
            # print(f"模式 {idx} 匹配失败")

    # print("所有模式都未匹配，返回None")
    return None