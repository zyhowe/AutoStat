# web/services/agent_service.py

"""Agent服务 - 大模型智能解读的Agent模式"""

import streamlit as st
import json
import pandas as pd
from typing import Dict, Any, List, Optional

from autostat.models.predictor import ModelPredictor
from web.services.model_training_service import load_model_for_inference, list_saved_models


class AgentService:
    """Agent服务 - 管理工具调用和模型预测"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.saved_models = None

    def refresh_models(self):
        """刷新已保存的模型列表"""
        self.saved_models = list_saved_models(self.session_id)

    def get_available_models(self) -> List[Dict]:
        """获取可用的模型列表"""
        if self.saved_models is None:
            self.refresh_models()
        return self.saved_models or []

    def get_model_info(self, model_key: str) -> Optional[Dict]:
        """获取模型信息"""
        models = self.get_available_models()
        for model in models:
            if model.get('model_key') == model_key:
                return model
        return None

    def get_model_by_name(self, model_name: str) -> Optional[Dict]:
        """根据名称模糊匹配模型"""
        models = self.get_available_models()
        model_name_lower = model_name.lower()
        for model in models:
            display_name = model.get('user_model_name', '').lower()
            model_key = model.get('model_key', '').lower()
            if model_name_lower in display_name or model_name_lower in model_key:
                return model
        return None

    def predict_with_model(self, model_key: str, input_values: Dict) -> Dict[str, Any]:
        """使用指定模型进行预测"""
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

            df = pd.DataFrame([valid_data])

            for col in df.columns:
                if col not in categorical_features:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception:
                        pass

            df = df[features]

            predictor = ModelPredictor(model, preprocessor)
            result = predictor.predict_with_confidence(df)

            prediction = result.get("prediction")
            if prediction is None:
                predictions = result.get("predictions")
                if predictions is not None and isinstance(predictions, list) and len(predictions) == 1:
                    prediction = predictions[0]

            return {
                "success": True,
                "model_name": metadata.get('user_model_name', model_key),
                "model_key": model_key,
                "prediction": prediction,
                "confidence": result.get("confidence"),
                "probabilities": result.get("probabilities"),
                "raw_result": result
            }

        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }


def get_agent_tools_description(available_models: List[Dict]) -> str:
    """获取Agent工具描述，用于大模型提示词"""
    if not available_models:
        return "当前没有可用的预测模型。"

    models_desc = []
    for model in available_models:
        config = model.get('config', {})
        task_type = config.get('task_type', 'unknown')
        features = config.get('features', [])
        user_model_name = model.get('user_model_name', model.get('model_key', 'unknown'))

        task_display = {
            "classification": "分类",
            "regression": "回归",
            "clustering": "聚类",
            "time_series": "时序"
        }.get(task_type, task_type)

        feature_str = ', '.join(features[:5])
        if len(features) > 5:
            feature_str += '...'

        models_desc.append(
            f"  - **{user_model_name}** (key: {model.get('model_key')})\n"
            f"    类型: {task_display}, 特征: {feature_str}"
        )

    return f"""
## 可用工具

### 1. 模型预测工具 predict_with_model
使用已训练的模型进行预测。

参数：
- model_key: 模型标识
- input_values: 字典，键为特征名，值为特征值

### 2. 根据名称查找模型 find_model
根据名称模糊匹配查找模型。

参数：
- model_name: 模型名称关键词

### 3. 列出所有模型 list_models
获取所有可用模型列表。

## 可用模型列表

{chr(10).join(models_desc) if models_desc else "暂无可用模型"}

## 使用说明
- 当用户需要进行预测时，请使用 predict_with_model 工具
- 使用工具前，先列出可用的模型让用户选择
- 预测结果需要清晰展示，包括预测值和置信度
- 如果用户没有指定模型，选择第一个可用的模型
- 用中文回答，结构清晰，友好专业
"""


def build_data_context(json_data: Dict) -> str:
    """构建数据上下文"""
    data_shape = json_data.get('data_shape', {})
    variable_types = json_data.get('variable_types', {})
    quality_report = json_data.get('quality_report', {})
    cleaning_suggestions = json_data.get('cleaning_suggestions', [])
    correlations = json_data.get('correlations', {})

    type_counts = {}
    type_display = {
        'continuous': '连续变量',
        'categorical': '分类变量',
        'categorical_numeric': '数值型分类',
        'ordinal': '有序分类',
        'datetime': '日期时间',
        'identifier': '标识符',
        'text': '文本'
    }
    for col, info in variable_types.items():
        typ = info.get('type', 'unknown')
        type_counts[typ] = type_counts.get(typ, 0) + 1

    type_summary = ", ".join([f"{type_display.get(t, t)}: {c}" for t, c in type_counts.items()])

    missing_list = quality_report.get('missing', [])
    missing_summary = "\n".join([f"  - {m['column']}: {m['percent']:.1f}%" for m in missing_list[:5]])

    high_corrs = correlations.get('high_correlations', [])
    corr_summary = "\n".join([f"  - {c['var1']} ↔ {c['var2']}: r={c['value']}" for c in high_corrs[:3]])

    return f"""
## 数据上下文

### 数据概览
- 总行数: {data_shape.get('rows', 0):,}
- 总列数: {data_shape.get('columns', 0)}
- 变量类型分布: {type_summary}

### 数据质量
缺失值:
{missing_summary if missing_summary else '  无缺失值'}

重复记录: {quality_report.get('duplicates', {}).get('count', 0)}条

### 清洗建议
{chr(10).join([f"  - {s}" for s in cleaning_suggestions[:3]]) if cleaning_suggestions else '  无清洗建议'}

### 相关性分析
{corr_summary if corr_summary else '  无强相关对'}
"""


def build_agent_system_prompt(
    analysis_type: str,
    source_name: str,
    json_data: Optional[Dict],
    html_content: Optional[str],
    raw_data_preview: Optional[Dict],
    available_models: List[Dict]
) -> str:
    """构建Agent系统提示词"""
    data_context = ""
    if json_data:
        data_context = build_data_context(json_data)

    models_context = get_agent_tools_description(available_models)

    return f"""你是专业的数据分析师，正在回答用户关于数据的问题。

## 分析信息
- 分析类型: {analysis_type}
- 数据源: {source_name}

{data_context}

{models_context}

## 重要说明
1. 当用户需要进行预测时，请使用 predict_with_model 工具
2. 使用工具前，先列出可用的模型让用户选择
3. 预测结果需要清晰展示，包括预测值和置信度
4. 如果用户没有指定模型，选择第一个可用的模型
5. 用中文回答，结构清晰，友好专业
"""


def process_agent_response(response: str, agent_service: AgentService) -> str:
    """处理Agent响应，解析工具调用"""
    if "[[TOOL_CALL]]" in response:
        import re
        tool_pattern = r'\[\[TOOL_CALL\]\](.*?)\[\[/TOOL_CALL\]\]'
        match = re.search(tool_pattern, response, re.DOTALL)

        if match:
            try:
                tool_call = json.loads(match.group(1))
                tool_name = tool_call.get('tool', '')
                params = tool_call.get('params', {})

                if tool_name == 'predict_with_model':
                    result = agent_service.predict_with_model(
                        params.get('model_key', ''),
                        params.get('input_values', {})
                    )
                    if result.get('success'):
                        confidence_str = ""
                        if result.get('confidence') is not None:
                            conf = result['confidence']
                            if isinstance(conf, list) and len(conf) > 0:
                                confidence_str = f"\n- 置信度: {conf[0]:.2%}"
                            elif isinstance(conf, (int, float)):
                                confidence_str = f"\n- 置信度: {conf:.2%}"

                        return f"""
**预测结果**

- 使用模型: {result.get('model_name')}
- 预测值: {result.get('prediction')}{confidence_str}
"""
                    else:
                        return f"预测失败: {result.get('error')}"

                elif tool_name == 'list_models':
                    models = agent_service.get_available_models()
                    if models:
                        model_list = "\n".join([
                            f"  - {m.get('user_model_name', m.get('model_key'))} (key: {m.get('model_key')})"
                            for m in models
                        ])
                        return f"**可用模型列表**\n\n{model_list}"
                    else:
                        return "当前没有可用的预测模型。"

                elif tool_name == 'find_model':
                    model = agent_service.get_model_by_name(params.get('model_name', ''))
                    if model:
                        return f"**找到模型**\n\n- 名称: {model.get('user_model_name')}\n- Key: {model.get('model_key')}"
                    else:
                        return f"未找到名称包含 '{params.get('model_name')}' 的模型。"

            except Exception as e:
                return f"工具调用失败: {str(e)}"

    return response