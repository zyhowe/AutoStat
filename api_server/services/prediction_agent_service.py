"""智能预测Agent服务 - 自然语言→模型匹配→预测→解释"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from autostat.models.storage import ModelStorage
from autostat.models.predictor import ModelPredictor


class PredictionAgentService:
    """智能预测Agent"""

    def __init__(self):
        self.model_storage = ModelStorage()

    def process(self, session_id: str, question: str) -> Dict[str, Any]:
        """
        处理用户的预测类问题

        Args:
            session_id: 会话ID
            question: 用户自然语言问题

        Returns:
            {
                "success": bool,
                "result": str,          # 自然语言结果描述
                "data": {...},          # 预测数据
                "model_used": str,      # 使用的模型名称
                "confidence": float,    # 置信度
                "error": str            # 错误信息
            }
        """
        # 1. 列出所有已训练模型
        models = self.model_storage.list_models(session_id)

        if not models:
            return {
                "success": False,
                "error": "暂无已训练模型，请先在智能预测页面训练模型",
                "result": "暂无可用的预测模型"
            }

        # 2. 解析用户意图，匹配模型
        intent = self._parse_intent(question, models)

        if not intent.get('matched'):
            return {
                "success": False,
                "error": "无法理解预测需求，请明确目标字段或选择已有模型",
                "result": "请描述您想预测什么，例如：'预测下个月的销售额'"
            }

        # 3. 执行预测
        try:
            model_key = intent.get('model_key')
            target = intent.get('target')
            features = intent.get('features', [])

            if not model_key:
                # 如果没有匹配到具体模型，用第一个
                model_key = models[0].get('model_key')

            # 加载模型
            model, preprocessor, metadata = self.model_storage.load_model(session_id, model_key)

            if model is None:
                return {
                    "success": False,
                    "error": f"模型加载失败: {model_key}",
                    "result": "模型加载失败，请重新训练"
                }

            # 获取特征和输入值（从问题中提取或使用默认值）
            input_values = self._extract_input_values(question, metadata)

            # 执行预测
            predictor = ModelPredictor(model, preprocessor,
                                      metadata.get('task_type'),
                                      model_key)

            result = predictor.predict_with_confidence(input_values)

            # 4. 生成自然语言解释
            explanation = self._generate_explanation(
                result,
                metadata,
                intent,
                question
            )

            return {
                "success": True,
                "result": explanation,
                "data": result,
                "model_used": metadata.get('user_model_name', model_key),
                "confidence": result.get('confidence_mean', result.get('confidence', 0.5)),
                "prediction": result.get('prediction'),
                "probabilities": result.get('probabilities')
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": f"预测执行失败: {str(e)}"
            }

    def _parse_intent(self, question: str, models: List[Dict]) -> Dict[str, Any]:
        """
        解析用户意图，匹配模型

        Args:
            question: 用户问题
            models: 已训练模型列表

        Returns:
            {
                "matched": bool,
                "model_key": str,
                "target": str,
                "features": list,
                "task_type": str
            }
        """
        question_lower = question.lower()

        # 检查是否是预测类问题
        predict_keywords = ['预测', '预估', '预报', '推演', 'forecast', 'predict']
        is_prediction = any(kw in question_lower for kw in predict_keywords)

        if not is_prediction:
            return {"matched": False}

        # 提取目标字段
        target = None
        for model in models:
            config = model.get('config', {})
            model_target = config.get('target_column', '')
            if model_target and model_target in question:
                target = model_target
                break

        # 如果没有从问题中提取到目标，从第一个模型获取
        if not target and models:
            target = models[0].get('config', {}).get('target_column', '')

        # 匹配模型
        matched_model = None
        for model in models:
            config = model.get('config', {})
            model_target = config.get('target_column', '')
            if model_target == target or (target and target in model_target):
                matched_model = model
                break

        # 如果没有精确匹配，使用第一个
        if not matched_model and models:
            matched_model = models[0]

        if matched_model:
            config = matched_model.get('config', {})
            return {
                "matched": True,
                "model_key": matched_model.get('model_key'),
                "target": config.get('target_column', ''),
                "features": config.get('features', []),
                "task_type": config.get('task_type', 'regression')
            }

        return {"matched": False}

    def _extract_input_values(self, question: str, metadata: Dict) -> Dict[str, Any]:
        """
        从问题中提取输入值

        Args:
            question: 用户问题
            metadata: 模型元数据

        Returns:
            输入值字典
        """
        input_values = {}
        features = metadata.get('features', [])

        # 尝试从问题中提取数值
        for feature in features:
            # 匹配模式: "feature是123" 或 "feature为123" 或 "feature=123"
            patterns = [
                rf'{feature}\s*[是为=:]\s*([\d.]+)',
                rf'{feature}\s*等于\s*([\d.]+)',
                rf'{feature}\s*(\d+\.?\d*)'
            ]
            for pattern in patterns:
                match = re.search(pattern, question)
                if match:
                    try:
                        input_values[feature] = float(match.group(1))
                    except ValueError:
                        input_values[feature] = match.group(1)
                    break

            # 如果没有提取到，尝试识别类别值
            if feature not in input_values:
                # 简单的关键词匹配
                words = question.split()
                for word in words:
                    if word in ['男', '女', '是', '否', '高', '中', '低']:
                        input_values[feature] = word
                        break

        # 如果还是空的，用默认值
        if not input_values and features:
            # 使用特征的中位数或均值（从metadata获取）
            for feature in features:
                input_values[feature] = 0  # 默认值

        return input_values

    def _generate_explanation(self, result: Dict, metadata: Dict,
                             intent: Dict, question: str) -> str:
        """
        生成自然语言解释

        Args:
            result: 预测结果
            metadata: 模型元数据
            intent: 意图解析结果
            question: 原始问题

        Returns:
            自然语言解释
        """
        prediction = result.get('prediction')
        confidence = result.get('confidence_mean', result.get('confidence', 0.5))
        model_name = metadata.get('user_model_name', '未知模型')
        target = intent.get('target', '目标变量')

        if prediction is None:
            return "预测失败，请检查输入值是否正确"

        # 格式化预测值
        if isinstance(prediction, float):
            pred_str = f"{prediction:.2f}"
        else:
            pred_str = str(prediction)

        # 构建解释
        conf_percent = confidence * 100 if confidence else 50

        explanation = f"根据「{model_name}」模型的预测，{target} 的预测值为 **{pred_str}**"

        if confidence:
            explanation += f"，置信度 {conf_percent:.1f}%"

        # 如果有概率分布（分类任务）
        probabilities = result.get('probabilities')
        if probabilities and len(probabilities) > 0:
            if isinstance(probabilities[0], list):
                probs = probabilities[0]
            else:
                probs = probabilities
            if len(probs) > 1:
                explanation += f"\n\n概率分布："
                for i, p in enumerate(probs[:5]):
                    explanation += f"\n- 类别 {i}: {(p * 100):.1f}%"

        # 如果有特征重要性
        feature_importances = result.get('feature_importances')
        if feature_importances:
            sorted_fi = sorted(feature_importances.items(),
                              key=lambda x: x[1], reverse=True)[:3]
            if sorted_fi:
                explanation += "\n\n🔑 **主要影响因素：**\n"
                for f, imp in sorted_fi:
                    explanation += f"- {f} (重要性: {imp:.2f})\n"

        return explanation