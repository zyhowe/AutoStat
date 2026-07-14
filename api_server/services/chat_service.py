"""AI对话服务 - 支持 query_data 和 predict 工具"""
import json
import pandas as pd
from typing import Dict, Any, List, Generator, Optional
from collections import defaultdict

from autostat.llm_client import LLMClient
from autostat.core.insight import InsightService
from autostat.models.storage import ModelStorage
from autostat.models.predictor import ModelPredictor
from api_server.services.config_service import ConfigService
from api_server.services.session_service import SessionService
from api_server.services.query_service import QueryService


class ChatService:
    """AI对话服务"""

    # dataKey → analysis_result 真实数据路径映射（用于非查询类问题）
    DATA_KEY_MAP = {
        'data_overview.distribution': 'variable_summaries',
        'data_overview.categorical': 'variable_summaries',
        'data_overview.continuous': 'variable_summaries',
        'data_overview.datetime': 'variable_summaries',
        'data_overview.missing': 'quality_report.missing',
        'data_overview.identifier': 'variable_summaries',
        'data_overview.text': 'variable_summaries',
        'data_overview.natural_query': 'data_shape',
        'data_overview.generate_sql': 'data_shape',
        'quality.overall': 'quality_report',
        'quality.completeness': 'quality_report.missing',
        'quality.accuracy': 'quality_report.outliers',
        'quality.consistency': 'quality_report.audit_rules',
        'quality.uniqueness': 'quality_report.duplicates',
        'data_validation.audit_rules': 'quality_report.audit_rules',
        'data_validation.outliers': 'quality_report.outliers',
        'data_validation.missing': 'quality_report.missing',
        'data_validation.duplicates': 'quality_report.duplicates',
        'data_validation.cleaning': 'cleaning_suggestions',
        'pattern_discovery.correlation': 'correlations.high_correlations',
        'pattern_discovery.timeseries': 'time_series_diagnostics',
        'pattern_discovery.trend': 'time_series_diagnostics',
        'pattern_discovery.categorical_pattern': 'variable_types',
        'pattern_discovery.distribution_insight': 'variable_summaries',
        'smart_prediction.model_recommend': 'model_recommendations',
        'smart_prediction.target_select': 'model_recommendations',
        'smart_prediction.feature_select': 'model_recommendations',
        'smart_prediction.forecast': 'time_series_diagnostics',
        'report_summary.overview': 'data_shape',
        'report_summary.conclusions': 'summary',
        'report_summary.insights': 'summary',
    }

    # query_data 工具定义
    QUERY_DATA_TOOL = {
        "type": "function",
        "function": {
            "name": "query_data",
            "description": """查询数据表中的原始数据，支持按条件筛选。
当用户需要查看具体数据行时使用此工具。
条件操作符：eq(等于), gt(大于), lt(小于), gte(大于等于), lte(小于等于), contains(包含)""",
            "parameters": {
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "array",
                        "description": "筛选条件列表，多个条件为 AND 关系",
                        "items": {
                            "type": "object",
                            "properties": {
                                "field": {"type": "string", "description": "字段名"},
                                "operator": {
                                    "type": "string",
                                    "enum": ["eq", "gt", "lt", "gte", "lte", "contains"],
                                    "description": "操作符"
                                },
                                "value": {"type": "string", "description": "条件值"}
                            },
                            "required": ["field", "operator", "value"]
                        }
                    },
                    "limit": {
                        "type": "integer",
                        "default": 100,
                        "description": "返回的最大行数"
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要返回的字段列表，不指定则返回所有"
                    },
                    "order_by": {
                        "type": "string",
                        "description": "排序字段，如 'age DESC'"
                    }
                },
                "required": ["filters"]
            }
        }
    }

    # predict 工具定义
    PREDICT_TOOL = {
        "type": "function",
        "function": {
            "name": "predict",
            "description": """使用已训练的模型进行预测。
当用户明确要求预测某个目标变量时使用此工具。
支持的模型列表会在对话上下文中提供。""",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_key": {
                        "type": "string",
                        "description": "要使用的模型标识符，必须从上下文中提供的模型列表中选择"
                    },
                    "input_values": {
                        "type": "object",
                        "description": "预测输入值，键为特征名，值为特征值"
                    }
                },
                "required": ["model_key", "input_values"]
            }
        }
    }

    def __init__(self):
        self.insight_service = InsightService()
        self._llm_client = None
        self._config_service = ConfigService()
        self._session_service = SessionService()
        self._query_service = QueryService()
        self._model_storage = ModelStorage()
        self._tools = [self.QUERY_DATA_TOOL, self.PREDICT_TOOL]

    def _get_llm_client(self):
        """获取大模型客户端 - 从配置读取"""
        if self._llm_client is None:
            configs = self._config_service.get_llm_configs()
            if configs:
                config = configs[0]
                print(f"使用大模型配置: {config.get('name')}")
                self._llm_client = LLMClient({
                    "api_base": config.get("api_base", ""),
                    "api_key": config.get("api_key", ""),
                    "model": config.get("model", ""),
                    "timeout": config.get("timeout", 60)
                })
            else:
                self._llm_client = LLMClient({
                    "api_base": "",
                    "api_key": "",
                    "model": "",
                    "timeout": 60
                })
        return self._llm_client

    def _get_models_for_session(self, session_id: str) -> List[Dict]:
        """获取当前会话的所有已训练模型，去重取最新"""
        models = self._model_storage.list_models(session_id)
        if not models:
            return []

        # 按目标变量分组，每组只保留最新的一个
        grouped = defaultdict(list)
        for m in models:
            config = m.get('config', {})
            target = config.get('target_column', 'unknown')
            grouped[target].append(m)

        latest_models = []
        for target, model_list in grouped.items():
            # 按创建时间排序，取最新的
            sorted_models = sorted(
                model_list,
                key=lambda x: x.get('created_at', ''),
                reverse=True
            )
            latest_models.append(sorted_models[0])

        return latest_models

    def _query_data_tool(self, arguments: Dict[str, Any], session_id: str) -> str:
        """执行 query_data 工具"""
        filters = arguments.get('filters', [])
        limit = arguments.get('limit', 100)
        fields = arguments.get('fields')
        order_by = arguments.get('order_by')

        success, message, df = self._query_service.query_data(
            session_id, filters, limit, fields, order_by
        )

        if not success:
            return f"查询失败：{message}"

        if len(df) == 0:
            return "查询结果为空，没有找到满足条件的记录。"

        # 转换为 Markdown 表格
        max_rows = min(200, len(df))
        display_df = df.head(max_rows)

        if len(display_df.columns) > 20:
            display_df = display_df.iloc[:, :20]

        headers = '| ' + ' | '.join(display_df.columns) + ' |'
        separator = '| ' + ' | '.join(['---'] * len(display_df.columns)) + ' |'

        rows = []
        for _, row in display_df.iterrows():
            row_strs = []
            for col in display_df.columns:
                val = row[col]
                if pd.isna(val):
                    row_strs.append('(空)')
                elif isinstance(val, (int, float)):
                    if isinstance(val, float):
                        row_strs.append(f'{val:.2f}')
                    else:
                        row_strs.append(str(val))
                else:
                    val_str = str(val)
                    if len(val_str) > 50:
                        val_str = val_str[:47] + '...'
                    row_strs.append(val_str)
            rows.append('| ' + ' | '.join(row_strs) + ' |')

        table = '\n'.join([headers, separator] + rows)
        info = f"📊 共 {len(df)} 行（显示前 {min(max_rows, len(df))} 行）\n\n"
        return f"以下是查询结果表格，请直接展示这个表格，不要添加任何额外的概况、摘要或统计说明：\n\n{info + table}"

    def _predict_tool(self, arguments: Dict[str, Any], session_id: str) -> str:
        """执行 predict 工具"""
        print(f"[PREDICT] 收到预测请求: {arguments}")

        model_key = arguments.get('model_key')
        input_values = arguments.get('input_values', {})

        if not model_key:
            error_msg = "错误：未指定模型"
            print(f"[PREDICT] {error_msg}")
            return error_msg

        if not input_values:
            error_msg = "错误：未提供输入值"
            print(f"[PREDICT] {error_msg}")
            return error_msg

        print(f"[PREDICT] 模型: {model_key}, 输入: {input_values}")

        try:
            # 加载模型
            model, preprocessor, metadata = self._model_storage.load_model(session_id, model_key)

            if model is None:
                error_msg = f"错误：模型 '{model_key}' 不存在或加载失败"
                print(f"[PREDICT] {error_msg}")
                return error_msg

            # 获取特征列表
            config = metadata.get('config', {})
            features = config.get('features', [])
            print(f"[PREDICT] 模型特征列表: {features}")

            if not features:
                error_msg = "错误：模型没有特征信息"
                print(f"[PREDICT] {error_msg}")
                return error_msg

            # 检查输入值是否完整
            missing_features = [f for f in features if f not in input_values]
            if missing_features:
                error_msg = f"错误：缺少以下特征值: {', '.join(missing_features)}"
                print(f"[PREDICT] {error_msg}")
                return error_msg

            # 构造 DataFrame
            input_data = {f: input_values.get(f, 0) for f in features}
            df = pd.DataFrame([input_data])
            df = df[features]
            print(f"[PREDICT] 构造的 DataFrame:\n{df}")

            # 执行预测
            predictor = ModelPredictor(
                model,
                preprocessor,
                metadata.get('task_type'),
                model_key
            )
            result = predictor.predict_with_confidence(df)
            print(f"[PREDICT] 预测结果: {result}")

            # 🔧 修复：获取预测值（支持 predictions 列表和单值 prediction）
            predictions = result.get('predictions')
            if predictions is not None and len(predictions) > 0:
                prediction = predictions[0]
            else:
                prediction = result.get('prediction')
                if prediction is None:
                    error_msg = "预测失败：未返回预测值"
                    print(f"[PREDICT] {error_msg}")
                    return error_msg

            confidence = result.get('confidence_mean', result.get('confidence'))

            # 格式化预测值
            pred_str = f"{prediction:.2f}" if isinstance(prediction, float) else str(prediction)

            # 概率分布（如果有）
            probs = result.get('probabilities')
            prob_str = ""
            if probs and len(probs) > 0:
                if isinstance(probs[0], list):
                    probs = probs[0]
                prob_str = "\n\n概率分布："
                for i, p in enumerate(probs[:5]):
                    prob_str += f"\n- 类别 {i}: {(p * 100):.1f}%"

            confidence_str = f"，置信度 {(confidence * 100):.1f}%" if confidence else ""

            model_name = metadata.get('user_model_name', model_key)

            success_msg = f"✅ 预测完成！\n\n📊 预测值: **{pred_str}**{confidence_str}{prob_str}\n\n🤖 模型: {model_name}"
            print(f"[PREDICT] 返回成功消息: {success_msg}")
            return success_msg

        except KeyError as e:
            error_msg = f"错误：特征匹配失败 - {str(e)}。请检查特征名称是否与模型训练时一致。"
            print(f"[PREDICT] {error_msg}")
            return error_msg
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"[PREDICT] 预测失败详情:\n{error_detail}")
            return f"预测失败：{str(e)}"

    def _get_schema_for_prompt(self, session_id: str) -> str:
        """获取表结构信息，用于 Prompt"""
        schema_info = self._query_service.get_schema_info(session_id)

        table_name = schema_info.get('table_name', '未知表')
        columns = schema_info.get('columns', [])
        total_rows = schema_info.get('total_rows', 0)
        sample_data = schema_info.get('sample_data', [])

        # 构建列信息
        col_info = []
        for col in columns[:30]:
            col_info.append(f"  - {col['name']}: {col['type']}")
        if len(columns) > 30:
            col_info.append(f"  ... 还有 {len(columns) - 30} 列")

        columns_str = '\n'.join(col_info) if col_info else "  暂无列信息"

        # 构建样例数据
        sample_str = ""
        if sample_data:
            sample_str = "\n## 样例数据（前5行）\n"
            sample_df = pd.DataFrame(sample_data)
            if len(sample_df.columns) > 10:
                sample_df = sample_df.iloc[:, :10]
            sample_str += sample_df.to_markdown(index=False)

        return f"""
## 表结构信息
- 表名: {table_name}
- 总行数: {total_rows:,}
- 列信息:
{columns_str}
{sample_str}
"""

    def _get_models_for_prompt(self, session_id: str) -> str:
        """获取已训练模型列表，用于 Prompt"""
        models = self._get_models_for_session(session_id)

        if not models:
            return "暂无已训练的模型"

        lines = ["## 已训练的预测模型"]
        for m in models:
            config = m.get('config', {})
            target = config.get('target_column', '未知')
            features = config.get('features', [])
            model_key = m.get('model_key', '')
            user_name = m.get('user_model_name', model_key)
            task_type = config.get('task_type', '未知')

            lines.append(f"\n- 模型名称: {user_name}")
            lines.append(f"  标识符: {model_key}")
            lines.append(f"  任务类型: {task_type}")
            lines.append(f"  目标变量: {target}")
            lines.append(f"  特征: {', '.join(features)}")

        return '\n'.join(lines)

    def _get_data_by_key(self, analysis_result: Dict[str, Any], data_key: str) -> Optional[Any]:
        """根据 dataKey 从 analysis_result 中提取真实数据"""
        if not data_key:
            return None

        real_path = self.DATA_KEY_MAP.get(data_key)
        if not real_path:
            return None

        keys = real_path.split('.')
        data = analysis_result
        for key in keys:
            if isinstance(data, dict) and key in data:
                data = data[key]
            else:
                return None
        return data

    def _build_system_prompt(self, session_id: str, analysis_result: Dict[str, Any],
                            context_data: Dict[str, Any] = None) -> str:
        """构建系统提示词"""
        data_shape = analysis_result.get("data_shape", {})
        variable_types = analysis_result.get("variable_types", {})
        quality = analysis_result.get("quality_report", {})
        correlations = analysis_result.get("correlations", {})
        ts_diag = analysis_result.get("time_series_diagnostics", {})
        summaries = analysis_result.get("variable_summaries", {})

        source_table = analysis_result.get("source_table", "未知表名")
        if context_data and context_data.get('source_table'):
            source_table = context_data.get('source_table')

        rows = data_shape.get('rows', 0)
        cols = data_shape.get('columns', 0)

        type_counts = {}
        type_display = {
            "continuous": "连续变量",
            "categorical": "分类变量",
            "categorical_numeric": "数值型分类",
            "ordinal": "有序分类",
            "datetime": "日期时间",
            "identifier": "标识符",
            "text": "文本"
        }
        for info in variable_types.values():
            typ = info if isinstance(info, str) else info.get("type", "unknown")
            type_counts[typ] = type_counts.get(typ, 0) + 1
        type_summary = "、".join(
            [f"{type_display.get(t, t)} {c}个" for t, c in type_counts.items() if t in type_display]
        )

        field_details = []
        for field_name, info in list(summaries.items())[:20]:
            detail_parts = []
            if info.get('type_desc'):
                detail_parts.append(f"类型: {info.get('type_desc')}")
            if info.get('mean') is not None:
                detail_parts.append(f"均值: {info.get('mean'):.2f}")
            if info.get('median') is not None:
                detail_parts.append(f"中位数: {info.get('median'):.2f}")
            if info.get('min') is not None and info.get('max') is not None:
                detail_parts.append(f"范围: {info.get('min')}~{info.get('max')}")
            if info.get('n_unique'):
                detail_parts.append(f"唯一值: {info.get('n_unique')}个")
            if info.get('min_date'):
                detail_parts.append(f"日期范围: {info.get('min_date')}~{info.get('max_date')}")
            if info.get('missing_pct') is not None:
                detail_parts.append(f"缺失率: {info.get('missing_pct'):.1f}%")
            if detail_parts:
                field_details.append(f"  - {field_name}: {', '.join(detail_parts)}")

        field_details_str = "\n".join(field_details) if field_details else "  暂无详细字段信息"

        # 处理 dataKey 提取
        data_key = context_data.get('dataKey') if context_data else None
        extracted_data_str = ""
        if data_key:
            extracted_data = self._get_data_by_key(analysis_result, data_key)
            if extracted_data is not None:
                if isinstance(extracted_data, list):
                    items = []
                    for item in extracted_data[:10]:
                        if isinstance(item, dict):
                            text = (item.get('text') or item.get('name') or
                                   item.get('description') or item.get('var1') or str(item))
                            if text:
                                if 'var1' in item and 'var2' in item and 'value' in item:
                                    text = f"{item.get('var1')} ↔ {item.get('var2')} (r={item.get('value', 0):.3f})"
                                items.append(f"  - {text}")
                        else:
                            items.append(f"  - {item}")
                    if items:
                        extracted_data_str = "\n".join(items)
                        real_path = self.DATA_KEY_MAP.get(data_key, data_key)
                        extracted_data_str = f"\n## 该问题对应的真实数据（来源: {data_key} → {real_path}）\n{extracted_data_str}\n"
                elif isinstance(extracted_data, dict):
                    items = []
                    for k, v in list(extracted_data.items())[:10]:
                        if isinstance(v, dict):
                            summary = []
                            if v.get('mean') is not None:
                                summary.append(f"均值: {v.get('mean'):.2f}")
                            if v.get('median') is not None:
                                summary.append(f"中位数: {v.get('median'):.2f}")
                            if v.get('n_samples'):
                                summary.append(f"样本量: {v.get('n_samples')}")
                            if summary:
                                items.append(f"  - {k}: {', '.join(summary)}")
                            else:
                                items.append(f"  - {k}: {str(v)[:50]}")
                        else:
                            items.append(f"  - {k}: {v}")
                    if items:
                        extracted_data_str = "\n".join(items)
                        real_path = self.DATA_KEY_MAP.get(data_key, data_key)
                        extracted_data_str = f"\n## 该问题对应的真实数据（来源: {data_key} → {real_path}）\n{extracted_data_str}\n"

        high_corrs = correlations.get("high_correlations", [])
        corr_summary = ""
        if high_corrs:
            top = high_corrs[0]
            corr_summary = f"{top.get('var1', '')} ↔ {top.get('var2', '')} (r={top.get('value', 0):.3f})"

        has_auto = any(v.get('has_autocorrelation') for v in ts_diag.values())
        ts_summary = "有" if has_auto else "无"

        dup_count = quality.get("duplicates", {}).get("count", 0)
        try:
            dup_count = int(dup_count) if dup_count else 0
        except (ValueError, TypeError):
            dup_count = 0

        missing_count = len(quality.get('missing', []))
        outlier_count = len(quality.get('outliers', {}))
        quality_score = quality.get('overall_score')

        # 获取表结构信息
        schema_info = self._get_schema_for_prompt(session_id)

        # 获取已训练模型列表
        models_info = self._get_models_for_prompt(session_id)

        prompt = f"""你是专业的数据分析师，正在回答用户关于数据的问题。

## 数据概况
- 表名: {source_table}
- 总行数: {rows:,}
- 总列数: {cols}
- 变量类型: {type_summary}

## 表结构
{field_details_str}

{extracted_data_str}

## 数据质量
- 缺失字段: {missing_count}个
- 异常字段: {outlier_count}个
- 重复记录: {dup_count}条
- 综合质量评分: {quality_score if quality_score is not None else '未评分'}

## 关键发现
- 强相关: {corr_summary if corr_summary else '无'}
- 时间序列: {ts_summary}

{schema_info}

{models_info}

## 回答原则（严格遵守）
1. 用户要什么就给什么，不要自作主张改变输出格式
2. 用户明确要求"表格"、"明细"、"所有数据"、"具体数据"等时，必须且只能输出表格，不得输出任何概况、摘要、统计、解释性文字
3. 如果用户没有明确要求表格但意图是"查看"数据（查询、列出、显示等），直接以表格形式呈现，任何概况、统计、解释性文字放在表格之后
4. 判断用户意图时，"查看数据"优先于"了解概况"——宁可多给数据，不要少给
5. 用中文回答，结构清晰，友好专业
6. 当用户需要预测时，使用 predict 工具，从上述模型列表中选择合适的模型
7. 调用 predict 工具时，model_key 必须使用上表中的"标识符"字段

## 工具使用说明
- query_data: 用于查询/筛选数据
- predict: 用于使用已训练模型进行预测
"""

        return prompt

    def chat(
        self,
        session_id: str,
        question: str,
        analysis_result: Dict[str, Any],
        context: List[str],
        context_data: Dict[str, Any] = None
    ) -> str:
        """非流式对话"""
        llm = self._get_llm_client()

        if not llm.api_key or not llm.api_base:
            return self._fallback_answer(question, analysis_result)

        # 构建消息
        system_prompt = self._build_system_prompt(session_id, analysis_result, context_data)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        # 调用 LLM（带工具）
        response = llm.chat(
            messages=messages,
            temperature=0.7,
            tools=self._tools,
            tool_choice="auto"
        )

        if "error" in response:
            return f"AI 回答失败: {response['error']}"

        # 检查是否有 tool_calls
        if 'choices' in response and len(response['choices']) > 0:
            choice = response['choices'][0]
            message = choice.get('message', {})
            tool_calls = message.get('tool_calls', [])

            if tool_calls:
                # 执行工具调用
                tool_call = tool_calls[0]
                function_name = tool_call.get('function', {}).get('name')
                arguments_str = tool_call.get('function', {}).get('arguments', '{}')
                try:
                    arguments = json.loads(arguments_str)
                except:
                    arguments = {}

                if function_name == 'query_data':
                    tool_result = self._query_data_tool(arguments, session_id)
                elif function_name == 'predict':
                    tool_result = self._predict_tool(arguments, session_id)
                else:
                    tool_result = f"未知工具: {function_name}"

                # 将工具结果加入消息历史
                messages.append(message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get('id'),
                    "content": tool_result
                })

                # 再次调用 LLM 获取最终回答
                final_response = llm.chat(
                    messages=messages,
                    temperature=0.7
                )

                if "error" in final_response:
                    return f"AI 回答失败: {final_response['error']}"

                if 'choices' in final_response and len(final_response['choices']) > 0:
                    return final_response['choices'][0].get('message', {}).get('content', '')

                return "处理失败"

            # 没有 tool_calls，直接返回内容
            return message.get('content', '')

        return "处理失败"

    def chat_stream(
        self,
        session_id: str,
        question: str,
        analysis_result: Dict[str, Any],
        context: List[str],
        context_data: Dict[str, Any] = None
    ) -> Generator[str, None, None]:
        """流式对话"""
        llm = self._get_llm_client()

        if not llm.api_key or not llm.api_base:
            yield self._fallback_answer(question, analysis_result)
            return

        system_prompt = self._build_system_prompt(session_id, analysis_result, context_data)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        try:
            # 先非流式判断是否需要调用工具
            response = llm.chat(
                messages=messages,
                temperature=0.7,
                tools=self._tools,
                tool_choice="auto"
            )

            if "error" in response:
                yield f"AI 回答失败: {response['error']}"
                return

            if 'choices' in response and len(response['choices']) > 0:
                choice = response['choices'][0]
                message = choice.get('message', {})
                tool_calls = message.get('tool_calls', [])

                if tool_calls:
                    # 有工具调用：执行工具，将结果拼入消息，再次调用（非流式）
                    tool_call = tool_calls[0]
                    function_name = tool_call.get('function', {}).get('name')
                    arguments_str = tool_call.get('function', {}).get('arguments', '{}')
                    try:
                        arguments = json.loads(arguments_str)
                    except:
                        arguments = {}

                    if function_name == 'query_data':
                        tool_result = self._query_data_tool(arguments, session_id)
                    elif function_name == 'predict':
                        tool_result = self._predict_tool(arguments, session_id)
                    else:
                        tool_result = f"未知工具: {function_name}"

                    messages.append(message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get('id'),
                        "content": tool_result
                    })

                    # 最终回答（非流式一次性输出）
                    final_response = llm.chat(
                        messages=messages,
                        temperature=0.7
                    )

                    if "error" in final_response:
                        yield f"AI 回答失败: {final_response['error']}"
                    else:
                        content = final_response.get('choices', [{}])[0].get('message', {}).get('content', '')
                        yield content
                    return

            # 没有工具调用：走流式输出
            for chunk in llm.chat_stream(
                messages=messages,
                temperature=0.7,
                tools=self._tools,
                tool_choice="none"
            ):
                if chunk:
                    yield chunk

        except Exception as e:
            yield f"AI 回答失败: {str(e)}"

    def _fallback_answer(self, question: str, analysis_result: Dict[str, Any]) -> str:
        """无大模型时的降级回答"""
        insights = self.insight_service.generate_rule_based_insights(analysis_result)

        if "数据整体" in question or "概况" in question:
            rows = analysis_result.get("data_shape", {}).get("rows", 0)
            cols = analysis_result.get("data_shape", {}).get("columns", 0)
            return f"数据共 {rows:,} 行，{cols} 列。建议配置大模型获取更详细的分析。"

        if "质量" in question:
            quality = analysis_result.get("quality_report", {})
            missing = len(quality.get("missing", []))
            outliers = len(quality.get("outliers", {}))
            return f"数据质量检查：发现 {missing} 个字段存在缺失值，{outliers} 个字段存在异常值。建议配置大模型获取详细解读。"

        if insights:
            return "\n".join(insights[:3]) + "\n\n💡 建议配置大模型获取更详细的解读。"

        return "请配置大模型 API 获取智能解读。"

    def get_scenarios(self, analysis_result: Dict[str, Any]) -> List[Dict]:
        """获取场景推荐"""
        variable_types = analysis_result.get("variable_types", {})
        ts_diag = analysis_result.get("time_series_diagnostics", {})
        quality = analysis_result.get("quality_report", {})
        correlations = analysis_result.get("correlations", {})

        has_continuous = any(info.get("type") == "continuous" for info in variable_types.values())
        has_categorical = any(
            info.get("type") in ["categorical", "categorical_numeric", "ordinal"] for info in variable_types.values()
        )
        has_datetime = any(info.get("type") == "datetime" for info in variable_types.values())
        has_auto = any(v.get("has_autocorrelation") for v in ts_diag.values())
        has_outliers = len(quality.get("outliers", {})) > 0

        numeric_cols = [col for col, info in variable_types.items() if info.get("type") == "continuous"]
        categorical_cols = [col for col, info in variable_types.items()
                            if info.get("type") in ["categorical", "categorical_numeric", "ordinal"]]
        high_corrs = correlations.get("high_correlations", [])

        scenarios = []

        if has_datetime and has_continuous:
            scenarios.append({
                "label": "📈 趋势分析场景",
                "question": "请从趋势分析角度解读数据，找出时间规律、周期性和异常波动。"
            })

        if has_categorical and has_continuous:
            scenarios.append({
                "label": "📊 对比分析场景",
                "question": "请从对比分析角度解读数据，找出各分类维度的差异和显著特征。"
            })

        if len(numeric_cols) >= 3:
            scenarios.append({
                "label": "🔗 相关性分析场景",
                "question": "请从相关性分析角度解读数据，找出各数值变量之间的关联关系。"
            })

        if has_auto:
            scenarios.append({
                "label": "🔮 预测分析场景",
                "question": "请从预测分析角度解读数据，评估哪些指标适合预测，给出建模建议。"
            })

        if has_outliers:
            scenarios.append({
                "label": "🚨 异常诊断场景",
                "question": "请从异常诊断角度解读数据，识别异常值和异常模式，分析可能的原因。"
            })

        if len(categorical_cols) >= 3:
            scenarios.append({
                "label": "🔗 关联规则场景",
                "question": "请从关联规则角度解读数据，找出各分类变量之间的关联模式。"
            })

        if len(high_corrs) > 0:
            scenarios.append({
                "label": "🔗 强相关分析场景",
                "question": f"请从强相关关系角度解读数据，找出 {len(high_corrs)} 对强相关关系背后的业务含义。"
            })

        return scenarios[:6]

    def get_recommended_questions(self, analysis_result: Dict[str, Any]) -> List[str]:
        """获取推荐问题"""
        variable_types = analysis_result.get("variable_types", {})
        ts_diag = analysis_result.get("time_series_diagnostics", {})
        quality = analysis_result.get("quality_report", {})
        high_corrs = analysis_result.get("correlations", {}).get("high_correlations", [])

        questions = [
            "📊 解读数据的主要特征和业务含义",
            "⚠️ 分析数据质量问题并给出清洗建议",
            "🔗 分析变量之间的相关性和关联关系",
            "🤖 推荐适合的建模方案和特征选择",
            "🎯 识别数据中的异常值和离群点",
            "📋 总结数据的关键洞察和行动建议"
        ]

        if any(v.get("has_autocorrelation") for v in ts_diag.values()):
            questions.insert(3, "📈 时间序列分析和预测建议")

        if high_corrs:
            questions.insert(4, f"🔗 发现 {len(high_corrs)} 对强相关关系，如何进行特征选择？")

        if quality.get("missing"):
            questions.append("📋 如何处理数据中的缺失值？")

        return questions