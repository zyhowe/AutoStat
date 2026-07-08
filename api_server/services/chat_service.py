"""AI对话服务 - 从配置读取大模型"""
import json
from typing import Dict, Any, List, Generator, Optional

from autostat.llm_client import LLMClient
from autostat.core.insight import InsightService
from api_server.services.config_service import ConfigService


class ChatService:
    """AI对话服务"""

    def __init__(self):
        self.insight_service = InsightService()
        self._llm_client = None
        self._config_service = ConfigService()

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

        try:
            system_prompt = self._build_system_prompt(analysis_result, context_data)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            return llm.chat(messages, temperature=0.7)
        except Exception as e:
            return f"AI 回答失败: {str(e)}"

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

        try:
            system_prompt = self._build_system_prompt(analysis_result, context_data)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            for chunk in llm.chat_stream(messages, temperature=0.7):
                yield chunk
        except Exception as e:
            yield f"AI 回答失败: {str(e)}"

    def _build_system_prompt(self, analysis_result: Dict[str, Any], context_data: Dict[str, Any] = None) -> str:
        """构建系统提示词"""
        data_shape = analysis_result.get("data_shape", {})
        variable_types = analysis_result.get("variable_types", {})
        quality = analysis_result.get("quality_report", {})
        correlations = analysis_result.get("correlations", {})
        ts_diag = analysis_result.get("time_series_diagnostics", {})
        summaries = analysis_result.get("variable_summaries", {})

        # 获取表名
        source_table = analysis_result.get("source_table", "未知表名")

        # 如果有 context_data，从中提取更详细的信息
        if context_data:
            rows = context_data.get('rows', data_shape.get('rows', 0))
            cols = context_data.get('columns', data_shape.get('columns', 0))
            variable_types = context_data.get('variableTypes', variable_types)
            field_summary = context_data.get('fieldSummary', [])
            top_correlations = context_data.get('topCorrelations', [])
            quality_score = context_data.get('qualityScore', quality.get('overall_score'))
            detailed_fields = context_data.get('detailedFields', {})
            if context_data.get('source_table'):
                source_table = context_data.get('source_table')
            # 从 context_data 中提取推荐问题对应的数据
            recommended_data = context_data.get('recommendedData', [])
            recommended_data_key = context_data.get('recommendedDataKey', '')
        else:
            rows = data_shape.get('rows', 0)
            cols = data_shape.get('columns', 0)
            field_summary = []
            top_correlations = []
            quality_score = quality.get('overall_score')
            detailed_fields = {}
            recommended_data = []
            recommended_data_key = ''

        # 变量类型统计
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

        # 构建字段列表
        field_list_str = ""
        if field_summary:
            field_list_str = "\n".join([
                f"  - {f.get('name')}: {f.get('type')} (缺失率: {f.get('missing_pct', 0):.1f}%)"
                for f in field_summary[:20]
            ])

        # 构建详细字段信息（供 SQL 生成使用）
        detailed_fields_str = ""
        if detailed_fields:
            for field_name, info in detailed_fields.items():
                detail_parts = []
                if info.get('type'):
                    detail_parts.append(f"类型: {info.get('type')}")
                if info.get('mean') is not None:
                    detail_parts.append(f"均值: {info.get('mean'):.2f}")
                if info.get('median') is not None:
                    detail_parts.append(f"中位数: {info.get('median'):.2f}")
                if info.get('min') is not None and info.get('max') is not None:
                    detail_parts.append(f"范围: {info.get('min')}~{info.get('max')}")
                if info.get('n_unique'):
                    detail_parts.append(f"唯一值: {info.get('n_unique')}个")
                if info.get('mode'):
                    detail_parts.append(f"众数: {info.get('mode')} ({info.get('mode_pct', 0):.1f}%)")
                if info.get('min_date'):
                    detail_parts.append(f"日期范围: {info.get('min_date')}~{info.get('max_date')}")
                if info.get('std') is not None:
                    detail_parts.append(f"标准差: {info.get('std'):.2f}")
                if info.get('skew') is not None:
                    detail_parts.append(f"偏度: {info.get('skew'):.2f}")
                detailed_fields_str += f"  - {field_name}: {', '.join(detail_parts)}\n"

        # 如果详细字段信息为空，从 summaries 补充
        if not detailed_fields_str and summaries:
            for field_name, info in list(summaries.items())[:15]:
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
                if detail_parts:
                    detailed_fields_str += f"  - {field_name}: {', '.join(detail_parts)}\n"

        # ==== 核心：从 context_data 中提取推荐问题对应的具体数据 ====
        recommended_data_str = ""
        if recommended_data:
            if isinstance(recommended_data, list):
                # 列表数据（如 distribution、correlation 等）
                recommended_data_str = "\n".join([
                    f"  - {item.get('text', '')}" if isinstance(item, dict) else f"  - {item}"
                    for item in recommended_data[:10]
                ])
            elif isinstance(recommended_data, dict):
                # 字典数据
                recommended_data_str = json.dumps(recommended_data, ensure_ascii=False, indent=2)[:2000]
            else:
                recommended_data_str = str(recommended_data)[:2000]

            if recommended_data_str:
                recommended_data_str = f"\n## 该问题对应的具体数据（来源: {recommended_data_key}）\n{recommended_data_str}\n"

        # 强相关
        corr_summary = ""
        if top_correlations:
            top = top_correlations[0]
            corr_summary = f"{top.get('var1', '')} ↔ {top.get('var2', '')} (r={top.get('value', 0):.3f})"
        elif correlations.get('high_correlations'):
            top = correlations['high_correlations'][0]
            corr_summary = f"{top.get('var1', '')} ↔ {top.get('var2', '')} (r={top.get('value', 0):.3f})"

        # 时间序列
        has_auto = any(v.get('has_autocorrelation') for v in ts_diag.values())
        ts_summary = "有" if has_auto else "无"

        # 重复记录
        dup_count = quality.get("duplicates", {}).get("count", 0)
        try:
            dup_count = int(dup_count) if dup_count else 0
        except (ValueError, TypeError):
            dup_count = 0

        # 缺失字段
        missing_count = len(quality.get('missing', []))
        outlier_count = len(quality.get('outliers', {}))

        prompt = f"""你是专业的数据分析师，正在回答用户关于数据的问题。

## 数据概况
- 表名: {source_table}
- 总行数: {rows:,}
- 总列数: {cols}
- 变量类型: {type_summary}

## 表结构
{field_list_str if field_list_str else '  暂无详细字段列表'}
{detailed_fields_str if detailed_fields_str else ''}
{recommended_data_str}

## 数据质量
- 缺失字段: {missing_count}个
- 异常字段: {outlier_count}个
- 重复记录: {dup_count}条
- 综合质量评分: {quality_score if quality_score else '未评分'}

## 关键发现
- 强相关: {corr_summary if corr_summary else '无'}
- 时间序列: {ts_summary}

## 重要说明
1. 用中文回答，结构清晰，友好专业
2. **回答时优先使用「该问题对应的具体数据」中的内容，这些是真实的数据**
3. 如果「具体数据」中存在用户问题的答案，直接引用
4. **生成SQL时，表名必须使用「{source_table}」**，不要使用占位符
5. 回答要具体、可执行，不要说空话套话
6. 如果有具体数值，直接给出，不要只说"较高""较低"
"""

        return prompt

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
                "question": f"请从相关性分析角度解读数据，找出各数值变量之间的关联关系。"
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