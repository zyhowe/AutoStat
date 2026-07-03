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
            # 🆕 从配置读取第一个可用的 LLM 配置
            configs = self._config_service.get_llm_configs()
            if configs:
                # 使用第一个配置
                config = configs[0]
                print(f"使用大模型配置: {config.get('name')}")
                self._llm_client = LLMClient({
                    "api_base": config.get("api_base", ""),
                    "api_key": config.get("api_key", ""),
                    "model": config.get("model", ""),
                    "timeout": config.get("timeout", 60)
                })
            else:
                # 没有配置，创建空客户端
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
        context: List[str]
    ) -> str:
        """非流式对话"""
        llm = self._get_llm_client()

        # 🆕 检查是否有有效的 API key
        if not llm.api_key or not llm.api_base:
            return self._fallback_answer(question, analysis_result)

        try:
            system_prompt = self._build_system_prompt(analysis_result)
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
        context: List[str]
    ) -> Generator[str, None, None]:
        """流式对话"""
        llm = self._get_llm_client()

        if not llm.api_key or not llm.api_base:
            yield self._fallback_answer(question, analysis_result)
            return

        try:
            system_prompt = self._build_system_prompt(analysis_result)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            for chunk in llm.chat_stream(messages, temperature=0.7):
                yield chunk
        except Exception as e:
            yield f"AI 回答失败: {str(e)}"


    def _build_system_prompt(self, analysis_result: Dict[str, Any]) -> str:
        """构建系统提示词"""
        data_shape = analysis_result.get("data_shape", {})
        variable_types = analysis_result.get("variable_types", {})
        quality = analysis_result.get("quality_report", {})
        correlations = analysis_result.get("correlations", {})
        ts_diag = analysis_result.get("time_series_diagnostics", {})

        # 变量类型统计
        type_counts = {}
        type_display = {
            "continuous": "连续变量",
            "categorical": "分类变量",
            "categorical_numeric": "数值型分类",
            "ordinal": "有序分类",
            "datetime": "日期时间",
            "identifier": "标识符"
        }
        for info in variable_types.values():
            typ = info.get("type", "unknown")
            type_counts[typ] = type_counts.get(typ, 0) + 1
        type_summary = "、".join(
            [f"{type_display.get(t, t)} {c}个" for t, c in type_counts.items() if t in type_display])

        # 强相关
        high_corrs = correlations.get("high_correlations", [])
        corr_summary = ""
        if high_corrs:
            top = high_corrs[0]
            corr_summary = f"{top.get('var1', '')} ↔ {top.get('var2', '')} (r={top.get('value', 0):.3f})"

        # 时间序列
        has_auto = any(v.get("has_autocorrelation") for v in ts_diag.values())
        ts_summary = "有" if has_auto else "无"

        # 重复记录
        dup_count = quality.get("duplicates", {}).get("count", 0)
        try:
            dup_count = int(dup_count) if dup_count else 0
        except (ValueError, TypeError):
            dup_count = 0

        return f"""你是专业的数据分析师，正在回答用户关于数据的问题。

## 数据概况
- 总行数: {data_shape.get('rows', 0):,}
- 总列数: {data_shape.get('columns', 0)}
- 变量类型: {type_summary}

## 数据质量
- 缺失字段: {len(quality.get('missing', []))}个
- 异常字段: {len(quality.get('outliers', {}))}个
- 重复记录: {dup_count}条

## 关键发现
- 强相关: {corr_summary if corr_summary else '无'}
- 时间序列: {ts_summary}

## 重要说明
1. 用中文回答，结构清晰，友好专业
2. 基于数据分析结果回答
3. 回答要具体、可执行
"""

    def _fallback_answer(self, question: str, analysis_result: Dict[str, Any]) -> str:
        """无大模型时的降级回答"""
        # 尝试从规则中匹配
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

        has_continuous = any(info.get("type") == "continuous" for info in variable_types.values())
        has_categorical = any(
            info.get("type") in ["categorical", "categorical_numeric", "ordinal"] for info in variable_types.values())
        has_datetime = any(info.get("type") == "datetime" for info in variable_types.values())
        has_auto = any(v.get("has_autocorrelation") for v in ts_diag.values())
        has_outliers = len(quality.get("outliers", {})) > 0

        numeric_cols = [col for col, info in variable_types.items() if info.get("type") == "continuous"]
        categorical_cols = [col for col, info in variable_types.items() if
                            info.get("type") in ["categorical", "categorical_numeric", "ordinal"]]

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

        return scenarios[:6]

    def get_recommended_questions(self, analysis_result: Dict[str, Any]) -> List[str]:
        """获取推荐问题"""
        return [
            "📊 解读数据的主要特征和业务含义",
            "⚠️ 分析数据质量问题并给出清洗建议",
            "🔗 分析变量之间的相关性和关联关系",
            "📈 时间序列分析和预测建议",
            "🤖 推荐适合的建模方案和特征选择",
            "🎯 识别数据中的异常值和离群点",
            "📋 总结数据的关键洞察和行动建议"
        ]