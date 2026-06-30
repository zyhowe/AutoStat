"""
自动故事生成模块

从数据中自动生成叙事性报告
"""

import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class StorySection:
    """故事章节"""
    title: str
    content: str
    chart_ref: Optional[str] = None
    data_ref: Optional[str] = None


@dataclass
class Story:
    """生成的故事"""
    title: str
    summary: str
    sections: List[StorySection]
    key_findings: List[str]
    recommendations: List[str]
    generated_at: str


class StoryGenerator:
    """
    自动故事生成器

    使用方式:
        generator = StoryGenerator()
        story = generator.generate(df, analysis_result)
    """

    def __init__(self, llm_client=None):
        """
        初始化

        参数:
        - llm_client: 大模型客户端（用于增强生成）
        """
        self.llm_client = llm_client

    def generate(
        self,
        df: pd.DataFrame,
        analysis_result: Dict[str, Any],
        charts: Optional[List[Dict]] = None,
        context: Optional[Dict] = None
    ) -> Story:
        """
        生成故事

        参数:
        - df: 数据框
        - analysis_result: 分析结果
        - charts: 图表列表
        - context: 额外上下文（业务背景等）

        返回: Story 对象
        """
        sections = []

        # 1. 数据概览章节
        sections.append(self._build_overview_section(df, analysis_result))

        # 2. 关键发现章节
        sections.append(self._build_findings_section(analysis_result))

        # 3. 数据质量章节
        sections.append(self._build_quality_section(analysis_result))

        # 4. 推荐模型章节
        sections.append(self._build_model_section(analysis_result))

        # 生成标题和摘要
        title = self._generate_title(df, analysis_result)
        summary = self._generate_summary(df, analysis_result)

        # 提取关键发现
        key_findings = self._extract_key_findings(analysis_result)

        # 生成建议
        recommendations = self._generate_recommendations(analysis_result)

        # 大模型增强
        if self.llm_client:
            enhanced = self._enhance_with_llm(title, summary, sections, key_findings)
            if enhanced:
                title = enhanced.get("title", title)
                summary = enhanced.get("summary", summary)
                key_findings = enhanced.get("findings", key_findings)
                recommendations = enhanced.get("recommendations", recommendations)

        return Story(
            title=title,
            summary=summary,
            sections=sections,
            key_findings=key_findings,
            recommendations=recommendations,
            generated_at=datetime.now().isoformat()
        )

    def _build_overview_section(
        self,
        df: pd.DataFrame,
        analysis_result: Dict[str, Any]
    ) -> StorySection:
        """构建数据概览章节"""
        rows, cols = df.shape
        variable_types = analysis_result.get("variable_types", {})

        type_counts = {}
        for info in variable_types.values():
            typ = info.get("type", "unknown")
            type_counts[typ] = type_counts.get(typ, 0) + 1

        type_desc = {
            "continuous": "连续变量",
            "categorical": "分类变量",
            "categorical_numeric": "数值型分类",
            "ordinal": "有序分类",
            "datetime": "日期时间",
            "identifier": "标识符"
        }

        type_summary = "、".join([f"{type_desc.get(t, t)} {c}个" for t, c in type_counts.items()])

        content = f"""
本次分析的数据集包含 **{rows:,}** 行、**{cols}** 列数据。
其中包含 {type_summary}。
数据来源: {analysis_result.get('source_table', '未知')}
分析时间: {analysis_result.get('analysis_time', '')[:19] if analysis_result.get('analysis_time') else '未知'}
        """.strip()

        return StorySection(
            title="📊 数据概览",
            content=content
        )

    def _build_findings_section(
        self,
        analysis_result: Dict[str, Any]
    ) -> StorySection:
        """构建关键发现章节"""
        findings = []

        # 从相关性中提取
        high_corrs = analysis_result.get("correlations", {}).get("high_correlations", [])
        if high_corrs:
            top_corr = high_corrs[0]
            findings.append(
                f"发现强相关关系：{top_corr.get('var1', '')} 与 {top_corr.get('var2', '')} 的相关系数为 {top_corr.get('value', 0)}，"
                f"表明两者存在{'正' if top_corr.get('value', 0) > 0 else '负'}向关联。"
            )

        # 从时间序列中提取
        ts_diag = analysis_result.get("time_series_diagnostics", {})
        ts_columns = [k for k, v in ts_diag.items() if v.get("has_autocorrelation")]
        if ts_columns:
            ts_str = "、".join(ts_columns[:3])
            findings.append(
                f"检测到 {len(ts_columns)} 个序列存在自相关性：{ts_str}"
                f"{'等' if len(ts_columns) > 3 else ''}，适合进行时间序列预测。"
            )

        # 从分布中提取
        skewed = analysis_result.get("distribution_insights", {}).get("skewed_variables", [])
        if skewed:
            skewed_names = [s.get("name", "") for s in skewed[:3] if s.get("name")]
            if skewed_names:
                findings.append(
                    f"发现 {len(skewed)} 个偏态变量：{', '.join(skewed_names)}"
                    f"{'等' if len(skewed) > 3 else ''}，建议使用中位数描述。"
                )

        if not findings:
            findings = ["数据分布较为均匀，未发现特别突出的模式或规律。"]

        return StorySection(
            title="🔍 关键发现",
            content="\n\n".join(findings)
        )

    def _build_quality_section(
        self,
        analysis_result: Dict[str, Any]
    ) -> StorySection:
        """构建数据质量章节"""
        quality = analysis_result.get("quality_report", {})
        missing = quality.get("missing", [])
        outliers = quality.get("outliers", {})
        duplicates = quality.get("duplicates", {})

        quality_parts = []

        if missing:
            high_missing = [m for m in missing if m.get("percent", 0) > 20]
            if high_missing:
                high_missing_names = [m.get("column", "") for m in high_missing[:3] if m.get("column")]
                quality_parts.append(
                    f"⚠️ 发现 {len(high_missing)} 个字段缺失率超过20%："
                    f"{', '.join(high_missing_names)}{'等' if len(high_missing) > 3 else ''}"
                )

        if outliers:
            outlier_fields = list(outliers.keys())
            outlier_names = ", ".join(outlier_fields[:3])
            quality_parts.append(
                f"⚠️ 发现 {len(outlier_fields)} 个字段存在异常值"
                f"（{outlier_names}{'等' if len(outlier_fields) > 3 else ''}）"
            )

        # 🆕 修复：确保 duplicates 的值是数字类型
        dup_count = duplicates.get("count", 0)
        try:
            dup_count = int(dup_count) if dup_count is not None else 0
        except (ValueError, TypeError):
            dup_count = 0

        if dup_count > 0:
            dup_pct = duplicates.get("percent", 0)
            try:
                dup_pct = float(dup_pct) if dup_pct is not None else 0
            except (ValueError, TypeError):
                dup_pct = 0
            quality_parts.append(
                f"⚠️ 发现 {dup_count} 条重复记录（占比 {dup_pct:.1f}%）"
            )

        if not quality_parts:
            quality_parts = ["✅ 数据质量良好，未发现明显问题。"]

        return StorySection(
            title="📋 数据质量",
            content="\n".join(quality_parts)
        )

    def _build_model_section(
        self,
        analysis_result: Dict[str, Any]
    ) -> StorySection:
        """构建模型推荐章节"""
        recommendations = analysis_result.get("model_recommendations", [])

        if not recommendations:
            return StorySection(
                title="🤖 模型建议",
                content="当前数据特征不足以生成模型推荐。建议增加更多字段或样本量。"
            )

        model_parts = []
        for rec in recommendations[:3]:
            task_type = rec.get("task_type", "")
            ml_model = rec.get("ml", "")
            target = rec.get("target_column", "无")
            reason = rec.get("reason", "")

            model_parts.append(
                f"**{task_type}**: {ml_model}\n"
                f"  目标: {target}\n"
                f"  原因: {reason[:100] if reason else '基于数据特征推荐'}"
            )

        return StorySection(
            title="🤖 模型建议",
            content="\n\n".join(model_parts)
        )

    def _generate_title(self, df: pd.DataFrame, analysis_result: Dict[str, Any]) -> str:
        """生成标题"""
        rows, cols = df.shape
        source = analysis_result.get("source_table", "数据")

        # 检测是否有时间序列
        ts_diag = analysis_result.get("time_series_diagnostics", {})
        has_ts = any(v.get("has_autocorrelation") for v in ts_diag.values())

        if has_ts:
            return f"📈 {source} 数据分析报告 - 含时间序列洞察"
        else:
            return f"📊 {source} 数据分析报告"

    def _generate_summary(self, df: pd.DataFrame, analysis_result: Dict[str, Any]) -> str:
        """生成摘要"""
        rows, cols = df.shape

        # 从关键发现中提取
        high_corrs = analysis_result.get("correlations", {}).get("high_correlations", [])
        ts_columns = [
            k for k, v in analysis_result.get("time_series_diagnostics", {}).items()
            if v.get("has_autocorrelation")
        ]

        parts = [f"本报告分析了 {rows:,} 行 × {cols} 列数据。"]

        if high_corrs:
            top_corr = high_corrs[0]
            parts.append(
                f"发现 {top_corr.get('var1', '')} 与 {top_corr.get('var2', '')} 存在强相关（r={top_corr.get('value', 0)}），"
                f"可重点关注此关系。"
            )

        if ts_columns:
            parts.append(
                f"检测到 {len(ts_columns)} 个序列具有可预测性，建议进行时间序列建模。"
            )

        return " ".join(parts)

    def _extract_key_findings(self, analysis_result: Dict[str, Any]) -> List[str]:
        """提取关键发现"""
        findings = []

        high_corrs = analysis_result.get("correlations", {}).get("high_correlations", [])
        for corr in high_corrs[:2]:
            findings.append(
                f"🔗 {corr.get('var1', '')} 与 {corr.get('var2', '')} 强相关 (r={corr.get('value', 0):.2f})"
            )

        ts_diag = analysis_result.get("time_series_diagnostics", {})
        for key, diag in ts_diag.items():
            if diag.get("has_autocorrelation"):
                findings.append(
                    f"📈 {key} 具有自相关性，可进行预测"
                )

        if not findings:
            findings = ["📊 数据整体分布较为均匀"]

        return findings[:5]

    def _generate_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []

        quality = analysis_result.get("quality_report", {})
        missing = quality.get("missing", [])

        if any(m.get("percent", 0) > 20 for m in missing):
            recommendations.append("建议处理缺失率超过20%的字段")

        if analysis_result.get("time_series_diagnostics"):
            recommendations.append("建议进行时间序列预测分析")

        model_recs = analysis_result.get("model_recommendations", [])
        if model_recs:
            rec = model_recs[0]
            ml_model = rec.get("ml", "机器学习")
            task_type = rec.get("task_type", "预测")
            recommendations.append(
                f"建议尝试 {ml_model} 模型进行 {task_type}"
            )

        if not recommendations:
            recommendations = ["建议进行数据探索性分析（EDA）"]

        return recommendations

    def _enhance_with_llm(
        self,
        title: str,
        summary: str,
        sections: List[StorySection],
        findings: List[str]
    ) -> Optional[Dict[str, Any]]:
        """使用大模型增强"""
        if not self.llm_client:
            return None

        # 构建章节摘要
        sections_summary = []
        for s in sections[:3]:
            content_preview = s.content[:100] if s.content else ""
            sections_summary.append(f"- {s.title}: {content_preview}...")

        prompt = f"""请优化以下数据分析故事：

## 标题
{title}

## 摘要
{summary}

## 章节
{chr(10).join(sections_summary)}

## 关键发现
{chr(10).join(findings)}

请返回JSON格式的优化结果：
{{"title": "优化标题", "summary": "优化摘要", "findings": ["优化发现1", ...], "recommendations": ["建议1", ...]}}
"""

        try:
            import json
            response = self.llm_client.chat([{"role": "user", "content": prompt}])
            # 尝试提取JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            result = json.loads(response.strip())
            return result
        except Exception as e:
            print(f"LLM增强失败: {e}")
            return None


def generate_story(
    df: pd.DataFrame,
    analysis_result: Dict[str, Any],
    **kwargs
) -> Story:
    """便捷函数：生成故事"""
    generator = StoryGenerator(**kwargs)
    return generator.generate(df, analysis_result)