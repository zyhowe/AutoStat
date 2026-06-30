"""
根因分析模块

定位异常的根本原因，通过维度下钻和归因分析
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RootCause:
    """根因结论"""
    description: str
    confidence: float  # 0-1
    evidence: List[Dict[str, Any]]
    dimensions: Dict[str, str]  # 下钻路径


@dataclass
class RootCauseResult:
    """根因分析结果"""
    anomaly: Dict[str, Any]
    root_causes: List[RootCause]
    summary: str
    drill_path: List[str]


class RootCauseAnalyzer:
    """
    根因分析器

    使用方式:
        analyzer = RootCauseAnalyzer()
        result = analyzer.analyze(anomaly, df, dimension_cols, metric_cols)
    """

    def __init__(self, max_depth: int = 3, min_contribution: float = 0.1):
        """
        初始化根因分析器

        参数:
        - max_depth: 最大下钻深度
        - min_contribution: 最小贡献度阈值
        """
        self.max_depth = max_depth
        self.min_contribution = min_contribution

    def analyze(
        self,
        anomaly: Dict[str, Any],
        df: pd.DataFrame,
        dimension_cols: List[str],
        metric_cols: List[str],
        time_col: Optional[str] = None
    ) -> RootCauseResult:
        """
        根因分析

        参数:
        - anomaly: 异常事件
        - df: 数据框
        - dimension_cols: 维度列（如地区、品类、渠道）
        - metric_cols: 指标列
        - time_col: 时间列（可选）

        返回: RootCauseResult
        """
        target = anomaly.get("target", metric_cols[0] if metric_cols else None)
        if not target or target not in df.columns:
            return self._empty_result(anomaly, "目标列不存在")

        # 1. 维度下钻
        drill_path, contributions = self._drill_down(df, target, dimension_cols)

        # 2. 归因分析
        root_causes = self._find_root_causes(
            df, target, dimension_cols, drill_path, contributions
        )

        # 3. 生成摘要
        summary = self._generate_summary(anomaly, root_causes)

        return RootCauseResult(
            anomaly=anomaly,
            root_causes=root_causes,
            summary=summary,
            drill_path=drill_path
        )

    def _drill_down(
        self,
        df: pd.DataFrame,
        target: str,
        dimension_cols: List[str]
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        维度下钻：找到贡献最大的维度组合
        """
        # 计算整体平均值
        overall_mean = df[target].mean()
        overall_std = df[target].std()

        # 记录下钻路径
        drill_path = []
        contributions = {}

        # 对每个维度进行下钻
        for dim in dimension_cols:
            if dim not in df.columns:
                continue

            # 计算各维度的贡献
            dim_contrib = self._calculate_dimension_contribution(
                df, target, dim, overall_mean
            )

            if dim_contrib:
                # 选择贡献最大的维度值
                best_dim, best_contrib = max(
                    dim_contrib.items(),
                    key=lambda x: abs(x[1])
                )

                if abs(best_contrib) > self.min_contribution:
                    drill_path.append(f"{dim}={best_dim}")
                    contributions[dim] = best_contrib

        return drill_path, contributions

    def _calculate_dimension_contribution(
        self,
        df: pd.DataFrame,
        target: str,
        dim: str,
        overall_mean: float
    ) -> Dict[str, float]:
        """
        计算单个维度的贡献
        """
        contributions = {}

        for value in df[dim].unique():
            subset = df[df[dim] == value]
            if len(subset) < 5:
                continue

            mean = subset[target].mean()
            contribution = (mean - overall_mean) / (abs(overall_mean) + 1e-6)
            contributions[str(value)] = contribution

        return contributions

    def _find_root_causes(
        self,
        df: pd.DataFrame,
        target: str,
        dimension_cols: List[str],
        drill_path: List[str],
        contributions: Dict[str, float]
    ) -> List[RootCause]:
        """
        查找根因
        """
        root_causes = []

        # 按贡献度排序
        sorted_contrib = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for dim, contrib in sorted_contrib[:3]:
            if abs(contrib) < self.min_contribution:
                continue

            # 提取证据
            evidence = self._extract_evidence(df, target, dim, contrib)

            # 计算置信度
            confidence = min(1.0, abs(contrib) * 2)

            root_causes.append(RootCause(
                description=f"{dim} 对 {target} 的贡献为 {contrib:.1%}",
                confidence=confidence,
                evidence=evidence[:3],
                dimensions={dim: f"{contrib:.1%}"}
            ))

        # 如果没有找到根因，尝试综合贡献
        if not root_causes and contributions:
            # 合并多个维度
            combined_desc = "、".join([f"{k}({v:.1%})" for k, v in sorted_contrib[:2]])
            root_causes.append(RootCause(
                description=f"综合因素: {combined_desc}",
                confidence=0.6,
                evidence=[],
                dimensions={k: f"{v:.1%}" for k, v in sorted_contrib[:2]}
            ))

        return root_causes

    def _extract_evidence(
        self,
        df: pd.DataFrame,
        target: str,
        dim: str,
        contribution: float
    ) -> List[Dict[str, Any]]:
        """
        提取证据
        """
        evidence = []

        # 计算该维度下各值的统计
        for value in df[dim].unique()[:5]:
            subset = df[df[dim] == value]
            if len(subset) < 5:
                continue

            mean = subset[target].mean()
            count = len(subset)
            pct = count / len(df) * 100

            evidence.append({
                "dimension": dim,
                "value": str(value),
                "mean": round(mean, 2),
                "count": count,
                "percentage": round(pct, 1),
                "contribution": round(contribution, 3)
            })

        # 按贡献度排序
        evidence.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        return evidence

    def _generate_summary(
        self,
        anomaly: Dict[str, Any],
        root_causes: List[RootCause]
    ) -> str:
        """生成摘要"""
        if not root_causes:
            return f"未找到 {anomaly.get('target', 'unknown')} 异常的明确根因，建议扩大分析范围"

        parts = [f"根因分析: {anomaly.get('message', '数据异常')}"]

        for i, rc in enumerate(root_causes[:2]):
            parts.append(f"  {i+1}. {rc.description} (置信度: {rc.confidence:.0%})")

        if len(root_causes) > 2:
            parts.append(f"  ... 还有 {len(root_causes) - 2} 个次要因素")

        return "\n".join(parts)

    def _empty_result(self, anomaly: Dict, message: str) -> RootCauseResult:
        """空结果"""
        return RootCauseResult(
            anomaly=anomaly,
            root_causes=[],
            summary=message,
            drill_path=[]
        )

    def analyze_with_time(
        self,
        anomaly: Dict[str, Any],
        df: pd.DataFrame,
        dimension_cols: List[str],
        metric_cols: List[str],
        time_col: str
    ) -> RootCauseResult:
        """
        带时间维度的根因分析

        参数:
        - anomaly: 异常事件
        - df: 数据框
        - dimension_cols: 维度列
        - metric_cols: 指标列
        - time_col: 时间列

        返回: RootCauseResult
        """
        # 先执行基础分析
        result = self.analyze(anomaly, df, dimension_cols, metric_cols, time_col)

        # 如果已有根因，直接返回
        if result.root_causes:
            return result

        # 增加时间维度分析
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
            df['year_month'] = df[time_col].dt.to_period('M')

            # 按月份分析
            target = anomaly.get("target", metric_cols[0] if metric_cols else None)
            if target and target in df.columns:
                monthly_avg = df.groupby('year_month')[target].mean()

                if len(monthly_avg) >= 3:
                    # 检测趋势变化点
                    changes = monthly_avg.pct_change().dropna()
                    max_change = changes.max()
                    min_change = changes.min()

                    time_evidence = []

                    if max_change > 0.2:
                        time_evidence.append(f"月度最大增长: {max_change:.1%}")
                    if min_change < -0.2:
                        time_evidence.append(f"月度最大下降: {min_change:.1%}")

                    if time_evidence:
                        result.root_causes.append(RootCause(
                            description=f"时间维度异常: {', '.join(time_evidence)}",
                            confidence=0.7,
                            evidence=[{"time_analysis": time_evidence}],
                            dimensions={"time": "月度变化"}
                        ))
                        result.summary = f"{result.summary}\n  时间维度: {', '.join(time_evidence)}"

        return result


def analyze_root_cause(
    anomaly: Dict[str, Any],
    df: pd.DataFrame,
    dimension_cols: List[str],
    metric_cols: List[str],
    **kwargs
) -> RootCauseResult:
    """便捷函数：根因分析"""
    analyzer = RootCauseAnalyzer(**kwargs)
    return analyzer.analyze(anomaly, df, dimension_cols, metric_cols)


def analyze_root_cause_with_time(
    anomaly: Dict[str, Any],
    df: pd.DataFrame,
    dimension_cols: List[str],
    metric_cols: List[str],
    time_col: str,
    **kwargs
) -> RootCauseResult:
    """便捷函数：带时间维度的根因分析"""
    analyzer = RootCauseAnalyzer(**kwargs)
    return analyzer.analyze_with_time(anomaly, df, dimension_cols, metric_cols, time_col)