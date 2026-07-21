"""
洞察分析器
执行场景后，基于明细数据生成聚合洞察
排行榜、趋势、交叉分析、集中度、群体画像
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class InsightAnalyzer:
    """洞察分析器 - 从明细数据中提取聚合洞察"""

    def __init__(self, df: pd.DataFrame, results: List[Dict], facts: Dict[str, Any],
                 field_mapping: Dict[str, str] = None):
        """
        初始化洞察分析器

        参数:
        - df: 原始数据 DataFrame
        - results: 场景执行结果列表
        - facts: 技术事实清单
        - field_mapping: 字段中文名映射
        """
        self.df = df
        self.results = results
        self.facts = facts
        self.field_mapping = field_mapping or {}

        # 缓存
        self._company_col = None
        self._year_col = None
        self._entity_cols = []

        self._init_dimensions()

    def _init_dimensions(self):
        """识别维度列"""
        # 公司列
        for col in ['companycode', 'companyCode', 'company_id', 'CompanyCode']:
            if col in self.df.columns:
                self._company_col = col
                break

        # 年份列
        for col in ['reportdate_year', 'reportdateYear', 'year', 'Year']:
            if col in self.df.columns:
                self._year_col = col
                break
        if not self._year_col:
            for col in ['reportdate', 'reportDate', 'date', 'Date']:
                if col in self.df.columns and pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    self._year_col = col
                    break

        # 实体列
        f15 = self.facts.get("F15", {})
        for col in f15.get("entity_columns", []):
            if isinstance(col, dict):
                name = col.get("name")
                if name and name in self.df.columns:
                    self._entity_cols.append(name)
            elif isinstance(col, str) and col in self.df.columns:
                self._entity_cols.append(col)

        logger.info(f"维度识别: 公司列={self._company_col}, 年份列={self._year_col}, 实体列={self._entity_cols}")

    def _get_field_display(self, field: str) -> str:
        """获取字段中文名"""
        return self.field_mapping.get(field, field)

    def analyze_all(self) -> Dict[str, Any]:
        """执行所有洞察分析"""
        logger.info("开始洞察分析...")

        insights = {
            "generated_at": pd.Timestamp.now().isoformat(),
            "summary": self._build_summary(),
            "rankings": self._build_rankings(),
            "trends": self._build_trends(),
            "cross": self._build_cross(),
            "concentration": self._build_concentration(),
            "clusters": self._build_clusters(),
            "correlations": self._build_correlations()
        }

        logger.info(f"洞察分析完成，共 {len(insights)} 个维度")
        return insights

    # ==================== 摘要统计 ====================

    def _build_summary(self) -> Dict[str, Any]:
        """构建摘要统计"""
        summary = {
            "total_violations": 0,
            "total_outliers": 0,
            "total_missing": 0,
            "affected_companies": 0,
            "total_rules": 0,
            "total_fields": 0
        }

        # 从 results 中统计
        for result in self.results:
            if result.get("status") != "completed":
                continue
            scenario_id = result.get("scenario_id", "")
            records = result.get("records", [])

            if scenario_id == "E1":
                summary["total_violations"] += len(records)
                # 规则数
                rules = set()
                for r in records:
                    if r.get("rule"):
                        rules.add(r.get("rule"))
                summary["total_rules"] = max(summary["total_rules"], len(rules))
            elif scenario_id == "E2":
                summary["total_outliers"] += len(records)
            elif scenario_id == "E3":
                summary["total_missing"] += len(records)

        # 从 facts 中获取字段信息
        f08 = self.facts.get("F08", {})
        summary["total_fields"] = f08.get("count_fields", 0)

        # 受影响公司数
        if self._company_col and self._company_col in self.df.columns:
            # 找出有异常/违反的公司
            affected = set()
            for result in self.results:
                if result.get("status") != "completed":
                    continue
                records = result.get("records", [])
                for r in records:
                    row = r.get("row", 0)
                    if row > 0 and row <= len(self.df):
                        company = self.df.iloc[row - 1][self._company_col]
                        if pd.notna(company):
                            affected.add(str(company))
            summary["affected_companies"] = len(affected)

        return summary

    # ==================== 排行榜 ====================

    def _build_rankings(self) -> Dict[str, List[Dict]]:
        """构建所有排行榜"""
        return {
            "violation_by_rule": self._rank_violations_by_rule(),
            "violation_by_company": self._rank_violations_by_company(),
            "outlier_by_field": self._rank_outliers_by_field(),
            "outlier_by_company": self._rank_outliers_by_company(),
            "missing_by_field": self._rank_missing_by_field()
        }

    def _rank_violations_by_rule(self) -> List[Dict]:
        """按规则聚合违反数量"""
        rule_counts = defaultdict(int)
        rule_details = {}

        for result in self.results:
            if result.get("scenario_id") != "E1" or result.get("status") != "completed":
                continue
            for r in result.get("records", []):
                rule = r.get("rule", "未知规则")
                rule_counts[rule] += 1
                if rule not in rule_details:
                    rule_details[rule] = {
                        "fields": r.get("fields", []),
                        "severity": r.get("severity", "medium")
                    }

        sorted_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)
        total = sum(rule_counts.values()) if rule_counts else 1

        result = []
        for rule, count in sorted_rules[:20]:
            detail = rule_details.get(rule, {})
            result.append({
                "name": rule[:80] + "..." if len(rule) > 80 else rule,
                "count": count,
                "rate": round(count / total * 100, 1),
                "fields": detail.get("fields", [])[:5],
                "severity": detail.get("severity", "medium")
            })

        return result

    def _rank_violations_by_company(self) -> List[Dict]:
        """按公司聚合违反数量"""
        if not self._company_col or self._company_col not in self.df.columns:
            return []

        company_counts = defaultdict(int)

        for result in self.results:
            if result.get("scenario_id") != "E1" or result.get("status") != "completed":
                continue
            for r in result.get("records", []):
                row = r.get("row", 0)
                if row > 0 and row <= len(self.df):
                    company = self.df.iloc[row - 1][self._company_col]
                    if pd.notna(company):
                        company_counts[str(company)] += 1

        sorted_companies = sorted(company_counts.items(), key=lambda x: x[1], reverse=True)
        total = sum(company_counts.values()) if company_counts else 1

        result = []
        for company, count in sorted_companies[:20]:
            result.append({
                "name": company[:50],
                "count": count,
                "rate": round(count / total * 100, 1)
            })

        return result

    def _rank_outliers_by_field(self) -> List[Dict]:
        """按字段聚合异常数量"""
        f08 = self.facts.get("F08", {})
        fields_data = f08.get("fields", [])

        result = []
        for item in fields_data[:20]:
            field = item.get("field", "") if isinstance(item, dict) else str(item)
            count = item.get("count", 0) if isinstance(item, dict) else 0
            percent = item.get("percent", 0) if isinstance(item, dict) else 0

            if field and count > 0:
                result.append({
                    "name": self._get_field_display(field),
                    "field": field,
                    "count": count,
                    "rate": round(percent, 1)
                })

        result.sort(key=lambda x: x["count"], reverse=True)
        return result[:20]

    def _rank_outliers_by_company(self) -> List[Dict]:
        """按公司聚合异常数量"""
        if not self._company_col or self._company_col not in self.df.columns:
            return []

        f08 = self.facts.get("F08", {})
        fields_data = f08.get("fields", [])

        # 收集所有异常字段
        outlier_fields = []
        for item in fields_data:
            if isinstance(item, dict):
                field = item.get("field", "")
                if field and field in self.df.columns:
                    outlier_fields.append(field)

        if not outlier_fields:
            return []

        # 按公司统计异常行数
        company_counts = defaultdict(int)
        for field in outlier_fields[:10]:  # 限制字段数避免性能问题
            try:
                # 使用IQR检测异常
                series = self.df[field].dropna()
                if len(series) < 5:
                    continue
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0:
                    continue
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                anomaly_mask = (self.df[field] < lower) | (self.df[field] > upper)
                anomaly_rows = self.df[anomaly_mask]
                for idx in anomaly_rows.index:
                    company = self.df.loc[idx, self._company_col]
                    if pd.notna(company):
                        company_counts[str(company)] += 1
            except Exception as e:
                logger.warning(f"异常值公司聚合失败 {field}: {e}")
                continue

        sorted_companies = sorted(company_counts.items(), key=lambda x: x[1], reverse=True)
        total = sum(company_counts.values()) if company_counts else 1

        result = []
        for company, count in sorted_companies[:20]:
            result.append({
                "name": company[:50],
                "count": count,
                "rate": round(count / total * 100, 1)
            })

        return result

    def _rank_missing_by_field(self) -> List[Dict]:
        """按字段聚合缺失数量"""
        f09 = self.facts.get("F09", {})
        missing_data = f09.get("fields", [])

        result = []
        for item in missing_data[:20]:
            field = item.get("column", "") if isinstance(item, dict) else str(item)
            count = item.get("count", 0) if isinstance(item, dict) else 0
            percent = item.get("percent", 0) if isinstance(item, dict) else 0

            if field and count > 0:
                result.append({
                    "name": self._get_field_display(field),
                    "field": field,
                    "count": count,
                    "rate": round(percent, 1)
                })

        result.sort(key=lambda x: x["count"], reverse=True)
        return result[:20]

    # ==================== 趋势分析 ====================

    def _build_trends(self) -> Dict[str, List[Dict]]:
        """构建趋势数据"""
        return {
            "violation_by_year": self._trend_violations_by_year(),
            "outlier_by_year": self._trend_outliers_by_year()
        }

    def _trend_violations_by_year(self) -> List[Dict]:
        """按年份统计违反趋势"""
        if not self._year_col or self._year_col not in self.df.columns:
            return []

        year_counts = defaultdict(int)

        for result in self.results:
            if result.get("scenario_id") != "E1" or result.get("status") != "completed":
                continue
            for r in result.get("records", []):
                row = r.get("row", 0)
                if row > 0 and row <= len(self.df):
                    year_val = self.df.iloc[row - 1][self._year_col]
                    if pd.notna(year_val):
                        year = str(int(year_val)) if isinstance(year_val, (int, float)) else str(year_val)
                        year_counts[year] += 1

        result = []
        for year, count in sorted(year_counts.items()):
            result.append({"period": year, "count": count})

        return result

    def _trend_outliers_by_year(self) -> List[Dict]:
        """按年份统计异常趋势"""
        if not self._year_col or self._year_col not in self.df.columns:
            return []

        f08 = self.facts.get("F08", {})
        fields_data = f08.get("fields", [])

        outlier_fields = []
        for item in fields_data:
            if isinstance(item, dict):
                field = item.get("field", "")
                if field and field in self.df.columns:
                    outlier_fields.append(field)

        if not outlier_fields:
            return []

        year_counts = defaultdict(int)

        for field in outlier_fields[:10]:
            try:
                series = self.df[field].dropna()
                if len(series) < 5:
                    continue
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0:
                    continue
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                anomaly_mask = (self.df[field] < lower) | (self.df[field] > upper)
                for idx in self.df[anomaly_mask].index:
                    year_val = self.df.loc[idx, self._year_col]
                    if pd.notna(year_val):
                        year = str(int(year_val)) if isinstance(year_val, (int, float)) else str(year_val)
                        year_counts[year] += 1
            except Exception:
                continue

        result = []
        for year, count in sorted(year_counts.items()):
            result.append({"period": year, "count": count})

        return result

    # ==================== 交叉分析 ====================

    def _build_cross(self) -> Dict[str, List[Dict]]:
        """构建交叉分析数据"""
        return {
            "rule_x_company": self._cross_rule_x_company(),
            "field_x_company": self._cross_field_x_company()
        }

    def _cross_rule_x_company(self) -> List[Dict]:
        """规则×公司交叉分析"""
        if not self._company_col or self._company_col not in self.df.columns:
            return []

        cross_data = defaultdict(lambda: defaultdict(int))

        for result in self.results:
            if result.get("scenario_id") != "E1" or result.get("status") != "completed":
                continue
            for r in result.get("records", []):
                rule = r.get("rule", "未知规则")[:50]
                row = r.get("row", 0)
                if row > 0 and row <= len(self.df):
                    company = self.df.iloc[row - 1][self._company_col]
                    if pd.notna(company):
                        cross_data[rule][str(company)] += 1

        # 转换为列表，取Top组合
        result = []
        for rule, companies in cross_data.items():
            for company, count in sorted(companies.items(), key=lambda x: x[1], reverse=True)[:5]:
                result.append({
                    "rule": rule,
                    "company": company[:50],
                    "count": count
                })

        result.sort(key=lambda x: x["count"], reverse=True)
        return result[:30]

    def _cross_field_x_company(self) -> List[Dict]:
        """字段×公司交叉分析"""
        if not self._company_col or self._company_col not in self.df.columns:
            return []

        f08 = self.facts.get("F08", {})
        fields_data = f08.get("fields", [])

        outlier_fields = []
        for item in fields_data:
            if isinstance(item, dict):
                field = item.get("field", "")
                if field and field in self.df.columns and item.get("count", 0) > 0:
                    outlier_fields.append(field)

        if not outlier_fields:
            return []

        cross_data = defaultdict(lambda: defaultdict(int))

        for field in outlier_fields[:10]:
            try:
                series = self.df[field].dropna()
                if len(series) < 5:
                    continue
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0:
                    continue
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                anomaly_mask = (self.df[field] < lower) | (self.df[field] > upper)
                for idx in self.df[anomaly_mask].index:
                    company = self.df.loc[idx, self._company_col]
                    if pd.notna(company):
                        cross_data[field][str(company)] += 1
            except Exception:
                continue

        result = []
        for field, companies in cross_data.items():
            for company, count in sorted(companies.items(), key=lambda x: x[1], reverse=True)[:5]:
                result.append({
                    "field": self._get_field_display(field),
                    "company": company[:50],
                    "count": count
                })

        result.sort(key=lambda x: x["count"], reverse=True)
        return result[:30]

    # ==================== 集中度分析 ====================

    def _build_concentration(self) -> Dict[str, Any]:
        """构建集中度分析"""
        result = {
            "hhi": 0,
            "cr4": 0,
            "level": "数据不足",
            "top_entities": []
        }

        # 找实体列
        entity_col = None
        for col in self._entity_cols:
            if col in self.df.columns:
                entity_col = col
                break

        if not entity_col:
            # 用公司列作为实体列
            if self._company_col and self._company_col in self.df.columns:
                entity_col = self._company_col

        if not entity_col:
            return result

        try:
            value_counts = self.df[entity_col].astype(str).value_counts()
            total = value_counts.sum()
            if total == 0:
                return result

            # HHI
            hhi = ((value_counts / total) ** 2).sum()
            result["hhi"] = round(hhi, 4)

            # CR4
            cr4 = value_counts.head(4).sum() / total
            result["cr4"] = round(cr4 * 100, 1)

            # 等级
            if hhi > 0.25:
                result["level"] = "高度集中"
            elif hhi > 0.15:
                result["level"] = "中度集中"
            else:
                result["level"] = "分散"

            # Top实体
            for val, count in value_counts.head(10).items():
                pct = count / total * 100
                result["top_entities"].append({
                    "name": val[:50],
                    "count": int(count),
                    "rate": round(pct, 1)
                })

        except Exception as e:
            logger.warning(f"集中度分析失败: {e}")

        return result

    # ==================== 群体画像 ====================

    def _build_clusters(self) -> Dict[str, Any]:
        """构建群体画像"""
        result = {
            "has_clusters": False,
            "profile": [],
            "key_dimensions": []
        }

        # 从D1结果中提取
        for res in self.results:
            if res.get("scenario_id") != "D1" or res.get("status") != "completed":
                continue

            records = res.get("records", [])
            if not records:
                continue

            # 统计各群体大小
            cluster_sizes = defaultdict(int)
            cluster_samples = defaultdict(list)

            for r in records:
                cluster_id = r.get("cluster_id", 0)
                cluster_sizes[cluster_id] += 1
                if len(cluster_samples[cluster_id]) < 5:
                    cluster_samples[cluster_id].append(r)

            total = sum(cluster_sizes.values()) if cluster_sizes else 1

            profiles = []
            for cluster_id, size in sorted(cluster_sizes.items()):
                samples = cluster_samples.get(cluster_id, [])
                features = []
                if samples:
                    # 提取特征
                    sample = samples[0]
                    if sample.get("features"):
                        feat_items = list(sample["features"].items())[:5]
                        features = [f"{self._get_field_display(k)}={v:.1f}" if isinstance(v, (int, float)) else f"{k}={v}"
                                   for k, v in feat_items]

                profiles.append({
                    "cluster_id": cluster_id,
                    "size": size,
                    "rate": round(size / total * 100, 1),
                    "features": features
                })

            result["has_clusters"] = True
            result["profile"] = profiles
            result["key_dimensions"] = ["群体大小", "特征分布"]
            break

        return result

    # ==================== 相关性分析 ====================

    def _build_correlations(self) -> Dict[str, List[Dict]]:
        """构建相关性分析"""
        f04 = self.facts.get("F04", {})
        high_corrs = f04.get("high_correlations", [])

        # 按相关性强度分组
        redundant = []
        strong = []

        for corr in high_corrs:
            val = abs(corr.get("value", 0))
            if val >= 0.95:
                redundant.append({
                    "var1": self._get_field_display(corr.get("var1", "")),
                    "var2": self._get_field_display(corr.get("var2", "")),
                    "value": corr.get("value", 0),
                    "type": "完全冗余" if val >= 0.99 else "高度冗余"
                })
            elif val >= 0.85:
                strong.append({
                    "var1": self._get_field_display(corr.get("var1", "")),
                    "var2": self._get_field_display(corr.get("var2", "")),
                    "value": corr.get("value", 0),
                    "type": "强相关"
                })

        return {
            "redundant_pairs": redundant[:20],
            "strong_pairs": strong[:20]
        }


def analyze_insights(
    df: pd.DataFrame,
    results: List[Dict],
    facts: Dict[str, Any],
    field_mapping: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    洞察分析入口函数

    参数:
    - df: 原始数据 DataFrame
    - results: 场景执行结果列表
    - facts: 技术事实清单
    - field_mapping: 字段中文名映射

    返回:
    - 洞察数据字典
    """
    analyzer = InsightAnalyzer(df, results, facts, field_mapping)
    return analyzer.analyze_all()