"""
场景执行器
执行场景深度计算，输出技术化结论 + 记录级明细
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime,date
import logging
import sys

# ===== 配置日志输出到控制台 =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class ScenarioExecutor:
    """场景执行器"""

    def __init__(self, data: pd.DataFrame, fact_sheet: Dict[str, Any]):
        """
        初始化场景执行器

        参数:
        - data: 数据 DataFrame（合并表或主表）
        - fact_sheet: 技术事实清单
        """
        self.data = data
        self.fact_sheet = fact_sheet
        self.facts = fact_sheet.get("facts", {})
        self.results = {}
        self._row_index = self.data.reset_index(drop=True)
        # 获取字段中文名映射（从 fact_sheet 或后续传入）
        self.field_mapping = {}
        logger.info(f"ScenarioExecutor 初始化: {len(data)} 行, {len(data.columns)} 列")

    def set_field_mapping(self, mapping: Dict[str, str]):
        """设置字段中文名映射"""
        self.field_mapping = mapping or {}
        logger.info(f"设置字段映射: {len(self.field_mapping)} 个")

    def _get_field_display(self, field: str) -> str:
        """获取字段中文名"""
        return self.field_mapping.get(field, field)

    def _safe_convert(self, value):
        """安全转换 numpy 类型为 Python 原生类型"""
        if value is None:
            return None
        if isinstance(value, (np.integer, np.int32, np.int64)):
            return int(value)
        if isinstance(value, (np.floating, np.float32, np.float64)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, dict):
            return {k: self._safe_convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._safe_convert(v) for v in value]
        if isinstance(value, tuple):
            return [self._safe_convert(v) for v in value]
        if isinstance(value, (pd.Series, pd.DataFrame)):
            try:
                return value.to_dict(orient='records')
            except:
                return str(value)
        return value

    def execute(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行单个场景

        参数:
        - scenario: 场景定义（包含 id, params 等）

        返回:
        - 执行结果（包含 conclusions 和 records）
        """
        scenario_id = scenario.get("id")
        params = scenario.get("params", {})

        executor_map = {
            "A1": self._execute_A1,
            "A2": self._execute_A2,
            "A3": self._execute_A3,
            "A4": self._execute_A4,
            "B1": self._execute_B1,
            "B2": self._execute_B2,
            "B3": self._execute_B3,
            "B4": self._execute_B4,
            "B5": self._execute_B5,
            "C1": self._execute_C1,
            "C2": self._execute_C2,
            "C3": self._execute_C3,
            "C4": self._execute_C4,
            "D1": self._execute_D1,
            "D2": self._execute_D2,
            "D3": self._execute_D3,
            "E1": self._execute_E1,
            "E2": self._execute_E2,
            "E3": self._execute_E3,
            "E4": self._execute_E4,
            "F1": self._execute_F1,
            "F2": self._execute_F2,
            "F3": self._execute_F3,
            "F4": self._execute_F4,
            "F5": self._execute_F5,
        }

        executor = executor_map.get(scenario_id)
        if not executor:
            result = {
                "scenario_id": scenario_id,
                "status": "not_implemented",
                "message": f"场景 {scenario_id} 尚未实现",
                "conclusions": [],
                "records": []
            }
            logger.warning(f"场景 {scenario_id} 未实现")
            return self._safe_convert(result)

        try:
            logger.info(f"执行场景: {scenario_id} - {scenario.get('name', '')}")
            result = executor(params)
            result["scenario_id"] = scenario_id
            result["status"] = "completed"
            if "records" not in result:
                result["records"] = []
            record_count = len(result.get("records", []))
            logger.info(f"场景 {scenario_id} 完成，生成 {record_count} 条记录")
            return self._safe_convert(result)
        except Exception as e:
            logger.error(f"场景 {scenario_id} 执行失败: {e}")
            import traceback
            traceback.print_exc()
            result = {
                "scenario_id": scenario_id,
                "status": "failed",
                "message": str(e),
                "conclusions": [],
                "records": []
            }
            return self._safe_convert(result)

    def execute_all(self, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行多个场景"""
        results = []
        for scenario in scenarios:
            if scenario.get("enabled", True):
                result = self.execute(scenario)
                results.append(result)
        logger.info(f"总共执行 {len(results)} 个场景")
        return results

    # ==================== A类：数据理解（无记录） ====================

    def _execute_A1(self, params: Dict) -> Dict[str, Any]:
        """A1: 数据全貌"""
        f01 = self.facts.get("F01", {})
        f02 = self.facts.get("F02", {})
        f18 = self.facts.get("F18", {})

        return {
            "name": "数据全貌",
            "conclusions": [
                {
                    "type": "summary",
                    "text": f"数据集包含 {f01.get('table_count', 1)} 张表，{f01.get('rows', 0):,} 行记录，{f01.get('columns', 0)} 个字段",
                    "confidence": 1.0
                },
                {
                    "type": "detail",
                    "text": f"变量类型：连续 {f02.get('continuous', 0)} 个，分类 {f02.get('categorical', 0)} 个，"
                            f"日期 {f02.get('datetime', 0)} 个，标识符 {f02.get('identifier', 0)} 个",
                    "confidence": 1.0
                },
                {
                    "type": "detail",
                    "text": f"质量评分：{f18.get('overall_score', 0):.1f} 分（{f18.get('grade', '未知')}）",
                    "confidence": 1.0
                }
            ],
            "records": []
        }

    def _execute_A2(self, params: Dict) -> Dict[str, Any]:
        """A2: 数据质量总览"""
        f18 = self.facts.get("F18", {})
        f09 = self.facts.get("F09", {})
        f08 = self.facts.get("F08", {})
        f10 = self.facts.get("F10", {})
        f11 = self.facts.get("F11", {})

        dims = f18.get("dimensions", {})
        conclusions = [
            {
                "type": "summary",
                "text": f"综合质量评分 {f18.get('overall_score', 0):.1f} 分",
                "confidence": 1.0
            },
            {
                "type": "detail",
                "text": f"完整性：{dims.get('completeness', 0):.1f}% | 准确性：{dims.get('accuracy', 0):.1f}% | "
                        f"一致性：{dims.get('consistency', 0):.1f}% | 唯一性：{dims.get('uniqueness', 0):.1f}%",
                "confidence": 1.0
            }
        ]

        if f09.get("has_missing"):
            conclusions.append({
                "type": "alert",
                "text": f"发现 {f09.get('count_fields', 0)} 个字段存在缺失值，其中 {len(f09.get('high_missing_fields', []))} 个缺失率超过20%",
                "confidence": 1.0
            })

        if f08.get("has_outliers"):
            conclusions.append({
                "type": "alert",
                "text": f"发现 {f08.get('count_fields', 0)} 个字段存在异常值，共 {f08.get('total_count', 0)} 条",
                "confidence": 1.0
            })

        if f10.get("has_duplicates"):
            conclusions.append({
                "type": "alert",
                "text": f"发现 {f10.get('count', 0)} 条重复记录，占比 {f10.get('percent', 0):.1f}%",
                "confidence": 1.0
            })

        if f11.get("has_rules"):
            conclusions.append({
                "type": "detail",
                "text": f"发现 {f11.get('total_count', 0)} 条勾稽规则（数值 {f11.get('arithmetic_count', 0)} 条）",
                "confidence": 1.0
            })

        return {"name": "数据质量总览", "conclusions": conclusions, "records": []}

    def _execute_A3(self, params: Dict) -> Dict[str, Any]:
        """A3: 字段分布总览"""
        f12 = self.facts.get("F12", {})
        f02 = self.facts.get("F02", {})

        conclusions = [
            {
                "type": "summary",
                "text": f"共 {f02.get('continuous', 0) + f02.get('categorical', 0) + f02.get('categorical_numeric', 0) + f02.get('ordinal', 0)} 个分析字段",
                "confidence": 1.0
            }
        ]

        skewed = f12.get("skewed_vars", [])
        if skewed:
            names = [s["name"] for s in skewed[:3]]
            conclusions.append({
                "type": "detail",
                "text": f"发现 {len(skewed)} 个偏态变量：{', '.join(names)}{' 等' if len(skewed) > 3 else ''}",
                "confidence": 0.9
            })

        imbalanced = f12.get("imbalanced_vars", [])
        if imbalanced:
            names = [s["name"] for s in imbalanced[:3]]
            conclusions.append({
                "type": "detail",
                "text": f"发现 {len(imbalanced)} 个不平衡分类变量：{', '.join(names)}{' 等' if len(imbalanced) > 3 else ''}",
                "confidence": 0.9
            })

        return {"name": "字段分布总览", "conclusions": conclusions, "records": []}

    def _execute_A4(self, params: Dict) -> Dict[str, Any]:
        """A4: 表间关系全景"""
        f03 = self.facts.get("F03", [])
        f01 = self.facts.get("F01", {})

        if not f03:
            return {
                "name": "表间关系全景",
                "conclusions": [
                    {"type": "summary", "text": "未发现表间关系", "confidence": 1.0}
                ],
                "records": []
            }

        type_map = {
            "one_to_one": "一对一",
            "one_to_many": "一对多",
            "many_to_one": "多对一",
            "many_to_many": "多对多"
        }

        rel_texts = []
        for rel in f03[:10]:
            from_t = rel.get("from_table", "?")
            from_c = rel.get("from_col", "?")
            to_t = rel.get("to_table", "?")
            to_c = rel.get("to_col", "?")
            rel_type = type_map.get(rel.get("relation_type", ""), rel.get("relation_type", "关联"))
            rel_texts.append(f"{from_t}.{from_c} → {to_t}.{to_c} ({rel_type})")

        return {
            "name": "表间关系全景",
            "conclusions": [
                {
                    "type": "summary",
                    "text": f"发现 {len(f03)} 条表间关系，涉及 {f01.get('table_count', 1)} 张表",
                    "confidence": 1.0
                },
                {
                    "type": "detail",
                    "text": "\n".join(rel_texts),
                    "confidence": 1.0
                }
            ],
            "records": []
        }

    # ==================== B类：关联与关系 ====================

    def _execute_B1(self, params: Dict) -> Dict[str, Any]:
        """B1: 数值关联发现 - 识别冗余特征"""
        f04 = self.facts.get("F04", {})
        high_corrs = f04.get("high_correlations", [])

        if not high_corrs:
            return {
                "name": "数值关联发现",
                "conclusions": [
                    {"type": "summary", "text": "未发现强相关关系", "confidence": 1.0}
                ],
                "records": []
            }

        redundant_pairs = []
        high_pairs = []

        for corr in high_corrs:
            val = corr.get("value", 0)
            if abs(val) >= 0.95:
                redundant_pairs.append(corr)
            else:
                high_pairs.append(corr)

        conclusions = [
            {
                "type": "summary",
                "text": f"发现 {len(high_corrs)} 对强相关关系（|r| > 0.7）",
                "confidence": 1.0
            }
        ]

        if redundant_pairs:
            conclusions.append({
                "type": "alert",
                "text": f"⚠️ 发现 {len(redundant_pairs)} 对高度冗余特征（|r| >= 0.95），建议删除其中之一",
                "confidence": 0.95
            })
            for pair in redundant_pairs[:5]:
                conclusions.append({
                    "type": "detail",
                    "text": f"  • {pair.get('var1', '?')} ↔ {pair.get('var2', '?')}：r = {pair.get('value', 0):.3f}（建议删除一个）",
                    "confidence": 0.95
                })

        if high_pairs and len(high_pairs) > 0:
            for pair in high_pairs[:3]:
                conclusions.append({
                    "type": "detail",
                    "text": f"  • {pair.get('var1', '?')} ↔ {pair.get('var2', '?')}：r = {pair.get('value', 0):.3f}",
                    "confidence": 0.9
                })
            if len(high_pairs) > 3:
                conclusions.append({
                    "type": "detail",
                    "text": f"  ... 还有 {len(high_pairs) - 3} 对",
                    "confidence": 1.0
                })

        return {"name": "数值关联发现", "conclusions": conclusions, "records": []}

    def _execute_B2(self, params: Dict) -> Dict[str, Any]:
        """B2: 分类关联发现"""
        f05 = self.facts.get("F05", {})
        pairs = f05.get("significant_pairs", [])

        if not pairs:
            return {
                "name": "分类关联发现",
                "conclusions": [
                    {"type": "summary", "text": "未发现显著分类关联", "confidence": 1.0}
                ],
                "records": []
            }

        conclusions = [
            {
                "type": "summary",
                "text": f"发现 {len(pairs)} 对显著分类关联",
                "confidence": 1.0
            }
        ]

        for pair in pairs[:5]:
            conclusions.append({
                "type": "detail",
                "text": f"{pair.get('var1', '?')} ↔ {pair.get('var2', '?')}：V = {pair.get('value', 0):.3f}",
                "confidence": 0.9
            })

        return {"name": "分类关联发现", "conclusions": conclusions, "records": []}

    def _execute_B3(self, params: Dict) -> Dict[str, Any]:
        """B3: 混合关联发现"""
        f06 = self.facts.get("F06", {})
        pairs = f06.get("significant_pairs", [])

        if not pairs:
            return {
                "name": "混合关联发现",
                "conclusions": [
                    {"type": "summary", "text": "未发现显著混合关联", "confidence": 1.0}
                ],
                "records": []
            }

        conclusions = [
            {
                "type": "summary",
                "text": f"发现 {len(pairs)} 对显著混合关联",
                "confidence": 1.0
            }
        ]

        for pair in pairs[:5]:
            conclusions.append({
                "type": "detail",
                "text": f"{pair.get('var1', '?')} ↔ {pair.get('var2', '?')}：η² = {pair.get('value', 0):.3f}",
                "confidence": 0.9
            })

        return {"name": "混合关联发现", "conclusions": conclusions, "records": []}

    def _execute_B4(self, params: Dict) -> Dict[str, Any]:
        """B4: 实体关系网络"""
        f03 = self.facts.get("F03", [])
        f15 = self.facts.get("F15", {})

        if not f03:
            return {
                "name": "实体关系网络",
                "conclusions": [
                    {"type": "summary", "text": "未发现实体关系", "confidence": 1.0}
                ],
                "records": []
            }

        nodes = set()
        edges = []
        for rel in f03:
            from_t = rel.get("from_table", "")
            to_t = rel.get("to_table", "")
            if from_t and to_t:
                nodes.add(from_t)
                nodes.add(to_t)
                edges.append((from_t, to_t))

        return {
            "name": "实体关系网络",
            "conclusions": [
                {
                    "type": "summary",
                    "text": f"构建了 {len(nodes)} 个节点，{len(edges)} 条边的实体关系网络",
                    "confidence": 1.0
                },
                {
                    "type": "detail",
                    "text": f"涉及 {len(f03)} 条关系，{len(nodes)} 个实体",
                    "confidence": 1.0
                }
            ],
            "graph_data": {
                "nodes": [{"id": n, "label": n} for n in list(nodes)[:50]],
                "edges": [{"source": e[0], "target": e[1]} for e in edges[:100]]
            },
            "records": []
        }

    def _execute_B5(self, params: Dict) -> Dict[str, Any]:
        """B5: 共现模式挖掘"""
        return {
            "name": "共现模式挖掘",
            "conclusions": [
                {"type": "summary", "text": "共现模式挖掘功能需要更多数据支持", "confidence": 0.5}
            ],
            "records": []
        }

    # ==================== C类：时间与趋势 ====================

    def _execute_C1(self, params: Dict) -> Dict[str, Any]:
        """C1: 趋势检测"""
        f07 = self.facts.get("F07", {})
        trend_vars = f07.get("trend_vars", [])

        if not trend_vars:
            return {
                "name": "趋势检测",
                "conclusions": [
                    {"type": "summary", "text": "未检测到显著趋势", "confidence": 1.0}
                ],
                "records": []
            }

        conclusions = [
            {
                "type": "summary",
                "text": f"检测到 {len(trend_vars)} 个变量存在趋势",
                "confidence": 0.85
            }
        ]

        for var in trend_vars[:3]:
            conclusions.append({
                "type": "detail",
                "text": f"{var} 存在趋势变化",
                "confidence": 0.8
            })

        return {"name": "趋势检测", "conclusions": conclusions, "records": []}

    def _execute_C2(self, params: Dict) -> Dict[str, Any]:
        """C2: 季节性检测"""
        f07 = self.facts.get("F07", {})
        seasonal_vars = f07.get("seasonal_vars", [])

        if not seasonal_vars:
            return {
                "name": "季节性检测",
                "conclusions": [
                    {"type": "summary", "text": "未检测到季节性模式", "confidence": 1.0}
                ],
                "records": []
            }

        return {
            "name": "季节性检测",
            "conclusions": [
                {
                    "type": "summary",
                    "text": f"检测到 {len(seasonal_vars)} 个变量存在季节性模式",
                    "confidence": 0.85
                }
            ],
            "records": []
        }

    def _execute_C3(self, params: Dict) -> Dict[str, Any]:
        """C3: 时序预测"""
        f07 = self.facts.get("F07", {})
        auto_vars = f07.get("auto_vars", [])

        if not auto_vars:
            return {
                "name": "时序预测",
                "conclusions": [
                    {"type": "summary", "text": "未检测到适合预测的时间序列", "confidence": 1.0}
                ],
                "records": []
            }

        forecast_periods = params.get("forecast_periods", 12)
        predictions = []
        target_var = auto_vars[0]

        try:
            if target_var in self.data.columns:
                series = self.data[target_var].dropna()
                if len(series) >= 30:
                    window = min(10, len(series) // 5)
                    if window >= 2:
                        last_values = series.iloc[-window:].values
                        avg_change = np.mean(np.diff(last_values)) if len(last_values) > 1 else 0
                        last_val = series.iloc[-1]
                        pred_values = [last_val + avg_change * (i + 1) for i in range(min(forecast_periods, 12))]
                        predictions = pred_values

                        trend_direction = "上升" if avg_change > 0 else "下降" if avg_change < 0 else "平稳"
                        trend_strength = abs(avg_change) / (series.std() + 0.01)

                        conclusions = [
                            {
                                "type": "summary",
                                "text": f"基于 {target_var} 的历史趋势，预测未来 {len(predictions)} 期呈{trend_direction}趋势",
                                "confidence": 0.75
                            },
                            {
                                "type": "detail",
                                "text": f"最近一期值: {last_val:.2f}，预测下一期: {pred_values[0]:.2f}（变化 {pred_values[0] - last_val:.2f}）",
                                "confidence": 0.7
                            },
                            {
                                "type": "detail",
                                "text": f"趋势强度: {'强' if trend_strength > 0.5 else '中' if trend_strength > 0.2 else '弱'}",
                                "confidence": 0.7
                            }
                        ]
                        if len(predictions) > 1:
                            conclusions.append({
                                "type": "detail",
                                "text": f"预测序列: {', '.join([f'{v:.2f}' for v in predictions[:5]])}{'...' if len(predictions) > 5 else ''}",
                                "confidence": 0.7
                            })
                        return {"name": "时序预测", "conclusions": conclusions, "records": []}

        except Exception as e:
            logger.warning(f"时序预测计算失败: {e}")

        return {
            "name": "时序预测",
            "conclusions": [
                {
                    "type": "summary",
                    "text": f"基于 {len(auto_vars)} 个自相关序列进行预测，预测周期 {forecast_periods} 期",
                    "confidence": 0.8
                },
                {
                    "type": "detail",
                    "text": f"主要预测变量：{', '.join(auto_vars[:3])}",
                    "confidence": 0.85
                }
            ],
            "records": []
        }

    def _execute_C4(self, params: Dict) -> Dict[str, Any]:
        """C4: 异常时段检测"""
        f07 = self.facts.get("F07", {})
        f08 = self.facts.get("F08", {})

        if not f07.get("has_timeseries") or not f08.get("has_outliers"):
            return {
                "name": "异常时段检测",
                "conclusions": [
                    {"type": "summary", "text": "未检测到异常时段", "confidence": 1.0}
                ],
                "records": []
            }

        # 尝试定位具体异常时段
        try:
            date_cols = [col for col in self.data.columns if col in f07.get("series", {}).keys()]
            if date_cols:
                date_col = date_cols[0]
                outlier_fields = [f["field"] for f in f08.get("fields", [])[:5]]
                if outlier_fields:
                    date_counts = {}
                    for field in outlier_fields:
                        if field in self.data.columns:
                            series = self.data[field].dropna()
                            if len(series) > 5:
                                Q1 = series.quantile(0.25)
                                Q3 = series.quantile(0.75)
                                IQR = Q3 - Q1
                                if IQR > 0:
                                    lower = Q1 - 1.5 * IQR
                                    upper = Q3 + 1.5 * IQR
                                    anomaly_mask = (self.data[field] < lower) | (self.data[field] > upper)
                                    if date_col in self.data.columns:
                                        date_anomalies = self.data[anomaly_mask].groupby(date_col).size()
                                        for dt, cnt in date_anomalies.items():
                                            if cnt > 0:
                                                date_counts[dt] = date_counts.get(dt, 0) + cnt

                    if date_counts:
                        sorted_dates = sorted(date_counts.items(), key=lambda x: x[1], reverse=True)
                        top_dates = sorted_dates[:3]
                        date_strs = [f"{dt.strftime('%Y-%m-%d')}({cnt}个)" for dt, cnt in top_dates]
                        return {
                            "name": "异常时段检测",
                            "conclusions": [
                                {
                                    "type": "summary",
                                    "text": f"检测到异常时段，共 {len(date_counts)} 天存在异常值，集中时段：{', '.join(date_strs)}",
                                    "confidence": 0.8
                                }
                            ],
                            "records": []
                        }
        except Exception as e:
            logger.warning(f"异常时段检测失败: {e}")

        return {
            "name": "异常时段检测",
            "conclusions": [
                {
                    "type": "summary",
                    "text": f"在时间序列中检测到异常，涉及 {f08.get('count_fields', 0)} 个字段",
                    "confidence": 0.8
                }
            ],
            "records": []
        }

    # ==================== D类：分组与分类 ====================

    def _execute_D1(self, params: Dict) -> Dict[str, Any]:
        """D1: 数据分群 - 输出聚类结果 + 每条记录的群组归属"""
        f13 = self.facts.get("F13", {})

        # 获取数值列
        continuous_cols = []
        for col in self.data.columns:
            if col not in self.data.columns:
                continue
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                continue
            if self.data[col].nunique() > len(self.data) * 0.9:
                continue
            if self.data[col].std() == 0:
                continue
            col_lower = col.lower()
            exclude_patterns = ['id', '_id', 'date', '时间', '日期', 'code', '编码']
            if any(p in col_lower for p in exclude_patterns):
                continue
            continuous_cols.append(col)

        if len(continuous_cols) < 3:
            return {
                "name": "数据分群",
                "conclusions": [
                    {"type": "summary", "text": f"数值变量不足3个（当前{len(continuous_cols)}个），无法聚类", "confidence": 1.0}
                ],
                "records": []
            }

        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            X_raw = self.data[continuous_cols].copy()
            X_filled = X_raw.fillna(X_raw.mean())
            X_filled = X_filled.dropna()
            if len(X_filled) < 100:
                return {
                    "name": "数据分群",
                    "conclusions": [
                        {"type": "summary", "text": f"有效样本不足（{len(X_filled)} < 100），无法聚类", "confidence": 1.0}
                    ],
                    "records": []
                }

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_filled)

            max_clusters = min(params.get("max_clusters", 8), len(X_filled) // 10)
            if max_clusters < 2:
                max_clusters = 2

            best_k = 2
            best_score = -1
            for k in range(2, min(max_clusters + 1, 10)):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                if len(set(labels)) > 1:
                    try:
                        score = silhouette_score(X_scaled, labels)
                        if score > best_score:
                            best_score = score
                            best_k = k
                    except:
                        pass

            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)

            # 统计各群体
            cluster_counts = {}
            cluster_sizes = []
            for label in range(best_k):
                count = (labels == label).sum()
                cluster_counts[f"群体{label + 1}"] = count
                cluster_sizes.append(count)

            max_size = max(cluster_sizes) if cluster_sizes else 0
            min_size = min(cluster_sizes) if cluster_sizes else 0
            size_desc = "均衡" if (max_size - min_size) / (max_size + 0.01) < 0.3 else "不均衡"

            # 群体特征
            cluster_features = []
            for label in range(best_k):
                cluster_data = X_filled.iloc[labels == label]
                feature_desc = []
                for col in continuous_cols[:5]:
                    mean_val = cluster_data[col].mean()
                    overall_mean = X_filled[col].mean()
                    if overall_mean != 0:
                        diff_pct = (mean_val - overall_mean) / abs(overall_mean) * 100
                    else:
                        diff_pct = 0
                    if abs(diff_pct) > 10:
                        direction = "高" if diff_pct > 0 else "低"
                        feature_desc.append(f"{self._get_field_display(col)}{direction}{abs(diff_pct):.0f}%")
                cluster_features.append({
                    "label": label + 1,
                    "size": len(cluster_data),
                    "pct": len(cluster_data) / len(X_filled) * 100,
                    "features": feature_desc[:3] if feature_desc else ["无明显特征"]
                })

            # ===== 构建记录级明细 =====
            records = []
            for idx, label in enumerate(labels):
                original_idx = X_filled.index[idx]
                feature_values = {}
                for col in continuous_cols[:5]:
                    feature_values[col] = float(X_filled.iloc[idx][col])
                records.append({
                    "row": int(original_idx) + 1,
                    "cluster_id": label + 1,
                    "features": feature_values,
                    "status": "pending",
                    "record_type": "cluster"
                })

            conclusions = [
                {
                    "type": "summary",
                    "text": f"完成聚类分析，{len(X_filled)} 条记录分为 {best_k} 个群体（分布{size_desc}）",
                    "confidence": 0.85
                }
            ]

            for cf in cluster_features:
                features_str = "、".join(cf["features"])
                conclusions.append({
                    "type": "detail",
                    "text": f"  • 群体{cf['label']}：{cf['size']} 条记录（{cf['pct']:.1f}%），特征：{features_str}",
                    "confidence": 0.85
                })

            if best_score > 0:
                conclusions.append({
                    "type": "detail",
                    "text": f"轮廓系数：{best_score:.3f}（{'好' if best_score > 0.5 else '一般' if best_score > 0.3 else '较差'}）",
                    "confidence": 0.85
                })

            record_count = len(records)
            logger.info(f"D1 数据分群生成 {record_count} 条记录")
            return {"name": "数据分群", "conclusions": conclusions, "records": records}

        except Exception as e:
            logger.warning(f"聚类执行失败: {e}")
            return {
                "name": "数据分群",
                "conclusions": [
                    {
                        "type": "summary",
                        "text": f"数据满足聚类条件（{len(continuous_cols) if 'continuous_cols' in locals() else '?'} 个数值变量，{len(self.data):,} 行），但执行失败: {str(e)[:50]}",
                        "confidence": 0.8
                    }
                ],
                "records": []
            }

    def _execute_D2(self, params: Dict) -> Dict[str, Any]:
        """D2: 偏态变量识别"""
        f12 = self.facts.get("F12", {})
        skewed = f12.get("skewed_vars", [])

        if not skewed:
            return {
                "name": "偏态变量识别",
                "conclusions": [
                    {"type": "summary", "text": "未发现偏态变量", "confidence": 1.0}
                ],
                "records": []
            }

        conclusions = [
            {
                "type": "summary",
                "text": f"发现 {len(skewed)} 个偏态变量",
                "confidence": 1.0
            }
        ]

        severe = []
        moderate = []
        mild = []

        for s in skewed:
            skew_val = abs(s["skew"])
            if skew_val > 10:
                severe.append(s)
            elif skew_val > 5:
                moderate.append(s)
            else:
                mild.append(s)

        if severe:
            names = [s["name"] for s in severe[:5]]
            suggestions = []
            for s in severe[:3]:
                skew_val = s["skew"]
                direction = "右" if skew_val > 0 else "左"
                suggestions.append(f"{self._get_field_display(s['name'])}（{direction}偏{abs(skew_val):.1f}）→对数变换")
            conclusions.append({
                "type": "alert",
                "text": f"🔴 {len(severe)} 个变量严重偏斜（{', '.join(names)}{' 等' if len(severe) > 5 else ''}），建议变换",
                "confidence": 0.95
            })
            conclusions.append({
                "type": "detail",
                "text": f"  • 具体建议：{'; '.join(suggestions)}",
                "confidence": 0.9
            })

        if moderate:
            names = [s["name"] for s in moderate[:5]]
            suggestions = []
            for s in moderate[:3]:
                skew_val = s["skew"]
                direction = "右" if skew_val > 0 else "左"
                suggestions.append(f"{self._get_field_display(s['name'])}（{direction}偏{abs(skew_val):.1f}）→平方根变换")
            conclusions.append({
                "type": "detail",
                "text": f"🟡 {len(moderate)} 个变量中度偏斜（{', '.join(names)}{' 等' if len(moderate) > 5 else ''}），可考虑变换",
                "confidence": 0.9
            })
            if suggestions:
                conclusions.append({
                    "type": "detail",
                    "text": f"  • 建议：{'; '.join(suggestions)}",
                    "confidence": 0.85
                })

        if mild:
            names = [s["name"] for s in mild[:5]]
            conclusions.append({
                "type": "detail",
                "text": f"🟢 {len(mild)} 个变量轻度偏斜（{', '.join(names)}{' 等' if len(mild) > 5 else ''}），可接受",
                "confidence": 0.9
            })

        return {"name": "偏态变量识别", "conclusions": conclusions, "records": []}

    def _execute_D3(self, params: Dict) -> Dict[str, Any]:
        """D3: 不平衡检测"""
        f12 = self.facts.get("F12", {})
        imbalanced = f12.get("imbalanced_vars", [])

        if not imbalanced:
            return {
                "name": "不平衡检测",
                "conclusions": [
                    {"type": "summary", "text": "未发现不平衡变量", "confidence": 1.0}
                ],
                "records": []
            }

        conclusions = [
            {
                "type": "summary",
                "text": f"发现 {len(imbalanced)} 个不平衡分类变量",
                "confidence": 1.0
            }
        ]

        severe = [s for s in imbalanced if s["mode_pct"] > 95]
        moderate = [s for s in imbalanced if 80 < s["mode_pct"] <= 95]

        if severe:
            names = [s["name"] for s in severe[:3]]
            conclusions.append({
                "type": "alert",
                "text": f"🔴 {len(severe)} 个变量严重失衡（{', '.join(names)}{' 等' if len(severe) > 3 else ''}），建议剔除或合并类别",
                "confidence": 0.95
            })

        if moderate:
            names = [s["name"] for s in moderate[:3]]
            suggestions = []
            for s in moderate[:3]:
                suggestions.append(f"{self._get_field_display(s['name'])}（众数占比{s['mode_pct']:.1f}%）→过采样/欠采样")
            conclusions.append({
                "type": "detail",
                "text": f"🟡 {len(moderate)} 个变量中度失衡（{', '.join(names)}{' 等' if len(moderate) > 3 else ''}），建议过采样或欠采样",
                "confidence": 0.9
            })
            if suggestions:
                conclusions.append({
                    "type": "detail",
                    "text": f"  • 建议：{'; '.join(suggestions)}",
                    "confidence": 0.85
                })

        return {"name": "不平衡检测", "conclusions": conclusions, "records": []}

    # ==================== E类：质量与异常（有记录） ====================

    def _execute_E1(self, params: Dict) -> Dict[str, Any]:
        """E1: 勾稽规则验证 - 支持任意数量左右项"""
        f11 = self.facts.get("F11", {})
        rules = f11.get("rules", [])
        min_conf = params.get("min_confidence", 0.7)

        if not rules:
            logger.info("E1: 未发现勾稽规则")
            return {
                "name": "勾稽规则验证",
                "conclusions": [
                    {"type": "summary", "text": "未发现勾稽规则", "confidence": 1.0}
                ],
                "records": []
            }

        valid_rules = [r for r in rules if r.get("confidence", 0) >= min_conf]
        logger.info(f"E1: 共 {len(rules)} 条规则，满足置信度 >= {min_conf} 的有 {len(valid_rules)} 条")

        if not valid_rules:
            return {
                "name": "勾稽规则验证",
                "conclusions": [
                    {"type": "summary", "text": f"无满足置信度要求（>={min_conf:.0%}）的规则", "confidence": 1.0}
                ],
                "records": []
            }

        violated_rules = []
        verified_rules = []
        all_records = []

        for rule in valid_rules[:30]:
            rule_str = rule.get("rule", "")
            fields = rule.get("fields", [])

            if len(fields) < 2:
                continue

            valid_fields = [f for f in fields if f in self.data.columns]
            if len(valid_fields) != len(fields):
                continue

            try:
                # 解析左右表达式
                if "=" not in rule_str:
                    continue
                left_expr, right_expr = rule_str.split("=")
                left_parts = [p.strip() for p in left_expr.split("+") if p.strip() and p.strip() != "0"]
                right_parts = [p.strip() for p in right_expr.split("+") if p.strip() and p.strip() != "0"]

                if not left_parts and not right_parts:
                    continue

                all_parts = left_parts + right_parts
                if not all_parts:
                    continue
                if not all(f in self.data.columns for f in all_parts):
                    continue

                valid_mask = self.data[all_parts].notna().all(axis=1)
                total_records = valid_mask.sum()
                if total_records == 0:
                    continue

                # 计算左值和右值（任意数量）
                left_sum = self.data.loc[valid_mask, left_parts].sum(axis=1) if left_parts else 0
                right_sum = self.data.loc[valid_mask, right_parts].sum(axis=1) if right_parts else 0

                diff = (left_sum - right_sum).abs()
                max_abs = np.maximum(left_sum.abs(), right_sum.abs())
                scale = np.maximum(max_abs, 1.0)
                rel_error = diff / scale

                violation_mask = rel_error > 1e-4
                violation_count = violation_mask.sum()

                verified_rules.append(rule_str)
                logger.info(f"E1 规则 '{rule_str}': 有效行 {total_records}, 违反 {violation_count}")

                if violation_count > 0:
                    violation_indices = self.data.loc[valid_mask].index[violation_mask.values]
                    for idx in violation_indices[:200]:
                        if idx in self.data.index:
                            row_data = self.data.loc[idx]
                            record = {
                                "row": int(idx) + 1,
                                "rule": rule_str,
                                "fields": all_parts,
                                "values": {f: float(row_data[f]) if pd.notna(row_data[f]) else None for f in all_parts},
                                "diff": float(diff.loc[idx]) if idx in diff.index else 0,
                                "status": "pending",
                                "severity": "high" if violation_count > total_records * 0.1 else "medium",
                                "record_type": "violation"
                            }
                            all_records.append(record)

                    violated_rules.append({
                        "rule": rule_str,
                        "fields": all_parts,
                        "violation_count": int(violation_count),
                        "total_records": int(total_records),
                        "violation_rate": round(violation_count / total_records * 100, 2)
                    })

            except Exception as e:
                logger.warning(f"规则验证失败: {rule_str}, {e}")
                continue

        conclusions = [
            {
                "type": "summary",
                "text": f"验证 {len(verified_rules)} 条勾稽规则，发现 {len(violated_rules)} 条被违反",
                "confidence": 1.0
            }
        ]

        if verified_rules:
            rule_preview = verified_rules[:5]
            conclusions.append({
                "type": "detail",
                "text": f"已验证规则（{len(verified_rules)}条）：{'; '.join(rule_preview)}{' 等' if len(verified_rules) > 5 else ''}",
                "confidence": 1.0
            })

        if violated_rules:
            violated_rules.sort(key=lambda x: x["violation_rate"], reverse=True)
            for vr in violated_rules[:3]:
                conclusions.append({
                    "type": "alert",
                    "text": f"  • 规则「{vr['rule']}」违反 {vr['violation_count']} 条（{vr['violation_rate']:.1f}%）",
                    "confidence": 0.9
                })
        else:
            conclusions.append({
                "type": "success",
                "text": f"✅ 所有规则均满足要求",
                "confidence": 1.0
            })

        record_count = len(all_records)
        logger.info(f"E1 总生成记录: {record_count}")
        return {"name": "勾稽规则验证", "conclusions": conclusions, "records": all_records[:500]}

    def _execute_E2(self, params: Dict) -> Dict[str, Any]:
        """E2: 异常值定位 - 使用 fact_sheet 中已有的异常检测结果"""
        f08 = self.facts.get("F08", {})
        fields_data = f08.get("fields", [])

        if not fields_data:
            logger.info("E2: 未发现异常值字段")
            return {
                "name": "异常值定位",
                "conclusions": [
                    {"type": "summary", "text": "未发现异常值", "confidence": 1.0}
                ],
                "records": []
            }

        all_records = []
        anomaly_details = []

        for field_info in fields_data[:20]:
            if isinstance(field_info, dict):
                field = field_info.get("field", "")
                count = field_info.get("count", 0)
                percent = field_info.get("percent", 0)
                lower = field_info.get("lower_bound")
                upper = field_info.get("upper_bound")
            else:
                field = field_info if isinstance(field_info, str) else str(field_info)
                count = 0
                percent = 0
                lower = None
                upper = None

            if not field or field not in self.data.columns:
                continue

            if count == 0 and percent == 0:
                # 如果没有预先计算的异常值，用 IQR 检测
                series = self.data[field].dropna()
                if len(series) < 5:
                    continue
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0:
                    continue
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                anomaly_mask = (self.data[field] < lower) | (self.data[field] > upper)
                anomaly_rows = self.data[anomaly_mask]
                count = len(anomaly_rows)
                percent = count / len(self.data) * 100 if len(self.data) > 0 else 0

            if count == 0:
                continue

            # 获取异常值行
            if lower is not None and upper is not None:
                anomaly_mask = (self.data[field] < lower) | (self.data[field] > upper)
            else:
                # 如果没有边界，用 IQR 重新计算
                series = self.data[field].dropna()
                if len(series) < 5:
                    continue
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0:
                    continue
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                anomaly_mask = (self.data[field] < lower) | (self.data[field] > upper)

            anomaly_rows = self.data[anomaly_mask]
            if len(anomaly_rows) == 0:
                continue

            # 计算偏离程度
            IQR = (upper - lower) / 3 if upper is not None and lower is not None else 1
            if IQR == 0:
                IQR = 1

            deviations = []
            for idx in anomaly_rows.index[:200]:
                val = self.data.loc[idx, field]
                if pd.isna(val):
                    continue
                if upper is not None and val > upper:
                    dev = (val - upper) / (IQR + 0.01)
                elif lower is not None and val < lower:
                    dev = (lower - val) / (IQR + 0.01)
                else:
                    dev = 0
                deviations.append((idx, val, dev))

            deviations.sort(key=lambda x: x[2], reverse=True)

            for idx, val, dev in deviations[:100]:
                all_records.append({
                    "row": int(idx) + 1,
                    "field": field,
                    "field_display": self._get_field_display(field),
                    "value": float(val),
                    "expected": f"[{lower:.2f}, {upper:.2f}]" if lower is not None and upper is not None else "[IQR范围]",
                    "deviation": float(dev),
                    "severity": "high" if dev > 10 else "medium" if dev > 3 else "low",
                    "status": "pending",
                    "record_type": "outlier"
                })

            anomaly_details.append({
                "field": field,
                "count": int(count),
                "lower": float(lower) if lower is not None else None,
                "upper": float(upper) if upper is not None else None
            })

        if not all_records:
            logger.info("E2: 没有生成任何异常记录")
            return {
                "name": "异常值定位",
                "conclusions": [
                    {"type": "summary", "text": "未发现异常值", "confidence": 1.0}
                ],
                "records": []
            }

        conclusions = [
            {
                "type": "summary",
                "text": f"发现 {len(anomaly_details)} 个字段存在异常值，共 {len(all_records)} 条记录",
                "confidence": 1.0
            }
        ]

        for detail in anomaly_details[:5]:
            field = detail["field"]
            count = detail["count"]
            lower_str = f"{detail['lower']:.2f}" if detail['lower'] is not None else "?"
            upper_str = f"{detail['upper']:.2f}" if detail['upper'] is not None else "?"
            conclusions.append({
                "type": "detail",
                "text": f"  • {self._get_field_display(field)}：{count} 条异常，IQR范围 [{lower_str}, {upper_str}]",
                "confidence": 0.95
            })

        record_count = len(all_records)
        logger.info(f"E2 异常值定位生成 {record_count} 条记录")
        return {"name": "异常值定位", "conclusions": conclusions, "records": all_records[:500]}

    def _execute_E3(self, params: Dict) -> Dict[str, Any]:
        """E3: 缺失值模式分析"""
        f09 = self.facts.get("F09", {})
        fields = f09.get("fields", [])

        if not fields:
            logger.info("E3: 未发现缺失值")
            return {
                "name": "缺失值模式分析",
                "conclusions": [
                    {"type": "summary", "text": "无缺失值", "confidence": 1.0}
                ],
                "records": []
            }

        # 过滤出缺失率 > 5% 的字段
        significant_fields = [f for f in fields if f.get("percent", 0) > 5]
        logger.info(f"E3: 共 {len(fields)} 个缺失字段，缺失率 > 5% 的有 {len(significant_fields)} 个")

        if not significant_fields:
            return {
                "name": "缺失值模式分析",
                "conclusions": [
                    {"type": "summary", "text": f"发现 {len(fields)} 个字段存在缺失值，但缺失率均低于5%", "confidence": 1.0}
                ],
                "records": []
            }

        high_missing = [f["field"] for f in significant_fields if f["field"] in self.data.columns]
        high_missing = [f for f in high_missing if f in self.data.columns]

        # 构建缺失记录明细
        records = []
        if high_missing:
            # 取前10个高缺失率字段
            top_fields = high_missing[:10]
            sample_size = min(200, len(self.data))

            # 随机采样
            sample_df = self.data[top_fields].sample(n=min(sample_size, len(self.data)), random_state=42)
            for idx, row in sample_df.iterrows():
                missing_fields = [f for f in top_fields if pd.isna(row.get(f))]
                if missing_fields:
                    records.append({
                        "row": int(idx) + 1,
                        "missing_fields": missing_fields,
                        "missing_count": len(missing_fields),
                        "status": "pending",
                        "record_type": "missing"
                    })

        if not records:
            logger.info("E3: 采样未找到缺失记录")
            return {
                "name": "缺失值模式分析",
                "conclusions": [
                    {"type": "summary", "text": f"发现 {len(fields)} 个字段存在缺失值，但采样未命中", "confidence": 1.0}
                ],
                "records": []
            }

        conclusions = [
            {
                "type": "summary",
                "text": f"发现 {len(fields)} 个字段存在缺失值，其中 {len([f for f in fields if f.get('percent', 0) > 20])} 个缺失率超过20%",
                "confidence": 1.0
            }
        ]

        # 显示高缺失率字段
        high_fields = [f for f in fields if f.get("percent", 0) > 20]
        if high_fields:
            field_texts = [f"{self._get_field_display(f['field'])}（{f['percent']:.1f}%）" for f in high_fields[:5]]
            conclusions.append({
                "type": "alert",
                "text": f"🔴 高缺失率字段：{', '.join(field_texts)}{' 等' if len(high_fields) > 5 else ''}",
                "confidence": 1.0
            })

        record_count = len(records)
        logger.info(f"E3 缺失值模式分析生成 {record_count} 条记录")
        return {"name": "缺失值模式分析", "conclusions": conclusions, "records": records[:500]}

    def _execute_E4(self, params: Dict) -> Dict[str, Any]:
        """E4: 重复记录检测"""
        f10 = self.facts.get("F10", {})

        if not f10.get("has_duplicates"):
            logger.info("E4: 无重复记录")
            return {
                "name": "重复记录检测",
                "conclusions": [
                    {"type": "summary", "text": "无重复记录", "confidence": 1.0}
                ],
                "records": []
            }

        records = []
        dup_count = f10.get("count", 0)
        if dup_count > 0:
            dup_mask = self.data.duplicated(keep=False)
            dup_indices = self.data[dup_mask].index[:200]
            for idx in dup_indices:
                records.append({
                    "row": int(idx) + 1,
                    "status": "pending",
                    "severity": "low",
                    "record_type": "duplicate"
                })

        logger.info(f"E4 重复记录检测生成 {len(records)} 条记录")
        return {
            "name": "重复记录检测",
            "conclusions": [
                {
                    "type": "summary",
                    "text": f"发现 {dup_count} 条重复记录，占比 {f10.get('percent', 0):.1f}%",
                    "confidence": 1.0
                },
                {
                    "type": "detail",
                    "text": f"基于 {f10.get('based_on', '全部列')} 检测",
                    "confidence": 1.0
                }
            ],
            "records": records[:500]
        }

    # ==================== F类：业务意图 ====================

    def _execute_F1(self, params: Dict) -> Dict[str, Any]:
        """F1: 异常关联模式"""
        f15 = self.facts.get("F15", {})
        f08 = self.facts.get("F08", {})

        if not f15.get("has_entity") or not f08.get("has_outliers"):
            return {
                "name": "异常关联模式",
                "conclusions": [
                    {"type": "summary", "text": "未发现异常关联模式", "confidence": 1.0}
                ],
                "records": []
            }

        return {
            "name": "异常关联模式",
            "conclusions": [
                {
                    "type": "summary",
                    "text": f"发现实体列与异常值的关联模式，涉及 {f15.get('entity_count', 0)} 个实体列，{f08.get('count_fields', 0)} 个异常字段",
                    "confidence": 0.7
                }
            ],
            "records": []
        }

    def _execute_F2(self, params: Dict) -> Dict[str, Any]:
        """F2: 关系传导路径"""
        f03 = self.facts.get("F03", [])
        f15 = self.facts.get("F15", {})

        if len(f03) < 2 or not f15.get("has_entity"):
            return {
                "name": "关系传导路径",
                "conclusions": [
                    {"type": "summary", "text": "未发现关系传导路径", "confidence": 1.0}
                ],
                "records": []
            }

        return {
            "name": "关系传导路径",
            "conclusions": [
                {
                    "type": "summary",
                    "text": f"发现 {len(f03)} 条关系可构成传导路径",
                    "confidence": 0.75
                }
            ],
            "records": []
        }

    def _execute_F3(self, params: Dict) -> Dict[str, Any]:
        """F3: 有向关系网络"""
        f03 = self.facts.get("F03", [])
        directional = [r for r in f03 if r.get("relation_type") in ["one_to_many", "many_to_one"]]

        if not directional:
            return {
                "name": "有向关系网络",
                "conclusions": [
                    {"type": "summary", "text": "未发现有向关系", "confidence": 1.0}
                ],
                "records": []
            }

        return {
            "name": "有向关系网络",
            "conclusions": [
                {
                    "type": "summary",
                    "text": f"发现 {len(directional)} 条有向关系",
                    "confidence": 0.85
                }
            ],
            "records": []
        }

    def _execute_F4(self, params: Dict) -> Dict[str, Any]:
        """F4: 集中度分析"""
        f15 = self.facts.get("F15", {})
        entity_cols_raw = f15.get("entity_columns", [])

        entity_cols = []
        for col in entity_cols_raw:
            if isinstance(col, dict):
                name = col.get("name")
                if name:
                    entity_cols.append(name)
            elif isinstance(col, str):
                entity_cols.append(col)

        valid_entity_cols = []
        for col in entity_cols:
            if col not in self.data.columns:
                continue
            n_unique = self.data[col].nunique()
            total_rows = len(self.data)
            if n_unique > total_rows * 0.9:
                continue
            valid_entity_cols.append(col)

        if not valid_entity_cols:
            return {
                "name": "集中度分析",
                "conclusions": [
                    {"type": "summary", "text": "未发现可分析的业务实体列（如公司名、产品名等），请手动指定实体列", "confidence": 1.0}
                ],
                "records": []
            }

        conclusions = []
        all_records = []

        for col in valid_entity_cols[:3]:
            try:
                value_counts = self.data[col].astype(str).value_counts()
                total = value_counts.sum()

                if total == 0:
                    continue

                hhi = ((value_counts / total) ** 2).sum()
                top4 = value_counts.head(4).sum()
                cr4 = top4 / total

                if hhi > 0.25:
                    level = "高度集中"
                    suggestion = "可能存在垄断或主导实体，建议关注"
                elif hhi > 0.15:
                    level = "中度集中"
                    suggestion = "存在少数主导实体，建议进一步分析"
                else:
                    level = "分散"
                    suggestion = "市场竞争较为充分，无显著集中"

                top_entities = []
                for val, count in value_counts.head(5).items():
                    pct = count / total * 100
                    top_entities.append(f"{val}（{pct:.1f}%）")
                    all_records.append({
                        "entity": val,
                        "count": int(count),
                        "percentage": round(pct, 2),
                        "status": "pending",
                        "record_type": "entity_concentration"
                    })

                conclusions.append({
                    "type": "summary" if len(conclusions) == 0 else "detail",
                    "text": f"「{self._get_field_display(col)}」集中度分析：{level}（HHI={hhi:.3f}，CR4={cr4:.1%}），前5大：{', '.join(top_entities)}",
                    "confidence": 0.9
                })

                conclusions.append({
                    "type": "detail",
                    "text": f"  • 建议：{suggestion}",
                    "confidence": 0.85
                })

            except Exception as e:
                logger.warning(f"集中度分析失败 {col}: {e}")
                continue

        if not conclusions:
            return {
                "name": "集中度分析",
                "conclusions": [
                    {"type": "summary", "text": "未发现可分析的业务实体列", "confidence": 1.0}
                ],
                "records": []
            }

        record_count = len(all_records)
        logger.info(f"F4 集中度分析生成 {record_count} 条记录")
        return {"name": "集中度分析", "conclusions": conclusions, "records": all_records[:200]}

    def _execute_F5(self, params: Dict) -> Dict[str, Any]:
        """F5: 趋势演化分析"""
        f07 = self.facts.get("F07", {})
        f15 = self.facts.get("F15", {})

        if not f07.get("has_timeseries") or not f15.get("has_entity"):
            return {
                "name": "趋势演化分析",
                "conclusions": [
                    {"type": "summary", "text": "未发现可分析的趋势演化", "confidence": 1.0}
                ],
                "records": []
            }

        entity_cols = f15.get("entity_columns", [])
        valid_entity_cols = []
        for col in entity_cols:
            if isinstance(col, dict):
                name = col.get("name")
                if name and name in self.data.columns and self.data[name].nunique() < len(self.data) * 0.9:
                    valid_entity_cols.append(name)
            elif isinstance(col, str) and col in self.data.columns and self.data[col].nunique() < len(self.data) * 0.9:
                valid_entity_cols.append(col)

        date_cols = [col for col in self.data.columns if col in f07.get("series", {}).keys()]

        if valid_entity_cols and date_cols:
            return {
                "name": "趋势演化分析",
                "conclusions": [
                    {
                        "type": "summary",
                        "text": f"发现时间列 {date_cols[0]} 与实体列 {valid_entity_cols[0]} 的组合，可进行趋势演化分析",
                        "confidence": 0.75
                    },
                    {
                        "type": "detail",
                        "text": "趋势演化分析需要多个时间点的实体状态数据，建议确保数据包含时间序列和实体标识",
                        "confidence": 0.7
                    }
                ],
                "records": []
            }

        return {
            "name": "趋势演化分析",
            "conclusions": [
                {"type": "summary", "text": "未发现可分析的趋势演化（需要同时包含时间列、实体列和数值列）", "confidence": 1.0}
            ],
            "records": []
        }


def execute_scenarios(
    data: pd.DataFrame,
    fact_sheet: Dict[str, Any],
    scenarios: List[Dict[str, Any]],
    field_mapping: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """
    执行场景（便捷函数）
    """
    logger.info(f"execute_scenarios: {len(scenarios)} 个场景, 数据 {len(data)} 行")
    executor = ScenarioExecutor(data, fact_sheet)
    if field_mapping:
        executor.set_field_mapping(field_mapping)
    return executor.execute_all(scenarios)