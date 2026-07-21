"""
场景推导引擎
基于技术事实清单推导候选场景
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


class ScenarioEngine:
    """场景推导引擎"""

    # 场景定义
    SCENARIO_DEFS = {
        # ===== A类：数据理解类（必出） =====
        "A1": {
            "id": "A1",
            "name": "数据全貌",
            "category": "数据理解",
            "description": "数据集的整体规模和结构概览",
            "default_enabled": True,
            "priority": 100,
            "requires_data": False
        },
        "A2": {
            "id": "A2",
            "name": "数据质量总览",
            "category": "数据理解",
            "description": "数据完整性、准确性、一致性、唯一性评分",
            "default_enabled": True,
            "priority": 90,
            "requires_data": False
        },
        "A3": {
            "id": "A3",
            "name": "字段分布总览",
            "category": "数据理解",
            "description": "各字段的数据分布和统计特征",
            "default_enabled": True,
            "priority": 80,
            "requires_data": False
        },
        "A4": {
            "id": "A4",
            "name": "表间关系全景",
            "category": "数据理解",
            "description": "多表之间的关联关系图",
            "default_enabled": True,
            "priority": 70,
            "requires_data": False
        },

        # ===== B类：关联与关系类 =====
        "B1": {
            "id": "B1",
            "name": "数值关联发现",
            "category": "关联与关系",
            "description": "发现数值变量之间的强相关关系，识别冗余特征",
            "default_enabled": True,
            "priority": 60,
            "requires_data": False,
            "min_trigger": {"high_correlations": 3}
        },
        "B2": {
            "id": "B2",
            "name": "分类关联发现",
            "category": "关联与关系",
            "description": "发现分类变量之间的显著关联",
            "default_enabled": True,
            "priority": 50,
            "requires_data": False,
            "min_trigger": {"categorical_significant": 2}
        },
        "B3": {
            "id": "B3",
            "name": "混合关联发现",
            "category": "关联与关系",
            "description": "发现数值变量与分类变量之间的关联",
            "default_enabled": True,
            "priority": 50,
            "requires_data": False,
            "min_trigger": {"eta_significant": 2}
        },
        "B4": {
            "id": "B4",
            "name": "实体关系网络",
            "category": "关联与关系",
            "description": "构建实体之间的关联关系网络",
            "default_enabled": True,
            "priority": 55,
            "requires_data": False,
            "min_trigger": {"relationships": 2, "entity_cols": 1}
        },
        "B5": {
            "id": "B5",
            "name": "共现模式挖掘",
            "category": "关联与关系",
            "description": "发现频繁共现的实体对",
            "default_enabled": False,
            "priority": 40,
            "requires_data": True,
            "min_trigger": {"entity_cols": 2, "rows": 100}
        },

        # ===== C类：时间与趋势类 =====
        "C1": {
            "id": "C1",
            "name": "趋势检测",
            "category": "时间与趋势",
            "description": "检测数据中的上升/下降趋势",
            "default_enabled": True,
            "priority": 65,
            "requires_data": False,
            "min_trigger": {"has_trend": True}
        },
        "C2": {
            "id": "C2",
            "name": "季节性检测",
            "category": "时间与趋势",
            "description": "检测数据中的周期性波动",
            "default_enabled": True,
            "priority": 55,
            "requires_data": False,
            "min_trigger": {"has_seasonality": True}
        },
        "C3": {
            "id": "C3",
            "name": "时序预测",
            "category": "时间与趋势",
            "description": "基于历史数据预测未来趋势",
            "default_enabled": True,
            "priority": 70,
            "requires_data": True,
            "min_trigger": {"has_autocorrelation": True, "min_samples": 30}
        },
        "C4": {
            "id": "C4",
            "name": "异常时段检测",
            "category": "时间与趋势",
            "description": "检测时间轴上的异常点",
            "default_enabled": True,
            "priority": 45,
            "requires_data": True,
            "min_trigger": {"has_timeseries": True, "has_outliers": True}
        },

        # ===== D类：分组与分类类 =====
        "D1": {
            "id": "D1",
            "name": "数据分群",
            "category": "分组与分类",
            "description": "将数据自动划分为不同的群体",
            "default_enabled": True,
            "priority": 60,
            "requires_data": True,
            "min_trigger": {"clustering_ready": True}
        },
        "D2": {
            "id": "D2",
            "name": "偏态变量识别",
            "category": "分组与分类",
            "description": "识别分布严重偏斜的变量",
            "default_enabled": True,
            "priority": 40,
            "requires_data": False,
            "min_trigger": {"skewed_count": 1}
        },
        "D3": {
            "id": "D3",
            "name": "不平衡检测",
            "category": "分组与分类",
            "description": "检测类别分布严重失衡的分类变量，给出处理建议",
            "default_enabled": True,
            "priority": 40,
            "requires_data": False,
            "min_trigger": {"imbalanced_count": 1}
        },

        # ===== E类：质量与异常类 =====
        "E1": {
            "id": "E1",
            "name": "勾稽规则验证",
            "category": "质量与异常",
            "description": "验证数据中的勾稽关系，定位违反记录",
            "default_enabled": True,
            "priority": 50,
            "requires_data": True,
            "min_trigger": {"has_rules": True}
        },
        "E2": {
            "id": "E2",
            "name": "异常值定位",
            "category": "质量与异常",
            "description": "定位各字段的异常值记录",
            "default_enabled": True,
            "priority": 45,
            "requires_data": True,
            "min_trigger": {"has_outliers": True}
        },
        "E3": {
            "id": "E3",
            "name": "缺失值模式分析",
            "category": "质量与异常",
            "description": "分析缺失值的分布和关联模式",
            "default_enabled": True,
            "priority": 40,
            "requires_data": True,
            "min_trigger": {"has_missing": True}
        },
        "E4": {
            "id": "E4",
            "name": "重复记录检测",
            "category": "质量与异常",
            "description": "检测数据中的重复记录",
            "default_enabled": True,
            "priority": 35,
            "requires_data": False,
            "min_trigger": {"has_duplicates": True}
        },

        # ===== F类：业务意图类 =====
        "F1": {
            "id": "F1",
            "name": "异常关联模式",
            "category": "业务意图",
            "description": "发现共现实体与数值异常的关联",
            "default_enabled": False,
            "priority": 30,
            "requires_data": True,
            "min_trigger": {"entity_cols": 2, "has_outliers": True}
        },
        "F2": {
            "id": "F2",
            "name": "关系传导路径",
            "category": "业务意图",
            "description": "发现实体间的关系传导路径",
            "default_enabled": False,
            "priority": 30,
            "requires_data": False,
            "min_trigger": {"relationships": 2, "entity_cols": 2}
        },
        "F3": {
            "id": "F3",
            "name": "有向关系网络",
            "category": "业务意图",
            "description": "构建有向实体关系网络",
            "default_enabled": False,
            "priority": 30,
            "requires_data": False,
            "min_trigger": {"relationships": 2, "has_directional": True}
        },
        "F4": {
            "id": "F4",
            "name": "集中度分析",
            "category": "业务意图",
            "description": "分析实体或数值的集中程度",
            "default_enabled": False,
            "priority": 25,
            "requires_data": False,
            "min_trigger": {"entity_cols": 1, "rows": 50}
        },
        "F5": {
            "id": "F5",
            "name": "趋势演化分析",
            "category": "业务意图",
            "description": "分析实体关系随时间的演化",
            "default_enabled": False,
            "priority": 25,
            "requires_data": True,
            "min_trigger": {"has_timeseries": True, "entity_cols": 1}
        }
    }

    def __init__(self, tech_fact_sheet: Dict[str, Any]):
        """
        初始化场景引擎

        参数:
        - tech_fact_sheet: 技术事实清单
        """
        self.tech_fact_sheet = tech_fact_sheet
        self.facts = tech_fact_sheet.get("facts", {})
        self.derived_scenarios = []

    def derive(self) -> List[Dict[str, Any]]:
        """
        推导候选场景

        返回:
        - 候选场景列表
        """
        self.derived_scenarios = []

        # 遍历所有场景定义
        for scenario_id, definition in self.SCENARIO_DEFS.items():
            if self._check_trigger(scenario_id, definition):
                self.derived_scenarios.append({
                    "id": scenario_id,
                    "name": definition["name"],
                    "category": definition["category"],
                    "description": definition["description"],
                    "enabled": definition.get("default_enabled", True),
                    "priority": definition.get("priority", 50),
                    "trigger_basis": self._get_trigger_basis(scenario_id),
                    "params": self._get_default_params(scenario_id),
                    "requires_data": definition.get("requires_data", False)
                })

        # 按优先级排序
        self.derived_scenarios.sort(key=lambda x: x["priority"], reverse=True)

        return self.derived_scenarios

    def _check_trigger(self, scenario_id: str, definition: Dict) -> bool:
        """检查场景是否满足触发条件"""
        min_trigger = definition.get("min_trigger", {})

        # 没有触发条件，默认触发
        if not min_trigger:
            return True

        # A类场景：始终触发
        if scenario_id.startswith("A"):
            return True

        # 检查每个触发条件
        for key, threshold in min_trigger.items():
            value = self._get_fact_value(key)
            if value is None:
                return False
            if isinstance(value, (int, float)):
                if value < threshold:
                    return False
            elif isinstance(value, bool):
                if not value:
                    return False
            elif isinstance(value, list):
                if len(value) < threshold:
                    return False

        return True

    def _get_fact_value(self, key: str) -> Any:
        """从技术事实中获取值"""
        mappings = {
            "rows": lambda: self.facts.get("F01", {}).get("rows", 0),
            "table_count": lambda: self.facts.get("F01", {}).get("table_count", 1),
            "high_correlations": lambda: self.facts.get("F04", {}).get("count_high", 0),
            "categorical_significant": lambda: self.facts.get("F05", {}).get("count_significant", 0),
            "eta_significant": lambda: self.facts.get("F06", {}).get("count_significant", 0),
            "relationships": lambda: len(self.facts.get("F03", [])),
            "has_trend": lambda: self.facts.get("F07", {}).get("has_trend", False),
            "has_seasonality": lambda: self.facts.get("F07", {}).get("has_seasonality", False),
            "has_autocorrelation": lambda: self.facts.get("F07", {}).get("has_autocorrelation", False),
            "min_samples": lambda: 30,
            "has_timeseries": lambda: self.facts.get("F07", {}).get("count", 0) > 0,
            "has_outliers": lambda: self.facts.get("F08", {}).get("has_outliers", False),
            "clustering_ready": lambda: self.facts.get("F13", {}).get("is_ready", False),
            "skewed_count": lambda: self.facts.get("F12", {}).get("skewed_count", 0),
            "imbalanced_count": lambda: self.facts.get("F12", {}).get("imbalanced_count", 0),
            "has_rules": lambda: self.facts.get("F11", {}).get("has_rules", False),
            "has_missing": lambda: self.facts.get("F09", {}).get("has_missing", False),
            "has_duplicates": lambda: self.facts.get("F10", {}).get("has_duplicates", False),
            "entity_cols": lambda: self.facts.get("F15", {}).get("entity_count", 0),
            "has_directional": lambda: len([r for r in self.facts.get("F03", []) if r.get("relation_type") in ["one_to_many", "many_to_one"]]) > 0,
        }

        if key in mappings:
            return mappings[key]()
        return None

    def _get_trigger_basis(self, scenario_id: str) -> str:
        """获取场景的触发依据描述"""
        basis_map = {
            "A1": "数据已加载，自动生成数据全貌",
            "A2": "数据已加载，自动生成质量总览",
            "A3": "数据已加载，自动生成字段分布",
            "A4": f"发现 {self._get_fact_value('relationships')} 条表间关系",
            "B1": f"发现 {self._get_fact_value('high_correlations')} 对强相关关系",
            "B2": f"发现 {self._get_fact_value('categorical_significant')} 对显著分类关联",
            "B3": f"发现 {self._get_fact_value('eta_significant')} 对显著混合关联",
            "B4": f"发现 {self._get_fact_value('relationships')} 条关系，{self._get_fact_value('entity_cols')} 个实体列",
            "B5": f"发现 {self._get_fact_value('entity_cols')} 个实体列，{self._get_fact_value('rows')} 条记录",
            "C1": "检测到数据趋势",
            "C2": "检测到季节性模式",
            "C3": "检测到自相关性，适合时序预测",
            "C4": "检测到时间序列数据且存在异常值",
            "D1": f"满足聚类条件（{self._get_fact_value('numeric_count')}个数值变量，{self._get_fact_value('rows')}行）",
            "D2": f"发现 {self._get_fact_value('skewed_count')} 个偏态变量",
            "D3": f"发现 {self._get_fact_value('imbalanced_count')} 个不平衡变量",
            "E1": f"发现 {self._get_fact_value('has_rules')} 条勾稽规则",
            "E2": "检测到异常值",
            "E3": "检测到缺失值",
            "E4": "检测到重复记录",
            "F1": f"发现 {self._get_fact_value('entity_cols')} 个实体列且存在异常值",
            "F2": f"发现 {self._get_fact_value('relationships')} 条关系，{self._get_fact_value('entity_cols')} 个实体列",
            "F3": f"发现 {self._get_fact_value('relationships')} 条有向关系",
            "F4": f"发现 {self._get_fact_value('entity_cols')} 个实体列，{self._get_fact_value('rows')} 条记录",
            "F5": f"发现时间序列数据和 {self._get_fact_value('entity_cols')} 个实体列",
        }
        return basis_map.get(scenario_id, "技术特征满足触发条件")

    def _get_default_params(self, scenario_id: str) -> Dict[str, Any]:
        """获取场景的默认参数"""
        params_map = {
            "B1": {"min_correlation": 0.7, "top_k": 10},
            "B2": {"min_cramers_v": 0.3, "top_k": 10},
            "B3": {"min_eta": 0.1, "top_k": 10},
            "B4": {"min_edges": 3},
            "C3": {"forecast_periods": 12, "confidence": 0.95},
            "C4": {"window": 30, "threshold": 2.5},
            "D1": {"n_clusters": None, "max_clusters": 8},
            "E1": {"min_confidence": 0.7},
            "E2": {"multiplier": 1.5},
        }
        return params_map.get(scenario_id, {})


def derive_scenarios(tech_fact_sheet: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    推导候选场景（便捷函数）

    参数:
    - tech_fact_sheet: 技术事实清单

    返回:
    - 候选场景列表
    """
    engine = ScenarioEngine(tech_fact_sheet)
    return engine.derive()