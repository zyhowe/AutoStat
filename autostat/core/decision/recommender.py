"""
行动建议生成器

基于根因分析结果，生成可执行的行动建议
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

class ActionPriority(Enum):
    """行动优先级"""
    HIGH = "高"
    MEDIUM = "中"
    LOW = "低"


class ActionDifficulty(Enum):
    """行动难度"""
    EASY = "低"
    MEDIUM = "中"
    HARD = "高"


@dataclass
class ActionSuggestion:
    """行动建议"""
    id: str
    title: str
    description: str
    priority: ActionPriority
    difficulty: ActionDifficulty
    expected_effect: str
    confidence: float  # 0-1
    steps: List[str]
    tags: List[str] = field(default_factory=list)


class ActionRecommender:
    """
    行动建议生成器

    使用方式:
        recommender = ActionRecommender()
        suggestions = recommender.recommend(root_cause_result, context)
    """

    def __init__(self, llm_client=None):
        """
        初始化建议生成器

        参数:
        - llm_client: 大模型客户端（可选，用于增强生成）
        """
        self.llm_client = llm_client

    def recommend(
        self,
        anomaly: Dict[str, Any],
        root_causes: List[Dict[str, Any]],
        context: Dict[str, Any],
        business_rules: Optional[List[Dict]] = None
    ) -> List[ActionSuggestion]:
        """
        生成行动建议

        参数:
        - anomaly: 异常事件
        - root_causes: 根因列表
        - context: 业务上下文
        - business_rules: 业务规则（用于匹配建议模板）

        返回: 建议列表
        """
        suggestions = []

        # 1. 基于规则模板匹配
        suggestions.extend(self._match_rule_templates(anomaly, root_causes, context))

        # 2. 基于根因生成针对性建议
        suggestions.extend(self._generate_from_root_causes(root_causes, context))

        # 3. 大模型增强（如果有）
        if self.llm_client:
            llm_suggestions = self._enhance_with_llm(anomaly, root_causes, context)
            suggestions.extend(llm_suggestions)

        # 4. 去重并按优先级排序
        suggestions = self._deduplicate(suggestions)
        suggestions.sort(key=lambda x: (x.priority.value == "高", x.confidence), reverse=True)

        return suggestions[:5]

    def _match_rule_templates(
        self,
        anomaly: Dict[str, Any],
        root_causes: List[Dict],
        context: Dict[str, Any]
    ) -> List[ActionSuggestion]:
        """匹配规则模板"""
        suggestions = []
        target = anomaly.get("target", "")

        # 销售相关模板
        if "销售" in target or "sales" in target.lower():
            suggestions.extend(self._get_sales_templates(anomaly, root_causes))

        # 用户相关模板
        if "用户" in target or "user" in target.lower() or "客户" in target:
            suggestions.extend(self._get_user_templates(anomaly, root_causes))

        # 质量相关模板
        if anomaly.get("type") in ["outlier", "rule_violation"]:
            suggestions.extend(self._get_quality_templates(anomaly, root_causes))

        # 通用模板
        if not suggestions:
            suggestions.append(self._get_general_template(anomaly))

        return suggestions

    def _get_sales_templates(
        self,
        anomaly: Dict[str, Any],
        root_causes: List[Dict]
    ) -> List[ActionSuggestion]:
        """销售相关建议模板"""
        suggestions = []

        # 检查是否有具体维度信息
        dim_info = self._extract_dimension_info(root_causes)

        if anomaly.get("type") == AnomalyType.DROP or "下降" in anomaly.get("message", ""):
            suggestions.append(ActionSuggestion(
                id=f"sales_promo_{datetime.now().timestamp()}",
                title="推出限时促销活动",
                description=f"针对{dim_info}销售额下降，建议推出限时折扣或满减活动",
                priority=ActionPriority.HIGH,
                difficulty=ActionDifficulty.MEDIUM,
                expected_effect="预计恢复销售额 15-20%",
                confidence=0.75,
                steps=[
                    "确定促销产品和折扣力度",
                    "制定促销时间表（建议2周）",
                    "准备促销物料和渠道投放",
                    "监测促销效果并实时调整"
                ],
                tags=["销售", "促销"]
            ))

            suggestions.append(ActionSuggestion(
                id=f"sales_channel_{datetime.now().timestamp()}",
                title="优化渠道投放策略",
                description=f"分析{dim_info}渠道表现，优化资源配置",
                priority=ActionPriority.MEDIUM,
                difficulty=ActionDifficulty.MEDIUM,
                expected_effect="预计提升渠道效率 10-15%",
                confidence=0.65,
                steps=[
                    "分析各渠道ROI表现",
                    "重新分配渠道预算",
                    "测试新的投放组合",
                    "持续优化投放策略"
                ],
                tags=["销售", "渠道"]
            ))

        if anomaly.get("type") == AnomalyType.SPIKE or "突增" in anomaly.get("message", ""):
            suggestions.append(ActionSuggestion(
                id=f"sales_analyze_{datetime.now().timestamp()}",
                title="分析增长原因并固化",
                description=f"深入分析{dim_info}销售额突增的原因，复制成功经验",
                priority=ActionPriority.HIGH,
                difficulty=ActionDifficulty.MEDIUM,
                expected_effect="巩固增长趋势，形成可复用的方法论",
                confidence=0.80,
                steps=[
                    "分析增长驱动因素（价格/促销/活动/竞品）",
                    "总结成功经验",
                    "制定复制推广计划",
                    "建立监控机制确保持续增长"
                ],
                tags=["销售", "增长"]
            ))

        return suggestions

    def _get_user_templates(
        self,
        anomaly: Dict[str, Any],
        root_causes: List[Dict]
    ) -> List[ActionSuggestion]:
        """用户相关建议模板"""
        dim_info = self._extract_dimension_info(root_causes)

        suggestions = [
            ActionSuggestion(
                id=f"user_recall_{datetime.now().timestamp()}",
                title=f"{dim_info}用户召回计划",
                description=f"针对{dim_info}用户活跃度下降，启动召回计划",
                priority=ActionPriority.HIGH,
                difficulty=ActionDifficulty.MEDIUM,
                expected_effect="预计召回 5-10% 流失用户",
                confidence=0.70,
                steps=[
                    "识别流失用户群体特征",
                    "制定召回策略（优惠/内容/活动）",
                    "执行召回动作",
                    "追踪召回效果"
                ],
                tags=["用户", "召回"]
            ),
            ActionSuggestion(
                id=f"user_retention_{datetime.now().timestamp()}",
                title="优化用户留存策略",
                description=f"通过优化用户体验和增值服务提升{dim_info}用户留存",
                priority=ActionPriority.MEDIUM,
                difficulty=ActionDifficulty.HARD,
                expected_effect="提升留存率 5-8%",
                confidence=0.60,
                steps=[
                    "分析用户流失原因",
                    "优化关键用户旅程节点",
                    "设计用户激励体系",
                    "持续监测留存指标"
                ],
                tags=["用户", "留存"]
            )
        ]

        return suggestions

    def _get_quality_templates(
        self,
        anomaly: Dict[str, Any],
        root_causes: List[Dict]
    ) -> List[ActionSuggestion]:
        """数据质量相关建议模板"""
        target = anomaly.get("target", "")

        suggestions = [
            ActionSuggestion(
                id=f"quality_fix_{datetime.now().timestamp()}",
                title=f"修复 {target} 数据质量问题",
                description=f"针对 {target} 的数据异常，执行数据清洗和修复",
                priority=ActionPriority.HIGH,
                difficulty=ActionDifficulty.MEDIUM,
                expected_effect="消除数据异常，提升质量评分",
                confidence=0.85,
                steps=[
                    f"检查 {target} 数据来源和采集逻辑",
                    "制定数据修复方案",
                    "执行数据清洗和补全",
                    "验证修复效果并建立监控"
                ],
                tags=["数据质量", "清洗"]
            ),
            ActionSuggestion(
                id=f"quality_rule_{datetime.now().timestamp()}",
                title="添加数据质量校验规则",
                description=f"为 {target} 添加数据质量监控规则，防止问题再次发生",
                priority=ActionPriority.MEDIUM,
                difficulty=ActionDifficulty.EASY,
                expected_effect="提前发现数据质量问题",
                confidence=0.90,
                steps=[
                    f"分析 {target} 的正常数据范围",
                    "配置校验规则（阈值/格式/勾稽）",
                    "集成到数据质量监控中",
                    "设置告警通知"
                ],
                tags=["数据质量", "规则"]
            )
        ]

        return suggestions

    def _get_general_template(self, anomaly: Dict[str, Any]) -> ActionSuggestion:
        """通用建议模板"""
        return ActionSuggestion(
            id=f"general_{datetime.now().timestamp()}",
            title=f"深入分析 {anomaly.get('target', 'unknown')} 异常",
            description=f"对 {anomaly.get('message', '数据异常')} 进行专项分析",
            priority=ActionPriority.MEDIUM,
            difficulty=ActionDifficulty.MEDIUM,
            expected_effect="明确问题根因，制定针对性方案",
            confidence=0.60,
            steps=[
                "收集更多相关数据",
                "进行多维度交叉分析",
                "验证根因假设",
                "制定并执行解决方案"
            ],
            tags=["分析"]
        )

    def _generate_from_root_causes(
        self,
        root_causes: List[Dict],
        context: Dict[str, Any]
    ) -> List[ActionSuggestion]:
        """从根因生成建议"""
        suggestions = []

        for rc in root_causes[:2]:
            if rc.get("confidence", 0) > 0.6:
                suggestions.append(ActionSuggestion(
                    id=f"root_{datetime.now().timestamp()}",
                    title=f"根因治理: {rc.get('description', '')[:20]}",
                    description=f"基于根因分析结果，制定针对性治理措施",
                    priority=ActionPriority.HIGH if rc.get("confidence", 0) > 0.8 else ActionPriority.MEDIUM,
                    difficulty=ActionDifficulty.MEDIUM,
                    expected_effect=f"预期可解决 {rc.get('confidence', 0):.0%} 的问题",
                    confidence=rc.get("confidence", 0.5),
                    steps=[
                        "制定详细治理方案",
                        "分配责任人",
                        "执行治理措施",
                        "验证治理效果"
                    ],
                    tags=["根因", "治理"]
                ))

        return suggestions

    def _enhance_with_llm(
        self,
        anomaly: Dict[str, Any],
        root_causes: List[Dict],
        context: Dict[str, Any]
    ) -> List[ActionSuggestion]:
        """大模型增强"""
        if not self.llm_client:
            return []

        prompt = f"""请根据以下信息，生成具体的行动建议：

## 异常
{anomaly.get('message', '')}
- 类型: {anomaly.get('type', 'unknown')}
- 目标: {anomaly.get('target', '')}

## 根因
{chr(10).join([f'- {rc.get("description", "")} (置信度: {rc.get("confidence", 0):.0%})' for rc in root_causes[:2]])}

## 上下文
{context}

请返回 2-3 条具体的行动建议，每条包括：标题、描述、预期效果、优先级（高/中/低）。
"""

        try:
            response = self.llm_client.chat([{"role": "user", "content": prompt}])
            # 简化解析：直接使用LLM返回的建议
            return [ActionSuggestion(
                id=f"llm_{datetime.now().timestamp()}",
                title=f"AI建议 {i+1}",
                description=response[:200],
                priority=ActionPriority.MEDIUM,
                difficulty=ActionDifficulty.MEDIUM,
                expected_effect="待评估",
                confidence=0.70,
                steps=["执行AI建议"],
                tags=["AI"]
            )]
        except Exception as e:
            print(f"LLM增强失败: {e}")
            return []

    def _extract_dimension_info(self, root_causes: List[Dict]) -> str:
        """从根因中提取维度信息"""
        if not root_causes:
            return "相关"

        dims = []
        for rc in root_causes:
            dims.extend(rc.get("dimensions", {}).keys())

        return "、".join(dims[:2]) if dims else "相关"

    def _deduplicate(self, suggestions: List[ActionSuggestion]) -> List[ActionSuggestion]:
        """去重"""
        seen = set()
        unique = []

        for s in suggestions:
            key = (s.title[:20], s.description[:30])
            if key not in seen:
                seen.add(key)
                unique.append(s)

        return unique