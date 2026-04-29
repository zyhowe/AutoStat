"""
关联规则挖掘模块 - 基于实体共现
"""

from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional
import math


class RelationMiner:
    """关联规则挖掘器"""

    def __init__(self):
        self.rules = []

    def mine_association_rules(self, entity_cooccurrence: List[List[str]],
                                min_support: float = 0.01,
                                min_confidence: float = 0.5) -> List[Dict]:
        """
        挖掘关联规则

        参数:
        - entity_cooccurrence: 每条文本中的实体列表（字符串列表）
        - min_support: 最小支持度
        - min_confidence: 最小置信度

        返回: [
            {
                "antecedent": "实体A",
                "consequent": "实体B",
                "support": 0.15,
                "confidence": 0.82,
                "lift": 2.3,
                "interpretation": "..."
            }
        ]
        """
        total_transactions = len(entity_cooccurrence)
        if total_transactions == 0:
            return []

        # 1. 统计实体频次
        entity_count = Counter()
        for transaction in entity_cooccurrence:
            unique_entities = set(transaction)
            for entity in unique_entities:
                entity_count[entity] += 1

        # 2. 统计实体对共现频次
        pair_count = defaultdict(int)
        for transaction in entity_cooccurrence:
            unique_entities = list(set(transaction))
            for i in range(len(unique_entities)):
                for j in range(i + 1, len(unique_entities)):
                    a, b = unique_entities[i], unique_entities[j]
                    pair_count[(a, b)] += 1

        # 3. 计算支持度、置信度、提升度
        rules = []
        for (a, b), count in pair_count.items():
            support = count / total_transactions
            if support < min_support:
                continue

            # 计算置信度
            confidence_ab = count / entity_count[a] if entity_count[a] > 0 else 0
            confidence_ba = count / entity_count[b] if entity_count[b] > 0 else 0

            # 取较高方向
            if confidence_ab > confidence_ba:
                antecedent, consequent = a, b
                confidence = confidence_ab
            else:
                antecedent, consequent = b, a
                confidence = max(confidence_ab, confidence_ba)

            if confidence < min_confidence:
                continue

            # 计算提升度
            p_consequent = entity_count[consequent] / total_transactions
            lift = confidence / p_consequent if p_consequent > 0 else 0

            if lift < 1.2:
                continue

            # 生成自然语言解读
            interpretation = self._generate_interpretation(antecedent, consequent, confidence, lift)

            rules.append({
                "antecedent": antecedent,
                "consequent": consequent,
                "support": round(support, 3),
                "confidence": round(confidence, 3),
                "lift": round(lift, 2),
                "interpretation": interpretation
            })

        # 按提升度排序
        rules.sort(key=lambda x: x["lift"], reverse=True)
        return rules[:20]

    def _generate_interpretation(self, antecedent: str, consequent: str,
                                   confidence: float, lift: float) -> str:
        """生成自然语言解读"""
        if lift > 3:
            strength = "极强关联"
        elif lift > 2:
            strength = "强关联"
        elif lift > 1.5:
            strength = "中等关联"
        else:
            strength = "弱关联"

        return f"当「{antecedent}」出现时，「{consequent}」有 {confidence:.0%} 的概率同时出现（{strength}，提升度 {lift:.1f} 倍）"

    def get_entity_cooccurrence_from_results(self, entity_results: List[Dict]) -> List[List[str]]:
        """从实体识别结果中提取实体共现矩阵"""
        cooccurrence = []
        for result in entity_results:
            entities = []
            for entity_type, entities_list in result.items():
                for entity_text, _, _ in entities_list:
                    # 添加类型前缀以便区分同名不同类实体
                    key = f"{entity_type}:{entity_text}"
                    entities.append(key)
            cooccurrence.append(entities)
        return cooccurrence