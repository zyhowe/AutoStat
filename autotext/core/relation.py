"""
关系发现模块 - 基于实体共现，不预设类型
"""

from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional
import math


class RelationDiscoverer:
    """关系发现器 - 基于实体共现"""

    def __init__(self):
        self.entity_results = None
        self.texts = None

    def discover(self, entity_results: List[Dict[str, List[Tuple[str, int, int]]]],
                 texts: List[str]) -> Dict[str, Any]:
        """
        发现实体间关系

        参数:
        - entity_results: 实体识别结果
        - texts: 原始文本列表

        返回:
        {
            'cooccurrence_pairs': [{'entity1': 'xx', 'entity2': 'xx', 'count': 10, 'pmi': 2.5, 'contexts': [...]}],
            'entity_freq': {'xx': 10, ...}
        }
        """
        self.entity_results = entity_results
        self.texts = texts

        # 统计实体频次
        entity_freq = self._count_entity_freq()

        # 统计共现对
        cooccurrence = self._count_cooccurrence()

        # 计算PMI
        total_pairs = sum(cooccurrence.values())
        total_entities = len(entity_freq)

        pairs_with_pmi = []
        for (e1, e2), count in cooccurrence.items():
            if e1 == e2:
                continue
            pmi = self._calculate_pmi(count, entity_freq[e1], entity_freq[e2], total_pairs)
            if pmi > 0.5:  # PMI阈值
                pairs_with_pmi.append({
                    'entity1': e1,
                    'entity2': e2,
                    'count': count,
                    'pmi': round(pmi, 2)
                })

        # 按PMI排序
        pairs_with_pmi.sort(key=lambda x: x['pmi'], reverse=True)

        # 添加上下文例句（前10个强关联对）
        for pair in pairs_with_pmi[:10]:
            contexts = self._find_contexts(pair['entity1'], pair['entity2'])
            pair['contexts'] = contexts

        return {
            'entity_frequency': dict(sorted(entity_freq.items(), key=lambda x: x[1], reverse=True)[:50]),
            'cooccurrence_pairs': pairs_with_pmi[:30]
        }

    def _count_entity_freq(self) -> Dict[str, int]:
        """统计实体频次"""
        freq = defaultdict(int)

        for result in self.entity_results:
            for entity_type, entities in result.items():
                for entity_text, _, _ in entities:
                    # 合并类型前缀
                    key = f"{entity_type}:{entity_text}"
                    freq[key] += 1

        return freq

    def _count_cooccurrence(self) -> Dict[Tuple[str, str], int]:
        """统计共现对"""
        cooccurrence = defaultdict(int)

        for result in self.entity_results:
            # 获取当前文本中的所有实体
            entities_in_text = []
            for entity_type, entities in result.items():
                for entity_text, _, _ in entities:
                    entities_in_text.append(f"{entity_type}:{entity_text}")

            # 统计共现对
            for i in range(len(entities_in_text)):
                for j in range(i + 1, len(entities_in_text)):
                    e1, e2 = sorted([entities_in_text[i], entities_in_text[j]])
                    cooccurrence[(e1, e2)] += 1

        return cooccurrence

    def _calculate_pmi(self, co_count: int, freq1: int, freq2: int, total: int) -> float:
        """计算点互信息 PMI = log(P(x,y) / (P(x) * P(y)))"""
        if co_count == 0:
            return 0
        p_xy = co_count / total
        p_x = freq1 / total
        p_y = freq2 / total
        if p_x == 0 or p_y == 0:
            return 0
        return math.log(p_xy / (p_x * p_y))

    def _find_contexts(self, entity1: str, entity2: str, max_contexts: int = 3) -> List[str]:
        """查找两个实体共同出现的上下文"""
        contexts = []

        for i, result in enumerate(self.entity_results):
            # 检查是否同时包含两个实体
            has_e1 = False
            has_e2 = False

            for entity_type, entities in result.items():
                for entity_text, _, _ in entities:
                    key = f"{entity_type}:{entity_text}"
                    if key == entity1:
                        has_e1 = True
                    if key == entity2:
                        has_e2 = True

            if has_e1 and has_e2:
                # 截取句子作为上下文
                text = self.texts[i]
                # 取前200字符
                context = text[:200] + "..." if len(text) > 200 else text
                contexts.append(context)

            if len(contexts) >= max_contexts:
                break

        return contexts