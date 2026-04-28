"""
洞察发现模块 - 自动发现数据中的规律
"""

from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
import numpy as np


class InsightDiscoverer:
    """洞察发现器 - 自动扫描数据中的显著模式"""

    def __init__(self):
        self.insights = []

    def discover_all(self, analyzer) -> List[Dict[str, Any]]:
        """
        发现所有洞察

        参数:
        - analyzer: TextAnalyzer 实例

        返回:
        [
            {
                'type': 'hot_entity',
                'title': 'XX公司近期提及量激增',
                'description': '...',
                'score': 0.85,
                'data': {...}
            },
            ...
        ]
        """
        self.insights = []

        # 1. 热点实体检测
        if hasattr(analyzer, 'entity_stats'):
            self._detect_hot_entities(analyzer)

        # 2. 强关联实体对
        if hasattr(analyzer, 'relation_result'):
            self._detect_strong_associations(analyzer)

        # 3. 情感倾向
        if hasattr(analyzer, 'sentiment_distribution'):
            self._detect_sentiment_trend(analyzer)

        # 4. 聚类特征
        if hasattr(analyzer, 'cluster_info') and analyzer.cluster_info:
            self._detect_cluster_insights(analyzer)

        # 5. 主题热度
        if hasattr(analyzer, 'topics') and analyzer.topics:
            self._detect_topic_insights(analyzer)

        # 按分数排序
        self.insights.sort(key=lambda x: x.get('score', 0), reverse=True)

        return self.insights[:15]

    def _detect_hot_entities(self, analyzer):
        """检测热点实体"""
        entity_stats = getattr(analyzer, 'entity_stats', {})

        # 统计实体总提及次数
        entity_counts = {}
        for entity_type in ['per', 'org', 'loc']:
            if entity_type in entity_stats:
                for name, count in entity_stats[entity_type].get('top', [])[:30]:
                    entity_counts[name] = entity_counts.get(name, 0) + count

        if not entity_counts:
            return

        # 找出高频实体
        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        for i, (entity, count) in enumerate(sorted_entities):
            # 计算分数（归一化）
            score = min(1.0, count / 100) if count > 0 else 0

            self.insights.append({
                'type': 'hot_entity',
                'title': f'🔥 高频实体：{entity}',
                'description': f'该实体在文本中被提及 {count} 次，是最受关注的对象之一。',
                'score': score,
                'priority': '高' if i == 0 else '中',
                'data': {'entity': entity, 'count': count}
            })

    def _detect_strong_associations(self, analyzer):
        """检测强关联实体对"""
        relation_result = getattr(analyzer, 'relation_result', {})

        pairs = relation_result.get('cooccurrence_pairs', [])
        if not pairs:
            return

        for pair in pairs[:5]:
            e1 = pair['entity1'].split(':', 1)[-1] if ':' in pair['entity1'] else pair['entity1']
            e2 = pair['entity2'].split(':', 1)[-1] if ':' in pair['entity2'] else pair['entity2']
            pmi = pair.get('pmi', 0)

            # 归一化分数
            score = min(1.0, pmi / 5)

            self.insights.append({
                'type': 'strong_association',
                'title': f'🔗 实体关联：{e1} ↔ {e2}',
                'description': f'这两个实体经常共同出现（PMI={pmi:.2f}），可能存在业务关联。',
                'score': score,
                'priority': '高' if score > 0.6 else '中',
                'data': {'entity1': e1, 'entity2': e2, 'pmi': pmi, 'contexts': pair.get('contexts', [])}
            })

    def _detect_sentiment_trend(self, analyzer):
        """检测情感趋势"""
        dist = getattr(analyzer, 'sentiment_distribution', {})

        pos = dist.get('positive_rate', 0)
        neg = dist.get('negative_rate', 0)
        neu = dist.get('neutral_rate', 0)

        # 判断情感倾向
        if pos > 0.5:
            title = '😊 整体情感偏积极'
            description = f'积极文本占比 {pos:.1%}，整体情绪正面。'
            score = pos
            priority = '高'
        elif neg > 0.3:
            title = '😞 整体情感偏消极'
            description = f'消极文本占比 {neg:.1%}，需要关注负面因素。'
            score = neg
            priority = '高'
        else:
            title = '😐 情感分布均衡'
            description = f'积极 {pos:.1%}，消极 {neg:.1%}，中性 {neu:.1%}，整体中性。'
            score = max(pos, neg)
            priority = '中'

        self.insights.append({
            'type': 'sentiment_trend',
            'title': title,
            'description': description,
            'score': score,
            'priority': priority,
            'data': {'positive_rate': pos, 'negative_rate': neg, 'neutral_rate': neu}
        })

    def _detect_cluster_insights(self, analyzer):
        """检测聚类洞察"""
        clusters = getattr(analyzer, 'cluster_info', [])

        if not clusters:
            return

        # 找出最大的簇
        largest = max(clusters, key=lambda x: x['size'])

        self.insights.append({
            'type': 'cluster_dominant',
            'title': f'📊 主要话题：{largest["top_words"][:3] if largest["top_words"] else "未知"}',
            'description': f'最大的文本簇包含 {largest["size"]} 篇文本（占比 {largest["percentage"]:.1%}），',
            'score': largest['percentage'],
            'priority': '高' if largest['percentage'] > 0.3 else '中',
            'data': {
                'size': largest['size'],
                'percentage': largest['percentage'],
                'keywords': largest['top_words'][:5],
                'center_text': largest.get('center_text', '')
            }
        })

    def _detect_topic_insights(self, analyzer):
        """检测主题洞察"""
        topics = getattr(analyzer, 'topics', [])

        if not topics:
            return

        # 找出最大的主题
        largest = max(topics, key=lambda x: x.get('texts_count', 0))

        self.insights.append({
            'type': 'topic_dominant',
            'title': f'📚 主要主题：{largest["keywords"][:3] if largest["keywords"] else "未知"}',
            'description': f'最大的主题包含 {largest["texts_count"]} 篇文本，核心词：{", ".join(largest["keywords"][:5])}。',
            'score': min(1.0, largest['texts_count'] / 100),
            'priority': '中',
            'data': {
                'texts_count': largest['texts_count'],
                'keywords': largest['keywords'][:10],
                'representative_text': largest.get('representative_text', '')
            }
        })


def format_insights_for_report(insights: List[Dict]) -> str:
    """格式化洞察为报告文本"""
    if not insights:
        return "未发现显著洞察"

    lines = []
    for i, insight in enumerate(insights[:10], 1):
        priority_icon = {'高': '🔴', '中': '🟠', '低': '🟢'}.get(insight.get('priority', '中'), '⚪')
        lines.append(f"**{i}. {priority_icon} {insight['title']}**")
        lines.append(f"   {insight['description']}\n")

    return '\n'.join(lines)