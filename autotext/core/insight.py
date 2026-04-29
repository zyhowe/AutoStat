"""
洞察发现模块 - 增强版（支持趋势、对比、关联规则）
"""

from collections import Counter
from typing import List, Dict, Any, Optional
import numpy as np


class InsightDiscoverer:
    """洞察发现器"""

    def __init__(self):
        self.insights = []

    def discover_all(self, analyzer) -> List[Dict[str, Any]]:
        """
        发现所有洞察
        """
        self.insights = []

        # 原有洞察
        self._detect_hot_entities(analyzer)
        self._detect_strong_associations(analyzer)
        self._detect_sentiment_trend(analyzer)
        self._detect_cluster_insights(analyzer)
        self._detect_topic_insights(analyzer)

        # 新增洞察
        self._detect_trend_insights(analyzer)
        self._detect_contrast_insights(analyzer)
        self._detect_association_rule_insights(analyzer)

        # 按分数排序
        self.insights.sort(key=lambda x: x.get('score', 0), reverse=True)
        return self.insights[:20]

    # ==================== 原有方法（保留） ====================

    def _detect_hot_entities(self, analyzer):
        """检测热点实体（原有）"""
        entity_stats = getattr(analyzer, 'entity_stats', {})
        entity_counts = {}
        for entity_type in ['per', 'org', 'loc']:
            if entity_type in entity_stats:
                for name, count in entity_stats[entity_type].get('top', [])[:30]:
                    entity_counts[name] = entity_counts.get(name, 0) + count

        for i, (entity, count) in enumerate(sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]):
            score = min(1.0, count / 100)
            self.insights.append({
                'type': 'hot_entity',
                'title': f'🔥 高频实体：{entity}',
                'description': f'该实体在文本中被提及 {count} 次，是最受关注的对象之一。',
                'score': score,
                'priority': '高' if i == 0 else '中',
                'data': {'entity': entity, 'count': count}
            })

    def _detect_strong_associations(self, analyzer):
        """检测强关联实体对（原有）"""
        relation_result = getattr(analyzer, 'relation_result', {})
        for pair in relation_result.get('cooccurrence_pairs', [])[:5]:
            e1 = pair['entity1'].split(':', 1)[-1] if ':' in pair['entity1'] else pair['entity1']
            e2 = pair['entity2'].split(':', 1)[-1] if ':' in pair['entity2'] else pair['entity2']
            score = min(1.0, pair.get('pmi', 0) / 5)
            self.insights.append({
                'type': 'strong_association',
                'title': f'🔗 实体关联：{e1} ↔ {e2}',
                'description': f'这两个实体经常共同出现（PMI={pair.get("pmi", 0):.2f}），可能存在业务关联。',
                'score': score,
                'priority': '高' if score > 0.6 else '中',
                'data': pair
            })

    def _detect_sentiment_trend(self, analyzer):
        """检测情感趋势（原有）"""
        dist = getattr(analyzer, 'sentiment_distribution', {})
        pos = dist.get('positive_rate', 0)
        neg = dist.get('negative_rate', 0)

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
            description = f'积极 {pos:.1%}，消极 {neg:.1%}，整体中性。'
            score = max(pos, neg)
            priority = '中'

        self.insights.append({
            'type': 'sentiment_trend',
            'title': title,
            'description': description,
            'score': score,
            'priority': priority,
            'data': dist
        })

    def _detect_cluster_insights(self, analyzer):
        """检测聚类洞察（原有）"""
        clusters = getattr(analyzer, 'cluster_info', [])
        if clusters:
            largest = max(clusters, key=lambda x: x.get('size', 0))
            self.insights.append({
                'type': 'cluster_dominant',
                'title': f'📊 主要话题：{largest.get("top_words", [])[:3] if largest.get("top_words") else "未知"}',
                'description': f'最大的文本簇包含 {largest.get("size", 0)} 篇文本（占比 {largest.get("percentage", 0):.1%}）。',
                'score': largest.get('percentage', 0),
                'priority': '高' if largest.get('percentage', 0) > 0.3 else '中',
                'data': largest
            })

    def _detect_topic_insights(self, analyzer):
        """检测主题洞察（原有）"""
        topics = getattr(analyzer, 'topics', [])
        if topics:
            largest = max(topics, key=lambda x: x.get('texts_count', 0))
            self.insights.append({
                'type': 'topic_dominant',
                'title': f'📚 主要主题：{largest.get("keywords", [])[:3] if largest.get("keywords") else "未知"}',
                'description': f'最大的主题包含 {largest.get("texts_count", 0)} 篇文本，核心词：{", ".join(largest.get("keywords", [])[:5])}。',
                'score': min(1.0, largest.get('texts_count', 0) / 100),
                'priority': '中',
                'data': largest
            })

    # ==================== 新增方法 ====================

    def _detect_trend_insights(self, analyzer):
        """检测趋势变化洞察"""
        # 从 analyzer 获取趋势数据
        event_timeline = getattr(analyzer, 'event_timeline', {})
        hot_topics = event_timeline.get('hot_topics', [])

        for topic in hot_topics[:5]:
            entity = topic.get('entity', '')
            entity_name = entity.split(':', 1)[-1] if ':' in entity else entity
            increase_pct = topic.get('increase_pct', 0)

            if increase_pct > 100:
                strength = "暴涨"
            elif increase_pct > 50:
                strength = "显著上升"
            else:
                strength = "上升"

            self.insights.append({
                'type': 'trend',
                'title': f'📈 热点话题：{entity_name}',
                'description': f'{entity_name} 提及次数 {strength} {increase_pct:.0f}%，近期达到峰值。',
                'score': min(1.0, increase_pct / 200),
                'priority': '高' if increase_pct > 100 else '中',
                'data': topic
            })

        # 情感转折点
        sentiment_shifts = event_timeline.get('sentiment_evolution', {}).get('turning_points', [])
        for shift in sentiment_shifts[:3]:
            entity = shift.get('entity', '整体')
            direction = shift.get('direction', '')
            if direction == '上升':
                desc = f'情感由负转正，舆情好转'
            elif direction == '下降':
                desc = f'情感由正转负，需关注'
            else:
                desc = f'情感发生转折'

            self.insights.append({
                'type': 'sentiment_shift',
                'title': f'🎭 情感转折：{entity}',
                'description': f'{entity} 于 {shift.get("shift_date", "")} 发生情感转折，{desc}。',
                'score': 0.7,
                'priority': '高',
                'data': shift
            })

    def _detect_contrast_insights(self, analyzer):
        """检测对比洞察"""
        clusters = getattr(analyzer, 'cluster_info', [])
        if len(clusters) < 2:
            return

        # 找出最大的两个簇
        sorted_clusters = sorted(clusters, key=lambda x: x.get('size', 0), reverse=True)
        if len(sorted_clusters) >= 2:
            c1 = sorted_clusters[0]
            c2 = sorted_clusters[1]

            keywords1 = set(c1.get('top_words', [])[:5])
            keywords2 = set(c2.get('top_words', [])[:5])

            # 计算差异度
            if keywords1 and keywords2:
                diff_ratio = len(keywords1 - keywords2) / len(keywords1) if keywords1 else 0

                if diff_ratio > 0.5:
                    diff_desc = "有较大差异"
                else:
                    diff_desc = "有一定差异"

                self.insights.append({
                    'type': 'contrast',
                    'title': f'📊 话题对比：簇{len(sorted_clusters)}个主要话题',
                    'description': f'最大的两个簇（{c1.get("size", 0)}条 vs {c2.get("size", 0)}条）关键词{diff_desc}，'
                                 f'前者聚焦{", ".join(list(keywords1)[:3])}，后者聚焦{", ".join(list(keywords2)[:3])}。',
                    'score': min(1.0, diff_ratio),
                    'priority': '中',
                    'data': {'cluster1': c1, 'cluster2': c2, 'diff_ratio': diff_ratio}
                })

    def _detect_association_rule_insights(self, analyzer):
        """检测关联规则洞察"""
        relation_result = getattr(analyzer, 'relation_result', {})
        pairs = relation_result.get('cooccurrence_pairs', [])

        for pair in pairs[:5]:
            e1 = pair['entity1'].split(':', 1)[-1] if ':' in pair['entity1'] else pair['entity1']
            e2 = pair['entity2'].split(':', 1)[-1] if ':' in pair['entity2'] else pair['entity2']
            pmi = pair.get('pmi', 0)
            count = pair.get('count', 0)

            if pmi > 3:
                strength = "极强"
            elif pmi > 2:
                strength = "很强"
            else:
                strength = "较强"

            self.insights.append({
                'type': 'association_rule',
                'title': f'🔗 关联规则：{e1} → {e2}',
                'description': f'当 {e1} 出现时，{e2} 有很高概率同时出现（共现{count}次，{strength}关联）。',
                'score': min(1.0, pmi / 5),
                'priority': '高' if pmi > 3 else '中',
                'data': pair
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