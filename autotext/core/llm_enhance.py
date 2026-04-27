"""
大模型增强模块 - 摘要、情感解释、簇命名、事件识别
"""

from typing import List, Dict, Any, Optional


class LLMEnhancer:
    """大模型增强器"""

    def __init__(self, llm_client=None):
        """
        初始化大模型增强器

        参数:
        - llm_client: 大模型客户端（需实现 chat 方法）
        """
        self.llm_client = llm_client

    def is_available(self) -> bool:
        """检查大模型是否可用"""
        return self.llm_client is not None

    def summarize_text(self, text: str, max_length: int = 200) -> Optional[str]:
        """
        生成文本摘要

        参数:
        - text: 文本
        - max_length: 摘要最大长度

        返回: 摘要文本
        """
        if not self.is_available():
            return None

        prompt = f"""请为以下文本生成一句话摘要，不超过{max_length}字：

文本：{text[:2000]}

摘要："""

        try:
            response = self.llm_client.chat([{"role": "user", "content": prompt}], temperature=0.5)
            return response.strip()[:max_length]
        except Exception as e:
            print(f"摘要生成失败: {e}")
            return None

    def explain_sentiment(self, text: str, sentiment_result: Dict) -> Optional[str]:
        """
        解释情感分析结果

        参数:
        - text: 文本
        - sentiment_result: 情感分析结果

        返回: 解释文本
        """
        if not self.is_available():
            return None

        prompt = f"""请解释为什么这段文本被判定为{sentiment_result.get('sentiment', 'neutral')}。

文本：{text[:500]}

正面词：{sentiment_result.get('positive_words', [])}
负面词：{sentiment_result.get('negative_words', [])}

请简要说明判断依据（50字以内）："""

        try:
            response = self.llm_client.chat([{"role": "user", "content": prompt}], temperature=0.5)
            return response.strip()
        except Exception as e:
            print(f"情感解释失败: {e}")
            return None

    def name_cluster(self, keywords: List[str], sample_texts: List[str]) -> Optional[str]:
        """
        为聚类命名

        参数:
        - keywords: 关键词列表
        - sample_texts: 代表性文本列表

        返回: 簇名称
        """
        if not self.is_available():
            return None

        sample_str = "、".join(sample_texts[:3])
        prompt = f"""请根据以下关键词和样本，为这个文本聚类起一个简短的名字（不超过10个字）：

关键词：{', '.join(keywords[:10])}

样本：{sample_str[:300]}

名称："""

        try:
            response = self.llm_client.chat([{"role": "user", "content": prompt}], temperature=0.5)
            return response.strip()
        except Exception as e:
            print(f"簇命名失败: {e}")
            return None

    def identify_events(self, time_series: List[Dict]) -> Optional[List[Dict]]:
        """
        识别事件

        参数:
        - time_series: 时间序列数据，每项包含 {"date": str, "summary": str, "keywords": List[str]}

        返回: [{"date": str, "event": str, "description": str}, ...]
        """
        if not self.is_available():
            return None

        # 构建输入
        timeline = "\n".join([f"- {item['date']}: {item['summary'][:100]}" for item in time_series[:20]])

        prompt = f"""请从以下时间线中识别重要事件：

{timeline}

返回 JSON 格式：
[
  {{"date": "日期", "event": "事件名称", "description": "事件描述"}}
]

只返回 JSON，不要其他内容："""

        try:
            import json
            response = self.llm_client.chat([{"role": "user", "content": prompt}], temperature=0.5)
            # 尝试提取 JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            events = json.loads(response)
            return events[:5]
        except Exception as e:
            print(f"事件识别失败: {e}")
            return None

    def generate_insights(self, stats: Dict, sentiment_dist: Dict, topics: List[Dict], clusters: List[Dict]) -> Optional[str]:
        """
        生成整体洞察

        参数:
        - stats: 基础统计
        - sentiment_dist: 情感分布
        - topics: 主题列表
        - clusters: 聚类信息

        返回: 洞察文本
        """
        if not self.is_available():
            return None

        prompt = f"""请根据以下文本分析结果，生成整体洞察（200字以内）：

## 基础统计
- 总文本数: {stats.get('total_count', 0)}
- 平均长度: {stats.get('char_length', {}).get('mean', 0):.0f} 字

## 情感分布
- 积极: {sentiment_dist.get('positive_rate', 0):.1%}
- 消极: {sentiment_dist.get('negative_rate', 0):.1%}
- 中性: {sentiment_dist.get('neutral_rate', 0):.1%}

## 主要主题
{chr(10).join([f"- {t['keywords'][:3]}" for t in topics[:3]]) if topics else '无'}

## 主要聚类
{chr(10).join([f"- 簇{cl.get('cluster_id')} ({cl.get('size')}条): {cl.get('top_words', [])[:5]}" for cl in clusters[:3]]) if clusters else '无'}

洞察："""

        try:
            response = self.llm_client.chat([{"role": "user", "content": prompt}], temperature=0.5)
            return response.strip()
        except Exception as e:
            print(f"洞察生成失败: {e}")
            return None