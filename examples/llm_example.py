# ==================== autotext/llm_extractor.py ====================
"""
大模型信息抽取客户端 - 基于给定的 schema 抽取实体、关系、事件、主题、归类
"""

import json
import time
import requests
from typing import List, Dict, Any, Optional


class InfoExtractorClient:
    """信息抽取客户端 - 调用大模型 API"""

    def __init__(self, api_base: str, api_key: str, model: str = "deepseek-chat"):
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.model = model

    def extract(self, text: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        从文本中抽取结构化信息

        返回格式:
        {
            "entities": [...],
            "relationships": [...],
            "events": [...],
            "themes": [...],
            "categorization": {...}
        }
        """
        prompt = self._build_prompt(text)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一个专业的信息抽取与分析系统。你必须只输出合法的JSON，不要有任何其他文字。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0  # 改为 0，确保输出稳定
            # 注意：移除了 response_format 参数，因为 DeepSeek 的 JSON 模式可能不稳定
        }

        for attempt in range(max_retries):
            try:
                print(f"  🔄 第{attempt + 1}次尝试调用大模型API...")

                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]

                    # 调试输出
                    print(f"  🔍 原始返回内容长度: {len(content)}")
                    print(f"  🔍 原始返回内容前300字符: {content[:300]}")

                    parsed = self._parse_json(content)
                    entity_count = len(parsed.get('entities', []))
                    relation_count = len(parsed.get('relationships', []))
                    static_count = len([r for r in parsed.get('relationships', []) if r.get('relation_type') == 'static'])
                    dynamic_count = len([r for r in parsed.get('relationships', []) if r.get('relation_type') == 'dynamic'])
                    print(f"  🔍 解析后实体数: {entity_count}, 关系数: {relation_count} (静态: {static_count}, 动态: {dynamic_count})")

                    if entity_count > 0:
                        return parsed
                    else:
                        print(f"  ⚠️ 第{attempt + 1}次尝试：返回的实体为空，重试中...")
                else:
                    print(f"  ⚠️ 第{attempt + 1}次尝试：HTTP {response.status_code}")
                    print(f"  🔍 响应内容: {response.text[:500]}")

            except requests.exceptions.Timeout:
                print(f"  ⚠️ 第{attempt + 1}次尝试：请求超时，重试中...")
            except requests.exceptions.ConnectionError as e:
                print(f"  ⚠️ 第{attempt + 1}次尝试：连接错误: {e}，重试中...")
            except Exception as e:
                print(f"  ⚠️ 第{attempt + 1}次尝试：异常: {e}，重试中...")

            if attempt < max_retries - 1:
                wait_time = 3
                print(f"  ⏳ 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)

        print(f"  ❌ {max_retries}次尝试均失败")
        return self._empty_result()

    def extract_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量抽取"""
        results = []
        for idx, text in enumerate(texts):
            if text and len(text) > 50:
                print(f"  📄 处理第 {idx + 1}/{len(texts)} 条文本")
                result = self.extract(text)
                results.append(result)
            else:
                results.append(self._empty_result())
        return results

    def _build_prompt(self, text: str) -> str:
        """构建提示词（根据文本结构动态识别包含关系）"""
        return f"""从以下文本中抽取结构化信息，输出JSON格式。

【重要】关系分为两类：
- static（静态关系）：实体之间的包含、属于、组成等层次关系。
- dynamic（动态关系）：实体之间的影响、导致、推动、抑制等因果关系。

【抽取规则】：
1. **实体类型**：Metric（指标如CPI、PPI）、Product（产品/商品）、Industry（行业）、EventName（事件名称）、Organization（组织）、Location（地点）、Person（人物）、Other（其他）

2. **静态关系（static）- 包含关系**：
   - 根据文本的章节结构判断包含关系。如果文本中有明确的章节标题（如"一、CPI温和回升"、"二、PPI涨幅扩大"），则该章节下描述的所有价格指标、产品、行业，都应该与该章节的主题实体建立"包含"关系
   - 例如：标题为"CPI温和回升"的章节下提到的"食品价格"、"能源价格"、"服务价格"，都应该与"CPI"建立"包含"关系
   - 例如：标题为"PPI涨幅扩大"的章节下提到的各个工业行业，都应该与"PPI"建立"包含"关系
   - 从属关系判断依据：段落归属（哪个章节下）、缩进层级、语义包含（如"食品价格"包含"猪肉价格"）

3. **动态关系（dynamic）**：抽取影响、导致、推动等因果关系
   - 例如："国际原油价格" 影响 "能源价格"
   - "国际大宗商品价格" 影响 "PPI"

4. 如果没有抽取到信息，对应字段使用空数组 []
5. 只输出JSON，不要有任何其他文字说明

【输出格式】：
{{
  "entities": [
    {{
      "entity_id": "E001",
      "entity_name": "实体名称",
      "entity_type": "Metric|Product|Industry|EventName|Organization|Location|Person|Other",
      "attributes": [
        {{"attr_name": "属性名", "attr_value": "属性值"}}
      ],
      "evidence": "原文片段"
    }}
  ],
  "relationships": [
    {{
      "relation_id": "R001",
      "subject_entity_id": "E001",
      "predicate": "关系谓词（包含、属于、影响、导致等）",
      "object_entity_id": "E002",
      "relation_type": "static|dynamic",
      "evidence": "原文证据句"
    }}
  ],
  "events": [
    {{
      "event_id": "EV001",
      "event_type": "事件类型",
      "trigger_word": "触发词",
      "time": "时间",
      "location": "地点",
      "participants": [
        {{"role": "角色", "entity_id": "E001"}}
      ],
      "summary": "事件描述",
      "evidence_sentences": ["原文句子"]
    }}
  ],
  "themes": [
    {{
      "theme_id": "T001",
      "parent_theme_id": null,
      "theme_name": "主题名称",
      "keywords": ["关键词1", "关键词2"],
      "summary": "主题摘要",
      "related_entity_ids": ["E001"],
      "related_event_ids": ["EV001"]
    }}
  ],
  "categorization": {{
    "categories": [
      {{
        "category_id": "C001",
        "category_name": "类别名称",
        "member_entity_ids": ["E001"],
        "shared_attributes": ["属性名"]
      }}
    ],
    "uncategorized": []
  }}
}}

【文本】：
{text[:6000]}

请输出JSON："""

    def _parse_json(self, content: str) -> Dict:
        """解析 JSON 响应"""
        # 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 尝试提取 JSON 代码块
        import re
        # 匹配 ```json ... ``` 或 ``` ... ```
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass

        # 尝试匹配第一个 { 到最后一个 }
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass

        print(f"  ❌ JSON解析失败，原始内容: {content[:500]}")
        return self._empty_result()

    def _empty_result(self) -> Dict:
        """空结果"""
        return {
            "entities": [],
            "relationships": [],
            "events": [],
            "themes": [],
            "categorization": {"categories": [], "uncategorized": []}
        }


# ==================== 便捷函数 ====================

def extract_info_from_texts(
        texts: List[str],
        api_base: str,
        api_key: str,
        model: str = "deepseek-chat"
) -> List[Dict]:
    """从文本列表抽取信息"""
    client = InfoExtractorClient(api_base, api_key, model)
    return client.extract_batch(texts)