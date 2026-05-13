# ==================== autotext/llm_extractor.py ====================
"""
大模型信息抽取客户端 - 基于给定的 schema 抽取实体、关系、事件、主题、归类
"""

import json
import time
import re
import requests
from typing import List, Dict, Any, Optional


class InfoExtractorClient:
    """信息抽取客户端 - 调用大模型 API"""

    def __init__(self, api_base: str, api_key: str, model: str = "deepseek-chat"):
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.model = model

    def extract(self, text: str, max_retries: int = 3) -> Dict[str, Any]:
        prompt = self._build_prompt(text)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system",
                 "content": "你是一个专业的信息抽取与分析系统。你必须只输出合法的完整JSON，不要有任何其他文字。先思考分析，再输出结果。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0,
            "seed": 42,
            "max_tokens": 80000
        }

        for attempt in range(max_retries):
            try:
                print(f"  🔄 第{attempt + 1}次尝试调用大模型API...")

                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=300
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]

                    print(f"  🔍 原始返回内容长度: {len(content)}")
                    print(f"  🔍 原始返回内容前300字符: {content[:300]}")
                    print(f"  🔍 原始返回内容后200字符: {content[-200:]}")

                    parsed = self._parse_json(content)
                    entity_count = len(parsed.get('entities', []))
                    relation_count = len(parsed.get('relationships', []))
                    event_count = len(parsed.get('events', []))
                    theme_count = len(parsed.get('themes', []))
                    static_count = len(
                        [r for r in parsed.get('relationships', []) if r.get('relation_type') == 'static'])
                    dynamic_count = len(
                        [r for r in parsed.get('relationships', []) if r.get('relation_type') == 'dynamic'])
                    print(
                        f"  🔍 解析后实体数: {entity_count}, 关系数: {relation_count} (静态: {static_count}, 动态: {dynamic_count}), 事件数: {event_count}, 主题数: {theme_count}")

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
        return f"""从以下文本中抽取结构化信息，输出完整JSON。

================================================================================
一、总体原则（先思考，后输出）
================================================================================
1. 不要猜测或编造信息。如果文本中没有明确的内容，输出空数组或null。
2. 一步步思考：先识别文本结构，再抽取实体，最后建立关系。
3. 每个判断都要有依据，在reason字段中说明。

================================================================================
二、引号使用规则
================================================================================
- JSON结构中的双引号（如"entities"、"entity_id"等字段名）使用英文双引号"
- 所有字段值（如实体名称、证据原文、属性值等）内部的双引号，使用中文全角双引号“和”

================================================================================
三、实体抽取规则（思考步骤）
================================================================================
【思考步骤】
第1步：通读文本，识别所有名词性短语（人名、地名、机构名、指标名、产品名、行业名）
第2步：判断这些名词性短语是否满足实体条件（有独立含义、可被引用）
第3步：为每个实体分配类型，提取属性

【实体类型】
根据实体在文中的实际含义选择：Metric、Product、Industry、EventName、Organization、Location、Person、Other

【实体名称】
- 直接使用原文中的表述，完全保持原样
- 不要添加、删减或修改任何字词
- 如果原文中实体出现多次，使用第一次出现时的完整名称

【属性】
- 只提取与该实体直接相关的数值属性
- 属性值保留原文完整形式，包括区间、范围
- 如果某个属性不存在，不要编造

【证据】
- 摘录最能证明该实体存在的原文片段，不超过30字
- 如果找不到明确的原文证据，不要创建该实体

【原因】
- 说明为什么将这段文本识别为实体，不超过15字

【负面示例】
- 错误：原文中没有的属性，不要编造
- 错误：原文中"鲜菜"，不要输出"鲜菜价格"
- 正确：原文中"鲜菜"，输出"鲜菜"

================================================================================
四、关系抽取规则（思考步骤）
================================================================================
【思考步骤】
第1步：遍历所有实体对
第2步：判断A和B之间是否存在关系
第3步：确定关系类型（static或dynamic）
第4步：确定关系谓词

【关系类型】
- static：A包含B、A属于B、A是B的组成部分
- dynamic：A引起B的变化、A影响B

【判断流程】
1. 检查A和B在文本中的位置关系
   - 如果A是章节标题，B是该章节下的内容 → static，原因"章节标题下属内容"
2. 检查A和B的语义关系
   - 如果A的语义范围包含B → static，原因"语义包含"
3. 检查A和B的因果关系
   - 如果A的出现引起了B的变化 → dynamic，原因"因果关系"
4. 以上都不满足 → 不建立关系

【谓词选择】
- static关系使用"包含"
- dynamic关系使用"影响"

【证据】
- 摘录最能证明该关系的原文句子，不超过30字

【原因】
- 说明判断依据，不超过15字

【负面示例】
- 错误：没有明确证据的关系，不要输出
- 错误：A和B只是同一段落中出现，没有明确关系，不要输出

================================================================================
五、事件抽取规则（思考步骤）
================================================================================
【思考步骤】
第1步：扫描文本，找出所有表示状态变化的动词（涨、跌、升、降、增、减、发布、宣布等）
第2步：确认该动词有明确的时间点
第3步：确认该动词有明确的参与者（主语或宾语）
第4步：如果三个条件都满足，创建事件

【事件判定条件】（三个条件必须同时满足，缺一不可）
1. 存在明确的触发词（状态变化动词）
2. 存在明确的时间点或时间段
3. 存在至少一个参与者实体

【事件合并条件】
- 同一时间点 + 同一参与者 + 同一触发词 → 合并为一个事件
- 例如："CPI环比上涨0.3%"和"CPI同比上涨1.2%"是不同事件（不同维度）

【事件类型】
- 使用触发词作为事件类型，如"上涨"、"下降"、"发布"

【不作为事件的情况】
- 没有触发词的陈述
- 没有明确时间的描述
- 纯粹的背景介绍或定义说明

【证据】
- 摘录原文中的证据句

【原因】
- 说明为什么判定为事件，不超过15字

【负面示例】
- 错误：没有触发词，不要输出事件
- 错误：没有明确时间，不要输出事件
- 错误：没有参与者，不要输出事件

================================================================================
六、主题抽取规则（思考步骤）
================================================================================
【思考步骤】
第1步：归纳文本的核心话题
第2步：将相关的实体和事件归入同一主题
第3步：为主题命名和撰写摘要

【主题定义】
- 主题是对文本中多个相关实体和事件的核心概念归纳
- 主题数量：根据文本内容自行判断，通常为2-5个
- 如果文本太短或内容单一，可以是1个主题

【主题层级】
- 如果主题之间有包含关系，设置parent_theme_id指向父主题
- 如果没有包含关系，parent_theme_id设为null

【关键词】
- 提取最能代表该主题的3-5个词语，从原文中选取

【摘要】
- 归纳该主题的核心内容，不超过50字
- 用原文中的表述，不要编造

【原因】
- 说明为什么归纳出这个主题，不超过15字

================================================================================
七、输出格式
================================================================================
每个实体、关系、事件、主题、分类都必须包含reason字段。

{{
  "entities": [
    {{
      "entity_id": "E001",
      "entity_name": "原文中的实体名称",
      "entity_type": "Metric",
      "attributes": [
        {{"attr_name": "属性名", "attr_value": "属性值"}}
      ],
      "evidence": "原文证据片段",
      "reason": "判断依据"
    }}
  ],
  "relationships": [
    {{
      "relation_id": "R001",
      "subject_entity_id": "E001",
      "predicate": "包含",
      "object_entity_id": "E002",
      "relation_type": "static",
      "evidence": "原文证据",
      "reason": "判断依据"
    }}
  ],
  "events": [
    {{
      "event_id": "EV001",
      "event_type": "上涨",
      "trigger_word": "上涨",
      "time": "4月份",
      "location": "全国",
      "participants": [
        {{"role": "受动者", "entity_id": "E001"}}
      ],
      "summary": "事件描述",
      "evidence_sentences": ["原文句子"],
      "reason": "有触发词、时间、参与者"
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
      "related_event_ids": ["EV001"],
      "reason": "核心概念归纳"
    }}
  ],
  "categorization": {{
    "categories": [
      {{
        "category_id": "C001",
        "category_name": "类别名称",
        "member_entity_ids": ["E001", "E002"],
        "shared_attributes": [],
        "reason": "实体共性归纳"
      }}
    ],
    "uncategorized": []
  }}
}}

================================================================================
八、文本
================================================================================

{text}

输出JSON："""

    def _parse_json(self, content: str) -> Dict:
        # 移除 markdown 代码块标记
        content = re.sub(r'^```(?:json)?\s*', '', content.strip())
        content = re.sub(r'\s*```$', '', content)

        # 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"  🔍 第一次解析失败: {e}")

        # 尝试提取 JSON 对象
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                print(f"  🔍 提取代码块后解析失败: {e}")

        print(f"  ❌ JSON解析失败")
        print(f"  🔍 原始内容前500字符: {content[:500]}")
        print(f"  🔍 原始内容后200字符: {content[-200:]}")
        return self._empty_result()

    def _empty_result(self) -> Dict:
        return {
            "entities": [],
            "relationships": [],
            "events": [],
            "themes": [],
            "categorization": {"categories": [], "uncategorized": []}
        }


def extract_info_from_texts(
        texts: List[str],
        api_base: str,
        api_key: str,
        model: str = "deepseek-chat"
) -> List[Dict]:
    client = InfoExtractorClient(api_base, api_key, model)
    return client.extract_batch(texts)