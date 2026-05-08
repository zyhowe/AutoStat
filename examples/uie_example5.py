"""
完整信息抽取 - DeepSeek API
事件 + 实体 + 属性 + 关系 + 主题
"""
import os
import json
from typing import List, Dict, Any
from openai import OpenAI

client = OpenAI(
    api_key="sk-c0e1f1ad1a3b41429a92f29251775ecf",
    base_url="https://api.deepseek.com"
)


def extract_full_information(text: str, model: str = "deepseek-chat") -> Dict[str, Any]:
    """
    从文本中一次性抽取：事件、实体、属性、关系、主题
    """
    prompt = f"""
从以下文本中抽取完整的信息结构，包括：事件、实体、属性、关系、主题。

文本：
{text}

输出格式（严格按此 JSON 结构）：
{{
  "summary": {{
    "topic": "全文主题（一句话概括）",
    "domain": "领域（如：财经、科技、政治、体育等）",
    "time_range": "文本覆盖的时间范围（如：单一时间点、多天、多年）"
  }},
  "entities": [
    {{
      "name": "实体名称",
      "type": "实体类型（人物/组织/地点/产品/时间/数值/其他）",
      "attributes": {{
        "属性名": "属性值"
      }},
      "mentions": ["原文提及片段"]
    }}
  ],
  "relations": [
    {{
      "subject": "关系主体",
      "predicate": "关系谓语",
      "object": "关系客体",
      "description": "关系描述"
    }}
  ],
  "events": [
    {{
      "event_id": 1,
      "summary": "事件一句话描述",
      "trigger": "触发词/动词",
      "subject": "事件主体（施事者）",
      "object": "事件客体（受事者）",
      "time": "发生时间",
      "location": "发生地点",
      "related_entities": ["关联的实体名"],
      "sentence": "原文依据"
    }}
  ],
  "statistics": {{
    "entity_count": 0,
    "relation_count": 0,
    "event_count": 0
  }}
}}

要求：
1. 实体去重：同名实体只保留一条，合并 mentions
2. 关系方向明确：subject -> predicate -> object
3. 事件按文本中发生的顺序编号
4. 属性提取：如人物的职位、公司的营收、产品的规格等
5. 只输出 JSON，不要其他内容
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个信息抽取专家，从文本中提取结构化知识。只输出 JSON。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        seed=42,  # 固定随机种子
        top_p=0.5,  # 降低采样范围
        response_format={"type": "json_object"}
    )
    result = json.loads(response.choices[0].message.content)

    # 自动统计
    if "statistics" in result:
        result["statistics"]["entity_count"] = len(result.get("entities", []))
        result["statistics"]["relation_count"] = len(result.get("relations", []))
        result["statistics"]["event_count"] = len(result.get("events", []))

    return result


def extract_information_streaming(text: str, return_all: bool = True) -> Dict[str, Any]:
    """
    流式/分步抽取：适合超长文本
    """
    steps = ["主题", "实体", "关系", "事件"]
    result = {"summary": {}, "entities": [], "relations": [], "events": []}

    for step in steps:
        prompt = f"""
从以下文本中抽取{step}信息。

文本：
{text}

抽取{step}：
{' - 识别全文主题、领域、时间范围' if step == '主题' else ''}
{' - 提取所有实体（人物/组织/地点/产品/数值）及其属性' if step == '实体' else ''}
{' - 抽取实体间的关系（subject-predicate-object）' if step == '关系' else ''}
{' - 提取所有事件（触发词、主体、客体、时间、地点）' if step == '事件' else ''}

输出格式：只输出 JSON，包含上述{step}信息。
"""
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        step_result = json.loads(response.choices[0].message.content)

        if step == "主题":
            result["summary"] = step_result
        elif step == "实体":
            result["entities"] = step_result.get("entities", [])
        elif step == "关系":
            result["relations"] = step_result.get("relations", [])
        elif step == "事件":
            result["events"] = step_result.get("events", [])

    return result


# 测试
if __name__ == "__main__":
    test_text = """
    2024年3月15日，宁德时代在福建宁德召开年度业绩说明会。
    会上，董事长曾毓群宣布，公司2024年全年营收847亿元，同比增长15%，净利润120亿元。
    同时，宁德时代与特斯拉签署长期供货协议，将从2025年起向特斯拉供应储能电池，合同金额约300亿元。
    曾毓群表示，公司计划在匈牙利投资50亿欧元建设第二座欧洲工厂，预计2026年投产，年产能100GWh。
    当天下午，宁德时代股价上涨3.5%，收于180元，市值突破8000亿元。
    此外，阿里巴巴同日宣布以50亿元收购某科技公司，加码AI业务布局。
    """

    print("=" * 80)
    print("完整信息抽取结果")
    print("=" * 80)

    result = extract_full_information(test_text)

    # 打印摘要
    print(f"\n【主题】{result['summary'].get('topic', 'N/A')}")
    print(f"【领域】{result['summary'].get('domain', 'N/A')}")
    print(f"【时间范围】{result['summary'].get('time_range', 'N/A')}")

    # 打印实体
    print(f"\n【实体】（共 {result['statistics']['entity_count']} 个）")
    for entity in result.get('entities', []):
        attrs = f", 属性: {entity['attributes']}" if entity.get('attributes') else ""
        print(f"  - {entity['name']} ({entity['type']}){attrs}")

    # 打印关系
    print(f"\n【关系】（共 {result['statistics']['relation_count']} 个）")
    for relation in result.get('relations', []):
        print(f"  - {relation['subject']} -> {relation['predicate']} -> {relation['object']}")

    # 打印事件
    print(f"\n【事件】（共 {result['statistics']['event_count']} 个）")
    for event in result.get('events', []):
        print(f"  {event['event_id']}. {event['summary']}")
        print(f"     触发词: {event['trigger']}, 主体: {event['subject']}, 客体: {event['object']}")
        if event.get('time'):
            print(f"     时间: {event['time']}")

    # 保存完整结果
    with open("extraction_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n完整结果已保存到 extraction_result.json")