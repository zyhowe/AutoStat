"""
大模型翻译服务
技术结论 → 业务语言
字段映射解析（从用户输入文本中抽取）
"""

import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import requests

logger = logging.getLogger(__name__)


class LLMTranslator:
    """大模型翻译器"""

    def __init__(self, api_base: Optional[str] = None, api_key: Optional[str] = None,
                 model: Optional[str] = None, timeout: int = 120):
        """
        初始化翻译器

        参数:
        - api_base: API地址
        - api_key: API密钥
        - model: 模型名称
        - timeout: 超时时间（秒），默认120秒
        """
        self.api_base = api_base or ""
        self.api_key = api_key or ""
        self.model = model or ""
        self.timeout = timeout
        self.max_retries = 2
        logger.info(f"LLMTranslator 初始化: timeout={self.timeout}s, model={self.model}")

    def translate_scenarios_batch(
        self,
        scenario_results: List[Dict[str, Any]],
        field_mapping: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        批量翻译所有场景（一次大模型调用）

        参数:
        - scenario_results: 场景执行结果列表
        - field_mapping: 字段映射 {字段名: 中文名}

        返回:
        - 翻译后的场景结果列表
        """
        if not self.api_base or not self.model:
            logger.warning("大模型未配置，使用降级翻译")
            return self._fallback_translate(scenario_results)

        # 过滤出已完成的场景
        completed_scenarios = [s for s in scenario_results if s.get("status") == "completed"]
        if not completed_scenarios:
            return scenario_results

        try:
            # 构建批量翻译 Prompt
            prompt = self._build_batch_translate_prompt(completed_scenarios, field_mapping)
            response = self._call_llm_with_retry(prompt)
            parsed = self._parse_batch_translate_response(response, completed_scenarios)

            # 将翻译结果合并回原始数据
            result_map = {r.get("scenario_id"): r for r in scenario_results}
            for scenario_id, translation in parsed.items():
                if scenario_id in result_map:
                    result_map[scenario_id]["business_summary"] = translation.get("summary", "")
                    result_map[scenario_id]["business_findings"] = translation.get("findings", [])
                    result_map[scenario_id]["business_actions"] = translation.get("actions", [])

            return scenario_results

        except Exception as e:
            logger.error(f"批量翻译失败: {e}")
            # 降级：逐个翻译
            logger.warning("降级到逐个翻译模式")
            return self.translate_scenarios_individual(scenario_results, field_mapping)

    def translate_scenarios_individual(
        self,
        scenario_results: List[Dict[str, Any]],
        field_mapping: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """逐个翻译（降级方案）"""
        if not self.api_base or not self.model:
            return self._fallback_translate(scenario_results)

        translated = []
        for result in scenario_results:
            if result.get("status") != "completed":
                translated.append(result)
                continue

            try:
                prompt = self._build_individual_translate_prompt(result, field_mapping)
                response = self._call_llm_with_retry(prompt)
                parsed = self._parse_individual_translate_response(response)

                result["business_summary"] = parsed.get("summary", result.get("name", ""))
                result["business_findings"] = parsed.get("findings", [])
                result["business_actions"] = parsed.get("actions", [])
                translated.append(result)
            except Exception as e:
                logger.error(f"翻译场景 {result.get('scenario_id')} 失败: {e}")
                result["business_summary"] = "翻译失败，请检查大模型配置或网络连接"
                result["business_findings"] = []
                result["business_actions"] = []
                result["_translation_error"] = str(e)
                translated.append(result)

        return translated

    def _build_batch_translate_prompt(
        self,
        scenarios: List[Dict[str, Any]],
        field_mapping: Dict[str, str]
    ) -> str:
        """构建批量翻译 Prompt"""
        # 字段映射文本
        mapping_items = list(field_mapping.items())[:30]
        mapping_text = "\n".join([f"  - {k} = {v}" for k, v in mapping_items])
        if len(field_mapping) > 30:
            mapping_text += f"\n  ... 还有 {len(field_mapping) - 30} 个字段"

        # 每个场景的技术结论
        scenarios_text = []
        for i, s in enumerate(scenarios, 1):
            name = s.get("name", f"场景{i}")
            conclusions = s.get("conclusions", [])
            tech_texts = []
            for c in conclusions:
                text = c.get("text", "")
                if text:
                    tech_texts.append(text)
            scenarios_text.append(f"【场景{i}: {name}】\n" + "\n".join([f"  - {t}" for t in tech_texts[:5]]))

        prompt = f"""
你是一个数据分析翻译助手。请把以下技术结论批量翻译成业务语言。

## 字段映射
{mapping_text}

## 需要翻译的场景
{chr(10).join(scenarios_text)}

## 要求
1. 用中文名替换技术字段名
2. 每个场景输出三段式：摘要(<=30字)、发现列表(2-3条)、建议列表(2-3条)
3. 给出具体数字

## 输出格式
请输出一个JSON对象，键为场景编号（如 "scene_1", "scene_2"），值为该场景的翻译结果。

示例格式：
{{
  "scene_1": {{
    "summary": "摘要内容",
    "findings": ["发现1", "发现2"],
    "actions": ["建议1", "建议2"]
  }},
  "scene_2": {{
    "summary": "摘要内容",
    "findings": ["发现1", "发现2"],
    "actions": ["建议1", "建议2"]
  }}
}}

请输出JSON：
"""
        return prompt

    def _build_individual_translate_prompt(
        self,
        result: Dict[str, Any],
        field_mapping: Dict[str, str]
    ) -> str:
        """构建单个翻译 Prompt"""
        conclusions = result.get("conclusions", [])
        tech_texts = []
        for c in conclusions:
            if c.get("type") == "summary":
                tech_texts.insert(0, c.get("text", ""))
            elif len(tech_texts) < 6:
                tech_texts.append(c.get("text", ""))

        mapping_items = list(field_mapping.items())[:10]
        mapping_text = "\n".join([f"  - {k} = {v}" for k, v in mapping_items])
        if len(field_mapping) > 10:
            mapping_text += f"\n  ... 还有 {len(field_mapping) - 10} 个字段"

        prompt = f"""
你是一个数据分析翻译助手。请把技术结论翻译成业务语言。

## 场景
{result.get('name', '数据分析')}

## 技术结论
{chr(10).join([f"- {t}" for t in tech_texts[:8]])}

## 字段映射
{mapping_text}

## 要求
1. 用中文名替换技术字段名
2. 输出三段式JSON：摘要(<=30字)、发现列表、建议列表
3. 给出具体数字

输出JSON：
"""
        return prompt

    def _call_llm_with_retry(self, prompt: str) -> str:
        """带重试的LLM调用"""
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return self._call_llm(prompt)
            except Exception as e:
                last_exception = e
                logger.warning(f"LLM调用失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
        raise last_exception or Exception("所有重试均失败")

    def _call_llm(self, prompt: str) -> str:
        """调用大模型 API"""
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 4000,  # 批量翻译需要更多token
            "stream": False
        }

        timeout_seconds = 120
        logger.info(f"调用大模型: {self.model}, timeout={timeout_seconds}s")

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout_seconds
            )
            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not content:
                    raise Exception("大模型返回空内容")
                logger.info("大模型调用成功")
                return content
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")
        except requests.exceptions.Timeout:
            logger.error(f"请求超时（{timeout_seconds}秒）")
            raise Exception(f"请求超时（{timeout_seconds}秒），请检查网络或增大超时时间")
        except Exception as e:
            logger.error(f"大模型调用失败: {e}")
            raise

    def _parse_batch_translate_response(
        self,
        response: str,
        scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """解析批量翻译响应"""
        try:
            response = response.strip()
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)

                # 将 scene_1, scene_2 映射到 scenario_id
                result = {}
                for i, s in enumerate(scenarios, 1):
                    key = f"scene_{i}"
                    if key in data:
                        result[s.get("scenario_id", key)] = data[key]
                    # 尝试其他可能的键名
                    elif s.get("scenario_id") in data:
                        result[s.get("scenario_id")] = data[s.get("scenario_id")]
                    elif s.get("name") in data:
                        result[s.get("scenario_id")] = data[s.get("name")]

                # 如果没有匹配到任何场景，尝试直接使用整个JSON
                if not result:
                    for k, v in data.items():
                        if isinstance(v, dict) and "summary" in v:
                            result[k] = v

                return result
            else:
                raise Exception("未找到JSON对象")
        except Exception as e:
            logger.error(f"解析批量翻译响应失败: {e}")
            # 返回空结果，触发降级
            return {}

    def _parse_individual_translate_response(self, response: str) -> Dict[str, Any]:
        """解析单个翻译响应"""
        try:
            response = response.strip()
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                raise Exception("未找到JSON对象")
        except Exception as e:
            logger.error(f"解析翻译响应失败: {e}")
            raise

    def _fallback_translate(self, scenario_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """降级翻译（无大模型时）"""
        for result in scenario_results:
            if result.get("status") == "completed":
                result["business_summary"] = result.get("name", "数据分析")
                result["business_findings"] = []
                result["business_actions"] = []
                for c in result.get("conclusions", []):
                    if c.get("type") == "summary":
                        result["business_findings"].append(c.get("text", ""))
        return scenario_results


def translate_scenarios(
    scenario_results: List[Dict[str, Any]],
    table_structure: Dict[str, Any],
    field_mapping: Dict[str, str],
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    翻译场景结果（便捷函数）
    默认使用批量翻译
    """
    translator = LLMTranslator(
        api_base=config.get("api_base"),
        api_key=config.get("api_key"),
        model=config.get("model"),
        timeout=120
    )
    return translator.translate_scenarios_batch(scenario_results, field_mapping)


def parse_field_mapping(text: str, config: Dict[str, Any]) -> Tuple[Dict[str, str], List[str]]:
    """
    用大模型从用户输入的文本中抽取字段映射
    """
    translator = LLMTranslator(
        api_base=config.get("api_base"),
        api_key=config.get("api_key"),
        model=config.get("model"),
        timeout=120
    )

    if not translator.api_base or not translator.model:
        logger.warning("大模型未配置，使用规则解析")
        return parse_field_mapping_with_rules(text)

    prompt = f"""
从以下文本中提取字段名和中文名的对应关系。

文本：
{text}

要求：
1. 字段名是英文/下划线/驼峰命名
2. 中文名是中文词汇
3. 输出JSON：{{"mapping": {{"字段名": "中文名"}}, "unmatched": ["未识别字段"]}}

输出JSON：
"""
    try:
        response = translator._call_llm_with_retry(prompt)
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end > start:
            json_str = response[start:end]
            data = json.loads(json_str)
            mapping = data.get("mapping", {})
            unmatched = data.get("unmatched", [])
            mapping = {k: v for k, v in mapping.items() if k and v}
            unmatched = [u for u in unmatched if u]
            return mapping, unmatched
        else:
            raise Exception("未找到JSON对象")
    except Exception as e:
        logger.error(f"大模型解析失败: {e}")
        return parse_field_mapping_with_rules(text)


def parse_field_mapping_with_rules(text: str) -> Tuple[Dict[str, str], List[str]]:
    """简单规则匹配（降级方案）"""
    mapping = {}
    lines = text.strip().split('\n')

    # 尝试检测表格格式
    design_idx = -1
    chinese_idx = -1
    for i, line in enumerate(lines):
        if '成员设计名' in line or '字段名' in line:
            parts = line.split()
            for j, p in enumerate(parts):
                if '设计名' in p or '字段名' in p:
                    design_idx = j
                if '中文名' in p:
                    chinese_idx = j
            if design_idx != -1 and chinese_idx != -1:
                for line2 in lines[i+1:]:
                    parts2 = line2.split()
                    if len(parts2) > max(design_idx, chinese_idx):
                        field = parts2[design_idx].strip()
                        chinese = parts2[chinese_idx].strip()
                        if field and chinese:
                            mapping[field] = chinese
                return mapping, []

    # 键值对
    for line in lines:
        line = line.strip()
        if not line:
            continue
        for sep in ['：', ':', '=', '→']:
            if sep in line:
                parts = line.split(sep, 1)
                key = parts[0].strip()
                value = parts[1].strip()
                if key and value and len(key) > 1:
                    mapping[key] = value
                    break

    return mapping, []