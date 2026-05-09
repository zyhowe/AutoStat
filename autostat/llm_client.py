"""
大模型客户端模块 - 支持 OpenAI 兼容 API
"""

import requests
import json
from typing import Dict, Any, Generator, List, Optional


class LLMClient:
    """大模型客户端 - 支持 OpenAI 兼容 API"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化大模型客户端

        参数:
        - config: 包含 api_base, api_key, model 的字典
        """
        self.api_base = config.get('api_base', '').rstrip('/')
        self.api_key = config.get('api_key', '')
        self.model = config.get('model', '')
        self.timeout = config.get('timeout', 30)

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat_stream(self, messages: List[Dict[str, str]],
                    temperature: float = 0.7,
                    tools: Optional[List[Dict]] = None,
                    tool_choice: Optional[Dict] = None) -> Generator[str, None, None]:
        """
        流式对话

        参数:
        - messages: 消息列表，格式 [{"role": "user", "content": "..."}]
        - temperature: 温度参数
        - tools: 工具/函数定义列表（可选）
        - tool_choice: 工具选择配置（可选）

        返回:
        - 生成器，逐字输出回答
        """
        if not self.api_base or not self.model:
            yield "错误：大模型配置不完整，请检查API地址和模型名称"
            return

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True
        }

        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout,
                stream=True
            )

            if response.status_code != 200:
                yield f"错误：HTTP {response.status_code} - {response.text[:200]}"
                return

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue

        except requests.exceptions.Timeout:
            yield "错误：请求超时"
        except requests.exceptions.ConnectionError:
            yield "错误：无法连接到API服务器"
        except Exception as e:
            yield f"错误：{str(e)}"

    def chat(self, messages: List[Dict[str, str]],
             temperature: float = 0.7,
             tools: Optional[List[Dict]] = None,
             tool_choice: Optional[Dict] = None) -> str:
        """
        非流式对话
        """
        if not self.api_base or not self.model:
            return "错误：大模型配置不完整"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }

        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout
            )

            if response.status_code != 200:
                return f"错误：HTTP {response.status_code}"

            result = response.json()

            if 'choices' in result and len(result['choices']) > 0:
                message = result['choices'][0].get('message', {})
                if 'tool_calls' in message and message['tool_calls']:
                    return ""
                return message.get('content', '')

            return ""

        except Exception as e:
            return f"错误：{str(e)}"

    def chat_complete(self, messages: List[Dict[str, str]],
                      temperature: float = 0.7,
                      tools: Optional[List[Dict]] = None,
                      tool_choice: Optional[Dict] = None) -> Dict[str, Any]:
        """
        返回完整的响应对象（用于解析 tool_calls）
        """
        if not self.api_base or not self.model:
            return {"error": "大模型配置不完整"}

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }

        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout
            )

            if response.status_code != 200:
                return {"error": f"HTTP {response.status_code}: {response.text[:200]}"}

            return response.json()

        except Exception as e:
            return {"error": str(e)}