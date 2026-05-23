"""
DeepSeekLLM - 深度求索（DeepSeek）模型实现。

支持能力：流式输出、函数调用、JSON模式、思考模式

API文档: https://api-docs.deepseek.com/
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from agent_framework.exceptions import StreamError
from agent_framework.models.message import Message
from agent_framework.models.response import StreamChunk, TokenUsage
from agent_framework.providers.base import BaseLLM, ModelCapabilities

logger = logging.getLogger(__name__)

# 思考等级到 budget_tokens 的映射
_THINKING_BUDGET_MAP = {
    "low": 1024,
    "medium": 4096,
    "high": 8192,
}


class DeepSeekLLM(BaseLLM):
    """深度求索模型实现"""

    _capabilities = ModelCapabilities(
        supports_streaming=True,
        supports_function_calling=True,
        supports_json_mode=True,
        supports_web_search=False,
        supports_thinking=True,
        supports_embedding=False,
        supports_vision=False,
        supports_audio_understand=False,
        supports_video=False,
        supports_document=False,
        supports_image_generation=False,
        supports_audio_generation=False,
        max_context_window=131072,
        supported_output_formats=("text", "json"),
    )

    _param_names = {
        "temperature", "top_p", "max_tokens", "stop",
        "frequency_penalty", "presence_penalty",
        "enable_thinking", "thinking_level",
        "tools", "tool_choice",
    }

    @property
    def provider_name(self) -> str:
        return "deepseek"

    @property
    def base_url(self) -> str:
        return "https://api.deepseek.com/v1"

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def _get_available_param_names(self) -> set[str]:
        return self._param_names

    async def chat(self, messages: list[Message], **kwargs) -> AsyncIterator[StreamChunk]:
        """
        流式聊天接口。

        DeepSeek特有参数映射:
            enable_thinking (bool) → extra_body.thinking
            thinking_level (str) → extra_body.thinking_budget:
                low → 1024
                medium → 4096 (默认)
                high → 8192
        """
        validated = self._validate_params(kwargs)
        client = self._get_client()

        request_params = {
            "model": self.model,
            "messages": self._messages_to_dicts(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        # 基础参数
        for key in ("temperature", "top_p", "max_tokens", "stop",
                     "frequency_penalty", "presence_penalty"):
            if key in validated:
                request_params[key] = validated[key]

        # 函数调用
        if "tools" in validated:
            request_params["tools"] = validated["tools"]
        if "tool_choice" in validated:
            request_params["tool_choice"] = validated["tool_choice"]

        # extra_body: 思考模式
        # DeepSeek 使用 extra_body.thinking = {"type": "enabled", "budget_tokens": N}
        extra_body = {}
        if validated.get("enable_thinking"):
            thinking_level = validated.get("thinking_level", "medium")
            budget = _THINKING_BUDGET_MAP.get(thinking_level, 4096)
            extra_body["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget,
            }
        if extra_body:
            request_params["extra_body"] = extra_body

        try:
            response = await client.chat.completions.create(**request_params)
            async for chunk in response:
                yield self._parse_chunk(chunk)
        except Exception as e:
            raise StreamError(f"DeepSeek 流式调用异常: {e}") from e

    def _parse_chunk(self, chunk) -> StreamChunk:
        """解析OpenAI SDK的流式chunk为统一的StreamChunk"""
        result = StreamChunk()
        if chunk.choices:
            delta = chunk.choices[0].delta
            result.content = delta.content
            result.finish_reason = chunk.choices[0].finish_reason
            # DeepSeek reasoner 模型的推理内容
            if hasattr(delta, "reasoning_content"):
                result.reasoning_content = delta.reasoning_content
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                result.tool_calls = delta.tool_calls
        if chunk.usage:
            result.usage = TokenUsage(
                prompt_tokens=chunk.usage.prompt_tokens or 0,
                completion_tokens=chunk.usage.completion_tokens or 0,
                total_tokens=chunk.usage.total_tokens or 0,
            )
        return result


# 自动注册到 ModelRegistry
from agent_framework.providers import ModelRegistry
ModelRegistry.register("deepseek", DeepSeekLLM)
