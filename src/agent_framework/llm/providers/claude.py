"""
ClaudeLLM - Anthropic Claude 模型实现（OpenAI 兼容模式）。

使用 Anthropic 提供的 OpenAI 兼容端点，复用 AsyncOpenAI 客户端。
支持能力：流式输出、函数调用、JSON 模式、思考模式、视觉理解、文档理解

API 文档: https://docs.anthropic.com/en/docs/about-claude/models
兼容层: https://docs.anthropic.com/en/api/openai-compatibility
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from agent_framework.llm.base.exceptions import StreamError
from agent_framework.llm.base.message import Message
from agent_framework.llm.base.response import StreamChunk, TokenUsage
from agent_framework.llm.providers.base import BaseLLM, ModelCapabilities

logger = logging.getLogger(__name__)


class ClaudeLLM(BaseLLM):
    """Anthropic Claude 模型实现"""

    _capabilities = ModelCapabilities(
        supports_streaming=True,
        supports_function_calling=True,
        supports_json_mode=True,
        supports_web_search=False,
        supports_thinking=True,
        supports_embedding=False,
        supports_vision=True,
        supports_audio_understand=False,
        supports_video=False,
        supports_document=True,
        supports_image_generation=False,
        supports_audio_generation=False,
        max_context_window=200000,
        supported_output_formats=("text", "json"),
    )

    _param_names = {
        "temperature", "top_p", "max_tokens", "stop",
        "enable_thinking", "thinking_budget",
        "tools", "tool_choice",
    }

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def base_url(self) -> str:
        return "https://api.anthropic.com/v1/"

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def _get_available_param_names(self) -> set[str]:
        return self._param_names

    async def chat(self, messages: list[Message], **kwargs) -> AsyncIterator[StreamChunk]:
        """
        流式聊天接口。

        Claude 特有参数映射:
            enable_thinking (bool) → extra_body.thinking
            thinking_budget (int)  → extra_body.thinking.budget_tokens
        """
        client = self._get_client()

        request_params = {
            "model": self.model,
            "messages": self._messages_to_dicts(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        validated = self._validate_params(kwargs)

        for key in ("temperature", "top_p", "max_tokens", "stop"):
            if key in validated:
                request_params[key] = validated[key]

        # 函数调用
        if "tools" in validated:
            request_params["tools"] = validated["tools"]
        if "tool_choice" in validated:
            request_params["tool_choice"] = validated["tool_choice"]

        # extra_body: 扩展思考
        # Claude 使用 thinking = {"type": "enabled", "budget_tokens": N}
        extra_body = {}
        if "enable_thinking" in validated and validated["enable_thinking"]:
            budget = validated.get("thinking_budget", 10000)
            extra_body["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget,
            }
            # Claude 要求开启思考时 temperature 必须为 1
            request_params["temperature"] = 1.0
        elif "enable_thinking" in validated and not validated["enable_thinking"]:
            extra_body["thinking"] = {"type": "disabled"}
        if extra_body:
            request_params["extra_body"] = extra_body

        try:
            response = await client.chat.completions.create(**request_params)
            async for chunk in response:
                yield self._parse_chunk(chunk)
        except Exception as e:
            raise StreamError(f"Claude 流式调用异常: {e}") from e

    def _parse_chunk(self, chunk) -> StreamChunk:
        """解析 OpenAI SDK 的流式 chunk 为统一的 StreamChunk"""
        result = StreamChunk()
        if chunk.choices:
            delta = chunk.choices[0].delta
            result.content = delta.content
            result.finish_reason = chunk.choices[0].finish_reason
            # Claude 扩展思考的推理内容
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
from agent_framework.llm.providers import ModelRegistry
ModelRegistry.register("anthropic", ClaudeLLM)
