"""
GeminiLLM - Google Gemini 模型实现（OpenAI 兼容模式）。

使用 Google 提供的 OpenAI 兼容端点，复用 AsyncOpenAI 客户端。
支持能力：流式输出、函数调用、JSON 模式、思考模式、视觉理解、文档理解

API 文档: https://ai.google.dev/gemini-api/docs/openai
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from milu.llm.base.exceptions import StreamError
from milu.llm.base.message import Message
from milu.llm.base.response import StreamChunk, TokenUsage
from milu.llm.providers.base import BaseLLM, ModelCapabilities

logger = logging.getLogger(__name__)


class GeminiLLM(BaseLLM):
    """Google Gemini 模型实现"""

    _capabilities = ModelCapabilities(
        supports_streaming=True,
        supports_function_calling=True,
        supports_json_mode=True,
        supports_web_search=False,
        supports_thinking=True,
        supports_embedding=True,
        supports_vision=True,
        supports_audio_understand=True,
        supports_video=True,
        supports_document=True,
        supports_image_generation=False,
        supports_audio_generation=False,
        max_context_window=1048576,   # 1M：Flash/Pro 系列默认，未收录模型的回退
        supported_output_formats=("text", "json"),
    )

    # 各模型真实上下文窗口（未收录回退到 1048576=1M）。数据截至 2026-06：
    # 当前 Gemini 3.x / 2.5 系列均为 1M（显式列出便于核对）；仅 1.5 Pro 为 2M。
    _context_windows = {
        "gemini-3.5-flash": 1_048_576,
        "gemini-3.1-pro": 1_048_576,
        "gemini-3-flash-preview": 1_048_576,
    }

    _param_names = {
        "temperature", "top_p", "max_tokens", "stop",
        "frequency_penalty", "presence_penalty",
        "enable_thinking",
        "tools", "tool_choice",
    }

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def base_url(self) -> str:
        return "https://generativelanguage.googleapis.com/v1beta/openai/"


    def _get_available_param_names(self) -> set[str]:
        return self._param_names

    async def chat(self, messages: list[Message], **kwargs) -> AsyncIterator[StreamChunk]:
        """
        流式聊天接口。

        Gemini 特有参数映射:
            enable_thinking=False → reasoning_effort="none"（关闭思考；OpenAI 兼容端点机制）
            enable_thinking=True  → 不显式设置（Gemini 2.5/3 默认即开启动态思考）
        """
        client = self._get_client()

        request_params = {
            "model": self.model,
            "messages": self._messages_to_dicts(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        validated = self._validate_params(kwargs)

        for key in ("temperature", "top_p", "max_tokens", "stop",
                     "frequency_penalty", "presence_penalty"):
            if key in validated:
                request_params[key] = validated[key]

        # 函数调用
        if "tools" in validated:
            request_params["tools"] = validated["tools"]
        if "tool_choice" in validated:
            request_params["tool_choice"] = validated["tool_choice"]

        # 思考控制：Gemini 的 OpenAI 兼容端点用顶层参数 reasoning_effort（"none" 关思考），
        # 【不是】extra_body.thinking——后者不是合法字段，会直接 400
        # "Unknown name 'thinking'"（旧实现的 bug；当时注释写的 thinking_config 也没落地）。
        # 约定：enable_thinking=False → reasoning_effort="none"；enable_thinking=True 不显式
        # 设置（Gemini 2.5/3 默认即开启动态思考，无需指定，也避开"某些 Gemini 3 拒绝
        # reasoning_effort=medium"的坑）。
        # 注意：reasoning_effort="none" 对 2.5 Flash 可彻底关闭思考；2.5 Pro / 3.x 思考模型
        # 不支持完全关闭，该值通常被忽略（思考保持开启）而不报错。
        # 如需更细粒度控制，可改用 extra_body={"google":{"thinking_config":{...}}}。
        if "enable_thinking" in validated and not validated["enable_thinking"]:
            request_params["reasoning_effort"] = "none"

        try:
            response = await client.chat.completions.create(**request_params)
            async for chunk in response:
                yield self._parse_chunk(chunk)
        except Exception as e:
            raise StreamError(f"Gemini 流式调用异常: {e}") from e

    def _parse_chunk(self, chunk) -> StreamChunk:
        """解析 OpenAI SDK 的流式 chunk 为统一的 StreamChunk"""
        result = StreamChunk()
        if chunk.choices:
            delta = chunk.choices[0].delta
            result.content = delta.content
            result.finish_reason = chunk.choices[0].finish_reason
            # Gemini 2.5 的推理内容（兼容层可能以 reasoning_content 传递）
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
from milu.llm.providers import ModelRegistry
ModelRegistry.register("gemini", GeminiLLM)
