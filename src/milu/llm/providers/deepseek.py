"""
DeepSeekLLM - 深度求索（DeepSeek）模型实现。

支持能力：流式输出、函数调用、JSON模式、思考模式

API文档: https://api-docs.deepseek.com/
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from milu.llm.base.exceptions import StreamError
from milu.llm.base.message import Message
from milu.llm.base.response import StreamChunk, TokenUsage
from milu.llm.providers.base import BaseLLM, ModelCapabilities

logger = logging.getLogger(__name__)


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

    # 各模型真实上下文窗口（未收录回退到 _capabilities 的 131072，覆盖 v3.x/chat/reasoner=128K）
    # 数据截至 2026-06：V4 全系（v4 / v4-flash / v4-pro）均为 1M 长上下文。
    _context_windows = {
        "deepseek-v4": 1_000_000,   # 子串匹配覆盖 v4 / v4-flash / v4-pro
    }

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


    def _get_available_param_names(self) -> set[str]:
        return self._param_names

    async def chat(self, messages: list[Message], **kwargs) -> AsyncIterator[StreamChunk]:
        """
        流式聊天接口。

        DeepSeek特有参数映射:
            enable_thinking (bool) → extra_body.thinking = {"type": "enabled"/"disabled"}
            thinking_level (str)   → 顶层 reasoning_effort（思考力度；DeepSeek 接受 high/max，
                low/medium 自动映射为 high）。【不是】thinking 对象里的 budget_tokens——
                官方 thinking 对象不含该字段。
        """
        client = self._get_client()

        request_params = {
            "model": self.model,
            "messages": self._messages_to_dicts(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        #=================参数解释====================#
        # 注释：
        # 所有从父类获得的参数加上运行时用户实际传过来的参数kwargs大集合。
        validated = self._validate_params(kwargs)
        # 注释：
        # 保留基础参数，过滤多余公共参数
        # 从父类已经继承了非常多的公共的参数包括默认值，但是未必是本模型能用的，所以要在这里过滤掉不能用的
        for key in ("temperature", "top_p", "max_tokens", "stop",
                     "frequency_penalty", "presence_penalty"):
            if key in validated:
                request_params[key] = validated[key]
        #============================================#

        # 函数调用
        if "tools" in validated:
            request_params["tools"] = validated["tools"]
        if "tool_choice" in validated:
            request_params["tool_choice"] = validated["tool_choice"]

        # extra_body: 思考模式
        # DeepSeek（v4-pro/flash）开关思考用 extra_body.thinking = {"type": "enabled"/"disabled"}。
        # ⚠️ thinking 对象【不含】budget_tokens（那是 Claude 的字段，DeepSeek 官方文档无此项，
        # 传了会被当未知字段）。思考力度由独立的顶层 reasoning_effort 控制
        # （DeepSeek 接受 high/max，low/medium 自动映射为 high）。
        extra_body = {}
        if "enable_thinking" in validated and validated["enable_thinking"]:
            extra_body["thinking"] = {"type": "enabled"}
            level = validated.get("thinking_level")
            if level:
                request_params["reasoning_effort"] = level
        elif "enable_thinking" in validated and not validated["enable_thinking"]:
            # 显式关闭思考
            extra_body["thinking"] = {"type": "disabled"}
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
            result.usage = TokenUsage.from_api_usage(chunk.usage)
        return result


# 自动注册到 ModelRegistry
from milu.llm.providers import ModelRegistry
ModelRegistry.register("deepseek", DeepSeekLLM)
