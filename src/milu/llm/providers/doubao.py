"""
DoubaoLLM - 豆包/火山引擎模型实现。

支持能力：流式输出、函数调用、JSON模式、联网搜索、
文本嵌入、视觉理解、图片生成

API文档: https://ark.cn-beijing.volces.com/api/v3
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from milu.llm.base.exceptions import StreamError
from milu.llm.base.message import Message
from milu.llm.base.response import StreamChunk, TokenUsage
from milu.llm.providers.base import BaseLLM, ModelCapabilities

logger = logging.getLogger(__name__)


class DoubaoLLM(BaseLLM):
    """豆包/火山引擎 Doubao 模型实现"""

    _capabilities = ModelCapabilities(
        supports_streaming=True,
        supports_function_calling=True,
        supports_json_mode=True,
        supports_web_search=True,
        supports_thinking=False,
        supports_embedding=True,
        supports_vision=True,
        supports_audio_understand=False,
        supports_video=False,
        supports_document=False,
        supports_image_generation=True,
        supports_audio_generation=False,
        max_context_window=131072,
        supported_output_formats=("text", "json"),
    )

    _param_names = {
        "temperature", "top_p", "max_tokens", "stop",
        "frequency_penalty", "presence_penalty",
        "web_search", "web_search_strategy",
        "tools", "tool_choice",
        "image_size", "image_quality", "num_images",
    }

    @property
    def provider_name(self) -> str:
        return "doubao"

    @property
    def base_url(self) -> str:
        return "https://ark.cn-beijing.volces.com/api/v3"

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def _get_available_param_names(self) -> set[str]:
        return self._param_names

    async def chat(self, messages: list[Message], **kwargs) -> AsyncIterator[StreamChunk]:
        """
        流式聊天接口。

        Doubao 特有参数映射:
            web_search (bool) → 不支持。豆包 OpenAI 兼容 chat completions 端点要求
                tools 内每一项均为 {"type":"function",...}，无法注入内置联网工具项；
                原生联网需走「应用(Bot)」端点或 Responses API（本框架未接入）。
                此处忽略该参数并一次性提示改用内置 web_search 工具联网。
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
            request_params["tools"] = list(validated["tools"])
        if "tool_choice" in validated:
            request_params["tool_choice"] = validated["tool_choice"]

        # 联网搜索：豆包的 OpenAI 兼容 chat completions 端点【不支持】把
        # {"type": "web_search"} 作为 tools 项注入——该端点校验 tools 内每一项都
        # 必须是 {"type": "function", "function": {...}}，否则直接 400
        # "missing tools.function parameter"，连带把同批的正常函数调用一起打挂
        # （即便只发一句「你好」，Agent 也会带上全套内置函数工具，故必崩）。
        # 豆包原生联网需走「应用(Bot)」端点或 Responses API（本框架统一使用
        # chat completions，未接入）。因此这里不再注入非法工具项，改为一次性提示
        # 用户使用内置 web_search 工具（BUILTIN_TOOLS，后端可配 bocha/tavily）联网。
        if validated.get("web_search") and "web_search" not in self._warned_params:
            logger.warning(
                "[doubao] 原生联网搜索在 chat completions 端点不可用，已忽略 web_search "
                "参数；如需联网请使用内置 web_search 工具（设置 WEB_SEARCH_PROVIDER=bocha/tavily）"
            )
            self._warned_params.add("web_search")

        try:
            response = await client.chat.completions.create(**request_params)
            async for chunk in response:
                yield self._parse_chunk(chunk)
        except Exception as e:
            raise StreamError(f"Doubao 流式调用异常: {e}") from e

    def _parse_chunk(self, chunk) -> StreamChunk:
        """解析 OpenAI SDK 的流式 chunk 为统一的 StreamChunk"""
        result = StreamChunk()
        if chunk.choices:
            delta = chunk.choices[0].delta
            result.content = delta.content
            result.finish_reason = chunk.choices[0].finish_reason
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
ModelRegistry.register("doubao", DoubaoLLM)
