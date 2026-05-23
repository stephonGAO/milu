"""
DoubaoLLM - 豆包/火山引擎模型实现。

支持能力：流式输出、函数调用、JSON模式、联网搜索、
文本嵌入、视觉理解、图片生成

API文档: https://ark.cn-beijing.volces.com/api/v3
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from agent_framework.llm.base.exceptions import StreamError
from agent_framework.llm.base.message import Message
from agent_framework.llm.base.response import StreamChunk, TokenUsage
from agent_framework.llm.providers.base import BaseLLM, ModelCapabilities

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
            web_search (bool) → tools列表追加 {"type": "web_search"}
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

        # extra_body: 联网搜索（通过 tools 列表注入 web_search 类型）
        if validated.get("web_search"):
            tools_list = request_params.get("tools", [])
            tools_list.append({"type": "web_search"})
            request_params["tools"] = tools_list

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
from agent_framework.llm.providers import ModelRegistry
ModelRegistry.register("doubao", DoubaoLLM)
