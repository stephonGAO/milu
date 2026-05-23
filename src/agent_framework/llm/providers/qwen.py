"""
QwenLLM - 通义千问（阿里云 DashScope）模型实现。

支持能力：流式输出、函数调用、JSON模式、联网搜索、
思考模式、文本嵌入、图片/音频/视频/文档理解、图片生成

API文档: https://help.aliyun.com/zh/model-studio/
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from agent_framework.llm.base.exceptions import StreamError
from agent_framework.llm.base.message import Message
from agent_framework.llm.base.response import StreamChunk, TokenUsage
from agent_framework.llm.providers.base import BaseLLM, ModelCapabilities

logger = logging.getLogger(__name__)


class QwenLLM(BaseLLM):
    """通义千问模型实现"""

    _capabilities = ModelCapabilities(
        supports_streaming=True,
        supports_function_calling=True,
        supports_json_mode=True,
        supports_web_search=True,
        supports_thinking=True,
        supports_embedding=True,
        supports_vision=True,
        supports_audio_understand=True,
        supports_video=True,
        supports_document=True,
        supports_image_generation=True,
        supports_audio_generation=False,
        max_context_window=131072,
        supported_output_formats=("text", "json"),
    )

    _param_names = {
        "temperature", "top_p", "max_tokens", "stop",
        "frequency_penalty", "presence_penalty",
        "web_search", "web_search_strategy",
        "enable_thinking", "thinking_level",
        "tools", "tool_choice",
    }

    @property
    def provider_name(self) -> str:
        return "qwen"

    @property
    def base_url(self) -> str:
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def _get_available_param_names(self) -> set[str]:
        return self._param_names

    async def chat(self, messages: list[Message], **kwargs) -> AsyncIterator[StreamChunk]:
        """
        流式聊天接口。

        Qwen特有参数映射:
            web_search (bool) → extra_body.enable_search
            enable_thinking (bool) → extra_body.enable_thinking
            thinking_level (str) → Qwen不支持，静默忽略
        """
        validated = self._validate_params(kwargs)
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

        # extra_body: 联网搜索和思考模式
        extra_body = {}
        if validated.get("web_search"):
            extra_body["enable_search"] = True
        if "enable_thinking" in validated:
            extra_body["enable_thinking"] = bool(validated["enable_thinking"])
        if extra_body:
            request_params["extra_body"] = extra_body

        try:
            response = await client.chat.completions.create(**request_params)
            async for chunk in response:
                yield self._parse_chunk(chunk)
        except Exception as e:
            raise StreamError(f"Qwen 流式调用异常: {e}") from e

    def _parse_chunk(self, chunk) -> StreamChunk:
        """解析OpenAI SDK的流式chunk为统一的StreamChunk"""
        result = StreamChunk()
        if chunk.choices:
            delta = chunk.choices[0].delta
            result.content = delta.content
            result.finish_reason = chunk.choices[0].finish_reason
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
ModelRegistry.register("qwen", QwenLLM)
