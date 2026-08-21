"""
GLMLLM - 智谱AI GLM模型实现。

支持能力：流式输出、函数调用、JSON模式、联网搜索、
思考模式、文本嵌入、图片/视频理解

API文档: https://open.bigmodel.cn/dev/api
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from milu.llm.base.exceptions import StreamError
from milu.llm.base.message import Message
from milu.llm.base.response import StreamChunk, TokenUsage
from milu.llm.providers.base import BaseLLM, ModelCapabilities

logger = logging.getLogger(__name__)


class GLMLLM(BaseLLM):
    """智谱AI GLM模型实现"""

    _capabilities = ModelCapabilities(
        supports_streaming=True,
        supports_function_calling=True,
        supports_json_mode=True,
        supports_web_search=True,
        supports_thinking=True,
        supports_embedding=True,
        supports_vision=True,
        supports_audio_understand=False,
        supports_video=True,
        supports_document=False,
        supports_image_generation=False,
        supports_audio_generation=False,
        max_context_window=131072,
        supported_output_formats=("text", "json"),
    )

    # 各模型真实上下文窗口（未收录回退到 _capabilities 的 131072=128K，覆盖 glm-4.5 等）。
    # 数据截至 2026-06：GLM-5 / 4.6 为 200K；GLM-5.1 官方基础规格 200K，但主流三方渠道
    # 以 1M 提供且实测可用，故取 1M（若部署仅 200K，用 context_window 覆盖回 200000）。
    _context_windows = {
        "glm-5.1": 1_000_000,   # 三方渠道 1M（最长片段优先于 glm-5）
        "glm-5": 200_000,
        "glm-4.6": 200_000,
    }

    _param_names = {
        "temperature", "top_p", "max_tokens", "stop",
        "frequency_penalty", "presence_penalty",
        "web_search", "web_search_strategy",
        "enable_thinking", "thinking_level",
        "tools", "tool_choice",
    }

    @property
    def provider_name(self) -> str:
        return "glm"

    @property
    def base_url(self) -> str:
        return "https://open.bigmodel.cn/api/paas/v4"


    def _get_available_param_names(self) -> set[str]:
        return self._param_names

    async def chat(self, messages: list[Message], **kwargs) -> AsyncIterator[StreamChunk]:
        """
        流式聊天接口。

        GLM特有参数映射:
            web_search (bool) → tools 列表追加
                {"type":"web_search","web_search":{"enable":"True","search_result":True}}
            enable_thinking (bool) → extra_body.enable_thinking=True
            thinking_level → GLM不支持等级设置，静默忽略
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

        # extra_body: 联网搜索和思考模式
        extra_body = {}
        if validated.get("web_search"):
            # GLM 通过 tools 列表中的 web_search 类型启用联网搜索。
            # ⚠️ 必须带【非空】的 web_search 子对象，否则 400
            # "tools[N].web_search 不能为空"（裸 {"type":"web_search"} 不合法）。
            # 字段按智谱官方示例：enable 用字符串 "True"；search_result=True 让搜索
            # 结果回填进上下文；search_engine 不显式指定，走平台默认（search_std）。
            tools_list = request_params.get("tools", [])
            tools_list.append({
                "type": "web_search",
                "web_search": {
                    "enable": "True",
                    "search_result": True,
                },
            })
            request_params["tools"] = tools_list
        if "enable_thinking" in validated:
            # GLM不支持thinking_level等级设置，仅支持开启/关闭
            extra_body["enable_thinking"] = bool(validated["enable_thinking"])
        if extra_body:
            request_params["extra_body"] = extra_body

        try:
            response = await client.chat.completions.create(**request_params)
            async for chunk in response:
                yield self._parse_chunk(chunk)
        except Exception as e:
            raise StreamError(f"GLM 流式调用异常: {e}") from e

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
            result.usage = TokenUsage.from_api_usage(chunk.usage)
        return result


# 自动注册到 ModelRegistry
from milu.llm.providers import ModelRegistry
ModelRegistry.register("glm", GLMLLM)
