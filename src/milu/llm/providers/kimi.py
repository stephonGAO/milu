"""
KimiLLM - 月之暗面Moonshot Kimi模型实现。

支持能力：流式输出、函数调用、JSON模式、联网搜索、
思考模式、图片/文档理解

API文档: https://platform.moonshot.cn/docs
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from milu.llm.base.exceptions import StreamError
from milu.llm.base.message import Message
from milu.llm.base.response import StreamChunk, TokenUsage
from milu.llm.providers.base import BaseLLM, ModelCapabilities

logger = logging.getLogger(__name__)


class KimiLLM(BaseLLM):
    """月之暗面Moonshot Kimi模型实现"""

    _capabilities = ModelCapabilities(
        supports_streaming=True,
        supports_function_calling=True,
        supports_json_mode=True,
        supports_web_search=True,
        supports_thinking=True,
        supports_embedding=False,
        supports_vision=True,
        supports_audio_understand=False,
        supports_video=False,
        supports_document=True,
        supports_image_generation=False,
        supports_audio_generation=False,
        max_context_window=262144,
        supported_output_formats=("text", "json"),
    )

    # 各模型真实上下文窗口（小写片段匹配）。数据截至 2026-06。
    # 当前在用的 Kimi K2 系列（k2.6 最新 / k2.5 / k2-thinking）均为 256K=262144；
    # moonshot-v1-* 是被 K2 取代的旧版模型名（窗口小得多），保留以防仍调用旧模型时溢出。
    _context_windows = {
        "kimi-k2.6": 262144,
        "kimi-k2.5": 262144,
        "kimi-k2-thinking": 262144,
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
        return "kimi"

    @property
    def base_url(self) -> str:
        return "https://api.moonshot.cn/v1"


    def _get_available_param_names(self) -> set[str]:
        return self._param_names

    def _messages_to_dicts(self, messages: list[Message]) -> list[dict]:
        """在基类序列化之上，为带 tool_calls 的 assistant 消息回填 reasoning_content。

        Kimi 的 thinking 模型有硬性约束：thinking 开启时，历史里每条带 tool_calls 的
        assistant 消息回传时【必须】携带【非空】reasoning_content，否则报
        400 "thinking is enabled but reasoning_content is missing in assistant tool
        call message at index N"。基类 Message.to_dict() 默认不带该字段（对 DeepSeek 等
        不能带 reasoning_content 的 provider 保持安全），故在此 Kimi 专属注入。

        注入范围：仅「带 tool_calls」的 assistant 消息（非工具调用的终结回答无需回传
        推理，避免上下文膨胀）。取值：有真实 reasoning_content 用真实值；模型偶有
        【不产出推理就直接发工具调用】的轮次（此时捕获到的是空），用占位兜底——
        否则该消息回传仍会触发上面的 400（子代理 researcher 用直接型角色提示词时易现）。
        """
        dicts = super()._messages_to_dicts(messages)
        for msg, d in zip(messages, dicts):
            if d.get("role") == "assistant" and d.get("tool_calls"):
                rc = getattr(msg, "reasoning_content", None)
                d["reasoning_content"] = rc if rc else "(本轮无显式推理)"
        return dicts

    async def chat(self, messages: list[Message], **kwargs) -> AsyncIterator[StreamChunk]:
        """
        流式聊天接口。

        Kimi特有参数映射:
            web_search (bool) → tools 列表中添加内置工具 {"type": "builtin_function", "function": {"name": "$web_search"}}
            enable_thinking (bool) → extra_body.thinking = {"type": "enabled"/"disabled"}
                （Kimi K2.5/K2.6 等思考模型的正式开关；不能用 OpenAI 的 reasoning_effort，
                 该参数无法表达"关闭"——"none" 会被这些模型拒绝，只接受 minimal/low/medium/high）
            thinking_level (str) → Kimi 思考开关无深度档位（只有 enabled/disabled），静默忽略
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
            # Kimi通过添加内置工具实现联网搜索
            tools = request_params.get("tools", [])
            tools.append({
                "type": "builtin_function",
                "function": {"name": "$web_search"},
            })
            request_params["tools"] = tools
        if "enable_thinking" in validated:
            # 思考开关走 Kimi 官方的 thinking.type（enabled/disabled）。
            # ⚠️ 不要用 reasoning_effort="none"——k2.6 等思考模型不支持 "none"
            # （只接受 minimal/low/medium/high），会直接 400。thinking.type 才是
            # 正式的开/关机制。开启时历史 assistant 消息仍需带 reasoning_content
            # （见本类 _messages_to_dicts 的回传处理）。
            extra_body["thinking"] = {
                "type": "enabled" if validated["enable_thinking"] else "disabled"
            }
        if extra_body:
            request_params["extra_body"] = extra_body

        try:
            response = await client.chat.completions.create(**request_params)
            async for chunk in response:
                yield self._parse_chunk(chunk)
        except Exception as e:
            raise StreamError(f"Kimi 流式调用异常: {e}") from e

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
ModelRegistry.register("kimi", KimiLLM)
