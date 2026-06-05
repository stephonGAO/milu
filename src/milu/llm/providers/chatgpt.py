"""
ChatGPTLLM - OpenAI ChatGPT 模型实现（Responses API）。

使用 OpenAI 最新的 Responses API（client.responses.create）替代传统 Chat Completions API。
支持能力：流式输出、函数调用、JSON 模式、视觉理解、推理模式

API 文档: https://platform.openai.com/docs/api-reference/responses
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from milu.llm.base.exceptions import StreamError
from milu.llm.base.message import Message, MessageRole
from milu.llm.base.response import StreamChunk, TokenUsage
from milu.llm.providers.base import BaseLLM, ModelCapabilities

logger = logging.getLogger(__name__)


class ChatGPTLLM(BaseLLM):
    """OpenAI ChatGPT 模型实现（Responses API）"""

    _capabilities = ModelCapabilities(
        supports_streaming=True,
        supports_function_calling=True,
        supports_json_mode=True,
        supports_web_search=False,
        supports_thinking=False,
        supports_embedding=False,
        supports_vision=True,
        supports_audio_understand=True,
        supports_video=False,
        supports_document=True,
        supports_image_generation=False,
        supports_audio_generation=False,
        max_context_window=200000,
        supported_output_formats=("text", "json"),
    )

    _param_names = {
        "temperature", "top_p", "max_tokens", "stop",
        "frequency_penalty", "presence_penalty",
        "reasoning_effort",
        "tools", "tool_choice",
    }

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def base_url(self) -> str:
        return "https://api.openai.com/v1"

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def _get_available_param_names(self) -> set[str]:
        return self._param_names

    def _messages_to_input(self, messages: list[Message]) -> tuple[str | None, list[dict]]:
        """将统一 Message 列表转换为 Responses API 的 (instructions, input) 格式。

        Responses API 与 Chat Completions 的消息格式差异:
          - system 消息 → instructions 参数
          - tool 消息 → function_call_output 类型
          - assistant 的 tool_calls → function_call 类型
        """
        instructions = None
        input_items: list[dict] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                instructions = msg.content if isinstance(msg.content, str) else None

            elif msg.role == MessageRole.USER:
                input_items.append({"role": "user", "content": msg.content})

            elif msg.role == MessageRole.ASSISTANT:
                if msg.tool_calls:
                    # 先输出文本（如果有）
                    if msg.content:
                        input_items.append({
                            "role": "assistant",
                            "content": msg.content,
                        })
                    # 输出每个工具调用（Responses API 独立条目）
                    for tc in msg.tool_calls:
                        input_items.append({
                            "type": "function_call",
                            "call_id": tc["id"],
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        })
                else:
                    input_items.append({
                        "role": "assistant",
                        "content": msg.content or "",
                    })

            elif msg.role == MessageRole.TOOL:
                input_items.append({
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id,
                    "output": msg.content,
                })

        return instructions, input_items

    def _convert_tools_to_responses_format(self, tools: list[dict]) -> list[dict]:
        """将 Chat Completions 格式的工具定义转换为 Responses API 格式。

        Chat Completions: {"type": "function", "function": {"name": "...", ...}}
        Responses API:    {"type": "function", "name": "...", ...}
        """
        converted = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                # 展平：将 function 下的字段提升到顶层
                fn_def = tool["function"]
                converted.append({
                    "type": "function",
                    "name": fn_def.get("name", ""),
                    "description": fn_def.get("description", ""),
                    "parameters": fn_def.get("parameters", {}),
                })
            else:
                # 已经是 Responses 格式或未知格式，直接传递
                converted.append(tool)
        return converted

    async def chat(self, messages: list[Message], **kwargs) -> AsyncIterator[StreamChunk]:
        """
        流式聊天接口（Responses API）。

        Responses API 特有参数映射:
            reasoning_effort (str) → reasoning_effort: "low" | "medium" | "high"
        """
        client = self._get_client()

        # 转换消息格式
        instructions, input_items = self._messages_to_input(messages)

        request_params = {
            "model": self.model,
            "input": input_items,
            "stream": True,
        }
        if instructions:
            request_params["instructions"] = instructions

        # 过滤参数
        validated = self._validate_params(kwargs)

        # 通用参数
        for key in ("temperature", "top_p", "max_tokens", "stop",
                     "frequency_penalty", "presence_penalty"):
            if key in validated:
                request_params[key] = validated[key]

        # 函数调用 - 转换为 Responses API 工具格式
        if "tools" in validated:
            request_params["tools"] = self._convert_tools_to_responses_format(validated["tools"])
        if "tool_choice" in validated:
            request_params["tool_choice"] = validated["tool_choice"]

        # reasoning_effort（o3/o4-mini 推理力度控制）
        if "reasoning_effort" in validated:
            request_params["reasoning_effort"] = validated["reasoning_effort"]

        try:
            stream = await client.responses.create(**request_params)

            # 跟踪正在构建的函数调用：output_index -> {call_id, name}
            pending_calls: dict[int, dict] = {}

            async for event in stream:
                event_type = getattr(event, "type", "")

                # ── 记录函数调用的 call_id 和 name ──
                if event_type == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if item and getattr(item, "type", "") == "function_call":
                        idx = getattr(event, "output_index", 0)
                        pending_calls[idx] = {
                            "call_id": getattr(item, "call_id", ""),
                            "name": getattr(item, "name", ""),
                        }
                    continue

                # ── 文本增量 ──
                if event_type == "response.output_text.delta":
                    yield StreamChunk(content=event.delta)
                    continue

                # ── 推理摘要增量（o3/o4-mini） ──
                if event_type == "response.reasoning_summary_text.delta":
                    yield StreamChunk(reasoning_content=event.delta)
                    continue

                # ── 工具调用完成 ──
                if event_type == "response.function_call_arguments.done":
                    idx = getattr(event, "output_index", 0)
                    info = pending_calls.get(idx, {})
                    call_id = info.get("call_id", "") or getattr(event, "call_id", "")
                    name = info.get("name", "")
                    arguments = getattr(event, "arguments", "{}")

                    tool_call = type("ToolCall", (), {
                        "index": idx,
                        "id": call_id,
                        "function": type("Function", (), {
                            "name": name,
                            "arguments": arguments,
                        })(),
                    })()
                    yield StreamChunk(tool_calls=[tool_call])
                    continue

                # ── 响应完成（包含 usage） ──
                if event_type == "response.completed":
                    response = getattr(event, "response", None)
                    usage_obj = getattr(response, "usage", None) if response else None
                    if usage_obj:
                        yield StreamChunk(
                            finish_reason="stop",
                            usage=TokenUsage(
                                prompt_tokens=getattr(usage_obj, "input_tokens", 0) or 0,
                                completion_tokens=getattr(usage_obj, "output_tokens", 0) or 0,
                                total_tokens=getattr(usage_obj, "total_tokens", 0) or 0,
                            ),
                        )
                    else:
                        yield StreamChunk(finish_reason="stop")
                    continue

                # 其他事件忽略

        except Exception as e:
            raise StreamError(f"ChatGPT (Responses API) 流式调用异常: {e}") from e

    def _parse_event(self, event) -> StreamChunk | None:
        """解析 Responses API 流式事件为统一的 StreamChunk。

        注意：此方法仅处理简单事件（文本增量、完成事件）。
        函数调用需要在 chat() 中通过状态跟踪处理。

        返回 None 表示该事件无需向上传递。
        """
        event_type = getattr(event, "type", "")

        # ── 文本增量 ──
        if event_type == "response.output_text.delta":
            return StreamChunk(content=event.delta)

        # ── 推理摘要增量（o3/o4-mini） ──
        if event_type == "response.reasoning_summary_text.delta":
            return StreamChunk(reasoning_content=event.delta)

        # ── 工具调用完成（简化版，chat() 中使用状态跟踪） ──
        if event_type == "response.function_call_arguments.done":
            tool_call = type("ToolCall", (), {
                "index": getattr(event, "output_index", 0),
                "id": getattr(event, "call_id", ""),
                "function": type("Function", (), {
                    "name": "",  # name 需要从 output_item.added 事件获取
                    "arguments": getattr(event, "arguments", "{}"),
                })(),
            })()
            return StreamChunk(tool_calls=[tool_call])

        # ── 响应完成（包含 usage） ──
        if event_type == "response.completed":
            response = getattr(event, "response", None)
            usage_obj = getattr(response, "usage", None) if response else None
            if usage_obj:
                return StreamChunk(
                    finish_reason="stop",
                    usage=TokenUsage(
                        prompt_tokens=getattr(usage_obj, "input_tokens", 0) or 0,
                        completion_tokens=getattr(usage_obj, "output_tokens", 0) or 0,
                        total_tokens=getattr(usage_obj, "total_tokens", 0) or 0,
                    ),
                )
            return StreamChunk(finish_reason="stop")

        # ── 其他事件忽略 ──
        return None


# 自动注册到 ModelRegistry
from milu.llm.providers import ModelRegistry
ModelRegistry.register("openai", ChatGPTLLM)
