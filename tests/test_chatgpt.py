"""测试 ChatGPTLLM - OpenAI Responses API 实现"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from milu.llm.base.message import Message, MessageRole
from milu.llm.providers.chatgpt import ChatGPTLLM


# ── Responses API Mock 事件 ────────────────────────────────


def _text_delta(text: str):
    """模拟 response.output_text.delta 事件"""
    return SimpleNamespace(type="response.output_text.delta", delta=text)


def _reasoning_delta(text: str):
    """模拟 response.reasoning_summary_text.delta 事件"""
    return SimpleNamespace(type="response.reasoning_summary_text.delta", delta=text)


def _function_call_done(call_id: str, name: str, arguments: str, output_index: int = 0):
    """模拟 response.function_call_arguments.done 事件（真实 SDK 格式，无 item 属性）"""
    return SimpleNamespace(
        type="response.function_call_arguments.done",
        call_id=call_id,
        arguments=arguments,
        output_index=output_index,
    )


def _completed(input_tokens=10, output_tokens=5, total_tokens=15):
    """模拟 response.completed 事件"""
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
    response = SimpleNamespace(usage=usage)
    return SimpleNamespace(type="response.completed", response=response)


def _noop_event(event_type: str):
    """模拟无需处理的中间事件"""
    return SimpleNamespace(type=event_type)


# ── 能力测试 ──────────────────────────────────────────────


class TestChatGPTCapabilities:

    def test_capabilities(self):
        llm = ChatGPTLLM(api_key="test", model="gpt-4o")
        caps = llm.capabilities

        assert caps.supports_streaming is True
        assert caps.supports_function_calling is True
        assert caps.supports_json_mode is True
        assert caps.supports_vision is True
        assert caps.supports_document is True
        assert caps.supports_web_search is False
        assert caps.supports_thinking is False
        assert caps.supports_embedding is False
        assert caps.supports_image_generation is False
        # 按模型解析：gpt-4o 真实窗口 128K（旧实现误报 200K）
        assert caps.max_context_window == 128000
        assert caps.supported_output_formats == ("text", "json")

    def test_provider_name(self):
        llm = ChatGPTLLM(api_key="test", model="gpt-4o")
        assert llm.provider_name == "openai"

    def test_base_url(self):
        llm = ChatGPTLLM(api_key="test", model="gpt-4o")
        assert llm.base_url == "https://api.openai.com/v1"

    def test_registered(self):
        from milu.llm.providers import ModelRegistry
        assert "openai" in ModelRegistry.list_providers()


# ── 参数测试 ──────────────────────────────────────────────


class TestChatGPTParams:

    def test_has_basic_params(self):
        llm = ChatGPTLLM(api_key="test", model="gpt-4o")
        params = llm._get_available_param_names()
        assert "temperature" in params
        assert "top_p" in params
        assert "max_tokens" in params
        assert "stop" in params

    def test_has_reasoning_effort(self):
        llm = ChatGPTLLM(api_key="test", model="o3")
        params = llm._get_available_param_names()
        assert "reasoning_effort" in params

    def test_no_thinking_params(self):
        llm = ChatGPTLLM(api_key="test", model="gpt-4o")
        params = llm._get_available_param_names()
        assert "enable_thinking" not in params
        assert "thinking_level" not in params

    def test_has_tools(self):
        llm = ChatGPTLLM(api_key="test", model="gpt-4o")
        params = llm._get_available_param_names()
        assert "tools" in params
        assert "tool_choice" in params

    def test_filters_unknown_params(self):
        llm = ChatGPTLLM(api_key="test", model="gpt-4o")
        validated = llm._validate_params({"temperature": 0.5, "unknown_param": "x"})
        assert "temperature" in validated
        assert "unknown_param" not in validated


# ── 消息格式转换测试 ──────────────────────────────────────


class TestMessagesToInput:

    def test_system_becomes_instructions(self):
        llm = ChatGPTLLM(api_key="test", model="gpt-4o")
        messages = [
            Message(role=MessageRole.SYSTEM, content="你是助手"),
            Message(role=MessageRole.USER, content="你好"),
        ]
        instructions, input_items = llm._messages_to_input(messages)
        assert instructions == "你是助手"
        assert len(input_items) == 1
        assert input_items[0] == {"role": "user", "content": "你好"}

    def test_user_and_assistant(self):
        llm = ChatGPTLLM(api_key="test", model="gpt-4o")
        messages = [
            Message(role=MessageRole.USER, content="问题"),
            Message(role=MessageRole.ASSISTANT, content="回答"),
        ]
        instructions, input_items = llm._messages_to_input(messages)
        assert instructions is None
        assert len(input_items) == 2
        assert input_items[0]["role"] == "user"
        assert input_items[1]["role"] == "assistant"

    def test_tool_messages(self):
        llm = ChatGPTLLM(api_key="test", model="gpt-4o")
        messages = [
            Message(role=MessageRole.TOOL, content="结果", tool_call_id="call_1", name="fn"),
        ]
        _, input_items = llm._messages_to_input(messages)
        assert len(input_items) == 1
        assert input_items[0]["type"] == "function_call_output"
        assert input_items[0]["call_id"] == "call_1"
        assert input_items[0]["output"] == "结果"

    def test_assistant_with_tool_calls(self):
        llm = ChatGPTLLM(api_key="test", model="gpt-4o")
        messages = [
            Message(
                role=MessageRole.ASSISTANT,
                content="我来查一下",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q":"test"}'},
                }],
            ),
        ]
        _, input_items = llm._messages_to_input(messages)
        assert len(input_items) == 2
        assert input_items[0] == {"role": "assistant", "content": "我来查一下"}
        assert input_items[1]["type"] == "function_call"
        assert input_items[1]["call_id"] == "call_1"
        assert input_items[1]["name"] == "search"

    def test_assistant_tool_calls_no_text(self):
        llm = ChatGPTLLM(api_key="test", model="gpt-4o")
        messages = [
            Message(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "fn", "arguments": "{}"},
                }],
            ),
        ]
        _, input_items = llm._messages_to_input(messages)
        # content 为 None 时不应输出 assistant 文本条目
        assert len(input_items) == 1
        assert input_items[0]["type"] == "function_call"

    def test_full_conversation(self):
        llm = ChatGPTLLM(api_key="test", model="gpt-4o")
        messages = [
            Message(role=MessageRole.SYSTEM, content="你是助手"),
            Message(role=MessageRole.USER, content="天气？"),
            Message(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city":"北京"}'},
                }],
            ),
            Message(role=MessageRole.TOOL, content="晴天", tool_call_id="call_1"),
        ]
        instructions, input_items = llm._messages_to_input(messages)
        assert instructions == "你是助手"
        assert len(input_items) == 3  # user + function_call + function_call_output
        assert input_items[0]["role"] == "user"
        assert input_items[1]["type"] == "function_call"
        assert input_items[2]["type"] == "function_call_output"


# ── 流式事件解析测试 ──────────────────────────────────────


class TestToolConversion:
    """测试工具格式转换"""

    def test_convert_chat_completions_format(self):
        llm = ChatGPTLLM(api_key="test", model="gpt-4o")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取天气",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        converted = llm._convert_tools_to_responses_format(tools)
        assert len(converted) == 1
        assert converted[0] == {
            "type": "function",
            "name": "get_weather",
            "description": "获取天气",
            "parameters": {"type": "object", "properties": {}},
        }
        assert "function" not in converted[0]

    def test_convert_preserves_already_converted(self):
        llm = ChatGPTLLM(api_key="test", model="gpt-4o")
        tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "获取天气",
                "parameters": {"type": "object"},
            }
        ]
        converted = llm._convert_tools_to_responses_format(tools)
        assert converted == tools

    def test_convert_multiple_tools(self):
        llm = ChatGPTLLM(api_key="test", model="gpt-4o")
        tools = [
            {"type": "function", "function": {"name": "tool_a", "description": "A", "parameters": {}}},
            {"type": "function", "function": {"name": "tool_b", "description": "B", "parameters": {}}},
        ]
        converted = llm._convert_tools_to_responses_format(tools)
        assert len(converted) == 2
        assert converted[0]["name"] == "tool_a"
        assert converted[1]["name"] == "tool_b"
        for c in converted:
            assert "function" not in c


# ── 流式事件解析测试 ──────────────────────────────────────


class TestParseEvent:

    def test_text_delta(self):
        llm = ChatGPTLLM(api_key="test", model="gpt-4o")
        chunk = llm._parse_event(_text_delta("你好"))
        assert chunk is not None
        assert chunk.content == "你好"

    def test_reasoning_delta(self):
        llm = ChatGPTLLM(api_key="test", model="o3")
        chunk = llm._parse_event(_reasoning_delta("让我想想"))
        assert chunk is not None
        assert chunk.reasoning_content == "让我想想"

    def test_function_call_done(self):
        llm = ChatGPTLLM(api_key="test", model="gpt-4o")
        event = _function_call_done("call_1", "weather", '{"city":"北京"}')
        chunk = llm._parse_event(event)
        assert chunk is not None
        assert chunk.tool_calls is not None
        assert len(chunk.tool_calls) == 1
        tc = chunk.tool_calls[0]
        assert tc.id == "call_1"
        # name 在 _parse_event 中为空（实际由 chat() 中的状态跟踪获取）
        assert tc.function.arguments == '{"city":"北京"}'

    def test_completed_with_usage(self):
        llm = ChatGPTLLM(api_key="test", model="gpt-4o")
        chunk = llm._parse_event(_completed(100, 50, 150))
        assert chunk is not None
        assert chunk.finish_reason == "stop"
        assert chunk.usage is not None
        assert chunk.usage.prompt_tokens == 100
        assert chunk.usage.completion_tokens == 50
        assert chunk.usage.total_tokens == 150

    def test_noop_events_return_none(self):
        llm = ChatGPTLLM(api_key="test", model="gpt-4o")
        for event_type in ("response.created", "response.in_progress",
                          "response.output_item.added", "response.output_item.done"):
            assert llm._parse_event(_noop_event(event_type)) is None


# ── 聊天集成测试 ──────────────────────────────────────────


class TestChatGPTChat:

    @pytest.mark.asyncio
    async def test_basic_streaming(self):
        """基础流式: 文本增量 + 完成事件"""
        events = [
            _text_delta("你"),
            _text_delta("好"),
            _noop_event("response.output_text.done"),
            _completed(10, 5, 15),
        ]

        async def mock_stream():
            for e in events:
                yield e

        llm = ChatGPTLLM(api_key="test-key", model="gpt-4o")
        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_stream())

        with patch.object(llm, "_get_client", return_value=mock_client):
            messages = [Message(role=MessageRole.USER, content="你好")]
            results = []
            async for chunk in llm.chat(messages):
                results.append(chunk)

        # 应得到 2 个文本 chunk + 1 个完成 chunk = 3
        assert len(results) == 3
        assert results[0].content == "你"
        assert results[1].content == "好"
        assert results[2].finish_reason == "stop"
        assert results[2].usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_passes_instructions(self):
        """system 消息应转为 instructions 参数"""
        async def mock_stream():
            yield _completed()

        llm = ChatGPTLLM(api_key="test-key", model="gpt-4o")
        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_stream())

        with patch.object(llm, "_get_client", return_value=mock_client):
            messages = [
                Message(role=MessageRole.SYSTEM, content="你是助手"),
                Message(role=MessageRole.USER, content="你好"),
            ]
            async for _ in llm.chat(messages):
                pass

        kwargs = mock_client.responses.create.call_args.kwargs
        assert kwargs["instructions"] == "你是助手"
        assert kwargs["input"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_passes_reasoning_effort(self):
        """reasoning_effort 应直接传递"""
        async def mock_stream():
            yield _completed()

        llm = ChatGPTLLM(api_key="test-key", model="o3")
        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_stream())

        with patch.object(llm, "_get_client", return_value=mock_client):
            messages = [Message(role=MessageRole.USER, content="思考")]
            async for _ in llm.chat(messages, reasoning_effort="high"):
                pass

        kwargs = mock_client.responses.create.call_args.kwargs
        assert kwargs["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_tool_call_streaming(self):
        """工具调用的流式输出"""
        events = [
            # output_item.added: 记录函数调用的 call_id 和 name
            SimpleNamespace(
                type="response.output_item.added",
                output_index=0,
                item=SimpleNamespace(type="function_call", call_id="call_1", name="search"),
            ),
            # function_call_arguments.done: 完整参数
            _function_call_done("call_1", "search", '{"q":"天气"}'),
            _completed(20, 10, 30),
        ]

        async def mock_stream():
            for e in events:
                yield e

        llm = ChatGPTLLM(api_key="test-key", model="gpt-4o")
        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_stream())

        with patch.object(llm, "_get_client", return_value=mock_client):
            messages = [Message(role=MessageRole.USER, content="搜索天气")]
            results = []
            async for chunk in llm.chat(messages):
                results.append(chunk)

        # 应得到 1 个 tool_call chunk + 1 个完成 chunk = 2
        assert len(results) == 2
        assert results[0].tool_calls is not None
        assert results[0].tool_calls[0].function.name == "search"
        assert results[0].tool_calls[0].id == "call_1"
        assert results[1].finish_reason == "stop"
