"""测试数据模型：Message, StreamChunk, TokenUsage"""

import pytest
from agent_framework.models.message import Message, MessageRole
from agent_framework.models.response import StreamChunk, TokenUsage


class TestMessageRole:
    def test_role_values(self):
        assert MessageRole.SYSTEM == "system"
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.TOOL == "tool"

    def test_role_is_string(self):
        role = MessageRole.USER
        assert isinstance(role, str)
        assert role == "user"


class TestMessage:
    def test_basic_text_message(self):
        msg = Message(role=MessageRole.USER, content="你好")
        assert msg.role == MessageRole.USER
        assert msg.content == "你好"
        assert msg.tool_calls is None
        assert msg.tool_call_id is None
        assert msg.name is None

    def test_system_message(self):
        msg = Message(role=MessageRole.SYSTEM, content="你是一个助手")
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "你是一个助手"

    def test_multimodal_message(self):
        content = [
            {"type": "text", "text": "这张图是什么？"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        msg = Message(role=MessageRole.USER, content=content)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_assistant_message_with_tool_calls(self):
        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "北京"}'}}]
        msg = Message(role=MessageRole.ASSISTANT, tool_calls=tool_calls)
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

    def test_tool_message(self):
        msg = Message(role=MessageRole.TOOL, content="晴天，25°C", tool_call_id="call_1", name="get_weather")
        assert msg.tool_call_id == "call_1"
        assert msg.name == "get_weather"

    def test_message_to_dict_basic(self):
        msg = Message(role=MessageRole.USER, content="你好")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "你好"}

    def test_message_to_dict_excludes_none(self):
        msg = Message(role=MessageRole.USER, content="你好")
        d = msg.to_dict()
        assert "tool_calls" not in d
        assert "tool_call_id" not in d
        assert "name" not in d

    def test_message_to_dict_with_tool_calls(self):
        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "test"}}]
        msg = Message(role=MessageRole.ASSISTANT, tool_calls=tool_calls)
        d = msg.to_dict()
        assert "tool_calls" in d

    def test_message_to_dict_with_tool_result(self):
        msg = Message(role=MessageRole.TOOL, content="结果", tool_call_id="call_1", name="func")
        d = msg.to_dict()
        assert d["tool_call_id"] == "call_1"
        assert d["name"] == "func"

    def test_message_to_dict_multimodal(self):
        content = [
            {"type": "text", "text": "描述图片"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        msg = Message(role=MessageRole.USER, content=content)
        d = msg.to_dict()
        assert d["content"] == content


class TestTokenUsage:
    def test_default_values(self):
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.reasoning_tokens == 0

    def test_custom_values(self):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, reasoning_tokens=30)
        assert usage.total_tokens == 150
        assert usage.reasoning_tokens == 30


class TestStreamChunk:
    def test_content_chunk(self):
        chunk = StreamChunk(content="你好")
        assert chunk.content == "你好"
        assert chunk.reasoning_content is None
        assert chunk.finish_reason is None

    def test_reasoning_chunk(self):
        chunk = StreamChunk(reasoning_content="让我想想...")
        assert chunk.reasoning_content == "让我想想..."
        assert chunk.content is None

    def test_finish_chunk_with_usage(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunk = StreamChunk(finish_reason="stop", usage=usage)
        assert chunk.finish_reason == "stop"
        assert chunk.usage.total_tokens == 15

    def test_tool_calls_chunk(self):
        chunk = StreamChunk(tool_calls=[{"id": "call_1", "function": {"name": "test"}}])
        assert len(chunk.tool_calls) == 1
