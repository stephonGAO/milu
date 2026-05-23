"""测试 AgentEvent 事件类型"""
import pytest
from agent_framework.models.events import (
    AgentEvent,
    TextDelta,
    ReasoningDelta,
    ToolCallStart,
    ToolResult,
    AgentDone,
    AgentError,
)
from agent_framework.models.response import TokenUsage


def test_text_delta():
    """TextDelta 应存储文本片段"""
    event = TextDelta(text="你好")
    assert event.text == "你好"
    assert isinstance(event, AgentEvent)


def test_text_delta_frozen():
    """TextDelta 应为不可变对象"""
    event = TextDelta(text="你好")
    with pytest.raises(AttributeError):
        event.text = "世界"


def test_reasoning_delta():
    """ReasoningDelta 应存储思考片段"""
    event = ReasoningDelta(text="让我想想")
    assert event.text == "让我想想"
    assert isinstance(event, AgentEvent)


def test_tool_call_start():
    """ToolCallStart 应存储工具调用信息"""
    event = ToolCallStart(
        tool_name="get_weather",
        tool_call_id="call_123",
        arguments='{"city": "北京"}'
    )
    assert event.tool_name == "get_weather"
    assert event.tool_call_id == "call_123"
    assert event.arguments == '{"city": "北京"}'


def test_tool_result():
    """ToolResult 应存储工具执行结果"""
    event = ToolResult(
        tool_name="get_weather",
        tool_call_id="call_123",
        output="北京：晴，25°C",
        is_error=False
    )
    assert event.tool_name == "get_weather"
    assert event.output == "北京：晴，25°C"
    assert event.is_error is False


def test_tool_result_error():
    """ToolResult 应能表示错误结果"""
    event = ToolResult(
        tool_name="get_weather",
        tool_call_id="call_123",
        output="工具不存在",
        is_error=True
    )
    assert event.is_error is True


def test_agent_done():
    """AgentDone 应存储完成信息"""
    usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    event = AgentDone(
        final_text="回答完成",
        total_usage=usage,
        turn_count=3
    )
    assert event.final_text == "回答完成"
    assert event.total_usage.total_tokens == 30
    assert event.turn_count == 3


def test_agent_error():
    """AgentError 应存储错误信息"""
    event = AgentError(
        error_type="max_turns",
        message="超出最大轮次(10)"
    )
    assert event.error_type == "max_turns"
    assert "超出最大轮次" in event.message
