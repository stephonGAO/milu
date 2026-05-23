"""测试 Agent 编排器"""
import pytest
from unittest.mock import AsyncMock
from agent_framework.agent import Agent, AgentConfig
from agent_framework.agent.events import TextDelta, ToolCallStart, ToolResult, AgentDone, AgentError
from agent_framework.llm.base.message import MessageRole
from agent_framework.llm.base.response import StreamChunk, TokenUsage
from agent_framework.tools import tool


@pytest.fixture
def simple_tool():
    @tool(name="get_time", description="获取时间")
    async def get_time() -> str:
        return "2026-05-23 10:00"
    return get_time


@pytest.mark.asyncio
async def test_simple_text_response():
    """无工具调用时应直接返回文本"""
    async def mock_chat(*args, **kwargs):
        yield StreamChunk(content="你好", finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    llm = AsyncMock()
    llm.chat = mock_chat

    agent = Agent(llm=llm, system_prompt="你是助手")

    events = []
    async for event in agent.run("你好"):
        events.append(event)

    assert any(isinstance(e, TextDelta) for e in events)
    assert any(isinstance(e, AgentDone) for e in events)

    done = next(e for e in events if isinstance(e, AgentDone))
    assert done.final_text == "你好"
    assert done.turn_count == 1


@pytest.mark.asyncio
async def test_tool_call_flow(simple_tool):
    """应正确处理工具调用流程"""
    call_count = 0

    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': 'call_1',
                    'function': type('obj', (), {'name': 'get_time', 'arguments': '{}'})()
                })()
            ])
            yield StreamChunk(finish_reason="tool_calls")
        else:
            yield StreamChunk(content="当前时间是 2026-05-23 10:00", finish_reason="stop")
            yield StreamChunk(usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30))

    llm = AsyncMock()
    llm.chat = mock_chat

    agent = Agent(llm=llm, system_prompt="你是助手", tools=[simple_tool])

    events = []
    async for event in agent.run("现在几点了？"):
        events.append(event)

    assert any(isinstance(e, ToolCallStart) for e in events)
    assert any(isinstance(e, ToolResult) for e in events)
    assert any(isinstance(e, TextDelta) for e in events)
    assert any(isinstance(e, AgentDone) for e in events)

    tool_start = next(e for e in events if isinstance(e, ToolCallStart))
    assert tool_start.tool_name == "get_time"

    tool_result = next(e for e in events if isinstance(e, ToolResult))
    assert tool_result.output == "2026-05-23 10:00"
    assert tool_result.is_error is False


@pytest.mark.asyncio
async def test_max_turns_limit(simple_tool):
    """超出最大轮次应返回 AgentError"""
    async def mock_chat(*args, **kwargs):
        yield StreamChunk(tool_calls=[
            type('obj', (), {
                'index': 0, 'id': 'call_x',
                'function': type('obj', (), {'name': 'get_time', 'arguments': '{}'})()
            })()
        ])

    llm = AsyncMock()
    llm.chat = mock_chat

    agent = Agent(llm=llm, system_prompt="你是助手", tools=[simple_tool], config=AgentConfig(max_turns=3))

    events = []
    async for event in agent.run("测试"):
        events.append(event)

    assert any(isinstance(e, AgentError) for e in events)
    error = next(e for e in events if isinstance(e, AgentError))
    assert error.error_type == "max_turns"


@pytest.mark.asyncio
async def test_conversation_history():
    """多轮对话应保留历史"""
    call_count = 0

    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        yield StreamChunk(content=f"回复{call_count}", finish_reason="stop")

    llm = AsyncMock()
    llm.chat = mock_chat

    agent = Agent(llm=llm, system_prompt="你是助手")

    async for event in agent.run("问题1"):
        pass
    async for event in agent.run("问题2"):
        pass
    async for event in agent.run("问题3"):
        pass

    messages = agent.history.all_messages
    assert messages[0].role == MessageRole.SYSTEM
    assert len(messages) == 7  # system + 3*(user + assistant)


@pytest.mark.asyncio
async def test_reset():
    """reset 应清空历史但保留 system"""
    async def mock_chat(*args, **kwargs):
        yield StreamChunk(content="回复", finish_reason="stop")

    llm = AsyncMock()
    llm.chat = mock_chat

    agent = Agent(llm=llm, system_prompt="你是助手")

    async for event in agent.run("你好"):
        pass

    assert len(agent.history.all_messages) == 3

    await agent.reset()
    assert len(agent.history.all_messages) == 1


@pytest.mark.asyncio
async def test_tool_call_limit(simple_tool):
    """工具调用次数超限应返回 AgentError"""
    async def mock_chat(*args, **kwargs):
        yield StreamChunk(tool_calls=[
            type('obj', (), {
                'index': 0, 'id': 'call_x',
                'function': type('obj', (), {'name': 'get_time', 'arguments': '{}'})()
            })()
        ])

    llm = AsyncMock()
    llm.chat = mock_chat

    agent = Agent(
        llm=llm,
        system_prompt="你是助手",
        tools=[simple_tool],
        config=AgentConfig(max_turns=100, tool_call_limit=2)
    )

    events = []
    async for event in agent.run("测试"):
        events.append(event)

    assert any(isinstance(e, AgentError) for e in events)
    error = next(e for e in events if isinstance(e, AgentError))
    assert error.error_type == "tool_limit"
