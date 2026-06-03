"""测试 token 用量遥测：finish_reason 之后的收尾 usage chunk 应被聚合进 AgentDone。

OpenAI 兼容流（stream_options.include_usage=True）会在 finish_reason 之后再发一个
「仅含 usage、choices 为空」的收尾 chunk。Agent 不能在 finish_reason=="stop" 时提前
break，否则丢失本次调用的 token 用量。
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agent_framework.agent import Agent, AgentConfig
from agent_framework.agent.events import AgentDone
from agent_framework.llm.base.response import StreamChunk, TokenUsage


def _agent(llm):
    return Agent(
        llm=llm,
        session_enabled=False,
        register_catalog=False,
        register_skills=False,
    )


@pytest.mark.asyncio
async def test_usage_aggregated_from_trailing_chunk():
    """收尾 usage chunk（在 finish_reason 之后）应被采集进 AgentDone.total_usage。"""
    async def chat(messages, **kwargs):
        yield StreamChunk(content="hi", finish_reason="stop")
        # 收尾 usage chunk：choices 为空，仅带 usage
        yield StreamChunk(usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    llm = AsyncMock()
    llm.chat = chat

    done = None
    async for ev in _agent(llm).run("hello"):
        if isinstance(ev, AgentDone):
            done = ev

    assert done is not None
    assert done.turn_count == 1
    assert done.total_usage.total_tokens == 15
    assert done.total_usage.prompt_tokens == 10
    assert done.total_usage.completion_tokens == 5


@pytest.mark.asyncio
async def test_usage_summed_across_tool_turns():
    """多轮（工具调用后再回复）应累加各轮 usage。"""
    import json
    call = 0

    async def chat(messages, **kwargs):
        nonlocal call
        call += 1
        if call == 1:
            # 第一轮：发起工具调用 + 收尾 usage
            yield StreamChunk(tool_calls=[
                type("o", (), {"index": 0, "id": "c1",
                               "function": type("f", (), {"name": "noop", "arguments": "{}"})()})()
            ])
            yield StreamChunk(finish_reason="tool_calls")
            yield StreamChunk(usage=TokenUsage(8, 2, 10))
        else:
            # 第二轮：基于工具结果回复 + 收尾 usage
            yield StreamChunk(content="done", finish_reason="stop")
            yield StreamChunk(usage=TokenUsage(20, 5, 25))

    llm = AsyncMock()
    llm.chat = chat

    from agent_framework.tools import tool

    @tool(name="noop", description="无操作", is_safe=True)
    async def noop() -> str:
        return "ok"

    agent = Agent(llm=llm, tools=[noop], session_enabled=False,
                  register_catalog=False, register_skills=False)

    done = None
    async for ev in agent.run("go"):
        if isinstance(ev, AgentDone):
            done = ev

    assert done is not None
    assert done.turn_count == 2
    assert done.total_usage.total_tokens == 35  # 10 + 25
