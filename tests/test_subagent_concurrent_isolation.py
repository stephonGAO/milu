"""Subagent 工具并发隔离测试 — 验证 _last_events 闭包 bug 修复

Task 7 (TDD red): 全部 4 个测试在当前代码下应失败，Tasks 8-9 重构后通过。
"""
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import AsyncIterator

import pytest

from agent_framework.agent.agent import Agent
from agent_framework.agent.config import AgentConfig
from agent_framework.agent.events import (
    AgentDone,
    AgentEvent,
    SubAgentDone,
)
from agent_framework.agent.subagent import SubAgentConfig, create_subagent_tools
from agent_framework.llm.providers.base import BaseLLM, ModelCapabilities
from agent_framework.llm.base.response import StreamChunk, TokenUsage


# ── Fake 数据结构 ─────────────────────────────────────────

@dataclass
class _FakeFunction:
    """Mock tool_call 中的 function 字段。"""
    name: str
    arguments: str = ""


@dataclass
class _FakeToolCall:
    """Mock tool_call 对象。"""
    function: _FakeFunction
    id: str
    index: int = 0


# ── Mock LLM ──────────────────────────────────────────────

class _EchoLLM(BaseLLM):
    """每次 chat() 调用返回一个固定文本 chunk。"""

    def __init__(self, text: str = "ok"):
        super().__init__(model="mock", provider="mock")
        self._text = text

    async def chat(
        self, messages, **kwargs
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(content=self._text, finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(1, 1, 2))

    def _get_available_param_names(self) -> frozenset:
        return frozenset()

    @property
    def base_url(self) -> str:
        return "mock://"

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(supports_streaming=True)


class _ParentLLM(BaseLLM):
    """父 Agent 的 mock LLM。

    第一次 chat()：发出 subagent tool_call（name=helper）
    第二次 chat()：发出终止文本
    """

    def __init__(self, label: str):
        super().__init__(model="mock", provider="mock")
        self._label = label
        self._call_count = 0

    async def chat(
        self, messages, **kwargs
    ) -> AsyncIterator[StreamChunk]:
        self._call_count += 1
        if self._call_count == 1:
            yield StreamChunk(tool_calls=[
                _FakeToolCall(
                    function=_FakeFunction(
                        name="helper",
                        arguments='{"task": "x"}',
                    ),
                    id="call_sub",
                    index=0,
                ),
            ])
            yield StreamChunk(finish_reason="tool_calls")
        else:
            yield StreamChunk(
                content=f"parent-{self._label}-done",
                finish_reason="stop",
            )
        yield StreamChunk(usage=TokenUsage(1, 1, 2))

    def _get_available_param_names(self) -> frozenset:
        return frozenset()

    @property
    def base_url(self) -> str:
        return "mock://"

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_streaming=True,
            supports_function_calling=True,
        )


# ── 测试 ──────────────────────────────────────────────────

def test_subagent_no_closure_state():
    """subagent 工具函数无 _last_events 闭包变量

    重构后：_last_events 应被移除，由 ContextVar 替代。
    """
    sub = create_subagent_tools(_EchoLLM("sub"), [
        SubAgentConfig(name="helper", description="test", system_prompt="x"),
    ])[0]
    closure = inspect.getclosurevars(sub)
    # Python 3.13: getclosurevars 返回 .nonlocals
    nonlocals = closure.nonlocals
    assert "_last_events" not in nonlocals, (
        f"_last_events 仍在闭包中（共享状态导致并发事件丢失）: {list(nonlocals)}"
    )


def test_subagent_factory_unchanged():
    """create_subagent_tools 工厂签名与返回值不变

    - 空列表入参 → 返回 []
    - 非空列表 → 返回等长工具列表，名称与 config.name 一一对应
    """
    tools = create_subagent_tools(_EchoLLM(), [])
    assert tools == []

    tools = create_subagent_tools(_EchoLLM(), [
        SubAgentConfig(name="a", description="A", system_prompt=""),
        SubAgentConfig(name="b", description="B", system_prompt=""),
    ])
    assert len(tools) == 2
    assert tools[0]._tool_wrapper.name == "a"
    assert tools[1]._tool_wrapper.name == "b"


@pytest.mark.asyncio
async def test_subagent_without_contextvar_raises():
    """不通过 Agent.run() 直接调 subagent 工具应抛 RuntimeError

    重构后：subagent 工具需从 ContextVar 读取 events list，
    缺失时必须显式报错（避免静默用全局共享列表）。
    """
    sub = create_subagent_tools(_EchoLLM("sub"), [
        SubAgentConfig(name="helper", description="test", system_prompt="x"),
    ])[0]

    with pytest.raises(RuntimeError, match="Agent"):
        await sub._tool_wrapper.func(task="x")


@pytest.mark.asyncio
async def test_subagent_concurrent_runs_events_isolated():
    """两个 Agent 并发调用同一 subagent 工具，事件不交叉、不丢失

    修复前：_last_events 是闭包共享变量，第二次 clear() 会清空第一次的事件，
    导致任一父 Agent 都无法获得自己子代理的 SubAgentDone。
    修复后：每次调用通过 ContextVar 获得独立 events 列表。
    """
    # 两个父 Agent 共用同一 subagent 工具（这正是触发闭包 bug 的场景）
    sub_tool = create_subagent_tools(_EchoLLM("sub"), [
        SubAgentConfig(
            name="helper",
            description="helper subagent",
            system_prompt="helper prompt",
        ),
    ])[0]

    parent_a = Agent(
        llm=_ParentLLM("A"),
        system_prompt="A",
        tools=[sub_tool],
        config=AgentConfig(max_turns=5), session_enabled=False,
    )
    parent_b = Agent(
        llm=_ParentLLM("B"),
        system_prompt="B",
        tools=[sub_tool],
        config=AgentConfig(max_turns=5), session_enabled=False,
    )

    # 并发跑两个父 Agent
    async def _collect(agen) -> list[AgentEvent]:
        out: list[AgentEvent] = []
        async for e in agen:
            out.append(e)
        return out

    events_a, events_b = await asyncio.gather(
        _collect(parent_a.run("hi")),
        _collect(parent_b.run("hi")),
    )

    # 每个父 Agent 应至少拿到 1 个 SubAgentDone
    sub_done_a = [e for e in events_a if isinstance(e, SubAgentDone)]
    sub_done_b = [e for e in events_b if isinstance(e, SubAgentDone)]

    assert len(sub_done_a) >= 1, (
        f"parent_a 缺少 SubAgentDone（events={len(events_a)}，"
        f"类型分布={[e.__class__.__name__ for e in events_a]}）"
    )
    assert len(sub_done_b) >= 1, (
        f"parent_b 缺少 SubAgentDone（events={len(events_b)}，"
        f"类型分布={[e.__class__.__name__ for e in events_b]}）"
    )

    # 每个 SubAgentDone 应有非空 final_text（事件真的拿到了，没被空列表覆盖）
    assert sub_done_a[0].final_text, "parent_a 的 SubAgentDone.final_text 为空"
    assert sub_done_b[0].final_text, "parent_b 的 SubAgentDone.final_text 为空"
