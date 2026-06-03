"""Todo 工具并发隔离测试 — 验证无状态重构后多用户并发安全"""
from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from agent_framework.agent import Agent, AgentConfig
from agent_framework.llm.base.response import StreamChunk, TokenUsage
from agent_framework.llm.providers.base import BaseLLM, ModelCapabilities
from agent_framework.tools.builtin.todo_write import (
    create_todo_write_tool,
    todo_read,
    todo_write,
)


# ── Fake 数据结构（替代 type("TC", ...) 动态类） ─────────

@dataclass
class _FakeFunction:
    """Mock tool_call 中的 function 字段。"""
    name: str
    arguments: str = ""


@dataclass
class _FakeToolCall:
    """Mock tool_call 对象（覆盖 Message.tool_calls 与流式 chunk.tool_calls 两种场景）。"""
    function: _FakeFunction
    id: str
    index: int = 0


# ── Mock LLM ────────────────────────────────────────────

class _MockLLM(BaseLLM):
    """Echo LLM：返回固定文本。"""
    def __init__(self):
        super().__init__(model="mock", provider="mock")
        self.calls: list[list[dict]] = []

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        yield StreamChunk(content="ok", finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(1, 1, 2))

    def _get_available_param_names(self):
        return frozenset()

    @property
    def base_url(self) -> str:
        return "mock://"

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def capabilities(self):
        return ModelCapabilities(supports_streaming=True)


class _TodoWriteLLM(BaseLLM):
    """Mock LLM：首轮发出一次 todo_write tool_call，下一轮结束。

    用于验证 Agent.run() 完整地穿越 todo_write 工具链路。
    """

    def __init__(self, items: list[dict]):
        super().__init__(model="mock", provider="mock")
        self._items = items
        self._call_count = 0

    async def chat(self, messages, **kwargs):
        self._call_count += 1
        if self._call_count == 1:
            yield StreamChunk(tool_calls=[
                _FakeToolCall(
                    function=_FakeFunction(
                        name="todo_write",
                        arguments=json.dumps({"items": self._items}, ensure_ascii=False),
                    ),
                    id="call_1",
                    index=0,
                )
            ])
            yield StreamChunk(finish_reason="tool_calls")
        else:
            yield StreamChunk(content="done", finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(1, 1, 2))

    def _get_available_param_names(self):
        return frozenset()

    @property
    def base_url(self) -> str:
        return "mock://"

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def capabilities(self):
        return ModelCapabilities(
            supports_streaming=True,
            supports_function_calling=True,
        )


# ── 测试 ────────────────────────────────────────────────

def test_todo_write_no_closure_state():
    """todo_write 工具函数无 TodoManager 闭包"""
    cw, cr = create_todo_write_tool()
    cw_closure = inspect.getclosurevars(cw)
    cr_closure = inspect.getclosurevars(cr)
    # 检查非局部变量和全局变量中不能有 TodoManager 相关
    # Python 3.13+: getclosurevars 返回 .nonlocals（更早版本字段名为 .cells）
    cw_nonlocals = cw_closure.nonlocals
    cr_nonlocals = cr_closure.nonlocals
    assert "mgr" not in cw_nonlocals
    assert "mgr" not in cr_nonlocals
    assert "manager" not in cw_nonlocals
    assert "manager" not in cr_nonlocals


def test_create_todo_write_tool_no_args():
    """v2 工厂函数不接受任何参数"""
    result = create_todo_write_tool()
    assert len(result) == 2
    assert result[0]._tool_wrapper.name == "todo_write"
    assert result[1]._tool_wrapper.name == "todo_read"


@pytest.mark.asyncio
async def test_todo_write_requires_contextvar(tmp_path: Path):
    """不通过 Agent.run() 直接调 todo_write 应抛 RuntimeError"""
    from agent_framework.tools.builtin.todo_write import _current_session_dir

    # 确保 ContextVar 未设置
    assert _current_session_dir.get() is None
    with pytest.raises(RuntimeError, match="Agent"):
        await todo_write._tool_wrapper.func(items=[
            {"content": "x", "status": "pending"},
        ])


@pytest.mark.asyncio
async def test_todo_read_without_file_returns_empty(tmp_path: Path):
    """无 plan.json 时 todo_read 返回 '暂无会话计划。'"""
    from agent_framework.tools.builtin.todo_write import _current_session_dir

    token = _current_session_dir.set(tmp_path)
    try:
        result = await todo_read._tool_wrapper.func()
        assert "暂无" in result
    finally:
        _current_session_dir.reset(token)


@pytest.mark.asyncio
async def test_two_agents_concurrent_writes_isolated(tmp_path: Path):
    """两个 Agent 并发跑 todo_write，各自写到独立的 plan.json，互不串味。"""
    items_a = [
        {"content": "任务 A1", "status": "pending"},
        {"content": "任务 A2", "status": "in_progress"},
    ]
    items_b = [
        {"content": "任务 B1", "status": "completed"},
        {"content": "任务 B2", "status": "pending"},
    ]

    llm_a = _TodoWriteLLM(items_a)
    llm_b = _TodoWriteLLM(items_b)

    agent_a = Agent(
        llm=llm_a,
        system_prompt="A",
        tools=list(create_todo_write_tool()),
        config=AgentConfig(
            session_dir=str(tmp_path / "A"),
            session_enabled=True,
            max_turns=5,
        ),
    )
    agent_b = Agent(
        llm=llm_b,
        system_prompt="B",
        tools=list(create_todo_write_tool()),
        config=AgentConfig(
            session_dir=str(tmp_path / "B"),
            session_enabled=True,
            max_turns=5,
        ),
    )

    # 两个 Agent 应有完全独立的 session 目录
    assert agent_a._session.dir_path != agent_b._session.dir_path

    # 真实并发：通过 asyncio.gather 同时跑两个 Agent.run()
    async def consume(agent: Agent, user_msg: str) -> None:
        async for _evt in agent.run(user_msg):
            pass

    await asyncio.gather(
        consume(agent_a, "请制定计划 A"),
        consume(agent_b, "请制定计划 B"),
    )

    # 各自的 plan.json 必须存在
    plan_a_path = agent_a._session.dir_path / "plan.json"
    plan_b_path = agent_b._session.dir_path / "plan.json"
    assert plan_a_path.exists(), f"agent A 的 plan.json 不存在: {plan_a_path}"
    assert plan_b_path.exists(), f"agent B 的 plan.json 不存在: {plan_b_path}"

    plan_a = json.loads(plan_a_path.read_text(encoding="utf-8"))
    plan_b = json.loads(plan_b_path.read_text(encoding="utf-8"))

    contents_a = [item["content"] for item in plan_a["items"]]
    contents_b = [item["content"] for item in plan_b["items"]]

    # agent A 的 plan 只包含自己的任务
    assert "任务 A1" in contents_a
    assert "任务 A2" in contents_a
    assert "任务 B1" not in contents_a
    assert "任务 B2" not in contents_a

    # agent B 的 plan 只包含自己的任务
    assert "任务 B1" in contents_b
    assert "任务 B2" in contents_b
    assert "任务 A1" not in contents_b
    assert "任务 A2" not in contents_b
