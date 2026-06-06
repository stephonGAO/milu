"""schedule 工具用户上下文隔离测试（对标 test_todo_concurrent_isolation.py）。

验证：ContextVar 注入落对应用户文件、未注入退化 default（与 memory 写拒绝
的**有意差异**，CLI 兼容约定）、Agent(schedule_user=...) 注入、并发隔离、
AgentPool 默认工厂派生。
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from milu.agent import Agent, AgentConfig
from milu.llm.base.response import StreamChunk, TokenUsage
from milu.llm.providers.base import BaseLLM, ModelCapabilities
from milu.tools.builtin.schedule_tool import (
    _current_schedule_user,
    _resolve_user,
    schedule_create,
    schedule_manage,
)


# ── Fake 数据结构 ────────────────────────────────────────

@dataclass
class _FakeFunction:
    name: str
    arguments: str = ""


@dataclass
class _FakeToolCall:
    function: _FakeFunction
    id: str
    index: int = 0


class _ScheduleCreateLLM(BaseLLM):
    """Mock LLM：首轮发出一次 schedule_create tool_call，下一轮结束。"""

    def __init__(self, task_name: str):
        super().__init__(model="mock", provider="mock")
        self._task_name = task_name
        self._call_count = 0

    async def chat(self, messages, **kwargs):
        self._call_count += 1
        if self._call_count == 1:
            yield StreamChunk(tool_calls=[
                _FakeToolCall(
                    function=_FakeFunction(
                        name="schedule_create",
                        arguments=json.dumps({
                            "name": self._task_name,
                            "prompt": "提醒喝水",
                            "trigger_type": "once",
                            "run_at": "2099-01-01T09:00:00",
                        }, ensure_ascii=False),
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


def _make_agent(llm: BaseLLM, schedule_user: str | None) -> Agent:
    # superwork 跳过安全检查（schedule_create 为不安全工具），聚焦上下文注入验证
    return Agent(
        llm=llm,
        system_prompt="test",
        tools=[schedule_create, schedule_manage],
        subagents=[],
        config=AgentConfig(max_turns=5),
        session_enabled=False,
        mode="superwork",
        schedule_user=schedule_user,
    )


async def _consume(agent: Agent, msg: str) -> None:
    async for _ in agent.run(msg):
        pass


# ── 工具层 ContextVar 行为 ────────────────────────────────

def test_resolve_user_default_when_unset():
    """ContextVar 未注入 → 退化为 "default"。

    与 memory 的「未注入即写拒绝」是**有意差异**：schedule 工具在
    BUILTIN_TOOLS 默认列表中，CLI 单人场景无注入必须保持旧行为兼容。
    """
    assert _current_schedule_user.get() is None
    assert _resolve_user() == "default"


def test_resolve_user_reads_contextvar():
    token = _current_schedule_user.set("alice")
    try:
        assert _resolve_user() == "alice"
    finally:
        _current_schedule_user.reset(token)


@pytest.mark.asyncio
async def test_tool_writes_to_contextvar_user_file(tmp_path: Path, monkeypatch):
    """注入 alice → schedule_create 落 schedules/alice.json。"""
    monkeypatch.setenv("MILU_HOME", str(tmp_path))
    token = _current_schedule_user.set("alice")
    try:
        result = await schedule_create._tool_wrapper.func(
            name="t1", prompt="p", trigger_type="once",
            run_at="2099-01-01T09:00:00",
        )
    finally:
        _current_schedule_user.reset(token)
    assert "已创建" in result
    assert (tmp_path / "schedules" / "alice.json").exists()


@pytest.mark.asyncio
async def test_tool_unset_writes_default_file(tmp_path: Path, monkeypatch):
    """未注入 → 落 schedules/default.json（CLI 兼容）。"""
    monkeypatch.setenv("MILU_HOME", str(tmp_path))
    result = await schedule_create._tool_wrapper.func(
        name="t1", prompt="p", trigger_type="once",
        run_at="2099-01-01T09:00:00",
    )
    assert "已创建" in result
    assert (tmp_path / "schedules" / "default.json").exists()


def test_safety_split():
    """安全性约定：create 整体不安全；manage 动态判定（list 只读安全）。"""
    assert schedule_create._tool_wrapper.is_safe is False
    assert schedule_create._tool_wrapper.safe_check is None

    check = schedule_manage._tool_wrapper.safe_check
    assert schedule_manage._tool_wrapper.is_safe is False
    assert check({"action": "list"}) is True
    for action in ("delete", "enable", "disable", "run_now"):
        assert check({"action": action}) is False
    assert check({}) is False  # 缺 action 视为不安全


@pytest.mark.asyncio
async def test_manage_actions_scoped_by_user(tmp_path: Path, monkeypatch):
    """manage 的删/启停/立即运行按用户隔离。"""
    monkeypatch.setenv("MILU_HOME", str(tmp_path))
    # alice 创建任务
    token = _current_schedule_user.set("alice")
    try:
        await schedule_create._tool_wrapper.func(
            name="t1", prompt="p", trigger_type="once",
            run_at="2099-01-01T09:00:00",
        )
    finally:
        _current_schedule_user.reset(token)

    # bob 删不到 alice 的任务
    token = _current_schedule_user.set("bob")
    try:
        result = await schedule_manage._tool_wrapper.func(action="delete", name="t1")
    finally:
        _current_schedule_user.reset(token)
    assert "不存在" in result

    # alice 自己可禁用/删除
    token = _current_schedule_user.set("alice")
    try:
        assert "已禁用" in await schedule_manage._tool_wrapper.func(
            action="disable", name="t1")
        assert "已删除" in await schedule_manage._tool_wrapper.func(
            action="delete", name="t1")
    finally:
        _current_schedule_user.reset(token)


# ── Agent 注入链路 ───────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_injects_schedule_user(tmp_path: Path, monkeypatch):
    """Agent(schedule_user="bob") 经 run() 注入，任务落 bob 文件。"""
    monkeypatch.setenv("MILU_HOME", str(tmp_path))
    agent = _make_agent(_ScheduleCreateLLM("bob-task"), schedule_user="bob")
    await _consume(agent, "创建任务")
    bob_file = tmp_path / "schedules" / "bob.json"
    assert bob_file.exists()
    data = json.loads(bob_file.read_text(encoding="utf-8"))
    assert data["tasks"][0]["name"] == "bob-task"
    assert data["tasks"][0]["user_id"] == "bob"


@pytest.mark.asyncio
async def test_two_agents_concurrent_isolated(tmp_path: Path, monkeypatch):
    """两个 Agent（alice/bob）并发创建任务，各落各的文件不串味。"""
    monkeypatch.setenv("MILU_HOME", str(tmp_path))
    agent_a = _make_agent(_ScheduleCreateLLM("task-a"), schedule_user="alice")
    agent_b = _make_agent(_ScheduleCreateLLM("task-b"), schedule_user="bob")

    await asyncio.gather(
        _consume(agent_a, "创建任务A"),
        _consume(agent_b, "创建任务B"),
    )

    data_a = json.loads(
        (tmp_path / "schedules" / "alice.json").read_text(encoding="utf-8"))
    data_b = json.loads(
        (tmp_path / "schedules" / "bob.json").read_text(encoding="utf-8"))
    assert [t["name"] for t in data_a["tasks"]] == ["task-a"]
    assert [t["name"] for t in data_b["tasks"]] == ["task-b"]
    assert all(t["user_id"] == "alice" for t in data_a["tasks"])
    assert all(t["user_id"] == "bob" for t in data_b["tasks"])


# ── AgentPool 默认工厂派生 ───────────────────────────────

@pytest.mark.asyncio
async def test_pool_factory_derives_schedule_user(tmp_path: Path, monkeypatch):
    """池上下文默认按 user_id 派生 schedule_user（schedule 工具默认在
    BUILTIN_TOOLS，不派生则全部用户共用 default 任务空间）。"""
    monkeypatch.setenv("MILU_HOME", str(tmp_path))
    from milu.serving import AgentPool

    pool = AgentPool.from_llm(
        _ScheduleCreateLLM("x"),
        agent_kwargs={"tools": [], "subagents": [], "session_enabled": False},
    )
    agent = await pool.get_or_create_agent("alice", "s1")
    assert agent._schedule_user == "alice"


@pytest.mark.asyncio
async def test_pool_factory_respects_explicit_schedule_user(tmp_path: Path, monkeypatch):
    """显式传字符串则尊重调用方（如多用户共享一份团队任务空间）。"""
    monkeypatch.setenv("MILU_HOME", str(tmp_path))
    from milu.serving import AgentPool

    pool = AgentPool.from_llm(
        _ScheduleCreateLLM("x"),
        agent_kwargs={
            "tools": [], "subagents": [], "session_enabled": False,
            "schedule_user": "team",
        },
    )
    agent = await pool.get_or_create_agent("alice", "s1")
    assert agent._schedule_user == "team"
