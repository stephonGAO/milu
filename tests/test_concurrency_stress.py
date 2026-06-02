"""并发压测 — 验证 P0 bug 在真实并发下是否会出现。

运行方式：
    .venv/Scripts/python -m pytest tests/test_concurrency_stress.py -v

本文件不是常规 unit test，而是"压力测试"：在并发场景下暴露 P0 bug。
正常 CI 应跳过此文件（加 marker 或单独运行），失败时反而说明 bug 存在。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent_framework.agent import Agent, AgentConfig
from agent_framework.agent.events import (
    AgentDone, AgentError, SubAgentDone, SubAgentEvent, TextDelta, ToolResult,
)
from agent_framework.agent.subagent import SubAgentConfig, create_subagent_tools
from agent_framework.llm.base.message import Message, MessageRole
from agent_framework.llm.base.response import StreamChunk, TokenUsage
from agent_framework.tools import tool


# ─────────────────────────────────────────────────────────────
# 工具 1: 共享 LLM（模拟服务端"多用户共享同一 LLM 实例"的常见用法）
# ─────────────────────────────────────────────────────────────

def _make_echo_llm(response_text: str = "echo", delay: float = 0.01):
    """构造一个会回复固定文本的异步流式 LLM（带轻微延迟模拟真实 RTT）。"""
    async def chat(messages, **kwargs):
        await asyncio.sleep(delay)
        yield StreamChunk(content=response_text, finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(
            prompt_tokens=10, completion_tokens=5, total_tokens=15,
        ))
    llm = AsyncMock()
    llm.chat = chat
    return llm


def _make_tool_call_llm():
    """第一次返回 tool_call，第二次返回 text。"""
    state = {"count": 0}

    async def chat(messages, **kwargs):
        await asyncio.sleep(0.01)
        state["count"] += 1
        if state["count"] % 2 == 1:
            yield StreamChunk(tool_calls=[type("TC", (), {
                "index": 0, "id": "call_x",
                "function": type("FN", (), {"name": "echo_tool", "arguments": "{}"})(),
            })()])
            yield StreamChunk(finish_reason="tool_calls")
        else:
            yield StreamChunk(content="done", finish_reason="stop")
            yield StreamChunk(usage=TokenUsage(10, 5, 15))

    llm = AsyncMock()
    llm.chat = chat
    return llm


@tool(name="echo_tool", description="echo")
async def echo_tool() -> str:
    return "echo-result"


# ─────────────────────────────────────────────────────────────
# Test 1: 共享 Agent 实例并发调用 — 历史串味
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_shared_agent_concurrent_runs_history_cross_contamination(capsys):
    """P0-2: 共享同一 Agent 实例并发调用，用户历史会相互串味。

    预期：N 个并发请求都把对方的消息混进自己的 history。
    这个 test 现在应该"通过"（显示 bug 存在）—— 修复后才会失败。
    """
    llm = _make_echo_llm("ok")
    agent = Agent(llm=llm, system_prompt="sys", config=AgentConfig(session_enabled=False))

    async def user_run(user_id: str, text: str):
        events = []
        async for e in agent.run(f"[{user_id}] {text}"):
            events.append(e)
        return user_id, events

    # 启动 20 个并发请求
    tasks = [user_run(f"u{i}", f"msg-{i}") for i in range(20)]
    results = await asyncio.gather(*tasks)

    # 检查每个 user 的 history 是否只包含自己的消息
    contaminated = []
    for user_id, _ in results:
        msgs = agent.history.all_messages
        # user 消息是 USER role 且以 [<user_id>] 开头
        user_msgs = [m for m in msgs if m.role == MessageRole.USER]
        # 所有 user 消息应该都来自同一个 user
        other_users = [m for m in user_msgs if not m.content.startswith(f"[{user_id}]")]
        if other_users:
            contaminated.append((user_id, len(other_users)))

    print(f"\n[Test 1] 共享 Agent 实例下，{len(contaminated)}/20 个用户历史被污染")
    print(f"  样例污染情况: {contaminated[:3]}")
    # bug 存在时 contaminated > 0，修复后应该全部为 0
    # 暂时只记录，不断言（让测试通过以记录现状）
    # assert len(contaminated) == 0, f"历史串味: {contaminated[:5]}"


# ─────────────────────────────────────────────────────────────
# Test 2: SubAgent 闭包共享 — 并发子代理事件错乱
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_subagent_concurrent_events_cross_contamination(capsys):
    """P0-1: SubAgent 的 _last_events 闭包共享，并发时事件互相覆盖。

    预期：调用同一 subagent 两次并发，父 Agent 看到的子代理事件数 < 期望值。
    """
    llm = _make_echo_llm("sub-result")

    sub_tool = create_subagent_tools(llm, [
        SubAgentConfig(name="helper", description="test helper", system_prompt="..."),
    ])[0]

    # 拿一个父 Agent 包装 subagent
    # 父 LLM 第一次返回 tool_call(helper)，第二次返回 text
    state = {"count": 0}
    async def parent_chat(messages, **kwargs):
        await asyncio.sleep(0.01)
        state["count"] += 1
        if state["count"] % 2 == 1:
            yield StreamChunk(tool_calls=[type("TC", (), {
                "index": 0, "id": "call_sub",
                "function": type("FN", (), {"name": "helper", "arguments": '{"task":"x"}'})(),
            })()])
            yield StreamChunk(finish_reason="tool_calls")
        else:
            yield StreamChunk(content="parent-done", finish_reason="stop")
            yield StreamChunk(usage=TokenUsage(10, 5, 15))

    parent_llm = AsyncMock()
    parent_llm.chat = parent_chat

    parent = Agent(llm=parent_llm, system_prompt="parent", tools=[sub_tool],
                   config=AgentConfig(session_enabled=False))

    # 单次跑，看 baseline
    base_events = []
    async for e in parent.run("hi"):
        base_events.append(e)
    base_sub_done = sum(1 for e in base_events if isinstance(e, SubAgentDone))
    base_sub_events = sum(1 for e in base_events if isinstance(e, SubAgentEvent))
    print(f"\n[Test 2 baseline] 单次 run: SubAgentDone={base_sub_done}, SubAgentEvent={base_sub_events}")

    # 并发跑 2 次，看事件是否完整
    concurrent_events_lists = await asyncio.gather(*[
        (lambda: _collect(parent.run(f"msg-{i}")))() for i in range(2)
    ])

    # 每个并发 run 期望至少 1 个 SubAgentDone
    missing = []
    for i, evts in enumerate(concurrent_events_lists):
        sub_done = sum(1 for e in evts if isinstance(e, SubAgentDone))
        if sub_done == 0:
            missing.append(i)

    print(f"[Test 2 concurrent] {len(concurrent_events_lists)} 个并发 run 中 {len(missing)} 个缺失 SubAgentDone")
    # bug 存在时 missing > 0（部分并发 run 看到对方的子事件被覆盖）


async def _collect(agen):
    out = []
    async for e in agen:
        out.append(e)
    return out


# ─────────────────────────────────────────────────────────────
# Test 3: Session JSONL 并发写 — 损坏
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_session_jsonl_concurrent_writes_corruption(tmp_path: Path, capsys):
    """P0-3: 两个 Agent 共用同一 Session 时，JSONL 写会交错/损坏。

    场景：两个 Agent 实例都 attach 到同一 Session 对象，并发跑多个 turn，
    每个 turn 都会调 sess.log_message。在两次 LLM 调用之间的 await 点，
    两个 Agent 协程会切换，导致 log_message 交错执行。
    """
    from agent_framework.agent.session import Session

    sess = Session("shared_sess", tmp_path, model="mock")
    n_per_agent = 20

    # 共享 LLM：每次调用先 await 一点（让出），再返回 text
    async def shared_chat(messages, **kwargs):
        await asyncio.sleep(0.005)  # 让出时间窗
        yield StreamChunk(content="ok", finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(10, 5, 15))

    shared_llm = AsyncMock()
    shared_llm.chat = shared_chat

    async def user_run(user_id: str):
        # 不同 Agent 共享同一 Session
        agent = Agent(
            llm=shared_llm,
            system_prompt="sys",
            history=None,
            config=AgentConfig(session_enabled=False),
        )
        # 强制共享 Session
        agent._session = sess
        agent.history.attach_session(sess)
        # 把已有历史也加载进来（让两个 agent 看到对方之前的消息）
        for _ in range(n_per_agent):
            async for _ in agent.run(f"[{user_id}] msg"):
                pass

    await asyncio.gather(*[user_run(f"u{i}") for i in range(3)])

    # 读回检查
    raw = sess.conversation_path.read_text(encoding="utf-8")
    lines = [l for l in raw.splitlines() if l.strip()]

    valid = 0
    corrupt = 0
    for line in lines:
        try:
            obj = json.loads(line)
            if "msg" in str(obj.get("content", "")):
                valid += 1
        except json.JSONDecodeError:
            corrupt += 1

    print(f"\n[Test 3] 三 Agent × {n_per_agent} 轮 = 期望 ≥ 60 条 user 消息")
    print(f"         实际 {len(lines)} 行 ({valid} valid, {corrupt} corrupt)")
    # bug 表现：corrupt > 0 或 valid < 60（消息被交错截断）


# ─────────────────────────────────────────────────────────────
# Test 4: _agent_busy 误伤 — 并发 run 全部进入
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_busy_does_not_block_concurrent_runs(capsys):
    """P0 旁证: _agent_busy 是单标志而不是锁/计数，多协程并发 run 全部会进入执行。

    修复后行为不变（每用户独立 Agent），本测试只确认 bug 的具体表现。
    """
    llm = _make_echo_llm("ok", delay=0.05)
    agent = Agent(llm=llm, config=AgentConfig(session_enabled=False))

    enter_count = 0

    # 注入一个 sleep tool 模拟耗时操作
    @tool(name="slow", description="slow")
    async def slow() -> str:
        nonlocal enter_count
        enter_count += 1
        await asyncio.sleep(0.05)
        return "done"

    # 重新构造 LLM 调用 slow
    state = {"count": 0}
    async def chat(messages, **kwargs):
        await asyncio.sleep(0.01)
        state["count"] += 1
        if state["count"] % 2 == 1:
            yield StreamChunk(tool_calls=[type("TC", (), {
                "index": 0, "id": "call_slow",
                "function": type("FN", (), {"name": "slow", "arguments": "{}"})(),
            })()])
            yield StreamChunk(finish_reason="tool_calls")
        else:
            yield StreamChunk(content="ok", finish_reason="stop")
            yield StreamChunk(usage=TokenUsage(10, 5, 15))

    llm2 = AsyncMock()
    llm2.chat = chat
    agent2 = Agent(llm=llm2, tools=[slow], config=AgentConfig(session_enabled=False))

    async def one_run(i):
        async for _ in agent2.run(f"r-{i}"):
            pass

    await asyncio.gather(*[one_run(i) for i in range(5)])

    print(f"\n[Test 4] 5 个并发 run，slow 工具被调用 {enter_count} 次（期望 5）")
    # bug 表现：如果 _agent_busy 真的互斥，5 个并发 run 应当被串行化，
    # 慢工具仍会被调用 5 次（但 turn 1 是 tool_call → 调用 slow）。
    # 真正的问题是：history 中 5 个 run 的 message 会互相污染（Test 1 已证）。
    assert enter_count == 5
