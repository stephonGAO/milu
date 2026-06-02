"""对比：相同压测场景下，AgentPool 是否真的隔离了用户。

复现 tests/test_concurrency_stress.py 的 Test 1 场景，但用 AgentPool 管理 Agent 实例。
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from agent_framework.agent import AgentConfig
from agent_framework.agent.events import TextDelta, AgentDone
from agent_framework.llm.base.message import MessageRole
from agent_framework.llm.base.response import StreamChunk, TokenUsage
from agent_framework.serving import AgentPool, AgentPoolConfig


def _make_echo_llm(delay: float = 0.01):
    async def chat(messages, **kwargs):
        await asyncio.sleep(delay)
        yield StreamChunk(content="ok", finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(10, 5, 15))
    llm = AsyncMock()
    llm.chat = chat
    return llm


@pytest.mark.asyncio
async def test_pool_isolates_concurrent_users_under_load(capsys):
    """20 个用户并发调用 AgentPool，断言：每个用户的 history 互不串味。"""
    llm = _make_echo_llm()
    def factory(u, s):
        return llm

    pool = AgentPool(
        llm_factory=factory,
        config=AgentPoolConfig(max_agents=50, max_concurrent_runs=20),
    )
    await pool.start()
    try:
        async def user_run(user_id: str):
            async with pool.acquire(user_id, "s1") as h:
                events = []
                async for e in h.agent.run(f"[{user_id}] hello"):
                    events.append(e)
                # 拿到 done 事件
                done = next((e for e in events if isinstance(e, AgentDone)), None)
                # 检查 history
                user_msgs = [
                    m for m in h.agent.history.all_messages
                    if m.role == MessageRole.USER
                ]
                bad = [m for m in user_msgs if not m.content.startswith(f"[{user_id}]")]
                return user_id, done is not None, len(bad)

        results = await asyncio.gather(*[user_run(f"u{i}") for i in range(20)])

        contaminated = [(uid, n) for uid, done, n in results if n > 0]
        failed = [uid for uid, done, _ in results if not done]

        print(f"\n[Pool隔离测试] 20 个用户并发:")
        print(f"  完成: {sum(1 for _, d, _ in results if d)}/20")
        print(f"  历史污染: {len(contaminated)}/20 (期望 0)")
        print(f"  创建 Agent: {pool.get_stats()['created']}")
        print(f"  复用 Agent: {pool.get_stats()['reused']}")

        assert len(contaminated) == 0, f"AgentPool 未隔离: {contaminated[:3]}"
        assert len(failed) == 0
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_pool_same_user_serial_runs_share_history(capsys):
    """同一用户连续 run，history 应累积（不丢失上下文）。"""
    llm = _make_echo_llm()
    def factory(u, s):
        return llm

    pool = AgentPool(
        llm_factory=factory,
        config=AgentPoolConfig(max_agents=10),
    )
    await pool.start()
    try:
        # 同一用户串行 3 轮
        for i in range(3):
            async with pool.acquire("u1", "s1") as h:
                async for _ in h.agent.run(f"turn-{i}"):
                    pass

        # 拿到 history 验证
        async with pool.acquire("u1", "s1") as h:
            user_msgs = [
                m for m in h.agent.history.all_messages
                if m.role == MessageRole.USER
            ]
            contents = [m.content for m in user_msgs]
            assert "turn-0" in contents
            assert "turn-1" in contents
            assert "turn-2" in contents
            assert len(user_msgs) == 3, f"同一用户 history 应累积, 实际 {len(user_msgs)}"

        assert pool.get_stats()["created"] == 1
        assert pool.get_stats()["reused"] == 3
    finally:
        await pool.stop()
