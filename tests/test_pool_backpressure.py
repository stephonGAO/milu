"""测试 AgentPool 的 P3 加固：背压（超时/拒绝）+ 观测性指标。"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from milu.llm.base.response import StreamChunk, TokenUsage
from milu.serving import (
    AgentPool,
    AgentPoolConfig,
    PoolBusyError,
    PoolExhaustedError,
)


def _echo_llm(delay: float = 0.0):
    async def chat(messages, **kwargs):
        if delay:
            await asyncio.sleep(delay)
        yield StreamChunk(content="ok", finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(10, 5, 15))
    llm = AsyncMock()
    llm.chat = chat
    return llm


def _blocking_llm():
    """run 会卡在 chat 里，直到 release 被 set；started 标记已进入。"""
    started = asyncio.Event()
    release = asyncio.Event()

    async def chat(messages, **kwargs):
        started.set()
        await release.wait()
        yield StreamChunk(content="ok", finish_reason="stop")

    llm = AsyncMock()
    llm.chat = chat
    return llm, started, release


@pytest.mark.asyncio
async def test_acquire_timeout_raises_pool_busy():
    """并发名额满 + acquire_timeout → 抛 PoolBusyError，并计入 rejected_busy。"""
    llm, started, release = _blocking_llm()
    pool = AgentPool(
        llm_factory=lambda u, s: llm,
        config=AgentPoolConfig(max_agents=10, max_concurrent_runs=1, acquire_timeout=0.2),
        agent_kwargs={"session_enabled": False},
    )
    await pool.start()
    try:
        async def hold():
            async with pool.acquire("a", "s1") as h:
                async for _ in h.agent.run("x"):
                    pass

        task = asyncio.create_task(hold())
        await asyncio.wait_for(started.wait(), 2.0)  # a 占住唯一名额

        # b 不同 key（避开 entry 锁），应在 ~0.2s 内被名额超时拒绝
        with pytest.raises(PoolBusyError):
            async with pool.acquire("b", "s1") as h:
                async for _ in h.agent.run("y"):
                    pass

        assert pool.get_stats()["rejected_busy"] >= 1

        release.set()
        await asyncio.wait_for(task, 2.0)
    finally:
        release.set()
        await pool.stop()


@pytest.mark.asyncio
async def test_pool_exhausted_when_full_and_busy():
    """池满（max_agents）且全部在用 → 抛 PoolExhaustedError，计入 rejected_full。"""
    llm, started, release = _blocking_llm()
    pool = AgentPool(
        llm_factory=lambda u, s: llm,
        config=AgentPoolConfig(max_agents=1, max_concurrent_runs=10),
        agent_kwargs={"session_enabled": False},
    )
    await pool.start()
    try:
        async def hold():
            async with pool.acquire("a", "s1") as h:
                async for _ in h.agent.run("x"):
                    pass

        task = asyncio.create_task(hold())
        await asyncio.wait_for(started.wait(), 2.0)  # a 占住唯一实例（active）

        with pytest.raises(PoolExhaustedError):
            async with pool.acquire("b", "s2") as h:  # 池满 1/1 且 a 在用，无法淘汰
                async for _ in h.agent.run("y"):
                    pass

        assert pool.get_stats()["rejected_full"] >= 1

        release.set()
        await asyncio.wait_for(task, 2.0)
    finally:
        release.set()
        await pool.stop()


@pytest.mark.asyncio
async def test_no_timeout_blocks_then_succeeds():
    """acquire_timeout=None（默认）时，名额满应阻塞等待而非拒绝。"""
    llm, started, release = _blocking_llm()
    pool = AgentPool(
        llm_factory=lambda u, s: llm,
        config=AgentPoolConfig(max_agents=10, max_concurrent_runs=1),  # 无 acquire_timeout
        agent_kwargs={"session_enabled": False},
    )
    await pool.start()
    try:
        async def hold():
            async with pool.acquire("a", "s1") as h:
                async for _ in h.agent.run("x"):
                    pass

        task = asyncio.create_task(hold())
        await asyncio.wait_for(started.wait(), 2.0)

        # b 启动后应阻塞（不抛异常）；放行 a 后 b 才能完成
        b_done = asyncio.Event()

        async def runner_b():
            async with pool.acquire("b", "s1") as h:
                async for _ in h.agent.run("y"):
                    pass
            b_done.set()

        # b 用 echo（不阻塞）：但它先要拿到名额
        # 这里直接复用阻塞 llm 不行（会卡），改为换一个 echo agent_factory 不便；
        # 简化：b 也走同一 pool，但 a 放行后整体很快完成
        tb = asyncio.create_task(runner_b())
        await asyncio.sleep(0.1)
        assert not b_done.is_set(), "名额满时 b 应阻塞等待"
        assert pool.get_stats()["waiting"] >= 1

        release.set()  # 放行 a，b 随后拿到名额
        await asyncio.wait_for(tb, 2.0)
        await asyncio.wait_for(task, 2.0)
        assert b_done.is_set()
    finally:
        release.set()
        await pool.stop()


@pytest.mark.asyncio
async def test_observability_fields_present():
    """get_stats 应包含 P3 新增的实时负载与时延指标。"""
    llm = _echo_llm()
    pool = AgentPool(
        llm_factory=lambda u, s: llm,
        config=AgentPoolConfig(max_agents=10, max_concurrent_runs=5),
        agent_kwargs={"session_enabled": False},
    )
    await pool.start()
    try:
        async with pool.acquire("a", "s1") as h:
            async for _ in h.agent.run("x"):
                pass

        st = pool.get_stats()
        for k in [
            "in_flight", "waiting", "available_slots", "mcp_connected_agents",
            "run_p50_ms", "run_p95_ms", "runs_sampled",
            "rejected_busy", "rejected_full", "completed_runs",
        ]:
            assert k in st, f"缺少指标 {k}"

        assert st["completed_runs"] == 1
        assert st["runs_sampled"] == 1
        assert st["in_flight"] == 0          # run 结束后归零
        assert st["waiting"] == 0
        assert st["available_slots"] == 5    # 空闲时名额全可用
        assert st["mcp_connected_agents"] == 0
        assert st["run_p50_ms"] >= 0
    finally:
        await pool.stop()
