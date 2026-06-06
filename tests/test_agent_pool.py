"""测试 AgentPool — 多用户隔离 + LRU 淘汰 + 限流。"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from milu.agent import Agent, AgentConfig
from milu.llm.base.response import StreamChunk, TokenUsage
from milu.serving import AgentPool, AgentPoolConfig


def _make_quick_llm():
    """构造一个会立即返回固定文本的 LLM。"""
    async def chat(messages, **kwargs):
        yield StreamChunk(content="ok", finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(10, 5, 15))
    llm = AsyncMock()
    llm.chat = chat
    return llm


@pytest.fixture
def llm_factory():
    """共享 LLM 工厂：返回同一个 LLM 实例（模拟实际部署）。"""
    llm = _make_quick_llm()
    def factory(user_id, session_id):
        return llm
    return factory


@pytest.mark.asyncio
async def test_pool_creates_agent_on_first_acquire(llm_factory):
    pool = AgentPool(llm_factory=llm_factory,
                     config=AgentPoolConfig(max_agents=10, max_concurrent_runs=5))
    await pool.start()
    try:
        async with pool.acquire("u1", "s1") as h:
            assert h.user_id == "u1"
            assert h.session_id == "s1"
            assert isinstance(h.agent, Agent)
        assert pool.get_stats()["created"] == 1
        assert pool.get_stats()["reused"] == 0
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_pool_reuses_agent_for_same_key(llm_factory):
    pool = AgentPool(llm_factory=llm_factory,
                     config=AgentPoolConfig(max_agents=10))
    await pool.start()
    try:
        async with pool.acquire("u1", "s1") as h1:
            agent_id_1 = id(h1.agent)
        async with pool.acquire("u1", "s1") as h2:
            agent_id_2 = id(h2.agent)
        assert agent_id_1 == agent_id_2, "相同 (user, session) 应复用同一 Agent"
        assert pool.get_stats()["created"] == 1
        assert pool.get_stats()["reused"] == 1
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_pool_isolates_different_users(llm_factory):
    pool = AgentPool(llm_factory=llm_factory,
                     config=AgentPoolConfig(max_agents=10))
    await pool.start()
    try:
        async with pool.acquire("u1", "s1") as h1:
            agent_id_1 = id(h1.agent)
        async with pool.acquire("u2", "s1") as h2:
            agent_id_2 = id(h2.agent)
        assert agent_id_1 != agent_id_2, "不同 user 应得到独立 Agent"
        assert pool.get_stats()["created"] == 2
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_pool_remove_evicts_and_rebuilds(llm_factory):
    """pool.remove() 关闭并从池移除指定实例；再次获取得到全新实例。"""
    pool = AgentPool(llm_factory=llm_factory, config=AgentPoolConfig(max_agents=10))
    await pool.start()
    try:
        a1 = await pool.get_or_create_agent("u1", "s1")
        assert ("u1", "s1") in pool._entries
        # 命中移除 → True，且从池字典摘除
        assert await pool.remove("u1", "s1") is True
        assert ("u1", "s1") not in pool._entries
        # 再次获取 → 按工厂重建新实例（非同一对象）
        a2 = await pool.get_or_create_agent("u1", "s1")
        assert id(a1) != id(a2)
        # 移除不存在的 → False（幂等、无副作用）
        assert await pool.remove("u1", "missing") is False
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_pool_lru_eviction(llm_factory):
    pool = AgentPool(llm_factory=llm_factory,
                     config=AgentPoolConfig(max_agents=2))
    await pool.start()
    try:
        # 创建 2 个填满池
        async with pool.acquire("u1", "s1") as h1:
            id1 = id(h1.agent)
        async with pool.acquire("u2", "s2") as h2:
            id2 = id(h2.agent)
        assert pool.get_stats()["active_entries"] == 2

        # 第 3 个应触发 LRU 淘汰（u1 最早）
        async with pool.acquire("u3", "s3") as h3:
            id3 = id(h3.agent)
        assert pool.get_stats()["evicted_lru"] == 1
        assert pool.get_stats()["active_entries"] == 2

        # u1 被淘汰，再次 acquire 应得到新实例
        async with pool.acquire("u1", "s1") as h1_new:
            assert id(h1_new.agent) != id1
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_pool_concurrent_runs_isolated(llm_factory):
    """核心场景：N 个用户并发 run，history 互不串味。"""
    pool = AgentPool(llm_factory=llm_factory,
                     config=AgentPoolConfig(max_agents=20, max_concurrent_runs=10),
                     # 只验证内存隔离，不依赖持久化；关闭 session 保持 hermetic
                     agent_kwargs={"session_enabled": False})
    await pool.start()
    try:
        async def user_run(user_id: str, text: str):
            async with pool.acquire(user_id, "s1") as h:
                events = []
                async for e in h.agent.run(f"[{user_id}] {text}"):
                    events.append(e)
                # 验证 history 只含本用户消息
                user_msgs = [
                    m for m in h.agent.history.all_messages
                    if m.role.value == "user"
                ]
                bad = [m for m in user_msgs if not m.content.startswith(f"[{user_id}]")]
                return user_id, len(events), len(bad)

        results = await asyncio.gather(*[
            user_run(f"u{i}", f"msg-{i}") for i in range(10)
        ])

        # 全部 10 个用户都应正确完成，且无历史串味
        for uid, n_events, n_bad in results:
            assert n_events > 0, f"{uid} 没有任何事件"
            assert n_bad == 0, f"{uid} 的 history 被污染（{n_bad} 条他人消息）"

        # 池中应有 10 个独立 Agent
        assert pool.get_stats()["created"] == 10
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_pool_concurrency_limit(llm_factory):
    """Semaphore 限流：max_concurrent_runs 之外的请求应阻塞。"""
    # 构造慢 LLM
    async def slow_chat(messages, **kwargs):
        await asyncio.sleep(0.2)
        yield StreamChunk(content="ok", finish_reason="stop")

    slow_llm = AsyncMock()
    slow_llm.chat = slow_chat
    def factory(user_id, session_id):
        return slow_llm

    pool = AgentPool(
        llm_factory=factory,
        config=AgentPoolConfig(max_agents=20, max_concurrent_runs=2),
    )
    await pool.start()
    try:
        start = time.monotonic()
        # 启动 5 个并发 run，限流 2 → 至少需要 3 批 × 0.2s ≈ 0.6s
        async def one_run(i):
            async with pool.acquire(f"u{i}", "s1") as h:
                async for _ in h.agent.run("x"):
                    pass
        await asyncio.gather(*[one_run(i) for i in range(5)])
        elapsed = time.monotonic() - start
        # 2 并发 / 5 任务 / 0.2s/任务 ≈ 至少 0.5s（实际 GIL 调度 + 上下文开销略大）
        assert elapsed >= 0.4, f"限流未生效，只耗时 {elapsed:.2f}s"
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_pool_idle_sweep(llm_factory):
    """空闲 TTL 过期后，后台清理任务应淘汰实例。"""
    pool = AgentPool(
        llm_factory=llm_factory,
        config=AgentPoolConfig(
            max_agents=10,
            idle_ttl_seconds=0.2,
            sweep_interval_seconds=0.1,
        ),
    )
    await pool.start()
    try:
        async with pool.acquire("u1", "s1"):
            pass
        assert pool.get_stats()["active_entries"] == 1

        # 等后台清理跑几轮
        await asyncio.sleep(0.5)
        assert pool.get_stats()["evicted_idle"] >= 1
        assert pool.get_stats()["active_entries"] == 0
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_pool_entry_lock_cleaned_on_lru_eviction(llm_factory):
    """回归：LRU 淘汰后 per-key 创建锁应随之移除，不随历史 key 数无限增长。

    _entry_locks 的不变量：始终与 _entries 的 key 集合一致——
    每个 entry 在「缓存未命中」创建时建一把锁，淘汰时连同 entry 一起删。
    """
    pool = AgentPool(llm_factory=llm_factory,
                     config=AgentPoolConfig(max_agents=2))
    await pool.start()
    try:
        # 制造 5 个不同 key 的 churn：池容量 2 → 触发 3 次 LRU 淘汰
        for i in range(5):
            async with pool.acquire(f"u{i}", f"s{i}"):
                pass

        assert pool.get_stats()["evicted_lru"] == 3
        # 关键断言：锁字典不随历史 key 数（5）增长，只保留在用的 2 个
        assert len(pool._entry_locks) <= 2
        assert set(pool._entry_locks.keys()) == set(pool._entries.keys()), \
            "_entry_locks 应始终与 _entries 同步（淘汰时同步清理）"
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_pool_entry_lock_cleaned_on_idle_sweep(llm_factory):
    """回归：空闲 TTL 清理掉所有 entry 后，per-key 创建锁也应清空。"""
    pool = AgentPool(
        llm_factory=llm_factory,
        config=AgentPoolConfig(
            max_agents=10,
            idle_ttl_seconds=0.2,
            sweep_interval_seconds=0.1,
        ),
    )
    await pool.start()
    try:
        for i in range(3):
            async with pool.acquire(f"u{i}", "s1"):
                pass
        assert len(pool._entry_locks) == 3

        # 等后台 sweep 清掉全部空闲实例
        await asyncio.sleep(0.5)
        assert pool.get_stats()["active_entries"] == 0
        assert len(pool._entry_locks) == 0, "空闲清理后锁字典应清空，否则内存泄漏"
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_pool_entry_locks_cleared_on_stop(llm_factory):
    """回归：stop()/_close_all 应清空 per-key 创建锁。"""
    pool = AgentPool(llm_factory=llm_factory,
                     config=AgentPoolConfig(max_agents=10))
    await pool.start()
    for i in range(3):
        async with pool.acquire(f"u{i}", "s1"):
            pass
    assert len(pool._entry_locks) == 3
    await pool.stop()
    assert len(pool._entry_locks) == 0, "关闭池后锁字典应清空"


@pytest.mark.asyncio
async def test_pool_get_or_create_agent_bypasses_semaphore(llm_factory):
    """get_or_create_agent 不占并发许可（供后台任务使用）。"""
    pool = AgentPool(
        llm_factory=llm_factory,
        config=AgentPoolConfig(max_agents=10, max_concurrent_runs=1),
    )
    await pool.start()
    try:
        # 不需要 acquire semaphore 即可创建
        agent = await pool.get_or_create_agent("bg-task", "s1")
        assert isinstance(agent, Agent)
        # 仍然计入池
        assert pool.get_stats()["created"] == 1
    finally:
        await pool.stop()
