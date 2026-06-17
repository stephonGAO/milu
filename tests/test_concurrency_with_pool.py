"""对比：相同压测场景下，AgentPool 是否真的隔离了用户。

复现 tests/test_concurrency_stress.py 的 Test 1 场景，但用 AgentPool 管理 Agent 实例。
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from milu.agent import Agent
from milu.agent.events import AgentDone
from milu.llm.base.message import MessageRole
from milu.llm.base.response import StreamChunk, TokenUsage
from milu.serving import AgentPool, AgentPoolConfig
from milu.tools import tool


def _make_echo_llm(delay: float = 0.01):
    async def chat(messages, **kwargs):
        await asyncio.sleep(delay)
        yield StreamChunk(content="ok", finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(10, 5, 15))
    llm = AsyncMock()
    llm.chat = chat
    return llm


def _make_tool_call_llm(tool_name: str, args: dict | None = None):
    """第一轮发起指定工具调用，第二轮收尾。"""
    import json
    call_count = 0
    args_str = json.dumps(args or {})

    async def chat(*a, **kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield StreamChunk(tool_calls=[
                type("obj", (), {
                    "index": 0, "id": "call_1",
                    "function": type("obj", (), {
                        "name": tool_name, "arguments": args_str,
                    })(),
                })()
            ])
            yield StreamChunk(finish_reason="tool_calls")
        else:
            yield StreamChunk(content="完成", finish_reason="stop")

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
        # 只验证内存隔离，不依赖持久化；关闭 session 保持 hermetic
        agent_kwargs={"session_enabled": False},
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

        print("\n[Pool隔离测试] 20 个用户并发:")
        print(f"  完成: {sum(1 for _, d, _ in results if d)}/20")
        print(f"  历史污染: {len(contaminated)}/20 (期望 0)")
        print(f"  创建 Agent: {pool.get_stats()['created']}")
        print(f"  复用 Agent: {pool.get_stats()['reused']}")

        assert len(contaminated) == 0, f"AgentPool 未隔离: {contaminated[:3]}"
        assert len(failed) == 0
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_pool_same_key_concurrent_runs_serialized(capsys):
    """P0-1：同一 (user, session) 的并发 run() 必须串行，history 不被污染。

    同会话共享同一个 Agent 实例，若并发执行 run() 会交错写 history/session。
    AgentPool 应通过 entry 锁把同 key 的请求串行化。
    """
    # 用带延迟的 echo LLM 制造交错窗口；并发计数在「被锁保护的临界区」内统计
    # （不依赖 async generator 的 finally，后者会延迟到生成器关闭才执行，造成误判）
    llm = _make_echo_llm(delay=0.02)

    pool = AgentPool(
        llm_factory=lambda u, s: llm,
        # max_concurrent_runs 远大于并发数 → 确保串行来自 entry 锁而非全局限流
        config=AgentPoolConfig(max_agents=10, max_concurrent_runs=10),
        # 本测试只验证串行化，不依赖持久化；关闭 session 避免读写 CWD 落盘
        agent_kwargs={"session_enabled": False},
    )
    await pool.start()
    try:
        N = 8
        active = 0
        max_active = 0

        async def one(i: int):
            nonlocal active, max_active
            async with pool.acquire("u1", "s1") as h:
                # 临界区：被 entry 锁保护，同 key 应串行进入
                active += 1
                max_active = max(max_active, active)
                try:
                    async for _ in h.agent.run(f"msg-{i}"):
                        pass
                finally:
                    active -= 1

        await asyncio.gather(*[one(i) for i in range(N)])

        # 串行性：同一 Agent 任意时刻只有 1 个 run 在执行
        assert max_active == 1, f"同会话 run() 未串行化，峰值并发={max_active}"

        # history 完整且无污染：恰好 N 条 user 消息，内容互异（无丢失/重复）
        async with pool.acquire("u1", "s1") as h:
            user_msgs = [
                m.content for m in h.agent.history.all_messages
                if m.role == MessageRole.USER
            ]
        assert len(user_msgs) == N, f"期望 {N} 条 user 消息，实际 {len(user_msgs)}"
        assert len(set(user_msgs)) == N, f"user 消息有重复/丢失：{user_msgs}"

        # 同 key 始终复用同一个 Agent
        assert pool.get_stats()["created"] == 1
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
        # 验证同一缓存实例内 history 累积，不依赖磁盘；关闭 session 保持 hermetic
        agent_kwargs={"session_enabled": False},
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


@pytest.mark.asyncio
async def test_agent_restores_history_from_session(tmp_path):
    """P1-3：显式 session_id + 已有日志 → 新建 Agent 应从磁盘恢复历史。"""
    sess_dir = str(tmp_path)
    llm = _make_echo_llm(0.0)

    a1 = Agent(llm=llm, session_dir=sess_dir, session_id="u1__s1",
               register_catalog=False, register_skills=False)
    async for _ in a1.run("hello-1"):
        pass

    # 用相同 session_id 新建 Agent（模拟淘汰/重启后重建）
    a2 = Agent(llm=llm, session_dir=sess_dir, session_id="u1__s1",
               register_catalog=False, register_skills=False)
    user_msgs = [m.content for m in a2.history.all_messages
                 if m.role == MessageRole.USER]
    assert "hello-1" in user_msgs, f"历史未恢复：{user_msgs}"


@pytest.mark.asyncio
async def test_pool_restores_history_after_eviction(tmp_path):
    """P1-3：LRU 淘汰后，按同一 (user, session) 重新 acquire 应恢复历史。"""
    llm = _make_echo_llm(0.0)
    pool = AgentPool(
        llm_factory=lambda u, s: llm,
        config=AgentPoolConfig(max_agents=1),
        agent_kwargs={"session_enabled": True, "session_dir": str(tmp_path)},
    )
    await pool.start()
    try:
        async with pool.acquire("u1", "s1") as h:
            async for _ in h.agent.run("remember-this"):
                pass

        # max_agents=1 → acquire 另一个 key 触发对 u1/s1 的 LRU 淘汰
        async with pool.acquire("u2", "s2") as h:
            async for _ in h.agent.run("other"):
                pass
        assert pool.get_stats()["evicted_lru"] == 1

        # 重新 acquire u1/s1 → 新建 Agent，应从磁盘恢复历史
        async with pool.acquire("u1", "s1") as h:
            user_msgs = [m.content for m in h.agent.history.all_messages
                         if m.role == MessageRole.USER]
            assert "remember-this" in user_msgs, f"淘汰后历史未恢复：{user_msgs}"
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_confirm_wait_does_not_hold_concurrency_slot():
    """P1-4：等待人工确认期间应释放全局并发许可，不阻塞其他用户。

    场景：max_concurrent_runs=1。用户 A 触发危险工具确认并长时间阻塞，
    用户 B 的请求应能拿到（A 让出的）并发名额并完成。
    """
    @tool(name="danger", description="危险操作", is_safe=False)
    async def danger(x: str) -> str:
        return f"done {x}"

    confirm_started = asyncio.Event()
    confirm_release = asyncio.Event()

    async def on_confirm_a(tool_name, args_str):
        confirm_started.set()
        await confirm_release.wait()   # 模拟人工长时间未响应
        return True

    echo_llm = _make_echo_llm(delay=0.0)

    def agent_factory(user_id, session_id, llm):
        if user_id == "A":
            return Agent(
                llm=_make_tool_call_llm("danger", {"x": "1"}),
                tools=[danger],
                on_confirm=on_confirm_a,
                mode="manual",  # 人工审批模式（不安全工具需确认）
                session_enabled=False,
            )
        return Agent(llm=echo_llm, session_enabled=False)

    pool = AgentPool(
        llm_factory=lambda u, s: echo_llm,
        agent_factory=agent_factory,
        config=AgentPoolConfig(max_agents=10, max_concurrent_runs=1),
    )
    await pool.start()
    try:
        async def run_a():
            async with pool.acquire("A", "s1") as h:
                async for _ in h.agent.run("do danger"):
                    pass

        async def run_b():
            async with pool.acquire("B", "s1") as h:
                done = False
                async for e in h.agent.run("hi"):
                    if isinstance(e, AgentDone):
                        done = True
                return done

        task_a = asyncio.create_task(run_a())

        # 等 A 进入确认等待（此时它应已释放 semaphore）
        await asyncio.wait_for(confirm_started.wait(), timeout=2.0)

        # 关键断言：B 能在 A 仍卡在确认时完成（说明名额已被释放）
        b_done = await asyncio.wait_for(run_b(), timeout=2.0)
        assert b_done, "用户 B 未能完成"
        assert not task_a.done(), "A 应仍卡在确认等待"

        # 放行 A，确认其也能正常收尾
        confirm_release.set()
        await asyncio.wait_for(task_a, timeout=2.0)
    finally:
        confirm_release.set()
        await pool.stop()
