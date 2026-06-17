"""测试 MCP 进程共享（P1-5）。

Level 1：Agent 注入「共享、外部拥有」的 MCPManager → 复用工具、不自建进程、不断开。
Level 2：AgentPool(shared_mcp=True) → 整池只连一组 MCP，注入每个 Agent，stop 时断开。

不依赖真实 MCP server：用 FakeMCPManager 模拟「已连接」的管理器。
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from milu.agent import Agent
from milu.llm.base.response import StreamChunk, TokenUsage
from milu.serving import AgentPool, AgentPoolConfig
from milu.tools import tool


def _wrappers():
    @tool(name="srv__fetch", description="抓取", is_safe=True)
    async def fetch(url: str) -> str:
        return "ok"

    @tool(name="srv__search", description="搜索", is_safe=True)
    async def search(q: str) -> str:
        return "ok"

    return [fetch._tool_wrapper, search._tool_wrapper]


class _FakeMCPManager:
    """模拟一个「已连接」的共享 MCPManager。"""

    def __init__(self, wrappers, configs=None):
        self._wrappers = list(wrappers)
        self.connect_calls = 0
        self.disconnect_calls = 0

    def get_tools(self):
        return list(self._wrappers)

    async def connect_all(self):
        self.connect_calls += 1
        return list(self._wrappers)

    async def disconnect_all(self):
        self.disconnect_calls += 1


def _echo_llm():
    async def chat(messages, **kwargs):
        yield StreamChunk(content="ok", finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(1, 1, 2))
    llm = AsyncMock()
    llm.chat = chat
    return llm


def _agent(mgr=None, **agent_kwargs):
    return Agent(
        llm=_echo_llm(),
        session_enabled=False,
        mcp_manager=mgr,
        register_catalog=False,
        register_skills=False,
        **agent_kwargs,  # 如 mcp_tools_active_by_default=True（现为 Agent 参数）
    )


# ── Level 1：Agent 注入共享 manager ──────────────────────────


def test_injected_manager_registers_tools_and_not_owned():
    mgr = _FakeMCPManager(_wrappers())
    agent = _agent(mgr)
    # 默认 mcp_tools_active_by_default=False → 进休眠池
    assert agent.tools.get_tool("srv__fetch") is not None
    assert agent.tools.is_dormant("srv__fetch")
    assert agent._owns_mcp is False


def test_two_agents_share_same_wrapper_objects():
    """两个 Agent 共享同一 manager → 工具是同一组 wrapper（同一组进程）。"""
    mgr = _FakeMCPManager(_wrappers())
    a1, a2 = _agent(mgr), _agent(mgr)
    assert a1.tools.get_tool("srv__fetch") is a2.tools.get_tool("srv__fetch")


@pytest.mark.asyncio
async def test_shared_agent_does_not_reconnect_or_disconnect():
    mgr = _FakeMCPManager(_wrappers())
    agent = _agent(mgr)
    await agent.connect_mcp()      # 共享：no-op，不重连
    await agent.disconnect_mcp()   # 共享：不断开（生命周期归注入方）
    assert mgr.connect_calls == 0
    assert mgr.disconnect_calls == 0


def test_active_by_default_registers_to_active_pool():
    mgr = _FakeMCPManager(_wrappers())
    agent = _agent(mgr, mcp_tools_active_by_default=True)
    assert "srv__fetch" in agent.tools.list_tools()  # 活跃池


def test_no_injection_owns_mcp():
    agent = _agent(None)
    assert agent._owns_mcp is True
    assert agent._mcp_manager is None


# ── Level 2：AgentPool(shared_mcp=True) ──────────────────────


@pytest.mark.asyncio
async def test_pool_shared_mcp_single_manager(monkeypatch):
    """整池只创建一个共享 manager，注入所有 Agent，stop 时断开一次。"""
    wrappers = _wrappers()
    import milu.tools.mcp.config as cfgmod
    import milu.tools.mcp.manager as mgrmod

    # 让配置加载返回非空（内容无所谓，FakeMgr 不用它）
    monkeypatch.setattr(cfgmod.MCPServerConfig, "load_file",
                        staticmethod(lambda path=None: ["fake-config"]))

    created: list = []

    class FakeMgr(_FakeMCPManager):
        def __init__(self, configs):
            super().__init__(wrappers, configs)
            created.append(self)

    monkeypatch.setattr(mgrmod, "MCPManager", FakeMgr)

    pool = AgentPool(
        llm_factory=lambda u, s: _echo_llm(),
        config=AgentPoolConfig(shared_mcp=True),
        agent_kwargs={"session_enabled": False},
    )
    await pool.start()
    try:
        assert len(created) == 1, "整池只应创建一个共享 manager"
        assert pool.shared_mcp_manager is created[0]
        assert created[0].connect_calls == 1

        async with pool.acquire("u1", "s1") as h:
            assert h.agent.tools.get_tool("srv__fetch") is not None
            assert h.agent._owns_mcp is False
        async with pool.acquire("u2", "s2") as h:
            assert h.agent.tools.get_tool("srv__fetch") is not None

        # 多用户后仍只有一个 manager（未随用户数增长）
        assert len(created) == 1
    finally:
        await pool.stop()

    assert created[0].disconnect_calls == 1, "stop 时应断开共享 MCP 一次"


@pytest.mark.asyncio
async def test_pool_shared_mcp_disabled_by_default():
    """默认 shared_mcp=False → 无共享 manager，Agent 各自拥有 MCP。"""
    pool = AgentPool(
        llm_factory=lambda u, s: _echo_llm(),
        config=AgentPoolConfig(),
        agent_kwargs={"session_enabled": False},
    )
    await pool.start()
    try:
        assert pool.shared_mcp_manager is None
        async with pool.acquire("u1", "s1") as h:
            assert h.agent._owns_mcp is True
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_pool_shared_mcp_connect_failure_degrades(monkeypatch):
    """共享 MCP 连接失败不应阻断池启动，退化为无 MCP。"""
    import milu.tools.mcp.config as cfgmod
    import milu.tools.mcp.manager as mgrmod

    monkeypatch.setattr(cfgmod.MCPServerConfig, "load_file",
                        staticmethod(lambda path=None: ["fake-config"]))

    class BoomMgr:
        def __init__(self, configs):
            pass

        async def connect_all(self):
            raise ConnectionError("all servers down")

    monkeypatch.setattr(mgrmod, "MCPManager", BoomMgr)

    pool = AgentPool(
        llm_factory=lambda u, s: _echo_llm(),
        config=AgentPoolConfig(shared_mcp=True),
        agent_kwargs={"session_enabled": False},
    )
    await pool.start()  # 不应抛异常
    try:
        assert pool.shared_mcp_manager is None
        async with pool.acquire("u1", "s1") as h:
            assert h.agent._owns_mcp is True  # 无共享 → 退回 per-agent
    finally:
        await pool.stop()
