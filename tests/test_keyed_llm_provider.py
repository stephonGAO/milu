"""测试 KeyedLLMProvider —— 多租户 API Key 隔离工厂。"""
from __future__ import annotations

import asyncio

import pytest

from milu.llm.base.response import StreamChunk, TokenUsage
from milu.llm.providers import ModelRegistry
from milu.llm.providers.base import BaseLLM, ModelCapabilities
from milu.serving import AgentPool, AgentPoolConfig, KeyedLLMProvider


class _FakeLLM(BaseLLM):
    """记录 api_key、可追踪关闭状态的假 LLM。"""

    def __init__(self, api_key=None, model="", **kwargs):
        super().__init__(api_key=api_key, model=model, **kwargs)
        self.closed = False

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def base_url(self) -> str:
        return "http://fake.local"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    def _get_available_param_names(self) -> set[str]:
        return set()

    async def chat(self, messages, **kwargs):
        yield StreamChunk(content="ok", finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(1, 1, 2))

    async def aclose(self) -> None:
        self.closed = True
        await super().aclose()


ModelRegistry.register("fake", _FakeLLM)


# ── 缓存语义 ────────────────────────────────────────────


async def test_same_key_reuses_instance():
    """同一 API Key 的多个用户复用同一个 LLM 实例。"""
    provider = KeyedLLMProvider("fake", resolve_api_key=lambda uid, sid: "shared-key")
    llm_a = provider("u1", "s1")
    llm_b = provider("u2", "s1")
    assert llm_a is llm_b, "相同 Key 应复用同一 LLM 实例"
    stats = provider.get_stats()
    assert stats["created"] == 1
    assert stats["reused"] == 1
    assert stats["active_clients"] == 1


async def test_different_keys_isolated():
    """不同 API Key 得到独立的 LLM 实例，且 Key 正确注入。"""
    keys = {"u1": "key-1", "u2": "key-2"}
    provider = KeyedLLMProvider("fake", resolve_api_key=lambda uid, sid: keys[uid])
    llm1 = provider("u1", "s1")
    llm2 = provider("u2", "s1")
    assert llm1 is not llm2
    assert llm1._api_key == "key-1"
    assert llm2._api_key == "key-2"
    assert provider.get_stats()["created"] == 2


async def test_none_key_falls_back_to_default():
    """resolve 返回 None 时使用 default_api_key，且多个 None 用户共享同一实例。"""
    provider = KeyedLLMProvider(
        "fake",
        resolve_api_key=lambda uid, sid: None,
        default_api_key="env-default",
    )
    llm_a = provider("u1", "s1")
    llm_b = provider("u2", "s2")
    assert llm_a is llm_b, "None Key 应共享默认实例"
    assert llm_a._api_key == "env-default"


async def test_model_and_kwargs_forwarded():
    """model 与额外 kwargs 透传给新建的 LLM。"""
    provider = KeyedLLMProvider(
        "fake", resolve_api_key=lambda uid, sid: "k", model="fake-pro", temperature=0.3,
    )
    llm = provider("u1", "s1")
    assert llm.model == "fake-pro"
    assert llm._extra_kwargs.get("temperature") == 0.3


# ── LRU 淘汰 + 连接池关闭 ─────────────────────────────────


async def test_lru_eviction_closes_connection():
    """超过 max_clients 时按 LRU 淘汰最久未用的 Key，并关闭其连接池。"""
    provider = KeyedLLMProvider(
        "fake", resolve_api_key=lambda uid, sid: uid, max_clients=2,
    )
    llm0 = provider("k0", "s")   # 最久未用
    llm1 = provider("k1", "s")
    assert provider.get_stats()["active_clients"] == 2

    llm2 = provider("k2", "s")   # 触发淘汰 k0
    assert provider.get_stats()["evicted"] == 1
    assert provider.get_stats()["active_clients"] == 2

    # 后台关闭任务跑完后，被淘汰的 llm0 连接池应被关闭
    await asyncio.sleep(0.05)
    assert llm0.closed is True, "被 LRU 淘汰的 LLM 应被关闭连接池"
    assert llm1.closed is False and llm2.closed is False


async def test_lru_refreshes_on_reuse():
    """复用会刷新 LRU 顺序，使最近用过的 Key 不被优先淘汰。"""
    provider = KeyedLLMProvider(
        "fake", resolve_api_key=lambda uid, sid: uid, max_clients=2,
    )
    llm0 = provider("k0", "s")
    llm1 = provider("k1", "s")
    provider("k0", "s")            # 复用 k0 → k0 变为最近使用
    provider("k2", "s")            # 淘汰最久未用的 k1（而非 k0）
    await asyncio.sleep(0.05)
    assert llm1.closed is True
    assert llm0.closed is False


async def test_aclose_closes_all():
    """aclose 关闭所有缓存的 LLM 并清空缓存。"""
    provider = KeyedLLMProvider("fake", resolve_api_key=lambda uid, sid: uid)
    llms = [provider(f"k{i}", "s") for i in range(3)]
    await provider.aclose()
    assert all(llm.closed for llm in llms)
    assert provider.get_stats()["active_clients"] == 0


async def test_max_clients_validation():
    with pytest.raises(ValueError):
        KeyedLLMProvider("fake", resolve_api_key=lambda uid, sid: "k", max_clients=0)


# ── 与 AgentPool 集成 ────────────────────────────────────


async def test_with_agent_pool_same_tenant_shares_client():
    """关键场景：同租户 Key 跨会话复用同一 LLM 客户端，不随会话数增长连接池。"""
    provider = KeyedLLMProvider("fake", resolve_api_key=lambda uid, sid: f"tenant-{uid}")
    pool = AgentPool(
        llm_factory=provider,
        config=AgentPoolConfig(max_agents=10, max_concurrent_runs=5),
        # 能力参数已上移为 Agent 直接参数，经 agent_kwargs 透传；关闭 session 保持 hermetic
        agent_kwargs={"session_enabled": False},
    )
    await pool.start()
    try:
        # 同一用户的两个不同会话 → 两个独立 Agent，但共享同一个 LLM 客户端
        for sid in ("s1", "s2"):
            async with pool.acquire("u1", sid) as h:
                async for _ in h.agent.run("hi"):
                    pass
        stats = provider.get_stats()
        assert stats["created"] == 1, "同租户 Key 只应建一个 LLM 客户端"
        assert stats["reused"] == 1, "第二个会话应复用该客户端"
    finally:
        await pool.stop()
        await provider.aclose()


async def test_with_agent_pool_different_tenants_isolated():
    """不同租户在 AgentPool 下使用各自独立的 LLM 客户端。"""
    seen_keys = {}

    def resolve(uid, sid):
        key = f"tenant-{uid}"
        seen_keys[uid] = key
        return key

    provider = KeyedLLMProvider("fake", resolve_api_key=resolve)
    pool = AgentPool(
        llm_factory=provider,
        config=AgentPoolConfig(max_agents=10, max_concurrent_runs=5),
        # 能力参数已上移为 Agent 直接参数，经 agent_kwargs 透传；关闭 session 保持 hermetic
        agent_kwargs={"session_enabled": False},
    )
    await pool.start()
    try:
        async def run_user(uid):
            async with pool.acquire(uid, "s1") as h:
                async for _ in h.agent.run("hi"):
                    pass

        await asyncio.gather(*[run_user(f"u{i}") for i in range(3)])
        assert provider.get_stats()["created"] == 3, "3 个不同租户应建 3 个客户端"
    finally:
        await pool.stop()
        await provider.aclose()


# ── BaseLLM.aclose 真实路径 ──────────────────────────────


async def test_basellm_aclose_real_client():
    """BaseLLM.aclose 关闭真实 AsyncOpenAI 客户端并复位为可重建状态。"""
    llm = ModelRegistry.create("qwen", api_key="sk-test", model="qwen-plus")
    client = llm._get_client()       # 懒建真实 AsyncOpenAI
    assert llm._client is client
    await llm.aclose()
    assert llm._client is None, "aclose 后客户端应复位，便于再次懒建"
    # 再次 aclose 应为安全的空操作
    await llm.aclose()
