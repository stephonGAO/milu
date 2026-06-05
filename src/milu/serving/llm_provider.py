"""多租户 API Key 隔离 —— 按 Key 缓存 LLM 实例的工厂。

用作 ``AgentPool`` 的 ``llm_factory``，让不同租户/用户使用各自的 API Key：

- **同 Key 复用**：相同 API Key 的多个用户复用同一个 ``BaseLLM`` 实例（及其底层
  ``AsyncOpenAI`` 连接池），不随用户数线性增长内存——这正是多租户的常见形态
  （Key 映射到「租户/组织」而非「单个用户」，distinct key 数远小于用户数）。
- **不同 Key 隔离**：不同 API Key 得到独立的 ``BaseLLM`` 实例，互不串用配额。
- **有界 + LRU**：缓存的客户端总数受 ``max_clients`` 上限约束，超出按 LRU 淘汰最久
  未用的 Key，并**异步关闭其连接池**（与 ``AgentPool`` 的「有界池 + LRU」哲学一致）。

为何不在框架内做「每请求覆盖 Key（共享单连接池）」？
    openai SDK 的 ``client.with_options(api_key=...)`` 可在共享一个连接池的前提下
    按请求覆盖 Key，理论上最省内存。但它需把 Key 沿 ``run() → chat()`` 请求链透传，
    侵入性强且耦合 SDK 内部行为。对绝大多数「Key=租户」场景，本工厂的有界缓存已足够；
    若确有「每用户唯一 Key 且海量并发」的需求，可自行基于 ``with_options`` 扩展。

典型用法::

    from milu import AgentPool, AgentPoolConfig, KeyedLLMProvider

    provider = KeyedLLMProvider(
        "qwen",
        model="qwen-plus",
        # Key 应取自环境变量/密钥管理服务（Vault、云 KMS），切勿明文硬编码在源码中
        resolve_api_key=lambda user_id, session_id: lookup_tenant_key(user_id),
        max_clients=256,
    )
    pool = AgentPool(llm_factory=provider, config=AgentPoolConfig(...))
    await pool.start()
    # ... 服务请求 ...
    await pool.stop()
    await provider.aclose()   # 关闭所有缓存的 LLM 连接池
"""
from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import Any, Callable, Optional

from milu.llm.providers import ModelRegistry
from milu.llm.providers.base import BaseLLM

logger = logging.getLogger(__name__)

# 解析函数：(user_id, session_id) -> api_key；返回 None 表示回退到默认 Key / 环境变量
ApiKeyResolver = Callable[[str, str], Optional[str]]


class KeyedLLMProvider:
    """按 API Key 缓存 ``BaseLLM`` 实例的工厂（多租户 Key 隔离）。

    作为可调用对象直接传给 ``AgentPool(llm_factory=...)``：每次创建用户 Agent 时，
    本工厂用 ``resolve_api_key(user_id, session_id)`` 解析出该用户的 Key，并据此
    复用或新建对应的 LLM 实例。

    线程/异步安全说明：``__call__`` 全程同步、无 ``await``，在单线程 asyncio 事件循环中
    原子执行；AgentPool 亦在其 per-key 创建锁下调用本工厂，故内部 ``OrderedDict``
    无需额外加锁。淘汰时的连接池关闭通过 ``loop.create_task`` 后台进行，不阻塞 ``__call__``。
    """

    # 解析返回 None（用默认 Key/环境变量）时使用的缓存哨兵键
    _DEFAULT_KEY = "__default__"

    def __init__(
        self,
        provider: str,
        *,
        resolve_api_key: ApiKeyResolver,
        model: str = "",
        max_clients: int = 256,
        default_api_key: str | None = None,
        **llm_kwargs: Any,
    ):
        """
        :param provider: 厂商名（传给 ``ModelRegistry.create``），如 "qwen"、"claude"。
        :param resolve_api_key: 必填。``(user_id, session_id) -> api_key | None``。
            返回 None 表示该用户用默认 Key（``default_api_key`` 或环境变量）。
        :param model: 模型名，透传给每个新建的 LLM。
        :param max_clients: 缓存的 LLM 客户端上限；超出按 LRU 淘汰并关闭连接池。须 ≥ 1。
        :param default_api_key: ``resolve_api_key`` 返回 None 时使用的兜底 Key；
            若也为 None，则交由 ``BaseLLM`` 回退到 ``{PROVIDER}_API_KEY`` 环境变量。
        :param llm_kwargs: 其余透传给 ``ModelRegistry.create`` 的构造参数。
        """
        if max_clients < 1:
            raise ValueError("max_clients 必须 ≥ 1")
        self._provider = provider
        self._resolve_api_key = resolve_api_key
        self._model = model
        self._max_clients = max_clients
        self._default_api_key = default_api_key
        self._llm_kwargs = llm_kwargs

        # api_key(或哨兵) -> BaseLLM；OrderedDict 维护 LRU（末尾=最近使用）
        self._clients: "OrderedDict[str, BaseLLM]" = OrderedDict()
        # 持有后台关闭任务的强引用，防止被 GC 提前回收
        self._closing: set[asyncio.Task] = set()
        self._stats = {"created": 0, "reused": 0, "evicted": 0}

    def __call__(self, user_id: str, session_id: str) -> BaseLLM:
        """AgentPool 的 ``llm_factory`` 入口：解析 Key → 复用或新建 LLM。"""
        api_key = self._resolve_api_key(user_id, session_id)
        cache_key = api_key if api_key is not None else self._DEFAULT_KEY

        # 命中：刷新 LRU 后复用
        llm = self._clients.get(cache_key)
        if llm is not None:
            self._clients.move_to_end(cache_key)
            self._stats["reused"] += 1
            return llm

        # 未命中：必要时 LRU 淘汰（并关闭被淘汰客户端的连接池）
        if len(self._clients) >= self._max_clients:
            _, evicted = self._clients.popitem(last=False)
            self._stats["evicted"] += 1
            self._schedule_close(evicted)
            logger.info("KeyedLLMProvider: LRU 淘汰一个 LLM 客户端（max_clients=%d）",
                        self._max_clients)

        # 新建
        create_kwargs = dict(self._llm_kwargs)
        if self._model:
            create_kwargs["model"] = self._model
        effective_key = api_key if api_key is not None else self._default_api_key
        if effective_key is not None:
            create_kwargs["api_key"] = effective_key
        # effective_key 为 None 时不传 api_key，交由 BaseLLM 回退到环境变量

        llm = ModelRegistry.create(self._provider, **create_kwargs)
        self._clients[cache_key] = llm
        self._stats["created"] += 1
        return llm

    def _schedule_close(self, llm: BaseLLM) -> None:
        """后台关闭一个被淘汰 LLM 的连接池（无运行中的事件循环则交给 GC）。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # 无事件循环：放弃主动关闭，依赖进程退出/GC 回收
        task = loop.create_task(self._safe_aclose(llm))
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    @staticmethod
    async def _safe_aclose(llm: BaseLLM) -> None:
        try:
            await llm.aclose()
        except Exception as e:  # 关闭失败不应影响主流程
            logger.warning("KeyedLLMProvider: 关闭被淘汰 LLM 连接池失败：%s", e)

    def get_stats(self) -> dict[str, Any]:
        """缓存统计。``hit_rate = reused / (reused + created)``，越高越说明 Key 复用充分。"""
        total = self._stats["reused"] + self._stats["created"]
        hit_rate = self._stats["reused"] / total if total else 0.0
        return {
            **self._stats,
            "active_clients": len(self._clients),
            "max_clients": self._max_clients,
            "hit_rate": round(hit_rate, 4),
        }

    async def aclose(self) -> None:
        """关闭所有缓存的 LLM 连接池（服务优雅关闭时调用）。"""
        clients = list(self._clients.values())
        self._clients.clear()
        for llm in clients:
            await self._safe_aclose(llm)
        # 等待此前 LRU 淘汰触发的后台关闭任务收尾
        pending = list(self._closing)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
