"""AgentPool — 多用户场景下的 Agent 资源池。

核心功能：
- 按 (user_id, session_id) 缓存 Agent 实例（同一用户多请求复用同一 Agent）
- LRU + idle TTL 淘汰：长时间未用自动清理，防止 OOM
- 全局并发上限：asyncio.Semaphore 限流，超出时阻塞
- async with 语法：自动管理 Agent 生命周期（计数空闲/激活）
- 不侵入 Agent 类本身，纯外层包装
"""


# ─────────────────────────────────────────────────────────────────────────────
# 多用户并发设计说明（per-user Agent 隔离方案）
# ─────────────────────────────────────────────────────────────────────────────
#
# 【问题背景】
# 原设计（multi_turn_chat.py 风格）假设服务器上只创建单个 Agent，多个用户共享
# 该实例并通过并发调用 agent.run(user_input) 来实现"多用户"。
# 但 Agent 内部存在若干"实例级状态"——它们在并发场景下会被多用户共享并相互污染。
#
# 【共享状态清单】（在并发 run() 下会出问题的 Agent 内部字段）
#
# 1. self.history : ConversationHistory
#    累积式消息列表（用户消息 + assistant 回复 + 工具结果）。多用户共享同一份
#    history → 用户 A 的消息被用户 B 看到，P0 致命。
#
# 2. self.session : Optional[Session]
#    会话持久化（JSONL 追加写）。多用户写同一份 Session 文件 → A 的回复被记到
#    B 的会话里，且 JSONL 行交叉，P0 致命。
#
# 3. self._work_started / self._plan_created : bool
#    run() 内的流程控制标记（控制"是否已注入工作清单"等）。两个并发 run 都会
#    看到 False → 重复进入"创建清单"分支，污染 LLM 上下文。
#

#
# 5. self._conn_retry_count
#    流式响应断连的重试计数。共享字段导致 A 的断连重试消耗 B 的重试额度。
#
# 6. self._mcp_manager : Optional[MCPManager]
#    MCP 子进程句柄（stdio pipe / HTTP 长连接）。多用户共用一组 MCP 进程
#    → MCP server 端没有 user_id 概念，工具调用无法按用户隔离（**真正的资源
#    大头**：每个 Agent 启 3-5 个 MCP server 占 15-50 MB）。
#
# 7. self.tools : ToolRegistry
#    工具 active/dormant pool。激活/休眠操作会修改 pool，并发下有竞态。
#
# 8. tools/todo.py 中的 TodoManager 模块级单例
#    全进程共用一份 todo 列表。A 创建的 todo 会显示在 B 的上下文里。
#
# 9. subagent.py 中的 _last_events 闭包变量
#    每次 _create_single_subagent_tool() 创建一个 ToolWrapper 闭包，绑一份
#    _last_events。和主 Agent 一起创建时绑定 → 多用户共享同一份 events 列表
#    → 并发调用时事件错乱/丢失。
#
# 【为什么不能"把状态塞进 run() 参数"？】
# 一些状态天然是"实例级"的，无法通过 kwargs 解决：
#   - history 是按时间累积的列表（不是一次性参数）
#   - Session 是持久化文件句柄（绑实例）
#   - subagent 闭包是 Agent 创建时绑定的（绑实例）
#   - _work_started 是 run() 内的流程标记（绑实例的 run 上下文）
# 因此 **唯一可行方案是 per-user Agent**：每个 (user_id, session_id) 一份独立
# Agent 实例。
#
# 【per-user Agent 的资源成本】
# 单 Agent 内存占用：
#   - 无 MCP 场景：~1 MB（Python 对象 + history + tools schema）
#   - 有 MCP 场景：~15-50 MB（**MCP 子进程是真正的瓶颈**）
#
# 16 GB 机器（Python 运行时占 4 GB，留 12 GB 给 Agent）：
#   - 无 MCP：~12,000 并发用户 ✅
#   - 轻度 MCP（3 server/Agent）：~800 用户 ⚠️
#   - 重度 MCP（5+ server + HTTP 长连接）：~240 用户 ❌ 需横向扩展
#
# 关键洞察：**MCP 才是瓶颈**，不是 Agent 本身。
#   - 如果 MCP 可按用户隔离 → 共享 MCPManager 进程，把 user_id 透传到 MCP server
#   - 否则 → 横向扩展（多机 + 负载均衡）
#
# 【per-user Agent 的 per-user 配置】
# AgentConfig（mode / max_turns / session_enabled 等）也是实例级共享字段。
# AgentPool 通过 agent_factory 回调支持按 user_id 注入不同配置：
#
#   def my_factory(user_id, session_id, llm):
#       if user_id.startswith("vip_"):
#           cfg = AgentConfig(mode="deep", max_turns=100)
#       else:
#           cfg = AgentConfig(mode="fast", max_turns=20)
#       return Agent(llm=llm, config=cfg)
#
#   pool = AgentPool(llm_factory=..., agent_factory=my_factory)
#
# 【AgentPool 的 4 个硬不变量】（任何代码改动都必须维持）
#   1. 每个 (user_id, session_id) 组合最多 1 个 Agent 实例
#   2. 实例数 ≤ config.max_agents；超出触发 LRU 淘汰
#   3. 全局 run() 并发 ≤ config.max_concurrent_runs（Semaphore 限流）
#   4. 空闲 ≥ config.idle_ttl_seconds 的实例被后台任务清理
#
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

from agent_framework.agent.agent import Agent
from agent_framework.agent.config import AgentConfig
from agent_framework.llm.providers.base import BaseLLM

logger = logging.getLogger(__name__)


# ── 配置 ──────────────────────────────────────────────────


@dataclass
class AgentPoolConfig:
    """AgentPool 配置。

    :param max_agents: 池容量上限。超出后 LRU 淘汰最久未用的实例。
    :param idle_ttl_seconds: Agent 空闲 TTL（秒）。超时未用自动关闭。
    :param max_concurrent_runs: 全局并发 run() 上限。超出时 acquire() 阻塞。
    :param sweep_interval_seconds: 后台清理任务巡检间隔。
    """
    max_agents: int = 200
    idle_ttl_seconds: float = 300.0
    max_concurrent_runs: int = 50
    sweep_interval_seconds: float = 60.0


# ── LLM 工厂类型 ──

# 用户提供的工厂：每次新 Agent 时调用得到 LLM（允许每用户独立 key/模型）
LLMFactory = Callable[[str, str], BaseLLM]  # (user_id, session_id) -> BaseLLM
AgentFactory = Callable[[str, str, BaseLLM], Agent]  # (user_id, session_id, llm) -> Agent


# ── 内部记录 ──


@dataclass
class _PoolEntry:
    """池中每个 Agent 实例的元数据。"""
    agent: Agent
    user_id: str
    session_id: str
    last_used: float = field(default_factory=time.monotonic)
    active_count: int = 0     # 当前持有/排队该 entry 的协程数（>0 时不被淘汰）
    # 同 (user_id, session_id) 的 run() 串行锁：
    # 同一会话的并发请求会复用同一个 Agent 实例，若并发执行 run() 会污染
    # 共享的 history / session / 流程标记。持有此锁可保证同会话请求顺序执行。
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def key(self) -> tuple[str, str]:
        return (self.user_id, self.session_id)


# ── PooledAgent 句柄 ──


@dataclass
class PooledAgent:
    """从 pool.get() 返回的句柄。

    用法：
        async with pool.acquire(user_id, session_id) as handle:
            async for evt in handle.agent.run(input_):
                ...
    """
    entry: _PoolEntry
    # 构造时由 acquire() 注入 pool 引用，repr/compare 时跳过
    _pool: Any = field(default=None, repr=False, compare=False)

    @property
    def agent(self) -> Agent:
        return self.entry.agent

    @property
    def user_id(self) -> str:
        return self.entry.user_id

    @property
    def session_id(self) -> str:
        return self.entry.session_id

    async def __aenter__(self) -> "PooledAgent":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._pool is not None:
            await self._pool._release(self)


# ── AgentPool 主体 ──


class AgentPool:
    """多用户 Agent 资源池。

    关键不变量：
    - 每个 (user_id, session_id) 组合最多一个 Agent 实例
    - 实例数 ≤ config.max_agents；超出触发 LRU 淘汰
    - 全局 run() 并发 ≤ config.max_concurrent_runs（通过 Semaphore 限流）
    - 空闲 ≥ config.idle_ttl_seconds 的实例被后台任务清理
    """

    def __init__(
        self,
        llm_factory: LLMFactory,
        agent_factory: Optional[AgentFactory] = None,
        config: Optional[AgentPoolConfig] = None,
        agent_config: Optional[AgentConfig] = None,
    ):
        """
        :param llm_factory: 必填。接收 (user_id, session_id) 返回 BaseLLM 实例。
            共享同一 LLM 实例是安全的（AsyncOpenAI 协程安全），
            所以工厂通常直接返回预先创建的实例。
        :param agent_factory: 可选。接收 (user_id, session_id, llm) 返回 Agent。
            不传则用默认工厂（注入 llm + agent_config）。
        :param config: 池配置。
        :param agent_config: 给默认 agent_factory 使用的 AgentConfig。
        """
        self._llm_factory = llm_factory
        self._agent_factory = agent_factory or self._default_agent_factory
        self._pool_config = config or AgentPoolConfig()
        self._agent_config = agent_config or AgentConfig()

        # 状态 — 单一 OrderedDict 兼顾"查找"与"LRU 顺序"
        # move_to_end(key) = 标记最近使用；popitem(last=False) = 弹最久未用
        self._entries: "OrderedDict[tuple[str, str], _PoolEntry]" = OrderedDict()
        self._semaphore = asyncio.Semaphore(self._pool_config.max_concurrent_runs)
        self._entry_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

        # 后台清理
        self._sweep_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # 指标（累计计数器，进程级别单调递增）
        # - created:          累计新建 Agent 数（缓存未命中触发 _get_or_create 创建）
        # - reused:           累计缓存命中复用次数（同一 user/session 再次请求时 +1）
        # - evicted_lru:      累计 LRU 淘汰次数（池满 len(_entries) >= max_agents 时触发）
        # - evicted_idle:     累计空闲 TTL 清理次数（后台 sweep 任务每 60s 巡检一次）
        # - concurrent_waits: 累计进入 semaphore 的请求数（每个 acquire() 调用 +1）
        #
        # 关键派生指标：hit_rate = reused / (reused + created)
        #   命中率越高越好，反映"per-user Agent 复用"的有效性
        self._stats = {
            "created": 0,
            "evicted_lru": 0,
            "evicted_idle": 0,
            "reused": 0,
            "concurrent_waits": 0,
        }

    # ── 公开 API ──────────────────────────────────────────

    async def start(self) -> None:
        """启动后台清理任务。"""
        if self._sweep_task is None:
            self._stop_event.clear()
            self._sweep_task = asyncio.create_task(self._sweep_loop())
            logger.info("AgentPool 已启动: %s", self._pool_config)

    async def stop(self) -> None:
        """停止后台任务并清理所有 Agent。"""
        if self._sweep_task:
            self._stop_event.set()
            try:
                await asyncio.wait_for(self._sweep_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._sweep_task.cancel()
            self._sweep_task = None

        await self._close_all()
        logger.info("AgentPool 已停止")

    def get_stats(self) -> dict[str, Any]:
        """获取池统计信息（监控用）。"""
        total = self._stats["reused"] + self._stats["created"]
        hit_rate = self._stats["reused"] / total if total > 0 else 0.0
        return {
            **self._stats,
            "active_entries": len(self._entries),
            "max_agents": self._pool_config.max_agents,
            "max_concurrent_runs": self._pool_config.max_concurrent_runs,
            "hit_rate": round(hit_rate, 4),
        }

    @asynccontextmanager
    async def acquire(
        self,
        user_id: str,
        session_id: str,
    ) -> AsyncIterator[PooledAgent]:
        """获取（或创建）一个用户的 Agent 句柄。

        行为：
        1. 查找缓存；命中则复用，未命中则创建
        2. 池满时按 LRU 淘汰最久未用的实例
        3. 对同一 (user_id, session_id) 持 entry 锁串行执行（避免并发 run()
           污染共享的 history / session）。锁在 Semaphore 之外获取，因此排队
           等待同会话锁的请求不会占用全局并发名额
        4. 通过 Semaphore 限制全局并发
        5. 退出 with 时释放并发许可、刷新 last_used、释放句柄

        例：
            async with pool.acquire(uid, sid) as h:
                async for evt in h.agent.run(text):
                    ...

        注意：同一 (user_id, session_id) 的并发 acquire() 会串行执行而非并行，
        这是有意为之——同会话共享一个 Agent，并发 run() 会相互污染。
        """
        # 1. 先取得 entry（创建期已由 _get_or_create 内的 per-key 锁串行化）
        entry = await self._get_or_create(user_id, session_id)

        # 2. active_count 立即 +1，保护 entry 在「排队等锁」期间不被淘汰。
        #    紧跟 _get_or_create 之后、无 await 间隔，故不会与淘汰发生竞态。
        entry.active_count += 1
        handle = PooledAgent(entry=entry, _pool=self)
        try:
            # 3. 同会话串行锁：放在 Semaphore 之前，排队请求不占全局并发名额
            async with entry.lock:
                # 4. 全局并发限流
                await self._semaphore.acquire()
                self._stats["concurrent_waits"] += 1
                try:
                    entry.last_used = time.monotonic()
                    yield handle
                finally:
                    self._semaphore.release()
        finally:
            # 释放：active_count -1、刷新 last_used、更新 LRU 顺序
            await self._release(handle)

    async def _release(self, handle: PooledAgent) -> None:
        """PooledAgent 退出 with 时调用。"""
        entry = handle.entry
        entry.active_count = max(0, entry.active_count - 1)
        entry.last_used = time.monotonic()
        self._entries.move_to_end(entry.key)

    async def get_or_create_agent(
        self,
        user_id: str,
        session_id: str,
    ) -> Agent:
        """直接拿一个 Agent（不绑定 semaphore）。

        适用场景：后台任务（不占并发许可），如调度器触发。
        调用方需自行管理生命周期。
        """
        entry = await self._get_or_create(user_id, session_id)
        return entry.agent

    # ── 内部 ──────────────────────────────────────────

    def _default_agent_factory(
        self, user_id: str, session_id: str, llm: BaseLLM
    ) -> Agent:
        """默认 Agent 工厂：注入 LLM + AgentConfig 模板。

        这里传入的是池级共享的 self._agent_config 模板，但 Agent.__init__ 会对其
        做浅拷贝（dataclasses.replace），因此每个 Agent 持有独立配置，
        set_mode() 等运行时修改不会跨用户串扰。
        """
        return Agent(
            llm=llm,
            config=self._agent_config,
        )

    async def _get_or_create(
        self, user_id: str, session_id: str
    ) -> _PoolEntry:
        key = (user_id, session_id)

        # 1. 缓存命中（无锁快路径）
        entry = self._entries.get(key)
        if entry is not None:
            self._stats["reused"] += 1
            self._entries.move_to_end(key)
            return entry

        # 2. 缓存未命中：per-key 锁串行化（避免多协程并发创建同 key）
        async with self._lock_for(key):
            # 双重检查：拿锁前别的协程可能已创建
            entry = self._entries.get(key)
            if entry is not None:
                return entry

            # 3. 池满 → LRU 淘汰
            if len(self._entries) >= self._pool_config.max_agents:
                await self._evict_lru()

            # 4. 创建
            llm = self._llm_factory(user_id, session_id)
            agent = self._agent_factory(user_id, session_id, llm)
            entry = _PoolEntry(agent=agent, user_id=user_id, session_id=session_id)
            self._entries[key] = entry
            self._stats["created"] += 1
            logger.info("AgentPool: 创建 Agent (user=%s, session=%s, total=%d)",
                        user_id, session_id, len(self._entries))
            return entry

    def _lock_for(self, key: tuple[str, str]) -> asyncio.Lock:
        """获取/创建一个 per-key 锁。"""
        lock = self._entry_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._entry_locks[key] = lock
        return lock

    def _touch_lru(self, key: tuple[str, str]) -> None:
        """刷新 LRU 顺序：移到末尾（最近使用）。"""
        self._entries.move_to_end(key)

    async def _evict_lru(self) -> None:
        """淘汰最久未用且当前未被占用的 Agent。"""
        async with self._global_lock:
            # OrderedDict 默认迭代顺序是插入顺序（最久未用在前面）
            for key, entry in self._entries.items():
                if entry.active_count == 0:
                    await self._close_entry(entry, key, reason="lru")
                    self._stats["evicted_lru"] += 1
                    return
            # 所有实例都在使用中：拒绝淘汰，强制后续 acquire 阻塞
            logger.warning("AgentPool: 所有 %d 个 Agent 都在使用中，无法淘汰",
                           len(self._entries))

    async def _sweep_loop(self) -> None:
        """后台定期清理空闲超时的 Agent。"""
        try:
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self._pool_config.sweep_interval_seconds,
                    )
                    break  # stop_event 被设置
                except asyncio.TimeoutError:
                    pass

                await self._sweep_idle()
        except asyncio.CancelledError:
            pass

    async def _sweep_idle(self) -> None:
        """清理空闲超时的 Agent。"""
        now = time.monotonic()
        ttl = self._pool_config.idle_ttl_seconds
        async with self._global_lock:
            # 拷贝 items 避免遍历时改 dict
            for key, entry in list(self._entries.items()):
                if entry.active_count > 0:
                    continue
                if (now - entry.last_used) > ttl:
                    await self._close_entry(entry, key, reason="idle")
                    self._stats["evicted_idle"] += 1
            logger.debug("AgentPool sweep done: %d entries remain", len(self._entries))

    async def _close_entry(
        self, entry: _PoolEntry, key: tuple[str, str], reason: str
    ) -> None:
        """关闭一个 Agent 实例（断 MCP 等后台资源）。

        由 _global_lock 保护，重复调用是安全的（OrderedDict.pop 已保证幂等）。
        """
        try:
            await entry.agent.disconnect_mcp()
        except Exception as e:
            logger.warning("AgentPool: 关闭 Agent 时断开 MCP 失败 (user=%s): %s",
                           entry.user_id, e)

        # pop 返回 None 说明已被其他路径关闭（无副作用）
        if self._entries.pop(key, None) is not None:
            logger.info("AgentPool: 淘汰 Agent (user=%s, session=%s, reason=%s)",
                        entry.user_id, entry.session_id, reason)

    async def _close_all(self) -> None:
        """关闭所有 Agent。"""
        async with self._global_lock:
            entries = list(self._entries.items())
            self._entries.clear()
            for key, entry in entries:
                try:
                    await entry.agent.disconnect_mcp()
                except Exception:
                    pass
            logger.info("AgentPool: 关闭全部 %d 个 Agent", len(entries))
