"""AgentRunner —— 把入站消息跑过 milu Agent 得到回复（接 milu 的唯一处）。

这是「平台无关核心」：每个渠道把消息规整成 InboundMessage 后交给它，它持有一个
`AgentPool`，按 (channel, user) 取到该用户独立的 Agent 实例跑一轮，取 `AgentDone`
的最终文本回出去。替代了此前手写 run.py 里的 `on_text` 接线——任何渠道复用同一个
AgentRunner，自动获得 per-用户隔离、milu 全配默认、以及（经 agent_kwargs 透传的）
strict 沙箱隔离与视觉输入能力。
"""
from __future__ import annotations

import logging
from typing import Callable

from milu.agent.events import AgentDone
from milu.serving.pool import AgentPool, AgentPoolConfig

from .base import InboundMessage, OutboundMessage

logger = logging.getLogger("milu.channels.runner")

# Agent 没产出有效回复时的兜底语（空回复会被渠道丢弃，体验不佳）
DEFAULT_FALLBACK = "（抱歉，我暂时没想到怎么回答，换个说法再试试？）"

# 自定义 (channel,user)->（user_id, session_id) 的映射钩子
SessionOf = Callable[[InboundMessage], "tuple[str, str]"]


class AgentRunner:
    """持有 AgentPool，把 InboundMessage 跑过 Agent 得到 OutboundMessage。

    既是 Gateway 的 dispatch 实现（`__call__`），也负责 pool 生命周期
    （`start`/`close`，由 Gateway 在 lifespan 中调用）。
    """

    def __init__(
        self,
        pool: AgentPool,
        *,
        fallback_text: str = DEFAULT_FALLBACK,
        session_of: SessionOf | None = None,
    ) -> None:
        self._pool = pool
        self._fallback = fallback_text
        self._session_of = session_of

    # ── 便利构造：单个共享 LLM 一行起池 ──────────────────────────────
    @classmethod
    def from_llm(
        cls,
        llm,
        *,
        agent_kwargs: dict | None = None,
        pool_config: AgentPoolConfig | None = None,
        fallback_text: str = DEFAULT_FALLBACK,
        session_of: SessionOf | None = None,
    ) -> "AgentRunner":
        """用单个共享 LLM 实例创建（AsyncOpenAI 协程安全，多用户共享安全）。

        :param agent_kwargs: 透传给每个 per-用户 Agent 的构造参数。strict 隔离
            （sandbox="docker" / workspace_jail=True）即经此显式传入——AgentPool
            默认工厂不会自动套用 config 的 multiuser=strict，必须在这里显式给。
        :param pool_config: 池配置（并发/容量/TTL）；None 时由 from_llm 按分层配置加载。
        """
        pool = AgentPool.from_llm(
            llm, config=pool_config, agent_kwargs=agent_kwargs or None
        )
        return cls(pool, fallback_text=fallback_text, session_of=session_of)

    @property
    def pool(self) -> AgentPool:
        return self._pool

    async def start(self) -> None:
        await self._pool.start()

    async def close(self) -> None:
        await self._pool.stop()

    # ── dispatch 实现 ─────────────────────────────────────────────────
    def _keys(self, inbound: InboundMessage) -> tuple[str, str]:
        """派生 (user_id, session_id)。默认按 渠道:用户 隔离、一人一会话。"""
        if self._session_of is not None:
            return self._session_of(inbound)
        uid = f"{inbound.channel}:{inbound.user_id}"
        sid = uid if inbound.session_id is None else f"{inbound.channel}:{inbound.session_id}"
        return uid, sid

    async def __call__(self, inbound: InboundMessage) -> OutboundMessage | None:
        """跑一轮 Agent，返回回复（异常/空回复回退兜底语，绝不向渠道抛异常）。"""
        user_id, session_id = self._keys(inbound)
        reply = ""
        try:
            async with self._pool.acquire(user_id=user_id, session_id=session_id) as h:
                async for evt in h.agent.run(
                    inbound.text, images=inbound.images or None
                ):
                    if isinstance(evt, AgentDone):
                        reply = evt.final_text
        except Exception:
            logger.exception("Agent 运行失败 channel=%s user=%s", inbound.channel, user_id)
            return OutboundMessage(text=self._fallback)
        return OutboundMessage(text=reply or self._fallback)
