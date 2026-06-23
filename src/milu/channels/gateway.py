"""Gateway —— 把若干 Channel 编排成一个可运行的多渠道接入服务。

职责：建 FastAPI app、把各 webhook 渠道的回调路由挂上、为各 polling 渠道起后台
长轮询任务、统一 `/healthz`，并在生命周期内拉起/收尾 dispatch（通常是 AgentRunner
的 pool）与各渠道资源。**所有平台差异都在 Channel 里**，Gateway 只做装配与生命周期。
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

from fastapi import FastAPI

from .base import Channel, Dispatch
from .runner import AgentRunner

logger = logging.getLogger("milu.channels.gateway")

Hook = Callable[[], Awaitable[None]]


class Gateway:
    """多渠道接入网关。

    :param dispatch: 入站消息处理器（吃 InboundMessage 回 OutboundMessage）。
    :param channels: 启用的渠道适配器列表。
    :param on_startup/on_shutdown: 生命周期钩子（如 AgentRunner 的 start/close）。
    :param title: FastAPI 文档标题。
    """

    def __init__(
        self,
        dispatch: Dispatch,
        channels: list[Channel],
        *,
        on_startup: Hook | None = None,
        on_shutdown: Hook | None = None,
        title: str = "milu Gateway",
    ) -> None:
        if not channels:
            raise ValueError("Gateway 至少需要一个 Channel")
        self._dispatch = dispatch
        self._channels = channels
        self._on_startup = on_startup
        self._on_shutdown = on_shutdown
        self._title = title

    @classmethod
    def from_runner(
        cls,
        runner: AgentRunner,
        channels: list[Channel],
        *,
        title: str = "milu Gateway",
    ) -> "Gateway":
        """用 AgentRunner 作 dispatch，并把它的 start/close 接到生命周期上。"""
        return cls(
            dispatch=runner,
            channels=channels,
            on_startup=runner.start,
            on_shutdown=runner.close,
            title=title,
        )

    @property
    def channels(self) -> list[Channel]:
        return self._channels

    # ── 装配 ──────────────────────────────────────────────────────────
    def build_app(self) -> FastAPI:
        """构建 FastAPI app（webhook 路由在此挂载，polling 任务在 lifespan 启动）。"""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            if self._on_startup is not None:
                await self._on_startup()
            # 为每个渠道起 run() 任务：polling 渠道在此长轮询；webhook 渠道的
            # 默认 run() 是 no-op，任务瞬时完成（取消已完成任务是安全 no-op）。
            tasks = [
                asyncio.create_task(ch.run(self._dispatch), name=f"channel:{ch.name}")
                for ch in self._channels
            ]
            try:
                yield
            finally:
                for t in tasks:
                    t.cancel()
                for t in tasks:
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        logger.exception("渠道轮询任务异常退出")
                for ch in self._channels:
                    try:
                        await ch.close()
                    except Exception:
                        logger.exception("渠道 %s 关闭失败", ch.name)
                if self._on_shutdown is not None:
                    try:
                        await self._on_shutdown()
                    except Exception:
                        logger.exception("on_shutdown 钩子失败")

        app = FastAPI(title=self._title, lifespan=lifespan)
        # webhook 路由必须在 app 启动前挂好（lifespan 启动阶段再加路由不生效）
        for ch in self._channels:
            ch.register(app, self._dispatch)

        @app.get("/healthz")
        async def healthz() -> dict:
            return {"ok": True, "channels": [c.name for c in self._channels]}

        return app

    # ── 直接起服务 ────────────────────────────────────────────────────
    def run(self, host: str = "0.0.0.0", port: int = 8800) -> None:
        """前台阻塞启动 uvicorn。"""
        import uvicorn

        uvicorn.run(self.build_app(), host=host, port=port)
