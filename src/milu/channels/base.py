"""网关核心契约：统一的入站/出站消息模型 + Channel 适配器抽象。

milu Gateway 采用 Ports & Adapters（六边形）架构：

    平台（微信客服 / 飞书 / Telegram …）
        │  各自的回调/长轮询、加解密、消息格式
        ▼
    Channel 适配器  ── 把平台消息规整成 InboundMessage ──▶  dispatch(msg)
        ▲                                                      │
        └────── 用平台 API 发出 reply.text  ◀── OutboundMessage ┘

`dispatch` 由 Gateway 提供（通常就是 AgentRunner）：吃一条规整后的
InboundMessage、跑 milu Agent、回一条 OutboundMessage。**所有平台差异都收在
Channel 里**，dispatch 一侧完全平台无关——新增平台 = 只写一个 Channel。

两种传输形态都收敛到同一条契约：
- **webhook 型**（微信客服 / 飞书）：重写 `register()` 往共享 FastAPI app 挂路由，
  路由里组 InboundMessage → `await dispatch(msg)` → 平台 API 回发。
- **polling 型**（Telegram）：重写 `run()` 跑长轮询循环，同样组 msg → dispatch → 回发。

本模块只依赖核心库（fastapi 类型注解），**不引入任何渠道专属依赖**
（cryptography 等只在具体渠道适配器里按需导入），故可随 `import milu` 安全加载。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from fastapi import FastAPI


# ── 规整后的消息模型 ──────────────────────────────────────────────────
@dataclass
class InboundMessage:
    """从某个平台规整出来的一条入站消息（平台无关）。

    :param channel: 来源渠道名（"wechat_kf" / "feishu" / "telegram"）。
    :param user_id: 平台原生用户标识（微信 external_userid / 飞书 open_id /
        Telegram chat_id 字符串）。与 channel 一起派生 milu 的 (user_id, session_id)。
    :param text: 规整后的纯文本内容。
    :param session_id: 会话标识；None 时由 AgentRunner 退化为 user_id（一人一会话）。
    :param images: 随消息附带的本地图片路径（接视觉用，可空）。
    :param files: 随消息附带的本地文档/数据文件路径（模型用 doc_read/file_read 读取，可空）。
    :param reply_to: **渠道私有**的回发路由上下文（如微信 {"open_kfid":..}、
        飞书 {"receive_id":..}、Telegram {"chat_id":..}）。dispatch 不解读，
        原样回到该渠道的 send 逻辑。
    :param raw: 平台原始报文（个别渠道需要时取用，dispatch 一般不碰）。
    :param msg_id: 平台消息 ID，用于去重（防回调重试/重启重放重复回复）。
    """

    channel: str
    user_id: str
    text: str
    session_id: str | None = None
    images: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    reply_to: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)
    msg_id: str = ""


@dataclass
class OutboundMessage:
    """一条出站回复（当前只做文本，预留富内容扩展）。"""

    text: str


# dispatch 契约：吃一条入站消息，回一条出站消息（None 表示不回复）。
Dispatch = Callable[[InboundMessage], Awaitable["OutboundMessage | None"]]


# ── Channel 适配器抽象 ────────────────────────────────────────────────
class Channel(ABC):
    """一个平台接入适配器。

    子类至少实现 `name`，并按传输形态二选一重写 `register`（webhook）或
    `run`（polling）；两个默认都是 no-op，故 polling 渠道不必实现 register、
    webhook 渠道不必实现 run。`close` 用于回收 http 客户端等资源。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """渠道名（同时作为 InboundMessage.channel 与状态命名空间）。"""

    def register(self, app: FastAPI, dispatch: Dispatch) -> None:
        """webhook 型渠道：往共享 FastAPI app 挂回调路由。默认 no-op。"""

    async def run(self, dispatch: Dispatch) -> None:
        """polling 型渠道：跑长轮询循环（由 Gateway 作为后台任务启动）。

        默认 no-op（webhook 渠道无需轮询）。实现方应当在内部循环里响应取消
        （asyncio.CancelledError）以便优雅停机。
        """

    async def close(self) -> None:
        """释放资源（http 客户端等）。默认 no-op。"""
