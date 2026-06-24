"""飞书长连接（websocket）渠道 —— 本地开发免隧道。

飞书事件订阅除 webhook 外还支持「长连接」：本进程**主动连飞书**收事件，无需公网
回调 URL / HTTPS / 内网穿透，最适合本地开发调试（和 Telegram 一样省事）。

飞书长连接走私有 protobuf 帧协议，自己手撸不现实，这里复用官方 **lark-oapi** SDK 的
ws 客户端作传输，**仅把收到的事件桥接到统一 dispatch**；发消息仍复用本包的
`FeishuClient`（httpx + tenant_access_token），不依赖 SDK 发送，与 webhook 版口径一致。

依赖：`pip install "milu[feishu-ws]"`（即 lark-oapi）。**仅 FEISHU_MODE=ws 时才需要**，
webhook 版（feishu.py）零额外依赖。

形态：长连接是 **polling 型** 渠道——实现 `run()`（在 daemon 线程跑 SDK 阻塞循环、
主循环侧 await 至取消），不实现 `register()`。`name` 仍为 "feishu"，使同一 open_id
的用户身份/会话在 webhook 与长连接两种模式间连续一致。
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading

from .base import Channel, Dispatch, InboundMessage
from .feishu import FeishuClient, FeishuConfig
from .state import InMemoryStateStore, StateStore

logger = logging.getLogger("milu.channels.feishu_ws")


class FeishuWsChannel(Channel):
    """飞书长连接渠道（polling 型，基于 lark-oapi ws 客户端）。"""

    def __init__(
        self,
        config: FeishuConfig,
        *,
        state: StateStore | None = None,
        client: FeishuClient | None = None,
    ) -> None:
        self._config = config
        self._client = client or FeishuClient(config)
        self._owns_client = client is None
        self._state = state or InMemoryStateStore(dedup_capacity=config.dedup_capacity)
        self._ws = None
        self._thread: threading.Thread | None = None

    @property
    def name(self) -> str:
        return "feishu"   # 与 webhook 版一致：同一 open_id → 同一用户/会话

    async def run(self, dispatch: Dispatch) -> None:
        """连飞书长连接收事件（SDK 阻塞循环放 daemon 线程），主循环 await 至取消。"""
        loop = asyncio.get_running_loop()
        # 友好依赖检查：用 find_spec **只探测、不导入**——关键！lark 的 ws/client.py
        # 在**模块导入时**就执行 `loop = asyncio.get_event_loop()` 存为模块全局并用于
        # run_until_complete；若在主线程（uvicorn 运行中的循环）里 import lark，它会
        # 绑定正在运行的主循环 → start() 时报 "This event loop is already running"。
        # 故主线程绝不 import lark，只探测存在性。
        import importlib.util

        if importlib.util.find_spec("lark_oapi") is None:
            raise RuntimeError(
                "飞书长连接需要 lark-oapi，请安装：pip install \"milu[feishu-ws]\""
            )

        def _thread_main() -> None:
            # 把 lark 的**导入 + 构造 + start()** 全部隔离到本 daemon 线程，并**先**给本
            # 线程新建并设定独立事件循环，**再** import lark——使 client.py 的模块级
            # `loop = get_event_loop()` 绑定到本线程这个（未运行的）新循环。
            import asyncio as _asyncio

            _asyncio.set_event_loop(_asyncio.new_event_loop())

            import lark_oapi as lark

            def on_message(data) -> None:
                # 本回调在 lark 线程触发 → 调度回主事件循环处理（dispatch/发消息都在主循环）
                try:
                    _asyncio.run_coroutine_threadsafe(
                        self._handle(data, dispatch), loop
                    )
                except Exception:
                    logger.exception("飞书长连接事件入队失败")

            builder = lark.EventDispatcherHandler.builder("", "")
            builder.register_p2_im_message_receive_v1(on_message)
            # 给「可能收到但不关心」的事件注册空处理器，消除 lark 的
            # "processor not found" ERROR 噪音（如已读回执、撤回）。用 getattr
            # 守护：老版本 lark 没有对应方法时跳过，不影响主流程。
            for _m in (
                "register_p2_im_message_message_read_v1",
                "register_p2_im_message_recalled_v1",
            ):
                _reg = getattr(builder, _m, None)
                if _reg is not None:
                    _reg(lambda data: None)
            handler = builder.build()
            self._ws = lark.ws.Client(
                self._config.app_id,
                self._config.app_secret,
                event_handler=handler,
                log_level=lark.LogLevel.WARNING,
            )
            self._ws.start()   # 阻塞本线程，用本线程的新循环跑

        # daemon 线程：不阻塞进程退出（lark ws 无干净 stop）
        self._thread = threading.Thread(
            target=_thread_main, name="feishu-ws", daemon=True
        )
        self._thread.start()
        logger.info("飞书长连接已启动（无需公网回调/隧道）")
        try:
            await asyncio.Event().wait()   # 阻塞至本任务被取消（停机）
        except asyncio.CancelledError:
            raise

    async def _handle(self, data, dispatch: Dispatch) -> None:
        """从 SDK 事件对象规整文本消息 → dispatch → 回发（整段不抛异常）。

        data 为 lark_oapi 的 P2ImMessageReceiveV1：以属性访问，故单测可用同形假对象。
        """
        try:
            event = data.event
            message = event.message
            mtype = getattr(message, "message_type", None)
            if mtype not in ("text", "image", "file"):  # 暂处理文本/图片/文件
                return
            event_id = getattr(data.header, "event_id", "") or ""
            if event_id and await self._state.seen(self.name, event_id):
                return
            open_id = getattr(event.sender.sender_id, "open_id", None)
            if not open_id:
                return
            try:
                content = json.loads(message.content or "{}")
            except (ValueError, TypeError):
                content = {}
            message_id = getattr(message, "message_id", "") or ""
            text, images, files = "", [], []
            if mtype == "text":
                text = (content.get("text") or "").strip()
                if not text:
                    return
            elif mtype == "image":  # 复用 webhook 版的资源下载落盘，走视觉
                from .feishu import save_image_resource

                path = await save_image_resource(
                    self._client, self.name, open_id,
                    message_id, content.get("image_key", ""),
                )
                if not path:
                    return
                images = [path]
            else:  # file：复用 webhook 版资源下载，模型用 doc_read/file_read 读取
                from .feishu import save_file_resource

                path = await save_file_resource(
                    self._client, self.name, open_id, message_id,
                    content.get("file_key", ""), content.get("file_name", ""),
                )
                if not path:
                    return
                files = [path]
            inbound = InboundMessage(
                channel=self.name,
                user_id=open_id,
                text=text,
                images=images,
                files=files,
                reply_to={
                    "receive_id": open_id,
                    "chat_id": getattr(message, "chat_id", None),
                },
                msg_id=message_id,
                raw={},
            )
            reply = await dispatch(inbound)
            if reply is None or not reply.text:
                return
            resp = await self._client.send_text(open_id, reply.text)
            if resp.get("code", 0) != 0:
                logger.warning("飞书 send 返回错误：%s", resp)
        except Exception:
            logger.exception("飞书长连接事件处理失败")

    async def close(self) -> None:
        # lark ws 客户端多数版本无干净 stop；daemon 线程随进程退出。尽力尝试。
        try:
            if self._ws is not None and hasattr(self._ws, "stop"):
                self._ws.stop()
        except Exception:
            pass
        if self._owns_client:
            await self._client.aclose()
        await self._state.close()
