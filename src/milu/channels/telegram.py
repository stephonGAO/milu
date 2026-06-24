"""Telegram 接入 —— Bot API 长轮询渠道适配器（polling 型，无加密）。

链路：
    Telegram 用户发消息
      → 本进程长轮询 getUpdates（服务端挂起最多 poll_timeout 秒）
      → 文本消息规整成 InboundMessage 交 dispatch（AgentRunner）
      → sendMessage 回发

与微信/飞书的 webhook 型不同，Telegram 这里走**主动拉取**：实现 `run()` 跑循环、
不实现 `register()`（无回调路由）。offset 经 StateStore 持久化，重启不重放积压。

注：国内服务器通常连不上 api.telegram.org，可用 TELEGRAM_API_BASE 指向代理或
自建 Bot API Server。

文档：https://core.telegram.org/bots/api#getupdates

配置（环境变量，见 TelegramConfig.from_env）：
    TELEGRAM_BOT_TOKEN（必填）/ TELEGRAM_API_BASE（默认 https://api.telegram.org）
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

import httpx

from milu.llm.base.vision import IMAGE_EXTENSIONS

from .base import Channel, Dispatch, InboundMessage
from .state import InMemoryStateStore, StateStore

logger = logging.getLogger("milu.channels.telegram")


@dataclass
class TelegramConfig:
    bot_token: str
    api_base: str = "https://api.telegram.org"
    poll_timeout: int = 50          # getUpdates 长轮询挂起秒数
    idle_backoff: float = 1.0       # 空轮询后的退避（防 poll_timeout=0 时 CPU 空转）
    dedup_capacity: int = 4096

    @classmethod
    def from_env(cls) -> "TelegramConfig":
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise RuntimeError("缺少环境变量 TELEGRAM_BOT_TOKEN（Telegram 接入必填）")
        base = os.environ.get("TELEGRAM_API_BASE", "https://api.telegram.org").strip()
        return cls(bot_token=token, api_base=(base or "https://api.telegram.org").rstrip("/"))


class TelegramChannel(Channel):
    """Telegram 渠道（polling 型）。"""

    def __init__(
        self,
        config: TelegramConfig,
        *,
        state: StateStore | None = None,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        # 读超时须大于长轮询挂起时间，否则每轮都被客户端超时打断
        self._http = http or httpx.AsyncClient(timeout=config.poll_timeout + 15)
        self._owns_http = http is None
        self._state = state or InMemoryStateStore(dedup_capacity=config.dedup_capacity)

    @property
    def name(self) -> str:
        return "telegram"

    def _url(self, method: str) -> str:
        return f"{self._config.api_base}/bot{self._config.bot_token}/{method}"

    async def run(self, dispatch: Dispatch) -> None:
        """长轮询循环：getUpdates → dispatch → sendMessage。可被取消优雅退出。"""
        offset_s = await self._state.get_cursor(self.name, "offset")
        offset = int(offset_s) if offset_s.isdigit() else 0
        logger.info("Telegram 长轮询启动 offset=%s", offset)
        while True:
            try:
                updates = await self._get_updates(offset)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("getUpdates 失败，3 秒后重试")
                await asyncio.sleep(3)
                continue

            # 空轮询退避：长轮询正常会被服务端挂起 poll_timeout 秒，但 poll_timeout=0
            # 或经 MockTransport 立即返回时，这里保证让出事件循环、不空转占满 CPU。
            if not updates:
                await asyncio.sleep(self._config.idle_backoff)
                continue

            for upd in updates:
                update_id = upd.get("update_id")
                if update_id is not None:
                    offset = update_id + 1  # 下次从此处续拉（同时向服务端确认已处理）
                msg = upd.get("message") or upd.get("edited_message")
                chat_id = (msg.get("chat") or {}).get("id") if msg else None
                if not msg or chat_id is None:
                    await self._state.set_cursor(self.name, "offset", str(offset))
                    continue
                if await self._state.seen(self.name, str(update_id)):
                    await self._state.set_cursor(self.name, "offset", str(offset))
                    continue

                # 文本 / 图片配文（caption）；photo 多尺寸数组取最大那张走视觉，
                # document 走文件（图片扩展名的文件仍按视觉处理）
                text = msg.get("text") or msg.get("caption") or ""
                images: list[str] = []
                files: list[str] = []
                photo = msg.get("photo")
                doc = msg.get("document")
                if photo:
                    file_id = (photo[-1] or {}).get("file_id", "")
                    path = await self._save_image(str(chat_id), file_id)
                    if path:
                        images.append(path)
                elif doc:
                    fid = doc.get("file_id", "")
                    fname = doc.get("file_name", "")
                    if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                        path = await self._save_image(str(chat_id), fid, filename=fname)
                        if path:
                            images.append(path)
                    else:
                        path = await self._save_file(str(chat_id), fid, filename=fname)
                        if path:
                            files.append(path)
                if not text and not images and not files:  # 贴纸/位置等不支持类型跳过
                    await self._state.set_cursor(self.name, "offset", str(offset))
                    continue

                inbound = InboundMessage(
                    channel=self.name,
                    user_id=str(chat_id),
                    text=text,
                    images=images,
                    files=files,
                    reply_to={"chat_id": chat_id},
                    msg_id=str(update_id),
                    raw=upd,
                )
                try:
                    reply = await dispatch(inbound)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("dispatch 生成回复失败 chat_id=%s", chat_id)
                    reply = None
                if reply is not None and reply.text:
                    try:
                        await self._send(chat_id, reply.text)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.exception("sendMessage 回复失败 chat_id=%s", chat_id)
                # 处理完一条即推进持久化游标（重启不重放已处理消息）
                await self._state.set_cursor(self.name, "offset", str(offset))

    async def _get_updates(self, offset: int) -> list[dict]:
        r = await self._http.get(
            self._url("getUpdates"),
            params={"offset": offset, "timeout": self._config.poll_timeout},
        )
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"getUpdates 返回错误：{data}")
        return data.get("result", [])

    async def _send(self, chat_id, text: str) -> dict:
        r = await self._http.post(
            self._url("sendMessage"),
            json={"chat_id": chat_id, "text": text},
        )
        return r.json()

    # ── 媒体下载（图片）────────────────────────────────────────────────
    async def _file_path(self, file_id: str) -> str | None:
        """getFile：把 file_id 换成服务端相对路径 file_path（再凭它下载）。"""
        r = await self._http.get(self._url("getFile"), params={"file_id": file_id})
        data = r.json()
        if not data.get("ok"):
            return None
        return (data.get("result") or {}).get("file_path")

    async def _download(self, file_path: str) -> bytes:
        """下载文件：注意走 /file/bot<token>/<file_path>，与 API 方法路径不同。"""
        url = f"{self._config.api_base}/file/bot{self._config.bot_token}/{file_path}"
        r = await self._http.get(url)
        return r.content

    async def _save_image(
        self, user_id: str, file_id: str, *, filename: str = ""
    ) -> str | None:
        """下载图片并落盘，返回绝对路径（失败/超限返回 None，不抛异常）。"""
        if not file_id:
            return None
        try:
            fp = await self._file_path(file_id)
            if not fp:
                return None
            data = await self._download(fp)
        except Exception:
            logger.exception("Telegram 图片下载失败 file_id=%s", file_id)
            return None
        from .media import save_image

        # 优先用消息里的原文件名（带正确扩展名），否则退回服务端 file_path
        return save_image(self.name, user_id, data, filename=filename or fp)

    async def _save_file(
        self, user_id: str, file_id: str, *, filename: str = ""
    ) -> str | None:
        """下载文件并落盘，返回绝对路径（失败/超限返回 None，不抛异常）。"""
        if not file_id:
            return None
        try:
            fp = await self._file_path(file_id)
            if not fp:
                return None
            data = await self._download(fp)
        except Exception:
            logger.exception("Telegram 文件下载失败 file_id=%s", file_id)
            return None
        from .media import save_file

        return save_file(self.name, user_id, data, filename=filename or fp)

    async def close(self) -> None:
        if self._owns_http:
            await self._http.aclose()
        await self._state.close()
