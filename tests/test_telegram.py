"""Telegram 渠道单测——用 httpx.MockTransport 假冒 Bot API，无真实网络。

覆盖：
1. run() 长轮询：getUpdates → dispatch → sendMessage；
2. offset 经 StateStore 持久化（重启从断点续拉）；
3. 同一 update_id 去重（只回一次）。
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx

from milu.channels.base import InboundMessage, OutboundMessage
from milu.channels.state import FileStateStore, InMemoryStateStore
from milu.channels.telegram import TelegramChannel, TelegramConfig


async def test_channel_defers_http_initialization():
    http = AsyncMock(spec=httpx.AsyncClient)
    config = TelegramConfig(bot_token="t", poll_timeout=10)
    with patch("milu.channels.telegram.httpx.AsyncClient", return_value=http) as factory:
        channel = TelegramChannel(config)
        factory.assert_not_called()

        first, second = await asyncio.gather(channel._get_http(), channel._get_http())
        assert first is http
        assert second is http
        factory.assert_called_once_with(timeout=25)

        await channel.close()
        http.aclose.assert_awaited_once()


def _update(uid: int, chat_id: int, text: str) -> dict:
    return {
        "update_id": uid,
        "message": {"message_id": uid, "chat": {"id": chat_id}, "text": text},
    }


def _make_http(batches: list[list[dict]], sent: list):
    """返回一个 AsyncClient：getUpdates 依次吐 batches（之后恒空），sendMessage 记账。"""
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "getUpdates" in url:
            i = state["i"]
            state["i"] += 1
            batch = batches[i] if i < len(batches) else []
            return httpx.Response(200, json={"ok": True, "result": batch})
        if "sendMessage" in url:
            body = json.loads(request.content)
            sent.append((body["chat_id"], body["text"]))
            return httpx.Response(200, json={"ok": True, "result": {}})
        return httpx.Response(404, json={"ok": False})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _echo(msg: InboundMessage) -> OutboundMessage:
    return OutboundMessage(text=f"echo:{msg.text}")


async def _run_until(channel: TelegramChannel, dispatch, predicate, timeout=2.0):
    """跑 run() 直到 predicate() 为真，然后取消任务收尾。"""
    task = asyncio.create_task(channel.run(dispatch))
    try:
        for _ in range(int(timeout / 0.01)):
            if predicate():
                break
            await asyncio.sleep(0.01)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def test_telegram_polls_and_replies():
    sent: list = []
    http = _make_http([[_update(1, 100, "你好"), _update(2, 100, "再问")]], sent)
    state = InMemoryStateStore()
    ch = TelegramChannel(
        TelegramConfig(bot_token="t", poll_timeout=0), state=state, http=http
    )
    await _run_until(ch, _echo, lambda: len(sent) >= 2)
    await ch.close()
    assert sent == [(100, "echo:你好"), (100, "echo:再问")]
    # offset 推进到末尾 update_id + 1
    assert await state.get_cursor("telegram", "offset") == "3"


async def test_telegram_offset_persisted_across_restart(tmp_path):
    sent: list = []
    http = _make_http([[_update(5, 7, "hi")]], sent)
    store = FileStateStore(tmp_path, flush_interval=0.02)
    ch = TelegramChannel(
        TelegramConfig(bot_token="t", poll_timeout=0), state=store, http=http
    )
    await _run_until(ch, _echo, lambda: len(sent) >= 1)
    await ch.close()   # 落盘 offset=6

    # 新进程：用同目录的 store，offset 应恢复为 6
    store2 = FileStateStore(tmp_path)
    assert await store2.get_cursor("telegram", "offset") == "6"
    await store2.close()


async def test_telegram_dedup_same_update_id():
    sent: list = []
    # 两批都含 update_id=2（模拟服务端重投）；seen() 应去重
    http = _make_http([[_update(2, 9, "once")], [_update(2, 9, "once")]], sent)
    ch = TelegramChannel(
        TelegramConfig(bot_token="t", poll_timeout=0),
        state=InMemoryStateStore(),
        http=http,
    )
    await _run_until(ch, _echo, lambda: len(sent) >= 1)
    await asyncio.sleep(0.1)   # 给第二批处理机会
    await ch.close()
    assert sent == [(9, "echo:once")]   # 只回一次
