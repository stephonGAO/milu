"""网关媒体（图片）接入单测——下载落盘 + 各渠道规整成 InboundMessage.images。

覆盖：
1. media.save_image：落 per-用户工作区 `_incoming`、超限拒绝、扩展名推断；
2. 微信客服 image 消息 → download_media → dispatch 带 images；
3. 飞书 image 事件 → download_resource → dispatch 带 images；
4. Telegram photo → getFile + 文件下载 → dispatch 带 images；
5. AgentRunner：只发图片无配文 → 补默认指令并把图片转给 agent.run。

不触真实网络/凭证；MILU_HOME/MILU_PROJECT_DIR 隔离到 tmp，下载落 tmp 工作区。
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from milu.channels import media
from milu.channels.base import InboundMessage, OutboundMessage
from milu.channels.feishu import FeishuChannel, FeishuClient, FeishuConfig
from milu.channels.runner import DEFAULT_IMAGE_PROMPT, AgentRunner
from milu.channels.state import InMemoryStateStore
from milu.channels.telegram import TelegramChannel, TelegramConfig
from milu.channels.wechat_kf import WeChatKfChannel, WeChatKfClient, WeChatKfConfig

# 最小合法 PNG（够过落盘，不校验像素）
_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
    "de0000000c49444154789c63f8cfc0f01f0005000150d2a0a30000000049454e44ae426082"
)


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """把数据/工作区/项目配置目录都隔离到 tmp，避免污染真实 ~/.milu。"""
    monkeypatch.setenv("MILU_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MILU_PROJECT_DIR", str(tmp_path / "proj"))
    monkeypatch.delenv("MILU_WORKSPACE", raising=False)
    (tmp_path / "proj").mkdir(parents=True, exist_ok=True)
    return tmp_path


# ── 1. media.save_image ───────────────────────────────────────────────
def test_save_image_lands_in_workspace_incoming():
    p = media.save_image("telegram", "u1", _PNG, content_type="image/png")
    path = Path(p)
    assert path.is_file()
    assert path.read_bytes() == _PNG
    assert path.suffix == ".png"
    assert path.parent.name == "_incoming"
    # 落在该用户工作区（channel:user 派生）下
    assert "telegram_u1" in str(path).replace("\\", "/")


def test_save_image_infers_ext_from_content_type():
    p = media.save_image("c", "u", _PNG, content_type="image/jpeg", media_id="m1")
    assert Path(p).suffix == ".jpg"


def test_save_image_keeps_valid_filename_ext():
    p = media.save_image("c", "u", _PNG, filename="photos/file_7.webp")
    assert Path(p).suffix == ".webp"


def test_save_image_rejects_oversize():
    big = b"x" * (media.MAX_IMAGE_BYTES + 1)
    assert media.save_image("c", "u", big, content_type="image/png") is None


def test_save_image_empty_returns_none():
    assert media.save_image("c", "u", b"") is None


# ── 2. 微信客服 image ─────────────────────────────────────────────────
def _kf_config() -> WeChatKfConfig:
    return WeChatKfConfig(
        corp_id="ww_test_corp", secret="s", token="test_token",
        encoding_aes_key="abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG",
        callback_path="/wechat/kf/callback",
    )


def _kf_image_msg(msgid: str, user: str, media_id: str) -> dict:
    return {
        "msgid": msgid, "msgtype": "image", "external_userid": user,
        "image": {"media_id": media_id}, "send_time": int(time.time()),
    }


async def test_wechat_image_dispatched_with_local_path():
    cfg = _kf_config()
    client = WeChatKfClient(cfg)
    client.fetch_messages = AsyncMock(
        return_value=([_kf_image_msg("m1", "userA", "MEDIA123")], "cur-1")
    )
    client.download_media = AsyncMock(return_value=(_PNG, "image/png"))
    client.send_text = AsyncMock(return_value={"errcode": 0})

    seen: dict = {}
    done = asyncio.Event()

    async def dispatch(msg: InboundMessage) -> OutboundMessage:
        seen["images"] = list(msg.images)
        seen["text"] = msg.text
        done.set()
        return OutboundMessage(text="看到了")

    ch = WeChatKfChannel(cfg, client=client, state=InMemoryStateStore())
    await ch._pull_and_dispatch("T", "wk1", dispatch)
    await asyncio.wait_for(done.wait(), timeout=2)

    client.download_media.assert_awaited_once_with("MEDIA123")
    assert len(seen["images"]) == 1
    assert Path(seen["images"][0]).is_file()
    assert seen["text"] == ""   # 纯图片无配文


async def test_wechat_image_download_failure_skips():
    cfg = _kf_config()
    client = WeChatKfClient(cfg)
    client.fetch_messages = AsyncMock(
        return_value=([_kf_image_msg("m1", "userA", "BAD")], "cur")
    )
    client.download_media = AsyncMock(side_effect=RuntimeError("media expired"))
    sent = []
    client.send_text = AsyncMock(side_effect=lambda *a: sent.append(a))

    async def dispatch(msg):
        raise AssertionError("下载失败不应 dispatch")

    ch = WeChatKfChannel(cfg, client=client, state=InMemoryStateStore())
    await ch._pull_and_dispatch("T", "wk1", dispatch)  # 不抛
    assert sent == []


# ── 3. 飞书 image ─────────────────────────────────────────────────────
def _fs_config() -> FeishuConfig:
    return FeishuConfig(app_id="cli_x", app_secret="sec", verify_token="vtok",
                        encrypt_key="", callback_path="/feishu/event")


def _fs_image_event(event_id: str, open_id: str, image_key: str) -> dict:
    return {
        "schema": "2.0",
        "header": {"event_id": event_id, "event_type": "im.message.receive_v1",
                   "token": "vtok"},
        "event": {
            "sender": {"sender_id": {"open_id": open_id}},
            "message": {"message_id": "om_" + event_id, "chat_id": "oc_1",
                        "message_type": "image",
                        "content": json.dumps({"image_key": image_key})},
        },
    }


async def test_feishu_image_dispatched_with_local_path():
    cfg = _fs_config()
    client = FeishuClient(cfg)
    client.download_resource = AsyncMock(return_value=(_PNG, "image/png"))
    sent, done = [], asyncio.Event()

    async def _send(open_id, text):
        sent.append((open_id, text))
        done.set()
        return {"code": 0}

    client.send_text = _send  # type: ignore[assignment]

    seen: dict = {}

    async def dispatch(msg: InboundMessage) -> OutboundMessage:
        seen["images"] = list(msg.images)
        return OutboundMessage(text="收到图片")

    ch = FeishuChannel(cfg, client=client, state=InMemoryStateStore())
    app = FastAPI()
    ch.register(app, dispatch)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.post("/feishu/event", json=_fs_image_event("e1", "ou_a", "img_key_1"))
        assert r.status_code == 200
    await asyncio.wait_for(done.wait(), timeout=2)

    client.download_resource.assert_awaited_once_with("om_e1", "img_key_1", "image")
    assert len(seen["images"]) == 1 and Path(seen["images"][0]).is_file()
    assert sent == [("ou_a", "收到图片")]


# ── 4. Telegram photo ─────────────────────────────────────────────────
def _photo_update(uid: int, chat_id: int, file_id: str, caption: str | None = None) -> dict:
    msg = {"message_id": uid, "chat": {"id": chat_id},
           "photo": [{"file_id": "thumb"}, {"file_id": file_id}]}
    if caption is not None:
        msg["caption"] = caption
    return {"update_id": uid, "message": msg}


def _make_tg_http(batches: list[list[dict]], sent: list):
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "getUpdates" in url:
            i = state["i"]; state["i"] += 1
            batch = batches[i] if i < len(batches) else []
            return httpx.Response(200, json={"ok": True, "result": batch})
        if "getFile" in url:
            return httpx.Response(200, json={
                "ok": True, "result": {"file_path": "photos/file_1.jpg"}})
        if "/file/bot" in url:                       # 实际文件下载
            return httpx.Response(200, content=_PNG)
        if "sendMessage" in url:
            body = json.loads(request.content)
            sent.append((body["chat_id"], body["text"]))
            return httpx.Response(200, json={"ok": True, "result": {}})
        return httpx.Response(404, json={"ok": False})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _run_until(channel, dispatch, predicate, timeout=2.0):
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


async def test_telegram_photo_downloaded_and_dispatched():
    sent: list = []
    http = _make_tg_http([[_photo_update(1, 100, "FILEID", caption="看这个")]], sent)
    seen: dict = {}

    async def dispatch(msg: InboundMessage) -> OutboundMessage:
        seen["images"] = list(msg.images)
        seen["text"] = msg.text
        return OutboundMessage(text=f"图片+{msg.text}")

    ch = TelegramChannel(
        TelegramConfig(bot_token="t", poll_timeout=0),
        state=InMemoryStateStore(), http=http,
    )
    await _run_until(ch, dispatch, lambda: len(sent) >= 1)
    await ch.close()

    assert len(seen["images"]) == 1 and Path(seen["images"][0]).is_file()
    assert Path(seen["images"][0]).suffix == ".jpg"
    assert seen["text"] == "看这个"           # caption 作为文本
    assert sent == [(100, "图片+看这个")]


# ── 5. AgentRunner：只发图片无配文 ────────────────────────────────────
class _RecAgent:
    def __init__(self):
        self.text = None
        self.images = None

    async def run(self, text, images=None):
        self.text = text
        self.images = images
        from milu.agent.events import AgentDone
        from milu.llm.base.response import TokenUsage
        yield AgentDone(final_text="ok", total_usage=TokenUsage(), turn_count=1)


class _Handle:
    def __init__(self, agent):
        self.agent = agent


class _Acquire:
    def __init__(self, h):
        self._h = h

    async def __aenter__(self):
        return self._h

    async def __aexit__(self, *exc):
        return False


class _Pool:
    def __init__(self, agent):
        self._a = agent

    async def start(self):
        ...

    async def stop(self):
        ...

    def acquire(self, user_id, session_id):
        return _Acquire(_Handle(self._a))


async def test_runner_image_only_uses_default_prompt():
    agent = _RecAgent()
    runner = AgentRunner(_Pool(agent))
    await runner(InboundMessage(channel="c", user_id="u", text="", images=["/a.png"]))
    assert agent.images == ["/a.png"]
    assert agent.text == DEFAULT_IMAGE_PROMPT


async def test_runner_image_with_caption_keeps_text():
    agent = _RecAgent()
    runner = AgentRunner(_Pool(agent))
    await runner(InboundMessage(channel="c", user_id="u", text="这是啥", images=["/a.png"]))
    assert agent.text == "这是啥" and agent.images == ["/a.png"]
