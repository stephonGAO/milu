"""飞书渠道单测——不依赖真实飞书凭证。

覆盖：
1. FeishuCrypto 加解密 round-trip；
2. url_verification 握手（明文 + 密文）回 challenge，verify_token 不符拒绝；
3. im.message.receive_v1 文本事件经 ASGITransport → 后台 dispatch → send 回发；
4. 同一 event_id 去重（只回一次）；密文事件但未配 ENCRYPT_KEY 拒绝。
"""
from __future__ import annotations

import asyncio
import json

import httpx
from fastapi import FastAPI

from milu.channels.base import InboundMessage, OutboundMessage
from milu.channels.feishu import FeishuChannel, FeishuClient, FeishuConfig, FeishuCrypto
from milu.channels.state import InMemoryStateStore

_ENCRYPT_KEY = "test_encrypt_key_123"
_VERIFY_TOKEN = "vtok"


def _config(encrypt: bool = True) -> FeishuConfig:
    return FeishuConfig(
        app_id="cli_x",
        app_secret="sec",
        verify_token=_VERIFY_TOKEN,
        encrypt_key=_ENCRYPT_KEY if encrypt else "",
        callback_path="/feishu/event",
    )


def _text_event(event_id: str, open_id: str, text: str) -> dict:
    return {
        "schema": "2.0",
        "header": {
            "event_id": event_id,
            "event_type": "im.message.receive_v1",
            "token": _VERIFY_TOKEN,
        },
        "event": {
            "sender": {"sender_id": {"open_id": open_id}, "sender_type": "user"},
            "message": {
                "message_id": "om_" + event_id,
                "chat_id": "oc_1",
                "message_type": "text",
                "content": json.dumps({"text": text}, ensure_ascii=False),
            },
        },
    }


async def _reply_dispatch(msg: InboundMessage) -> OutboundMessage:
    return OutboundMessage(text=f"[milu] {msg.text}")


def _channel_app(channel: FeishuChannel, dispatch=_reply_dispatch) -> FastAPI:
    app = FastAPI()
    channel.register(app, dispatch)
    return app


# ── 1. 加解密 ──────────────────────────────────────────────────────────
def test_feishu_crypto_roundtrip():
    c = FeishuCrypto(_ENCRYPT_KEY)
    plain = '你好 hello 🦞 {"text":"x"}'
    assert c.decrypt(c.encrypt(plain)) == plain


# ── 2. url_verification 握手 ──────────────────────────────────────────
async def test_url_verification_plaintext():
    ch = FeishuChannel(_config(encrypt=False), state=InMemoryStateStore())
    app = _channel_app(ch)
    payload = {"type": "url_verification", "challenge": "abc123", "token": _VERIFY_TOKEN}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.post("/feishu/event", json=payload)
    assert r.status_code == 200
    assert r.json() == {"challenge": "abc123"}


async def test_url_verification_encrypted():
    cfg = _config(encrypt=True)
    ch = FeishuChannel(cfg, state=InMemoryStateStore())
    app = _channel_app(ch)
    crypto = FeishuCrypto(_ENCRYPT_KEY)
    inner = {"type": "url_verification", "challenge": "ch-xyz", "token": _VERIFY_TOKEN}
    body = {"encrypt": crypto.encrypt(json.dumps(inner))}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.post("/feishu/event", json=body)
    assert r.status_code == 200
    assert r.json() == {"challenge": "ch-xyz"}


async def test_url_verification_bad_token_rejected():
    ch = FeishuChannel(_config(encrypt=False), state=InMemoryStateStore())
    app = _channel_app(ch)
    payload = {"type": "url_verification", "challenge": "x", "token": "WRONG"}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.post("/feishu/event", json=payload)
    assert r.status_code == 403


async def test_encrypted_body_without_key_acked_and_ignored():
    # 渠道未配 encrypt_key 却收到密文 → ack 200 忽略（不 403，避免飞书重投风暴）
    ch = FeishuChannel(_config(encrypt=False), state=InMemoryStateStore())
    app = _channel_app(ch)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.post("/feishu/event", json={"encrypt": "AAAA"})
    assert r.status_code == 200


async def test_event_bad_token_acked_not_processed():
    """非 url_verification 的事件 token 不符 → 回 200 ack 但不处理（不再 403）。"""
    cfg = _config(encrypt=False)
    done = asyncio.Event()
    client, sent = _mock_client(cfg, done)
    ch = FeishuChannel(cfg, client=client, state=InMemoryStateStore())
    app = _channel_app(ch)
    bad = _text_event("evtX", "ou_a", "hi")
    bad["header"]["token"] = "WRONG_TOKEN"   # 伪造 / 非本应用来源

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.post("/feishu/event", json=bad)
    assert r.status_code == 200          # ack，不再 403
    await asyncio.sleep(0.15)
    assert sent == []                    # 但不处理、不回发


# ── 3. 文本消息事件 → dispatch → 回发 ─────────────────────────────────
def _mock_client(cfg: FeishuConfig, done: asyncio.Event):
    client = FeishuClient(cfg)
    sent = []

    async def _send(open_id, text):
        sent.append((open_id, text))
        done.set()
        return {"code": 0}

    client.send_text = _send  # type: ignore[assignment]
    return client, sent


async def test_message_event_dispatch_and_reply():
    cfg = _config(encrypt=False)
    done = asyncio.Event()
    client, sent = _mock_client(cfg, done)
    ch = FeishuChannel(cfg, client=client, state=InMemoryStateStore())
    app = _channel_app(ch)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.post("/feishu/event", json=_text_event("evt1", "ou_abc", "你好"))
        assert r.status_code == 200          # 立即 ack
    await asyncio.wait_for(done.wait(), timeout=2)
    assert sent == [("ou_abc", "[milu] 你好")]


async def test_message_event_encrypted_dispatch():
    cfg = _config(encrypt=True)
    done = asyncio.Event()
    client, sent = _mock_client(cfg, done)
    ch = FeishuChannel(cfg, client=client, state=InMemoryStateStore())
    app = _channel_app(ch)
    crypto = FeishuCrypto(_ENCRYPT_KEY)
    body = {"encrypt": crypto.encrypt(json.dumps(_text_event("evt2", "ou_x", "在吗")))}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.post("/feishu/event", json=body)
        assert r.status_code == 200
    await asyncio.wait_for(done.wait(), timeout=2)
    assert sent == [("ou_x", "[milu] 在吗")]


async def test_message_event_dedup_same_event_id():
    cfg = _config(encrypt=False)
    done = asyncio.Event()
    client, sent = _mock_client(cfg, done)
    ch = FeishuChannel(cfg, client=client, state=InMemoryStateStore())
    app = _channel_app(ch)

    ev = _text_event("dup1", "ou_a", "hi")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        await ac.post("/feishu/event", json=ev)
        await asyncio.wait_for(done.wait(), timeout=2)
        # 同一 event_id 再投一次
        await ac.post("/feishu/event", json=ev)
        await asyncio.sleep(0.2)   # 给后台任务机会（若未去重会再发一次）
    assert sent == [("ou_a", "[milu] hi")]   # 只回一次
