"""微信客服 echo 链路本地验证（无需真实企业微信凭证）。

覆盖三层：
1. WeComCrypto 加解密 round-trip + 签名 / URL 验证 / 消息解密；
2. handle_event echo 逻辑（mock sync_msg/send_text，含去重、非文本跳过、on_text 接缝）；
3. HTTP 层经 httpx ASGITransport：GET 回 echostr 明文、POST 回 success 且后台 echo 发出。

注：本地用同一套 WeComCrypto 构造「企业微信发来的」加密请求，验证 verify/decrypt
正确逆转 encrypt/sign。最终是否与企业微信完全一致，仍需真实回调联调确认。
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import httpx
import pytest

from milu.channels.wecom_crypto import WeComCrypto, WeComCryptError, sha1_signature
from milu.channels.wechat_kf import (
    WeChatKfChannel,
    WeChatKfClient,
    WeChatKfConfig,
    build_router,
    handle_event,
)
from milu.channels.base import InboundMessage, OutboundMessage
from milu.channels.state import InMemoryStateStore
from fastapi import FastAPI

# 测试用凭证（EncodingAESKey 必须是合法的 43 位 base64）
_AESKEY = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG"  # 43 位
_TOKEN = "test_token"
_CORPID = "ww_test_corp"


def _crypto() -> WeComCrypto:
    return WeComCrypto(token=_TOKEN, encoding_aes_key=_AESKEY, receive_id=_CORPID)


def _config() -> WeChatKfConfig:
    return WeChatKfConfig(
        corp_id=_CORPID,
        secret="test_secret",
        token=_TOKEN,
        encoding_aes_key=_AESKEY,
        callback_path="/wechat/kf/callback",
    )


# ── 1. 加解密 ──────────────────────────────────────────────────────────
def test_crypto_roundtrip():
    c = _crypto()
    plain = "你好，世界 hello 🦞 <xml>edge</xml>"
    enc = c.encrypt(plain)
    assert c.decrypt(enc) == plain


def test_crypto_receive_id_mismatch():
    c = _crypto()
    enc = c.encrypt("hi")
    wrong = WeComCrypto(token=_TOKEN, encoding_aes_key=_AESKEY, receive_id="other_corp")
    with pytest.raises(WeComCryptError):
        wrong.decrypt(enc)


def test_verify_url():
    c = _crypto()
    echo_plain = "1616140317555161061"
    echostr = c.encrypt(echo_plain)
    ts, nonce = "1411443780", "123456"
    sig = sha1_signature(_TOKEN, ts, nonce, echostr)
    assert c.verify_url(sig, ts, nonce, echostr) == echo_plain
    # 签名错误必须拒绝
    with pytest.raises(WeComCryptError):
        c.verify_url("deadbeef", ts, nonce, echostr)


def test_decrypt_message():
    c = _crypto()
    inner = (
        "<xml><ToUserName><![CDATA[ww]]></ToUserName>"
        "<Event><![CDATA[kf_msg_or_event]]></Event>"
        "<Token><![CDATA[ENCTOKEN]]></Token>"
        "<OpenKfId><![CDATA[wk123]]></OpenKfId></xml>"
    )
    encrypt = c.encrypt(inner)
    body = f"<xml><Encrypt><![CDATA[{encrypt}]]></Encrypt></xml>"
    ts, nonce = "1411443780", "123456"
    sig = sha1_signature(_TOKEN, ts, nonce, encrypt)
    out = c.decrypt_message(body, sig, ts, nonce)
    assert "kf_msg_or_event" in out and "wk123" in out


# ── 2. handle_event echo 逻辑 ─────────────────────────────────────────
def _text_msg(msgid: str, user: str, content: str, send_time: int | None = None) -> dict:
    return {
        "msgid": msgid,
        "msgtype": "text",
        "external_userid": user,
        "text": {"content": content},
        # 默认给当前时间，使其在冷启动新鲜窗口内（不被 backlog 保护跳过）
        "send_time": int(time.time()) if send_time is None else send_time,
    }


async def test_handle_event_echo():
    client = WeChatKfClient(_config())
    client.sync_msg = AsyncMock(return_value=[_text_msg("m1", "userA", "在吗")])
    client.send_text = AsyncMock(return_value={"errcode": 0})

    await handle_event(client, token="T", open_kfid="wk1")

    client.send_text.assert_awaited_once_with("userA", "wk1", "在吗")


async def test_handle_event_dedup_and_skip_nontext():
    client = WeChatKfClient(_config())
    client.sync_msg = AsyncMock(
        return_value=[
            _text_msg("m1", "userA", "hi"),
            _text_msg("m1", "userA", "hi"),  # 重复 msgid，应去重
            {"msgid": "m2", "msgtype": "image", "external_userid": "userA"},  # 非文本跳过
        ]
    )
    client.send_text = AsyncMock(return_value={"errcode": 0})

    await handle_event(client, token="T", open_kfid="wk1")

    # 只应回一次（去重 + 跳过非文本）
    client.send_text.assert_awaited_once()


async def test_handle_event_on_text_hook():
    """on_text 接缝：返回值替换 echo（模拟后续接 milu）。"""
    client = WeChatKfClient(_config())
    client.sync_msg = AsyncMock(return_value=[_text_msg("m1", "userA", "1+1")])
    client.send_text = AsyncMock(return_value={"errcode": 0})

    async def on_text(user: str, text: str) -> str:
        return f"[milu] 收到 {user}: {text}"

    await handle_event(client, token="T", open_kfid="wk1", on_text=on_text)

    client.send_text.assert_awaited_once_with("userA", "wk1", "[milu] 收到 userA: 1+1")


# ── 3. HTTP 层（ASGITransport，无真实网络）────────────────────────────
def _build_app(client: WeChatKfClient) -> FastAPI:
    app = FastAPI()
    app.include_router(build_router(_config(), client=client))
    return app


async def test_http_verify_get():
    client = WeChatKfClient(_config())
    app = _build_app(client)
    c = _crypto()
    echo_plain = "echo-12345"
    echostr = c.encrypt(echo_plain)
    ts, nonce = "1411443780", "abc"
    sig = sha1_signature(_TOKEN, ts, nonce, echostr)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get(
            "/wechat/kf/callback",
            params={
                "msg_signature": sig,
                "timestamp": ts,
                "nonce": nonce,
                "echostr": echostr,
            },
        )
    assert r.status_code == 200
    assert r.text == echo_plain


async def test_http_receive_post_triggers_echo():
    client = WeChatKfClient(_config())
    # mock 掉真实 API 调用：sync_msg 返回一条文本，send_text 用 Event 标记完成
    done = asyncio.Event()
    client.sync_msg = AsyncMock(return_value=[_text_msg("m1", "userA", "你好milu")])

    async def _send(touser, open_kfid, content):
        _send.calls.append((touser, open_kfid, content))
        done.set()
        return {"errcode": 0}

    _send.calls = []  # type: ignore[attr-defined]
    client.send_text = _send  # type: ignore[assignment]

    app = _build_app(client)
    c = _crypto()
    inner = (
        "<xml><Event><![CDATA[kf_msg_or_event]]></Event>"
        "<Token><![CDATA[ENCTOKEN]]></Token>"
        "<OpenKfId><![CDATA[wk1]]></OpenKfId></xml>"
    )
    encrypt = c.encrypt(inner)
    body = f"<xml><Encrypt><![CDATA[{encrypt}]]></Encrypt></xml>"
    ts, nonce = "1411443780", "abc"
    sig = sha1_signature(_TOKEN, ts, nonce, encrypt)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post(
            "/wechat/kf/callback",
            params={"msg_signature": sig, "timestamp": ts, "nonce": nonce},
            content=body,
        )
    # 必须立即回 success
    assert r.status_code == 200
    assert r.text == "success"
    # 后台任务完成后应已 echo
    await asyncio.wait_for(done.wait(), timeout=2)
    assert _send.calls == [("userA", "wk1", "你好milu")]  # type: ignore[attr-defined]


# ── 4. WeChatKfChannel（统一网关适配器 + StateStore）──────────────────
async def _post_kf_event(app: FastAPI, open_kfid: str = "wk1") -> httpx.Response:
    """构造一条加密的 kf_msg_or_event 回调并 POST。"""
    c = _crypto()
    inner = (
        "<xml><Event><![CDATA[kf_msg_or_event]]></Event>"
        f"<Token><![CDATA[ENCTOKEN]]></Token>"
        f"<OpenKfId><![CDATA[{open_kfid}]]></OpenKfId></xml>"
    )
    encrypt = c.encrypt(inner)
    body = f"<xml><Encrypt><![CDATA[{encrypt}]]></Encrypt></xml>"
    ts, nonce = "1411443780", "abc"
    sig = sha1_signature(_TOKEN, ts, nonce, encrypt)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        return await ac.post(
            "/wechat/kf/callback",
            params={"msg_signature": sig, "timestamp": ts, "nonce": nonce},
            content=body,
        )


async def test_channel_dispatch_reply_and_cursor():
    cfg = _config()
    client = WeChatKfClient(cfg)
    # fetch_messages 返回 (消息列表, next_cursor)
    client.fetch_messages = AsyncMock(
        return_value=([_text_msg("m1", "userA", "你好milu")], "cursor-1")
    )
    done = asyncio.Event()
    sent = []

    async def _send(touser, open_kfid, content):
        sent.append((touser, open_kfid, content))
        done.set()
        return {"errcode": 0}

    client.send_text = _send  # type: ignore[assignment]
    state = InMemoryStateStore()
    ch = WeChatKfChannel(cfg, client=client, state=state)

    async def dispatch(msg: InboundMessage) -> OutboundMessage:
        assert msg.channel == "wechat_kf"
        assert msg.reply_to == {"open_kfid": "wk1"}
        return OutboundMessage(text=f"[milu] {msg.text}")

    app = FastAPI()
    ch.register(app, dispatch)

    r = await _post_kf_event(app)
    assert r.status_code == 200 and r.text == "success"

    await asyncio.wait_for(done.wait(), timeout=2)
    assert sent == [("userA", "wk1", "[milu] 你好milu")]
    # 游标推进并落到 StateStore
    assert await state.get_cursor("wechat_kf", "wk1") == "cursor-1"


async def test_channel_cold_start_skips_backlog():
    """冷启动（空游标）只回新鲜消息，跳过数天前的历史积压（防 95001 刷屏）。"""
    cfg = _config()
    client = WeChatKfClient(cfg)
    old = _text_msg("old1", "userA", "几天前的历史", send_time=int(time.time()) - 99999)
    fresh = _text_msg("new1", "userA", "你好", send_time=int(time.time()))
    client.fetch_messages = AsyncMock(return_value=([old, fresh], "cur"))
    done = asyncio.Event()
    sent = []

    async def _send(touser, open_kfid, content):
        sent.append(content)
        done.set()
        return {"errcode": 0}

    client.send_text = _send  # type: ignore[assignment]
    state = InMemoryStateStore()   # 空游标 = 冷启动
    ch = WeChatKfChannel(cfg, client=client, state=state)

    async def dispatch(msg):
        return OutboundMessage(text=f"[milu] {msg.text}")

    app = FastAPI()
    ch.register(app, dispatch)
    await _post_kf_event(app)
    await asyncio.wait_for(done.wait(), timeout=2)
    await asyncio.sleep(0.1)
    assert sent == ["[milu] 你好"]   # 只回新鲜消息，老消息被跳过
    # 冷启动后游标已推进，下次即非冷启动、正常回复全部
    assert await state.get_cursor("wechat_kf", "wk1") == "cur"


async def test_channel_dedup_via_state():
    cfg = _config()
    client = WeChatKfClient(cfg)
    client.fetch_messages = AsyncMock(
        return_value=(
            [
                _text_msg("m1", "userA", "hi"),
                _text_msg("m1", "userA", "hi"),   # 同 msgid 去重
                {"msgid": "m2", "msgtype": "image", "external_userid": "userA"},  # 非文本
            ],
            "cur",
        )
    )
    done = asyncio.Event()
    calls = []

    async def _send(touser, open_kfid, content):
        calls.append(content)
        done.set()
        return {"errcode": 0}

    client.send_text = _send  # type: ignore[assignment]
    ch = WeChatKfChannel(cfg, client=client, state=InMemoryStateStore())

    async def dispatch(msg):
        return OutboundMessage(text=msg.text)

    app = FastAPI()
    ch.register(app, dispatch)
    await _post_kf_event(app)
    await asyncio.wait_for(done.wait(), timeout=2)
    await asyncio.sleep(0.1)   # 给潜在的多余回复机会
    assert calls == ["hi"]     # 只回一次


async def test_http_receive_bad_signature_rejected():
    client = WeChatKfClient(_config())
    app = _build_app(client)
    c = _crypto()
    encrypt = c.encrypt("<xml><Event><![CDATA[kf_msg_or_event]]></Event></xml>")
    body = f"<xml><Encrypt><![CDATA[{encrypt}]]></Encrypt></xml>"

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post(
            "/wechat/kf/callback",
            params={"msg_signature": "wrong", "timestamp": "1", "nonce": "n"},
            content=body,
        )
    assert r.status_code == 403
