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
from unittest.mock import AsyncMock

import httpx
import pytest

from milu.channels.wecom_crypto import WeComCrypto, WeComCryptError, sha1_signature
from milu.channels.wechat_kf import (
    WeChatKfClient,
    WeChatKfConfig,
    build_router,
    handle_event,
)
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
def _text_msg(msgid: str, user: str, content: str) -> dict:
    return {
        "msgid": msgid,
        "msgtype": "text",
        "external_userid": user,
        "text": {"content": content},
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
