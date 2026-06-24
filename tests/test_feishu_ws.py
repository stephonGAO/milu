"""飞书长连接渠道单测——只测事件规整/dispatch/发送/去重逻辑（不连真实 ws、不需 lark-oapi）。

`FeishuWsChannel._handle` 以**属性访问**读 SDK 事件对象，故这里用同形的假对象驱动；
真正的 websocket 传输由 lark-oapi 负责，需真实飞书应用 + 安装 lark-oapi 才能联调。
"""
from __future__ import annotations

import json

from milu.channels.base import InboundMessage, OutboundMessage
from milu.channels.feishu import FeishuClient, FeishuConfig
from milu.channels.feishu_ws import FeishuWsChannel
from milu.channels.state import InMemoryStateStore


class _Obj:
    """轻量属性容器，模拟 lark-oapi 事件模型的属性访问。"""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _fake_event(event_id: str, open_id: str, text: str, mtype: str = "text") -> _Obj:
    return _Obj(
        header=_Obj(event_id=event_id),
        event=_Obj(
            sender=_Obj(sender_id=_Obj(open_id=open_id)),
            message=_Obj(
                message_type=mtype,
                message_id="om_" + event_id,
                chat_id="oc_1",
                content=json.dumps({"text": text}, ensure_ascii=False),
            ),
        ),
    )


def _channel():
    cfg = FeishuConfig(app_id="x", app_secret="y")
    client = FeishuClient(cfg)
    sent: list = []

    async def _send(open_id, text):
        sent.append((open_id, text))
        return {"code": 0}

    client.send_text = _send  # type: ignore[assignment]
    ch = FeishuWsChannel(cfg, client=client, state=InMemoryStateStore())
    return ch, sent


async def _reply(msg: InboundMessage) -> OutboundMessage:
    assert msg.channel == "feishu"
    return OutboundMessage(text=f"[milu] {msg.text}")


async def test_ws_handle_dispatch_and_reply():
    ch, sent = _channel()
    await ch._handle(_fake_event("e1", "ou_a", "你好"), _reply)
    assert sent == [("ou_a", "[milu] 你好")]


async def test_ws_handle_skips_nontext():
    ch, sent = _channel()
    await ch._handle(_fake_event("e2", "ou_a", "x", mtype="image"), _reply)
    assert sent == []


async def test_ws_handle_dedup_same_event_id():
    ch, sent = _channel()
    ev = _fake_event("dup", "ou_a", "hi")
    await ch._handle(ev, _reply)
    await ch._handle(ev, _reply)   # 同 event_id 第二次被去重
    assert sent == [("ou_a", "[milu] hi")]


async def test_ws_handle_empty_reply_not_sent():
    ch, sent = _channel()

    async def _none(msg):
        return None

    await ch._handle(_fake_event("e3", "ou_a", "在吗"), _none)
    assert sent == []


async def test_ws_channel_name_is_feishu():
    # 与 webhook 版同名，保证用户身份/会话在两种模式间一致
    ch, _ = _channel()
    assert ch.name == "feishu"


async def test_ws_handle_never_raises_on_bad_dispatch():
    ch, sent = _channel()

    async def _boom(msg):
        raise RuntimeError("dispatch 炸了")

    # 整段不抛异常（只记日志）
    await ch._handle(_fake_event("e4", "ou_a", "hi"), _boom)
    assert sent == []
