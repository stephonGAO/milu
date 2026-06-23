"""网关核心（base / state / runner / gateway）单测——不依赖任何真实平台凭证。

覆盖：
1. StateStore：内存版去重/游标；文件版**跨实例持久化** + 原子写。
2. AgentRunner：跑过（假）AgentPool 取 AgentDone 文本；异常/空回复回退兜底语。
3. Gateway：/healthz 列渠道；webhook 渠道经 dispatch 路由（ASGITransport）；
   lifespan 起停钩子 + polling 渠道 run() 被拉起/取消。
"""
from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

from milu.agent.events import AgentDone
from milu.llm.base.response import TokenUsage
from milu.channels import (
    AgentRunner,
    Channel,
    Gateway,
    InboundMessage,
    OutboundMessage,
)
from milu.channels.runner import DEFAULT_FALLBACK
from milu.channels.state import FileStateStore, InMemoryStateStore


# ── 1. StateStore ─────────────────────────────────────────────────────
async def test_memory_state_dedup_and_cursor():
    s = InMemoryStateStore()
    assert await s.get_cursor("c", "k") == ""
    await s.set_cursor("c", "k", "cur-1")
    assert await s.get_cursor("c", "k") == "cur-1"
    assert await s.seen("c", "m1") is False
    assert await s.seen("c", "m1") is True       # 第二次见到
    assert await s.seen("c", "") is False        # 空 id 恒 False
    # 渠道命名空间隔离
    assert await s.get_cursor("other", "k") == ""


async def test_memory_state_lru_eviction():
    s = InMemoryStateStore(dedup_capacity=2)
    assert await s.seen("c", "a") is False
    assert await s.seen("c", "b") is False
    assert await s.seen("c", "c") is False   # 越界淘汰最旧 a
    assert await s.seen("c", "b") is True     # b 仍在
    assert await s.seen("c", "c") is True     # c 仍在
    # a 已被淘汰，再次见到视为未见（此查询会把 a 重新插入并淘汰 b，故放最后）
    assert await s.seen("c", "a") is False


async def test_file_state_persists_across_instances(tmp_path):
    s = FileStateStore(tmp_path, flush_interval=0.05)
    await s.set_cursor("wechat_kf", "wk1", "cursor-123")
    assert await s.seen("wechat_kf", "m1") is False
    assert await s.seen("wechat_kf", "m1") is True
    await s.close()   # 强制落盘

    # 文件确实写出来了
    assert (tmp_path / "wechat_kf.json").exists()

    # 新实例从磁盘恢复游标与去重集合
    s2 = FileStateStore(tmp_path, flush_interval=0.05)
    assert await s2.get_cursor("wechat_kf", "wk1") == "cursor-123"
    assert await s2.seen("wechat_kf", "m1") is True    # 已持久化为已见
    assert await s2.seen("wechat_kf", "m2") is False
    await s2.close()


async def test_file_state_flush_writes_without_close(tmp_path):
    s = FileStateStore(tmp_path, flush_interval=10.0)  # 去抖窗口很长
    await s.set_cursor("telegram", "offset", "42")
    await s.flush()                                    # 主动刷盘，不等去抖
    s2 = FileStateStore(tmp_path)
    assert await s2.get_cursor("telegram", "offset") == "42"
    await s.close()
    await s2.close()


# ── 2. AgentRunner（假 pool）──────────────────────────────────────────
class _FakeAgent:
    def __init__(self, reply: str, *, boom: bool = False, empty: bool = False):
        self._reply = reply
        self._boom = boom
        self._empty = empty
        self.seen_images = None

    async def run(self, text: str, images=None):
        self.seen_images = images
        if self._boom:
            raise RuntimeError("agent exploded")
        if self._empty:
            return
        yield AgentDone(final_text=f"{self._reply}:{text}",
                        total_usage=TokenUsage(), turn_count=1)


class _FakeHandle:
    def __init__(self, agent):
        self.agent = agent


class _FakeAcquire:
    def __init__(self, handle):
        self._h = handle

    async def __aenter__(self):
        return self._h

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, agent: _FakeAgent):
        self._agent = agent
        self.calls: list[tuple[str, str]] = []
        self.started = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.started = False

    def acquire(self, user_id: str, session_id: str):
        self.calls.append((user_id, session_id))
        return _FakeAcquire(_FakeHandle(self._agent))


async def test_runner_returns_agent_done_text():
    pool = _FakePool(_FakeAgent("回复"))
    runner = AgentRunner(pool)
    out = await runner(InboundMessage(channel="wechat_kf", user_id="bob", text="在吗"))
    assert isinstance(out, OutboundMessage)
    assert out.text == "回复:在吗"
    # 默认 (user_id, session_id) 按 渠道:用户 派生
    assert pool.calls == [("wechat_kf:bob", "wechat_kf:bob")]


async def test_runner_passes_images():
    agent = _FakeAgent("ok")
    runner = AgentRunner(_FakePool(agent))
    await runner(InboundMessage(channel="c", user_id="u", text="t", images=["/x.png"]))
    assert agent.seen_images == ["/x.png"]


async def test_runner_fallback_on_empty_and_error():
    # 空回复 → 兜底语
    r1 = AgentRunner(_FakePool(_FakeAgent("", empty=True)))
    out1 = await r1(InboundMessage(channel="c", user_id="u", text="t"))
    assert out1.text == DEFAULT_FALLBACK
    # Agent 抛异常 → 兜底语（绝不向渠道抛）
    r2 = AgentRunner(_FakePool(_FakeAgent("x", boom=True)), fallback_text="稍后再试")
    out2 = await r2(InboundMessage(channel="c", user_id="u", text="t"))
    assert out2.text == "稍后再试"


async def test_runner_custom_session_of():
    pool = _FakePool(_FakeAgent("ok"))
    runner = AgentRunner(pool, session_of=lambda m: ("U", "S"))
    await runner(InboundMessage(channel="c", user_id="u", text="t"))
    assert pool.calls == [("U", "S")]


# ── 3. Gateway ────────────────────────────────────────────────────────
class _WebhookChannel(Channel):
    """最小 webhook 渠道：POST /hook 把 body 当文本走 dispatch，回复直接返回。"""

    @property
    def name(self) -> str:
        return "hook"

    def register(self, app: FastAPI, dispatch) -> None:
        async def hook(request: Request):
            text = (await request.body()).decode()
            reply = await dispatch(
                InboundMessage(channel=self.name, user_id="webuser", text=text)
            )
            return PlainTextResponse(reply.text if reply else "")
        app.add_api_route("/hook", hook, methods=["POST"])

    async def close(self) -> None:
        self.closed = True


class _PollChannel(Channel):
    """最小 polling 渠道：run() 里 dispatch 一条消息后阻塞，直到被取消。"""

    def __init__(self):
        self.ran = asyncio.Event()
        self.reply = None
        self.closed = False

    @property
    def name(self) -> str:
        return "poll"

    async def run(self, dispatch) -> None:
        self.reply = await dispatch(
            InboundMessage(channel=self.name, user_id="pu", text="ping")
        )
        self.ran.set()
        await asyncio.Event().wait()   # 阻塞直到 lifespan 关停取消本任务

    async def close(self) -> None:
        self.closed = True


async def _echo_dispatch(msg: InboundMessage):
    return OutboundMessage(text=f"echo:{msg.text}")


async def test_gateway_healthz_and_webhook_dispatch():
    ch = _WebhookChannel()
    gw = Gateway(dispatch=_echo_dispatch, channels=[ch])
    app = gw.build_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        h = await ac.get("/healthz")
        assert h.status_code == 200
        assert h.json() == {"ok": True, "channels": ["hook"]}
        r = await ac.post("/hook", content="你好")
        assert r.text == "echo:你好"


async def test_gateway_lifespan_startup_polling_shutdown():
    poll = _PollChannel()
    flags = {"started": False, "stopped": False}

    async def on_start():
        flags["started"] = True

    async def on_stop():
        flags["stopped"] = True

    gw = Gateway(dispatch=_echo_dispatch, channels=[poll],
                 on_startup=on_start, on_shutdown=on_stop)
    app = gw.build_app()
    async with app.router.lifespan_context(app):
        assert flags["started"] is True
        await asyncio.wait_for(poll.ran.wait(), timeout=2)
        assert poll.reply.text == "echo:ping"
    # 退出 lifespan：polling 任务被取消、渠道关闭、on_shutdown 调用
    assert poll.closed is True
    assert flags["stopped"] is True


def test_gateway_requires_a_channel():
    with pytest.raises(ValueError):
        Gateway(dispatch=_echo_dispatch, channels=[])


async def test_gateway_from_runner_wires_lifecycle():
    pool = _FakePool(_FakeAgent("ok"))
    runner = AgentRunner(pool)
    poll = _PollChannel()
    gw = Gateway.from_runner(runner, [poll])
    app = gw.build_app()
    async with app.router.lifespan_context(app):
        assert pool.started is True               # runner.start → pool.start
        await asyncio.wait_for(poll.ran.wait(), timeout=2)
        assert poll.reply.text == "ok:ping"       # dispatch 走的是 runner
    assert pool.started is False                  # runner.close → pool.stop
