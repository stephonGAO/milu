"""观测大屏单测：聚合口径 / 池活跃快照 / 鉴权门控 / 实时枢纽 / 跨用户枚举。"""
from __future__ import annotations

import json
import types

import pytest

from milu.observability.analytics import (
    group_stats,
    percentile,
    summarize_runs,
    timeseries,
)
from milu.serving import AgentPool
from milu.serving.web import dashboard as dash


# ── 测试夹具数据 ──────────────────────────────────────────

def _run(**kw):
    base = dict(
        trace_id="t" + str(kw.get("i", 0)).zfill(31),
        started_at=1_700_000_000.0,
        provider="qwen", model="qwen-plus", mode="auto",
        session_id=None, user_id="default", status="ok", error_type=None,
        duration_ms=1000.0, turn_count=1, input_tokens=100, output_tokens=50,
        reasoning_tokens=0, cost_estimate=0.001, cost_currency="CNY",
        ttft_ms=200.0, llm_time_ms=800.0, tool_time_ms=100.0, judge_time_ms=0.0,
        wait_time_ms=0.0, tool_calls=2, tool_errors=0, judge_denies=0,
        judge_confirms=0, fail_opens=0, compactions=0, subagent_runs=0,
        trace_path=None,
    )
    base.update(kw)
    return base


# ── analytics ─────────────────────────────────────────────

def test_percentile():
    assert percentile([], 0.5) == 0.0
    assert percentile([5], 0.5) == 5
    vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(vals, 0.5) in (5, 6)
    assert percentile(vals, 0.95) == 10


def test_summarize_runs_basic():
    rows = [
        _run(status="ok", input_tokens=100, output_tokens=50, cost_estimate=0.002),
        _run(status="error", input_tokens=200, output_tokens=80, cost_estimate=0.003,
             fail_opens=1, judge_denies=1, tool_errors=1),
    ]
    s = summarize_runs(rows)
    assert s["runs"] == 2
    assert s["ok"] == 1
    assert s["error"] == 1
    assert s["input_tokens"] == 300
    assert s["output_tokens"] == 130
    assert s["cost"]["CNY"] == pytest.approx(0.005)
    assert s["fail_opens"] == 1
    assert s["judge_denies"] == 1
    assert s["tool_errors"] == 1


def test_summarize_empty():
    s = summarize_runs([])
    assert s["runs"] == 0
    assert s["cost"] == {}
    assert s["p50_ms"] == 0.0
    assert s["avg_ttft_ms"] is None


def test_summarize_multi_currency():
    rows = [_run(cost_estimate=0.01, cost_currency="CNY"),
            _run(cost_estimate=0.02, cost_currency="USD")]
    s = summarize_runs(rows)
    assert s["cost"] == pytest.approx({"CNY": 0.01, "USD": 0.02})


def test_group_stats_by_model_and_provider():
    rows = [
        _run(provider="qwen", model="qwen-plus"),
        _run(provider="qwen", model="qwen-plus"),
        _run(provider="deepseek", model="deepseek-chat"),
    ]
    g = group_stats(rows, "provider")
    by = {x["key"]: x for x in g}
    assert by["qwen"]["runs"] == 2
    assert by["deepseek"]["runs"] == 1
    gm = {x["key"]: x for x in group_stats(rows, "model")}
    assert "qwen/qwen-plus" in gm
    assert gm["qwen/qwen-plus"]["runs"] == 2


def test_timeseries_buckets_deterministic():
    now = 1_700_003_600.0  # 固定基准便于断言
    rows = [
        _run(started_at=now - 30),       # 当前小时桶
        _run(started_at=now - 3600 - 30),  # 上一个小时桶
        _run(started_at=now - 999999),   # 超出窗口，应被忽略
    ]
    series = timeseries(rows, bucket="hour", buckets=3, now_ts=now)
    assert len(series) == 3
    assert sum(b["runs"] for b in series) == 2
    # 最新桶（最后一个）应含 now-30 那条
    assert series[-1]["runs"] == 1


# ── 池活跃快照 ────────────────────────────────────────────

class _FakeAgent:
    def __init__(self):
        self.mode = types.SimpleNamespace(value="auto")
        self._mcp_manager = None

    async def disconnect_mcp(self):
        pass

    async def close_knowledge(self):
        pass


async def test_pool_list_active_sessions():
    pool = AgentPool(
        llm_factory=lambda u, s: None,
        agent_factory=lambda u, s, llm: _FakeAgent(),
    )
    assert pool.list_active_sessions() == []
    await pool.get_or_create_agent("alice", "main")
    await pool.get_or_create_agent("bob", "main")
    active = pool.list_active_sessions()
    assert {a["user_id"] for a in active} == {"alice", "bob"}
    a0 = active[0]
    assert a0["mode"] == "auto"
    assert a0["mcp_connected"] is False
    assert a0["active_count"] == 0
    assert a0["idle_seconds"] >= 0


# ── 鉴权门控 ──────────────────────────────────────────────

def _req(host="127.0.0.1", headers=None, query=None):
    return types.SimpleNamespace(
        client=types.SimpleNamespace(host=host),
        headers=headers or {},
        query_params=query or {},
    )


def test_admin_no_token_local_ok(monkeypatch):
    monkeypatch.delenv("MILU_ADMIN_TOKEN", raising=False)
    dash.check_admin(_req(host="127.0.0.1"))  # 不抛即通过


def test_admin_no_token_remote_denied(monkeypatch):
    from fastapi import HTTPException
    monkeypatch.delenv("MILU_ADMIN_TOKEN", raising=False)
    with pytest.raises(HTTPException) as ei:
        dash.check_admin(_req(host="10.0.0.9"))
    assert ei.value.status_code == 403


def test_admin_token_required(monkeypatch):
    from fastapi import HTTPException
    monkeypatch.setenv("MILU_ADMIN_TOKEN", "s3cr3t")
    # 远程无令牌 → 拒
    with pytest.raises(HTTPException) as ei:
        dash.check_admin(_req(host="10.0.0.9"))
    assert ei.value.status_code == 401
    # header 令牌正确 → 通过
    dash.check_admin(_req(host="10.0.0.9", headers={"X-Admin-Token": "s3cr3t"}))
    # query 令牌正确 → 通过
    dash.check_admin(_req(host="10.0.0.9", query={"token": "s3cr3t"}))
    # Bearer 令牌正确 → 通过
    dash.check_admin(_req(host="10.0.0.9", headers={"Authorization": "Bearer s3cr3t"}))
    # 错误令牌 → 拒
    with pytest.raises(HTTPException):
        dash.check_admin(_req(host="10.0.0.9", headers={"X-Admin-Token": "wrong"}))


# ── 实时枢纽 ──────────────────────────────────────────────

async def test_hub_publish_subscribe():
    hub = dash.DashboardHub(buffer_size=10, queue_size=4)
    q = hub.subscribe()
    hub.publish({"type": "span", "kind": "tool"})
    ev = await q.get()
    assert ev["kind"] == "tool"
    assert ev["seq"] == 1
    assert hub.subscriber_count == 1
    hub.unsubscribe(q)
    assert hub.subscriber_count == 0


async def test_hub_drop_on_full():
    hub = dash.DashboardHub(buffer_size=50, queue_size=2)
    q = hub.subscribe()
    for i in range(6):  # 远超队列容量，应丢最旧不报错
        hub.publish({"type": "span", "i": i})
    # 队列仍可取，且 recent 缓冲保留全部
    assert q.qsize() <= 2
    assert len(hub.recent(100)) == 6


def test_span_to_event_generation():
    span = {
        "kind": "generation", "name": "chat qwen-plus", "trace_id": "abc",
        "span_id": "s1", "parent_id": "p1", "start_time": 1.0, "duration_ms": 12.0,
        "status": "ok",
        "attributes": {
            "gen_ai.request.model": "qwen-plus",
            "gen_ai.usage.input_tokens": 100,
            "gen_ai.usage.output_tokens": 40,
        },
    }
    ev = dash.span_to_event(span, "alice")
    assert ev["type"] == "span"
    assert ev["user_id"] == "alice"
    assert ev["model"] == "qwen-plus"
    assert ev["input_tokens"] == 100
    assert ev["output_tokens"] == 40


def test_span_to_event_guardrail_fail_open():
    span = {
        "kind": "guardrail", "name": "guardrail judge", "trace_id": "abc",
        "span_id": "s2", "parent_id": "p1", "start_time": 1.0, "duration_ms": 5.0,
        "status": "ok",
        "attributes": {"milu.judge.fail_open": True,
                       "milu.judge.verdicts": [{"tool": "shell", "decision": "deny"}]},
    }
    ev = dash.span_to_event(span, "bob")
    assert ev["fail_open"] is True
    assert ev["verdicts"] == [{"tool": "shell", "decision": "deny"}]


# ── 跨用户枚举 ────────────────────────────────────────────

def test_enumerate_user_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("MILU_HOME", str(tmp_path))
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "alice.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "runs" / "runs.jsonl.migrated").write_text("x", encoding="utf-8")  # 应忽略
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "bob.json").write_text("[]", encoding="utf-8")
    (tmp_path / "schedules").mkdir()
    (tmp_path / "schedules" / "alice.json").write_text("{}", encoding="utf-8")
    (tmp_path / "knowledge" / "carol").mkdir(parents=True)
    users = dash._enumerate_user_ids()
    assert users == {"alice", "bob", "carol"}


def test_memory_count(tmp_path, monkeypatch):
    monkeypatch.setenv("MILU_HOME", str(tmp_path))
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "dave.json").write_text(
        json.dumps({"items": [{"content": "a"}, {"content": "b"}]}), encoding="utf-8")
    assert dash._memory_count("dave") == 2
    assert dash._memory_count("nobody") == 0


# ── 用户下钻 / 对话浏览（P4）──────────────────────────────

def test_content_text():
    assert dash._content_text("hi") == "hi"
    assert dash._content_text([{"type": "text", "text": "a"},
                               {"type": "image_path", "path": "x.png"}]) == "a\n[图片]"
    assert dash._content_text(None) == ""


def test_memory_items(tmp_path, monkeypatch):
    monkeypatch.setenv("MILU_HOME", str(tmp_path))
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "u.json").write_text(
        json.dumps({"items": [{"content": "fact", "category": "fact"}]}), encoding="utf-8")
    items = dash._memory_items("u")
    assert len(items) == 1 and items[0]["content"] == "fact"
    assert dash._memory_count("u") == 1


def test_session_messages_path_traversal_guarded(tmp_path, monkeypatch):
    monkeypatch.setenv("MILU_HOME", str(tmp_path))
    # 非法字符（路径穿越）→ None，绝不读盘
    assert dash._session_messages("../../etc/passwd") is None
    assert dash._session_messages("a/b") is None
    assert dash._session_messages("has space") is None
    assert dash._session_messages("") is None
    # 合法 id 但目录不存在 → None
    assert dash._session_messages("u__missing") is None


def test_session_messages_reads_conversation(tmp_path, monkeypatch):
    monkeypatch.setenv("MILU_HOME", str(tmp_path))
    from milu.agent.session import Session
    from milu.llm.base.message import Message, MessageRole
    base = tmp_path / "sessions"
    sess = Session("u__s1", base, model="qwen-plus")
    sess.log_message(Message(role=MessageRole.USER, content="你好"))
    sess.log_message(Message(role=MessageRole.ASSISTANT, content="你好，有什么可以帮你"))
    msgs = dash._session_messages("u__s1")
    assert msgs is not None and len(msgs) == 2
    assert msgs[0]["role"] == "user" and msgs[0]["content"] == "你好"
    assert msgs[1]["role"] == "assistant"


# ── 路由注册（构造期，不触发 lifespan）────────────────────

def test_dashboard_routes_registered():
    from milu.serving.web.app import create_app
    app = create_app(provider="qwen", use_scheduler=False, use_mcp=False)
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/dashboard" in paths
    assert "/api/dashboard/overview" in paths
    assert "/api/dashboard/stream" in paths
    assert "/api/dashboard/trace/{trace_id}" in paths
    assert "/api/dashboard/user/{uid}" in paths
    assert "/api/dashboard/session/{session_id}" in paths
