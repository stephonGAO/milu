"""milu 数据中心观测大屏 —— 跨用户全局监控的后端（路由 + 实时枢纽 + 聚合）。

与现有单用户对话 SPA（`/`）正交：大屏是**管理员/运维视角**，跨所有用户俯瞰
运行指标、资源池、Agent 链路、安全审计、定时任务与每用户数据画像。

三块能力：
- **DashboardHub**：进程级实时事件枢纽。每个 Agent 的 TraceConfig.extra_sinks
  注入一个 CallbackSink，span 完成即推入枢纽，再经 `/api/dashboard/stream`(SSE)
  广播给大屏；调度任务结果也走同一枢纽。环形缓冲 + 订阅队列，满则丢最旧保活。
- **跨用户聚合**：复用 observability.analytics 的口径，叠加资源池实时态、
  会话/记忆/知识库/定时任务的每用户 rollup（短 TTL 缓存，避免轮询打爆磁盘）。
- **鉴权**：跨用户读全局数据须门控。设 `MILU_ADMIN_TOKEN` 后校验令牌
  （header `X-Admin-Token` / `Authorization: Bearer` / 查询参数 `token`）；
  未设置时默认**仅 localhost** 可访问。敏感正文（对话/记忆）默认不进聚合，
  仅在显式下钻（trace span / 会话加载）时按需返回。
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import secrets
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from milu.observability import CallbackSink

logger = logging.getLogger("milu.serving.web.dashboard")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# 本机回环地址（未配置 admin token 时仅放行这些来源）
_LOCAL_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"})


# ── 实时事件枢纽 ──────────────────────────────────────────

class DashboardHub:
    """进程级实时事件枢纽：扇出 span / 调度结果给所有大屏 SSE 订阅者。

    - `publish()` 同步、非阻塞（在事件循环内由 tracer sink / 调度回调调用）；
      订阅队列满则丢最旧再塞，宁可丢历史也不阻塞业务路径。
    - `subscribe()/unsubscribe()` 管理订阅队列；SSE 协程用 try/finally 配对。
    - 环形缓冲保留最近事件，新订阅者连上即可回放近况（避免空屏）。
    """

    def __init__(self, buffer_size: int = 300, queue_size: int = 256):
        self._recent: deque[dict] = deque(maxlen=buffer_size)
        self._subscribers: set[asyncio.Queue] = set()
        self._queue_size = queue_size
        self._seq = 0

    def publish(self, event: dict) -> None:
        self._seq += 1
        event = {**event, "seq": self._seq}
        self._recent.append(event)
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # 丢最旧一条再塞新的（保实时性）
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except Exception:
                    pass

    def recent(self, limit: int = 100) -> list[dict]:
        items = list(self._recent)
        return items[-limit:] if limit else items

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


def span_to_event(span: dict, user_id: str | None) -> dict:
    """把一条已完成 span（to_dict）压成大屏事件流的精简事件。

    只取展示需要的字段，避免把可能含截断正文的全量 attributes 推给前端。
    """
    attrs = span.get("attributes", {}) or {}
    kind = span.get("kind")
    ev: dict[str, Any] = {
        "type": "span",
        "kind": kind,
        "name": span.get("name"),
        "user_id": user_id,
        "trace_id": span.get("trace_id"),
        "span_id": span.get("span_id"),
        "parent_id": span.get("parent_id"),
        "ts": span.get("start_time"),
        "duration_ms": span.get("duration_ms"),
        "status": span.get("status"),
    }
    if kind == "generation":
        ev["model"] = attrs.get("gen_ai.request.model")
        ev["input_tokens"] = attrs.get("gen_ai.usage.input_tokens")
        ev["output_tokens"] = attrs.get("gen_ai.usage.output_tokens")
    elif kind == "tool":
        ev["tool"] = attrs.get("gen_ai.tool.name")
    elif kind == "guardrail":
        ev["fail_open"] = bool(attrs.get("milu.judge.fail_open"))
        ev["verdicts"] = [
            {"tool": v.get("tool"), "decision": v.get("decision")}
            for v in (attrs.get("milu.judge.verdicts") or [])
        ]
    elif kind == "confirmation":
        ev["decision"] = attrs.get("milu.decision")
    elif kind == "agent":
        ev["provider"] = attrs.get("gen_ai.provider.name")
        ev["model"] = attrs.get("gen_ai.request.model")
        ev["mode"] = attrs.get("milu.mode")
        ev["turns"] = attrs.get("milu.turn_count")
    return ev


def make_span_sink(hub: DashboardHub, user_id: str | None) -> CallbackSink:
    """为某 Agent 造一个把 span 推进枢纽的 sink（user_id 绑定在闭包里）。"""
    def _emit(span) -> None:
        try:
            hub.publish(span_to_event(span.to_dict(), user_id))
        except Exception:  # 观测增强，绝不影响业务（与 tracer fail-silent 一致）
            pass
    return CallbackSink(_emit)


# ── 鉴权 ──────────────────────────────────────────────────

def _admin_token() -> str | None:
    tok = os.environ.get("MILU_ADMIN_TOKEN")
    tok = tok.strip() if tok else ""
    return tok or None


def _client_is_local(request: Request) -> bool:
    client = request.client
    host = client.host if client else ""
    if host in _LOCAL_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _extract_token(request: Request) -> str | None:
    h = request.headers.get("X-Admin-Token")
    if h:
        return h.strip()
    auth = request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    q = request.query_params.get("token")
    return q.strip() if q else None


def check_admin(request: Request) -> None:
    """大屏鉴权门控：未配置 token → 仅本机；配置了 → 校验令牌（常量时间比较）。"""
    token = _admin_token()
    if token is None:
        if _client_is_local(request):
            return
        raise HTTPException(
            403, "观测大屏跨用户读取全局数据，远程访问需先设置 MILU_ADMIN_TOKEN")
    provided = _extract_token(request)
    if provided is not None and secrets.compare_digest(provided, token):
        return
    raise HTTPException(401, "需要有效的管理员令牌（X-Admin-Token / Bearer / ?token=）")


# ── 短 TTL 缓存（避免 5s 轮询打爆磁盘扫描）────────────────

def _cached(state, key: str, ttl: float, fn: Callable[[], Any]) -> Any:
    cache = getattr(state, "_dash_cache", None)
    if cache is None:
        cache = state._dash_cache = {}
    now = time.monotonic()
    hit = cache.get(key)
    if hit is not None and hit[0] > now:
        return hit[1]
    val = fn()
    cache[key] = (now + ttl, val)
    return val


# ── 跨用户枚举与画像 ──────────────────────────────────────

def _enumerate_user_ids() -> set[str]:
    """从磁盘各用户级数据目录归并出已知用户（文件名即 safe_user_id）。"""
    from milu.resources import user_data_dir
    base = user_data_dir()
    users: set[str] = set()
    for sub, pat in (("runs", "*.jsonl"), ("memory", "*.json"),
                     ("schedules", "*.json")):
        d = base / sub
        if d.is_dir():
            for f in d.glob(pat):
                if f.suffix == ".migrated" or f.name.endswith(".migrated"):
                    continue
                users.add(f.stem)
    kd = base / "knowledge"
    if kd.is_dir():
        for d in kd.iterdir():
            if d.is_dir():
                users.add(d.name)
    return users


def _memory_count(uid: str) -> int:
    from milu.tools.builtin.memory_tool import memory_file_path
    p = memory_file_path(uid)
    if not p.exists():
        return 0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    items = data if isinstance(data, list) else data.get("items", [])
    return len(items)


def _kb_rollup(uid: str) -> dict:
    try:
        from milu.knowledge import KnowledgeStore, knowledge_dir
        d = knowledge_dir(uid)
        if not d.is_dir():
            return {"chunks": 0, "sources": 0, "model": ""}
        s = KnowledgeStore(d).stats()
        return {"chunks": s.get("count", 0), "sources": s.get("sources", 0),
                "model": s.get("model", "")}
    except Exception:
        return {"chunks": 0, "sources": 0, "model": ""}


def _session_rollup(uid: str) -> dict:
    from milu.agent.session import Session
    from milu.resources import default_session_dir
    from milu.serving import AgentPool
    prefix = AgentPool._derive_session_id(uid, "")
    sessions = [s for s in Session.list_sessions(default_session_dir())
                if str(s.get("session_id", "")).startswith(prefix)]
    return {
        "count": len(sessions),
        "messages": sum(int(s.get("message_count", 0) or 0) for s in sessions),
        "updated_at": max((s.get("updated_at", 0) or 0 for s in sessions), default=0),
    }


def _build_users(state) -> list[dict]:
    """每用户画像：运行/token/成本/会话/消息/记忆/知识库/定时任务/在线态。"""
    from milu.observability import load_run_index, summarize_runs
    online = {e["user_id"] for e in state.pool.list_active_sessions()} \
        if getattr(state, "pool", None) else set()
    store = getattr(state, "store", None)
    out: list[dict] = []
    for uid in sorted(_enumerate_user_ids()):
        runs = load_run_index(limit=0, user_id=uid)
        s = summarize_runs(runs)
        sess = _session_rollup(uid)
        kb = _kb_rollup(uid)
        tasks = store.list_user(uid) if store else []
        out.append({
            "user_id": uid,
            "online": uid in online,
            "runs": s["runs"],
            "ok": s["ok"],
            "error": s["error"],
            "input_tokens": s["input_tokens"],
            "output_tokens": s["output_tokens"],
            "cost": s["cost"],
            "p50_ms": s["p50_ms"],
            "fail_opens": s["fail_opens"],
            "last_run": runs[0].get("started_at") if runs else None,
            "sessions": sess["count"],
            "messages": sess["messages"],
            "memory": _memory_count(uid),
            "knowledge": kb,
            "tasks": len(tasks),
            "tasks_enabled": sum(1 for t in tasks if t.enabled),
            "last_active": max(
                (runs[0].get("started_at") or 0) if runs else 0,
                sess["updated_at"] or 0,
            ),
        })
    out.sort(key=lambda u: u.get("last_active") or 0, reverse=True)
    return out


def _providers_status() -> list[dict]:
    from milu.cli.config import DEFAULT_MODELS, env_key_name
    from milu.llm.providers import ModelRegistry
    items = []
    for name in ModelRegistry.list_providers():
        env_name = env_key_name(name)
        items.append({
            "name": name,
            "default_model": DEFAULT_MODELS.get(name, ""),
            "env_key": env_name,
            "key_configured": bool(os.environ.get(env_name)),
        })
    return items


def _today_start() -> float:
    from datetime import datetime
    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def _parse_range(span: str) -> tuple[str, int, float]:
    """把 "24h"/"7d"/"30d" 解析成 (bucket, buckets, cutoff_ts)。"""
    span = (span or "24h").strip().lower()
    now = time.time()
    try:
        if span.endswith("d"):
            n = max(1, int(span[:-1] or 7))
            return "day", n, now - n * 86400
        n = max(1, int(span[:-1] or 24)) if span.endswith("h") else 24
        return "hour", n, now - n * 3600
    except ValueError:
        return "hour", 24, now - 24 * 3600


# ── 路由注册 ──────────────────────────────────────────────

def register_dashboard_routes(app) -> None:
    """把大屏页面与 `/api/dashboard/*` 端点挂到既有 FastAPI app 上（全部经鉴权）。"""

    @app.get("/dashboard")
    async def dashboard_page(request: Request):
        check_admin(request)
        return FileResponse(
            STATIC_DIR / "dashboard.html",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    @app.get("/api/dashboard/overview")
    async def dash_overview(request: Request):
        check_admin(request)
        from milu.observability import load_run_index, summarize_runs
        st = request.app.state
        rows = load_run_index(limit=0, all_users=True)
        today_cut = _today_start()
        today_rows = [r for r in rows if (r.get("started_at") or 0) >= today_cut]
        store = getattr(st, "store", None)
        tasks = store.list_all() if store else []
        pool = getattr(st, "pool", None)
        return {
            "server_time": time.time(),
            "observability_enabled": bool(
                getattr(st, "observability_cfg", {}).get("enabled")),
            "pool": pool.get_stats() if pool else {},
            "active_sessions": pool.list_active_sessions() if pool else [],
            "summary_all": summarize_runs(rows),
            "summary_today": summarize_runs(today_rows),
            "users_count": len(_enumerate_user_ids()),
            "providers": _providers_status(),
            "scheduler": {
                "tasks_total": len(tasks),
                "enabled": sum(1 for t in tasks if t.enabled),
                "recent_results": len(getattr(st, "recent_results", []) or []),
            },
        }

    @app.get("/api/dashboard/pool")
    async def dash_pool(request: Request):
        check_admin(request)
        pool = getattr(request.app.state, "pool", None)
        if pool is None:
            return {"stats": {}, "active_sessions": []}
        return {"stats": pool.get_stats(),
                "active_sessions": pool.list_active_sessions()}

    @app.get("/api/dashboard/users")
    async def dash_users(request: Request):
        check_admin(request)
        st = request.app.state
        return {"users": _cached(st, "users", 4.0, lambda: _build_users(st))}

    @app.get("/api/dashboard/runs")
    async def dash_runs(request: Request, limit: int = 50, user: str = "",
                        model: str = "", status: str = ""):
        check_admin(request)
        from milu.observability import load_run_index
        rows = load_run_index(limit=0, all_users=True)
        if user:
            rows = [r for r in rows if (r.get("user_id") or "default") == user]
        if model:
            rows = [r for r in rows if model in (r.get("model") or "")]
        if status:
            rows = [r for r in rows if r.get("status") == status]
        return {"runs": rows[:limit] if limit else rows}

    @app.get("/api/dashboard/trace/{trace_id}")
    async def dash_trace(request: Request, trace_id: str):
        check_admin(request)
        from milu.observability import load_run_index, load_trace
        run = next((r for r in load_run_index(limit=0, all_users=True)
                    if r.get("trace_id", "").startswith(trace_id)), None)
        if run is None or not run.get("trace_path"):
            return {"run": None, "spans": []}
        return {"run": run, "spans": load_trace(run["trace_path"], run["trace_id"])}

    @app.get("/api/dashboard/analytics")
    async def dash_analytics(request: Request, range: str = "24h", by: str = "model"):
        check_admin(request)
        from milu.observability import group_stats, load_run_index, timeseries
        rows = load_run_index(limit=0, all_users=True)
        bucket, buckets, cutoff = _parse_range(range)
        rows_in = [r for r in rows if (r.get("started_at") or 0) >= cutoff]
        return {
            "range": range,
            "bucket": bucket,
            "series": timeseries(rows_in, bucket, buckets),
            "by_model": group_stats(rows_in, "model"),
            "by_provider": group_stats(rows_in, "provider"),
            "by": by,
            "groups": group_stats(rows_in, by),
        }

    @app.get("/api/dashboard/audit")
    async def dash_audit(request: Request):
        check_admin(request)
        from milu.observability import group_stats, load_run_index, summarize_runs
        rows = load_run_index(limit=0, all_users=True)
        s = summarize_runs(rows)
        by_mode = [{"key": m["key"], "runs": m["runs"]}
                   for m in group_stats(rows, "mode")]
        return {
            "summary": s,
            "by_mode": by_mode,
            "fail_opens": s["fail_opens"],
            "judge_denies": s["judge_denies"],
            "judge_confirms": s["judge_confirms"],
            "tool_calls": s["tool_calls"],
            "tool_errors": s["tool_errors"],
        }

    @app.get("/api/dashboard/scheduler")
    async def dash_scheduler(request: Request):
        check_admin(request)
        st = request.app.state
        store = getattr(st, "store", None)
        tasks = store.list_all() if store else []
        recent = list(reversed(getattr(st, "recent_results", []) or []))[:50]
        return {"tasks": [asdict(t) for t in tasks], "recent_results": recent}

    @app.get("/api/dashboard/stream")
    async def dash_stream(request: Request):
        check_admin(request)
        hub: DashboardHub | None = getattr(request.app.state, "dashboard_hub", None)
        if hub is None:
            raise HTTPException(503, "事件枢纽未就绪")

        async def gen():
            q = hub.subscribe()
            try:
                for ev in hub.recent(50):
                    yield {"event": "feed", "data": json.dumps(ev, ensure_ascii=False)}
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        continue  # 让 EventSourceResponse 的 ping 保活，回环检查断开
                    yield {"event": "feed", "data": json.dumps(ev, ensure_ascii=False)}
            finally:
                hub.unsubscribe(q)

        return EventSourceResponse(gen(), ping=15)
