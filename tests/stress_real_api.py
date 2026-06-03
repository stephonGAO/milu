"""真实 API 多用户并发压测 —— 验证 AgentPool 的并发吞吐、隔离与稳定性。

这是一个独立运行脚本（不随 pytest 默认集运行，因为它会真实调用 LLM、产生费用）。

运行：
    .venv/Scripts/python tests/stress_real_api.py

可选环境变量：
    STRESS_PROVIDER   提供商（默认 qwen）
    STRESS_MODEL      模型（默认 qwen-turbo）
    STRESS_USERS      并发用户数（默认 20）
    STRESS_REQS       每用户串行请求数（默认 2）
    STRESS_CONCURRENCY  全局并发上限 max_concurrent_runs（默认 10）
    STRESS_MAX_AGENTS   池容量 max_agents（默认 50）
    STRESS_MAX_TOKENS   每次回复 max_tokens（默认 24，省钱）

覆盖维度：
    Phase A  多用户并发吞吐 + 跨用户历史隔离
    Phase B  同一 (user, session) 并发 → 串行化、history 不污染（P0-1）
    Phase C  背压：acquire_timeout + 小并发上限 → PoolBusyError 优雅拒绝（P3）
    全程校验 AgentPool 硬不变量（实例数 ≤ max_agents 等）
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import statistics
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agent_framework.agent import Agent, AgentConfig
from agent_framework.agent.events import AgentDone, AgentError
from agent_framework.llm.base.message import Message, MessageRole
from agent_framework.llm.providers import ModelRegistry
from agent_framework.serving import AgentPool, AgentPoolConfig, PoolBusyError

PROVIDER = os.environ.get("STRESS_PROVIDER", "qwen")
MODEL = os.environ.get("STRESS_MODEL", "qwen-turbo")
N_USERS = int(os.environ.get("STRESS_USERS", "20"))
REQS = int(os.environ.get("STRESS_REQS", "2"))
CONCURRENCY = int(os.environ.get("STRESS_CONCURRENCY", "10"))
MAX_AGENTS = int(os.environ.get("STRESS_MAX_AGENTS", "50"))
MAX_TOKENS = int(os.environ.get("STRESS_MAX_TOKENS", "24"))

SYSTEM_PROMPT = "你是一个复读机：原样复述用户发来的内容，不要添加任何多余文字。"


def _sid(user_id: str, session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{user_id}__{session_id}")


def _make_factory(session_dir: str):
    def factory(user_id: str, session_id: str, llm):
        # 精简 Agent：无工具/无元工具/无技能 → 纯 LLM 往返，聚焦并发本身
        return Agent(
            llm=llm,
            system_prompt=SYSTEM_PROMPT,
            config=AgentConfig(
                session_enabled=True, session_dir=session_dir, max_turns=2
            ),
            session_id=_sid(user_id, session_id),
            register_catalog=False,
            register_skills=False,
        )
    return factory


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


async def _smoke(llm) -> None:
    parts = []
    async for c in llm.chat([Message(role=MessageRole.USER, content="只回复：OK")],
                            max_tokens=8):
        if c.content:
            parts.append(c.content)
        if c.finish_reason == "stop":
            break
    print(f"  冒烟通过：{''.join(parts)[:20]!r}")


async def _one_run(pool: AgentPool, user_id: str, text: str) -> dict:
    """一次请求：acquire + run，返回结果记录。"""
    t = time.monotonic()
    rec = {"user": user_id, "ok": False, "latency": 0.0, "tokens": 0,
           "error": None, "rejected": False}
    try:
        async with pool.acquire(user_id, "main") as h:
            async for ev in h.agent.run(text, max_tokens=MAX_TOKENS):
                if isinstance(ev, AgentDone):
                    rec["tokens"] = ev.total_usage.total_tokens
                elif isinstance(ev, AgentError):
                    rec["error"] = f"AgentError:{ev.message[:80]}"
        rec["ok"] = rec["error"] is None
    except PoolBusyError as e:
        rec["rejected"] = True
        rec["error"] = f"PoolBusy:{e}"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}:{str(e)[:100]}"
    rec["latency"] = time.monotonic() - t
    return rec


# ─────────────────────────────────────────────────────────────
# Phase A：多用户并发吞吐 + 跨用户历史隔离
# ─────────────────────────────────────────────────────────────
async def phase_a(pool: AgentPool) -> dict:
    print(f"\n[Phase A] {N_USERS} 用户并发 × 每用户 {REQS} 次串行请求 "
          f"（全局并发上限 {CONCURRENCY}）...")

    async def user_workload(i: int) -> list[dict]:
        uid = f"u{i}"
        recs = []
        for r in range(REQS):
            marker = f"{uid}#{r}-OK"   # 每用户独有前缀，用于隔离校验
            recs.append(await _one_run(pool, uid, marker))
        return recs

    t0 = time.monotonic()
    nested = await asyncio.gather(*[user_workload(i) for i in range(N_USERS)])
    wall = time.monotonic() - t0
    recs = [r for sub in nested for r in sub]

    # 跨用户隔离校验：每个用户的 history 里 USER 消息必须全是自己的前缀
    violations = []
    for i in range(N_USERS):
        uid = f"u{i}"
        async with pool.acquire(uid, "main") as h:
            for m in h.agent.history.all_messages:
                if m.role == MessageRole.USER and not m.content.startswith(f"{uid}#"):
                    violations.append((uid, m.content[:40]))

    oks = [r for r in recs if r["ok"]]
    errs = [r for r in recs if not r["ok"]]
    lats = [r["latency"] for r in oks]
    tokens = sum(r["tokens"] for r in oks)

    return {
        "total": len(recs), "ok": len(oks), "err": len(errs),
        "wall": wall, "throughput": len(recs) / wall if wall else 0,
        "p50": _pct(lats, 50), "p95": _pct(lats, 95),
        "max": max(lats) if lats else 0, "avg": statistics.mean(lats) if lats else 0,
        "tokens": tokens, "violations": violations,
        "errors": [r["error"] for r in errs][:5],
    }


# ─────────────────────────────────────────────────────────────
# Phase B：同一 (user, session) 并发 → 必须串行、history 不污染
# ─────────────────────────────────────────────────────────────
async def phase_b(pool: AgentPool) -> dict:
    K = 5
    print(f"\n[Phase B] 同一会话并发 {K} 个请求（应被串行化，history 不丢不乱）...")

    async def fire(i: int):
        return await _one_run(pool, "solo", f"solo#{i}-OK")

    recs = await asyncio.gather(*[fire(i) for i in range(K)])

    async with pool.acquire("solo", "main") as h:
        user_msgs = [m.content for m in h.agent.history.all_messages
                     if m.role == MessageRole.USER]

    return {
        "fired": K,
        "ok": sum(1 for r in recs if r["ok"]),
        "history_user_msgs": len(user_msgs),
        "distinct": len(set(user_msgs)),
        "intact": len(user_msgs) == K and len(set(user_msgs)) == K,
    }


# ─────────────────────────────────────────────────────────────
# Phase C：背压 —— 小并发上限 + acquire_timeout，过载时优雅拒绝
# ─────────────────────────────────────────────────────────────
async def phase_c(session_dir: str, llm) -> dict:
    M = 12
    print(f"\n[Phase C] 背压：max_concurrent_runs=2 + acquire_timeout=0.5s，"
          f"瞬时并发 {M}（部分应被 PoolBusyError 拒绝）...")
    pool = AgentPool(
        llm_factory=lambda u, s: llm,
        agent_factory=_make_factory(session_dir),
        config=AgentPoolConfig(max_agents=50, max_concurrent_runs=2,
                               acquire_timeout=0.5),
    )
    await pool.start()
    try:
        recs = await asyncio.gather(*[_one_run(pool, f"c{i}", f"c{i}#0-OK")
                                      for i in range(M)])
        st = pool.get_stats()
        return {
            "fired": M,
            "ok": sum(1 for r in recs if r["ok"]),
            "rejected": sum(1 for r in recs if r["rejected"]),
            "other_err": sum(1 for r in recs if not r["ok"] and not r["rejected"]),
            "rejected_busy_stat": st["rejected_busy"],
        }
    finally:
        await pool.stop()


async def main() -> None:
    print("=" * 64)
    print(f"  AgentPool 真实 API 并发压测  [{PROVIDER}/{MODEL}]")
    print(f"  users={N_USERS} reqs/user={REQS} concurrency={CONCURRENCY} "
          f"max_agents={MAX_AGENTS} max_tokens={MAX_TOKENS}")
    print("=" * 64)

    llm = ModelRegistry.create(PROVIDER, model=MODEL)
    print("[冒烟] 验证 API 可用...")
    await _smoke(llm)

    session_dir = tempfile.mkdtemp(prefix="stress_sessions_")
    pool = AgentPool(
        llm_factory=lambda u, s: llm,
        agent_factory=_make_factory(session_dir),
        config=AgentPoolConfig(max_agents=MAX_AGENTS,
                               max_concurrent_runs=CONCURRENCY),
    )
    await pool.start()

    inv_violations = []

    def check_invariants(tag: str):
        st = pool.get_stats()
        if st["active_entries"] > st["max_agents"]:
            inv_violations.append(f"{tag}: active_entries {st['active_entries']} > max_agents {st['max_agents']}")
        if st["in_flight"] > st["max_concurrent_runs"]:
            inv_violations.append(f"{tag}: in_flight {st['in_flight']} > max_concurrent_runs {st['max_concurrent_runs']}")

    try:
        a = await phase_a(pool)
        check_invariants("after A")
        b = await phase_b(pool)
        check_invariants("after B")
        stats_main = pool.get_stats()
    finally:
        await pool.stop()

    c = await phase_c(session_dir, llm)

    # 清理临时会话目录
    shutil.rmtree(session_dir, ignore_errors=True)

    # ── 报告 ──
    print("\n" + "=" * 64)
    print("  压测结果汇总")
    print("=" * 64)
    print(f"\n● Phase A（多用户并发吞吐 + 隔离）")
    print(f"    请求总数      : {a['total']}  (成功 {a['ok']} / 失败 {a['err']})")
    print(f"    成功率        : {a['ok'] / a['total'] * 100:.1f}%")
    print(f"    总耗时        : {a['wall']:.2f}s")
    print(f"    吞吐          : {a['throughput']:.2f} req/s")
    print(f"    时延 p50/p95  : {a['p50']:.2f}s / {a['p95']:.2f}s （avg {a['avg']:.2f}s, max {a['max']:.2f}s）")
    print(f"    总 tokens     : {a['tokens']}")
    print(f"    跨用户隔离违例: {len(a['violations'])}  （期望 0）")
    if a["violations"]:
        print(f"      例: {a['violations'][:3]}")
    if a["errors"]:
        print(f"    失败样例      : {a['errors']}")

    print(f"\n● Phase B（同会话并发串行化 / P0-1）")
    print(f"    并发请求      : {b['fired']}  (成功 {b['ok']})")
    print(f"    history 用户消息: {b['history_user_msgs']}  (互异 {b['distinct']})")
    print(f"    完整无污染    : {'✅ 是' if b['intact'] else '❌ 否'}  （期望 = {b['fired']} 条且互异）")

    print(f"\n● Phase C（背压 / P3）")
    print(f"    瞬时并发      : {c['fired']}")
    print(f"    成功          : {c['ok']}")
    print(f"    被拒(PoolBusy): {c['rejected']}  (池计数 rejected_busy={c['rejected_busy_stat']})")
    print(f"    其它错误      : {c['other_err']}")
    print(f"    优雅降级      : {'✅ 是' if c['other_err'] == 0 and (c['ok'] + c['rejected'] == c['fired']) else '⚠️ 见上'}")

    print(f"\n● AgentPool 不变量")
    print(f"    违例          : {len(inv_violations)}  （期望 0）")
    for v in inv_violations:
        print(f"      - {v}")

    print(f"\n● Phase A 结束时池指标快照")
    for k in ("created", "reused", "hit_rate", "completed_runs", "evicted_lru",
              "evicted_idle", "in_flight", "waiting", "rejected_busy",
              "rejected_full", "run_p50_ms", "run_p95_ms", "active_entries"):
        print(f"    {k:18}: {stats_main.get(k)}")

    # 总评
    healthy = (
        a["err"] == 0 and not a["violations"] and b["intact"]
        and c["other_err"] == 0 and not inv_violations
    )
    print("\n" + "=" * 64)
    print(f"  总体：{'✅ 全部健康' if healthy else '⚠️ 存在需关注项（见上）'}")
    print("=" * 64)


if __name__ == "__main__":
    asyncio.run(main())
