"""跨运行指标聚合 —— CLI `milu trace stats` 与 Web 观测大屏共用一套口径。

只做纯计算：输入是 RunReport.to_dict() 行列表（load_run_index 的返回），输出是
聚合字典 / 分组 / 时间序列。不涉及渲染、不读盘、无副作用，故 CLI 终端渲染与
大屏 SVG 图表都能复用同一份数字，避免两处口径漂移。
"""
from __future__ import annotations

from datetime import datetime, timedelta


def percentile(values: list[float], q: float) -> float:
    """近似百分位（q ∈ [0,1]）。values 无需预排序；空列表返回 0.0。"""
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[idx]


def summarize_runs(rows: list[dict]) -> dict:
    """把一批运行行聚合成 KPI 字典（成功率/token/成本/延迟分位/安全计数等）。

    成本按币种分桶（CNY/USD 不可相加）；价格缺失的运行不计入成本。
    """
    n = len(rows)
    durs = sorted((r.get("duration_ms") or 0) for r in rows)
    costs: dict[str, float] = {}
    for r in rows:
        if r.get("cost_estimate") is not None:
            cur = r.get("cost_currency") or "?"
            costs[cur] = round(costs.get(cur, 0.0) + float(r["cost_estimate"]), 6)
    ttfts = [r["ttft_ms"] for r in rows if r.get("ttft_ms") is not None]
    return {
        "runs": n,
        "ok": sum(1 for r in rows if r.get("status") == "ok"),
        "error": sum(1 for r in rows if r.get("status") == "error"),
        "interrupted": sum(1 for r in rows if r.get("status") == "interrupted"),
        "input_tokens": sum(r.get("input_tokens", 0) for r in rows),
        "cached_input_tokens": sum(r.get("cached_input_tokens", 0) for r in rows),
        "output_tokens": sum(r.get("output_tokens", 0) for r in rows),
        "reasoning_tokens": sum(r.get("reasoning_tokens", 0) for r in rows),
        "cost": costs,
        "p50_ms": percentile(durs, 0.5),
        "p95_ms": percentile(durs, 0.95),
        "avg_ms": round(sum(durs) / n, 1) if n else 0.0,
        "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 1) if ttfts else None,
        "tool_calls": sum(r.get("tool_calls", 0) for r in rows),
        "tool_errors": sum(r.get("tool_errors", 0) for r in rows),
        "judge_denies": sum(r.get("judge_denies", 0) for r in rows),
        "judge_confirms": sum(r.get("judge_confirms", 0) for r in rows),
        "fail_opens": sum(r.get("fail_opens", 0) for r in rows),
        "compactions": sum(r.get("compactions", 0) for r in rows),
        "subagent_runs": sum(r.get("subagent_runs", 0) for r in rows),
    }


def _day_key(r: dict) -> str:
    """运行所属日期键 "MM-DD"（与 CLI _fmt_time(ts)[:5] 同结果）。"""
    try:
        return datetime.fromtimestamp(r.get("started_at") or 0).strftime("%m-%d")
    except (TypeError, ValueError, OSError):
        return "?"


# 分组维度 → 取键函数（与 trace_cmd stats 的 group_key 对齐）
GROUPERS = {
    "model": lambda r: f"{r.get('provider', '?')}/{r.get('model') or '?'}",
    "provider": lambda r: r.get("provider") or "?",
    "mode": lambda r: r.get("mode") or "?",
    "user": lambda r: r.get("user_id") or "default",
    "status": lambda r: r.get("status") or "?",
    "day": _day_key,
}


def group_runs(rows: list[dict], by: str = "model") -> dict[str, list[dict]]:
    """按维度分组（未知维度回退 model）。"""
    keyfn = GROUPERS.get(by, GROUPERS["model"])
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(keyfn(r), []).append(r)
    return groups


def group_stats(rows: list[dict], by: str = "model") -> list[dict]:
    """分组聚合：每组一份 summarize_runs，附 key 字段，按 key 升序。"""
    return [{"key": k, **summarize_runs(rs)}
            for k, rs in sorted(group_runs(rows, by).items())]


def timeseries(rows: list[dict], bucket: str = "hour",
               buckets: int = 24, now_ts: float | None = None) -> list[dict]:
    """把运行按时间桶聚合（最近 `buckets` 个桶，旧→新）。

    :param bucket: "hour"（按整点）或 "day"（按自然日）。
    :param now_ts: 基准时刻（epoch 秒），默认当前时间——传入便于测试确定性。
    返回每桶 {ts, label, runs, ok, error, input_tokens, output_tokens}。
    """
    now = datetime.fromtimestamp(now_ts) if now_ts else datetime.now()
    if bucket == "day":
        cur = now.replace(hour=0, minute=0, second=0, microsecond=0)
        step = timedelta(days=1)
        fmt = "%m-%d"
    else:
        cur = now.replace(minute=0, second=0, microsecond=0)
        step = timedelta(hours=1)
        fmt = "%H:%M"
    starts = [cur - step * i for i in range(buckets - 1, -1, -1)]
    out = [{"ts": s.timestamp(), "label": s.strftime(fmt), "runs": 0, "ok": 0,
            "error": 0, "input_tokens": 0, "cached_input_tokens": 0,
            "output_tokens": 0} for s in starts]
    if not starts:
        return out
    start0 = starts[0].timestamp()
    step_s = step.total_seconds()
    for r in rows:
        ts = r.get("started_at") or 0
        if ts < start0:
            continue
        i = int((ts - start0) // step_s)
        if 0 <= i < buckets:
            b = out[i]
            b["runs"] += 1
            status = r.get("status")
            if status == "ok":
                b["ok"] += 1
            elif status == "error":
                b["error"] += 1
            b["input_tokens"] += r.get("input_tokens", 0)
            b["cached_input_tokens"] += r.get("cached_input_tokens", 0)
            b["output_tokens"] += r.get("output_tokens", 0)
    return out
