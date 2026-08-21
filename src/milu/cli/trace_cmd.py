"""`milu trace` 子命令 —— 运行追踪的终端查看（list / show / compare / stats）。

数据源：~/.milu/runs/{user}.jsonl 运行索引（按用户分文件，CLI 聚合全部）
+ 各运行的 trace.jsonl（RunReport.trace_path
定位）。只读，不依赖 LLM；trace_id 支持前缀匹配（list 输出的短 id 即可用）。
"""
from __future__ import annotations

import sys
from datetime import datetime

from milu.i18n import t
from milu.cli.render import DIVIDER, c
from milu.observability import load_run_index, load_trace

# 各 span 类型在树状图中的颜色
_KIND_COLOR = {
    "agent": "bold",
    "generation": "cyan",
    "tool": "yellow",
    "guardrail": "magenta",
    "confirmation": "magenta",
    "compaction": "blue",
}

_CURRENCY_SYMBOL = {"CNY": "¥", "USD": "$"}


# ── 格式化助手 ────────────────────────────────────────────

def _fmt_dur(ms) -> str:
    if ms is None:
        return "—"
    return f"{ms:.0f}ms" if ms < 1000 else f"{ms / 1000:.1f}s"


def _fmt_tokens(i, o) -> str:
    return f"{int(i or 0):,}→{int(o or 0):,}"


def _fmt_cache(tokens) -> str:
    count = int(tokens or 0)
    return f" {c('dim', f'(cache {count:,})')}" if count else ""


def _fmt_cost(amount, currency) -> str:
    if amount is None:
        return "—"
    return f"{_CURRENCY_SYMBOL.get(currency, (currency or '') + ' ')}{amount:.4f}"


def _fmt_time(ts) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return "?"


def _status_str(status, error_type=None) -> str:
    if status == "ok":
        return c("green", "ok")
    if status == "interrupted":
        return c("yellow", t("中断"))
    return c("red", f"error:{error_type}" if error_type else "error")


def _bar(dur_ms, total_ms, width: int = 10) -> str:
    """耗时占比条（相对根 span 总时长）。"""
    if not dur_ms or not total_ms:
        return ""
    n = max(1, min(width, round(dur_ms / total_ms * width)))
    return c("dim", "█" * n + "░" * (width - n))


def _find_run(tid_prefix: str) -> dict | None:
    """按 trace_id 前缀在运行索引中定位（取最新命中）。

    CLI 跨所有用户文件聚合查找（本机即真相，任何 id 都应可定位）。
    """
    for r in load_run_index(limit=0, all_users=True):
        if r.get("trace_id", "").startswith(tid_prefix):
            return r
    return None


# ── list ──────────────────────────────────────────────────

def _trace_list(args) -> int:
    # CLI 聚合所有用户文件（保持"看全部"语义；运行索引已按用户分文件、各自轮转）
    rows = load_run_index(limit=getattr(args, "n", 20) or 20, all_users=True)
    model_filter = getattr(args, "model", None)
    if model_filter:
        rows = [r for r in rows if model_filter in (r.get("model") or "")]
    if not rows:
        print(c("dim", t("暂无运行记录（开启 observability 后运行一次对话即有）")))
        return 0
    print(f"{DIVIDER}\n  " + t("近期运行 ({n} 条)", n=len(rows)) + f"\n{DIVIDER}")
    for r in rows:
        print(
            f"  {c('cyan', r['trace_id'][:12])}  {c('dim', _fmt_time(r.get('started_at')))}  "
            f"{r.get('provider', '?')}/{r.get('model') or '?'}  {c('dim', r.get('mode', ''))}  "
            f"{_status_str(r.get('status'), r.get('error_type'))}  "
            f"{_fmt_dur(r.get('duration_ms'))}  "
            f"{_fmt_tokens(r.get('input_tokens'), r.get('output_tokens'))} tok"
            f"{_fmt_cache(r.get('cached_input_tokens'))}  "
            f"{_fmt_cost(r.get('cost_estimate'), r.get('cost_currency'))}  "
            + t("{t} 轮 / {k} 工具", t=r.get('turn_count', 0), k=r.get('tool_calls', 0))
        )
    print(c("dim", "\n  " + t("详情：milu trace show <trace_id 前缀>")))
    return 0


# ── show ──────────────────────────────────────────────────

def _span_line(s: dict, total_ms) -> str:
    a = s.get("attributes", {})
    kind = s.get("kind", "")
    rnd = a.get("milu.round")
    rnd_str = c("dim", f"#{rnd}") + "  " if rnd else ""
    bar = _bar(s.get("duration_ms"), total_ms)
    dur = _fmt_dur(s.get("duration_ms"))
    name = c(_KIND_COLOR.get(kind, "cyan"), s.get("name", "?"))

    if kind == "generation":
        ttfc = a.get("gen_ai.response.time_to_first_chunk")
        ttfc_str = f"  TTFT {ttfc * 1000:.0f}ms" if ttfc is not None else ""
        fin = a.get("gen_ai.response.finish_reasons")
        fin_str = c("dim", f"  →{fin[0]}") if fin else ""
        return (f"{name}  {rnd_str}{bar} {dur}{ttfc_str}  "
                f"{_fmt_tokens(a.get('gen_ai.usage.input_tokens'), a.get('gen_ai.usage.output_tokens'))} tok"
                f"{_fmt_cache(a.get('milu.cached_input_tokens'))}{fin_str}")
    if kind == "tool":
        status = c("red", "err") if s.get("status") == "error" else c("green", "ok")
        size = a.get("milu.output_len")
        size_str = c("dim", f"  {size}B") if size is not None else ""
        return f"{name}  {rnd_str}{bar} {dur}  {status}{size_str}"
    if kind == "guardrail":
        verdicts = a.get("milu.judge.verdicts") or []
        if a.get("milu.judge.fail_open"):
            v_str = c("red", t("判定失败 → fail-open 放行"))
        else:
            v_str = "; ".join(
                f"{v.get('tool')}→{v.get('decision')}"
                + (f"（{v.get('reason')}）" if v.get("reason") else "")
                for v in verdicts
            )
        return f"{name}  {rnd_str}{bar} {dur}  {v_str}"
    if kind == "confirmation":
        decision = a.get("milu.decision", "?")
        color = "green" if decision == "approved" else "red"
        return f"{name}  {rnd_str}" + t("等待 {d}", d=dur) + f" → {c(color, decision)}"
    if kind == "compaction":
        return (f"{name}  {a.get('milu.compact.strategy', '?')}  "
                + t("{b} → {a} 条消息", b=a.get('milu.compact.before', '?'),
                    a=a.get('milu.compact.after', '?')))
    if kind == "agent":
        return (f"{name}  {a.get('gen_ai.provider.name', '?')}/{a.get('gen_ai.request.model') or '?'}  "
                f"{c('dim', a.get('milu.mode', ''))}  {dur}  "
                f"{_fmt_tokens(a.get('gen_ai.usage.input_tokens'), a.get('gen_ai.usage.output_tokens'))} tok"
                f"{_fmt_cache(a.get('milu.cached_input_tokens'))}  "
                + t("{n} 轮", n=a.get('milu.turn_count', '?')))
    return f"{name}  {bar} {dur}"


def _render_tree(spans: list[dict], node: dict, children: dict,
                 total_ms, prefix: str = "", is_last: bool = True) -> None:
    elbow = "" if not prefix and node.get("parent_id") is None else ("└─ " if is_last else "├─ ")
    print(f"  {prefix}{elbow}{_span_line(node, total_ms)}")
    kids = sorted(children.get(node["span_id"], []), key=lambda s: s.get("start_time", 0))
    child_prefix = prefix + ("" if not prefix and node.get("parent_id") is None
                             else ("   " if is_last else "│  "))
    for i, k in enumerate(kids):
        _render_tree(spans, k, children, total_ms, child_prefix, i == len(kids) - 1)


def _trace_show(args) -> int:
    run = _find_run(args.trace_id)
    if run is None:
        print(c("red", t("未找到运行: {id}（trace_id 支持前缀匹配）", id=args.trace_id)),
              file=sys.stderr)
        return 1
    trace_path = run.get("trace_path")
    # 按 trace_id 过滤：会话级 trace.jsonl 累积同会话所有轮次，须只取本次运行
    spans = load_trace(trace_path, run.get("trace_id")) if trace_path else []
    if not spans:
        print(c("red", t("trace 文件缺失或为空: {p}", p=trace_path)), file=sys.stderr)
        return 1

    children: dict[str, list[dict]] = {}
    roots = []
    by_id = {s["span_id"]: s for s in spans}
    for s in spans:
        pid = s.get("parent_id")
        if pid and pid in by_id:
            children.setdefault(pid, []).append(s)
        else:
            roots.append(s)

    print(f"{DIVIDER}\n  trace {c('cyan', run['trace_id'])}  "
          f"{c('dim', _fmt_time(run.get('started_at')))}  "
          f"{_status_str(run.get('status'), run.get('error_type'))}  "
          + t("成本 {x}", x=_fmt_cost(run.get('cost_estimate'), run.get('cost_currency')))
          + f"\n{DIVIDER}")
    total_ms = max((r.get("duration_ms") or 0) for r in roots) or None
    for r in sorted(roots, key=lambda s: s.get("start_time", 0)):
        _render_tree(spans, r, children, total_ms)
    # 时间分解（"时间花在哪"）
    print(c("dim",
            "\n  " + t("时间分解：LLM {a} / 工具 {b} / 安全判定 {c} / 审批等待 {d}",
                       a=_fmt_dur(run.get("llm_time_ms")),
                       b=_fmt_dur(run.get("tool_time_ms")),
                       c=_fmt_dur(run.get("judge_time_ms")),
                       d=_fmt_dur(run.get("wait_time_ms")))))
    if run.get("fail_opens"):
        print(c("red", "  " + t("⚠ 本次运行发生 {n} 次安全判定 fail-open（判定失败放行）",
                                n=run["fail_opens"])))
    return 0


# ── compare ───────────────────────────────────────────────

_COMPARE_ROWS = [
    ("provider/model", lambda r: f"{r.get('provider', '?')}/{r.get('model') or '?'}"),
    (t("状态"), lambda r: r.get("status", "?")),
    (t("耗时"), lambda r: _fmt_dur(r.get("duration_ms"))),
    ("TTFT", lambda r: _fmt_dur(r.get("ttft_ms"))),
    (t("轮数"), lambda r: str(r.get("turn_count", 0))),
    ("input tok", lambda r: f"{r.get('input_tokens', 0):,}"),
    ("cached tok", lambda r: f"{r.get('cached_input_tokens', 0):,}"),
    ("output tok", lambda r: f"{r.get('output_tokens', 0):,}"),
    (t("成本"), lambda r: _fmt_cost(r.get("cost_estimate"), r.get("cost_currency"))),
    (t("工具调用"), lambda r: f"{r.get('tool_calls', 0)} ({r.get('tool_errors', 0)} err)"),
    ("LLM 时间", lambda r: _fmt_dur(r.get("llm_time_ms"))),
    (t("工具时间"), lambda r: _fmt_dur(r.get("tool_time_ms"))),
    (t("子代理"), lambda r: str(r.get("subagent_runs", 0))),
]


def _display_width(s: str) -> int:
    """终端显示宽度：CJK 全角字符按 2 列计（含 ANSI 码剥离）。"""
    import unicodedata
    plain = _strip_ansi(s)
    return sum(2 if unicodedata.east_asian_width(ch) in "FW" else 1 for ch in plain)


def _pad(s: str, width: int) -> str:
    """按显示宽度右补空格（ljust 不识别 ANSI 码与全角字符）。"""
    return s + " " * max(0, width - _display_width(s))


def _trace_compare(args) -> int:
    runs = []
    for tid in args.trace_ids:
        run = _find_run(tid)
        if run is None:
            print(c("red", t("未找到运行: {id}", id=tid)), file=sys.stderr)
            return 1
        runs.append(run)
    label_w = 16
    col_w = max(22, *(len(f"{r.get('provider')}/{r.get('model')}") + 2 for r in runs))
    header = " " * (label_w + 2) + "".join(_pad(c("cyan", r["trace_id"][:12]), col_w) for r in runs)
    print(f"{DIVIDER}\n{header}\n{DIVIDER}")
    for label, fn in _COMPARE_ROWS:
        cells = "".join(_pad(fn(r), col_w) for r in runs)
        print(f"  {_pad(label, label_w)}{cells}")
    return 0


def _strip_ansi(s: str) -> str:
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# ── stats ─────────────────────────────────────────────────

def _trace_stats(args) -> int:
    from milu.observability.analytics import group_stats

    rows = load_run_index(limit=0, all_users=True)
    if not rows:
        print(c("dim", t("暂无运行记录")))
        return 0
    by = getattr(args, "by", None) or "model"
    # 与大屏共用同一套聚合口径（group_stats 内部 summarize_runs）
    print(f"{DIVIDER}\n  " + t("运行统计（共 {n} 次，按 {by} 分组）", n=len(rows), by=by)
          + f"\n{DIVIDER}")
    for g in group_stats(rows, by):
        cost_str = " + ".join(_fmt_cost(v, k) for k, v in g["cost"].items()) or "—"
        line = (
            f"  {_pad(c('cyan', g['key']), 38)} "
            + t("{n} 次", n=g["runs"]) + f"  ok {g['ok']}/{g['runs']}  "
            f"p50 {_fmt_dur(g['p50_ms'])}  p95 {_fmt_dur(g['p95_ms'])}  "
            f"{_fmt_tokens(g['input_tokens'], g['output_tokens'])} tok  {cost_str}"
        )
        if g["fail_opens"]:
            line += c("red", f"  fail-open×{g['fail_opens']}")
        print(line)
    return 0


# ── 分发 ──────────────────────────────────────────────────

def cmd_trace(args) -> int:
    action = getattr(args, "trace_action", None) or "list"
    return {
        "list": _trace_list,
        "show": _trace_show,
        "compare": _trace_compare,
        "stats": _trace_stats,
    }[action](args)
