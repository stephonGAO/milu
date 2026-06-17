"""RunReport —— 单次运行的指标聚合（可测量、可比较的载体）。

顶层运行结束时由 Tracer 收集到的 span 树聚合而成，追加一行到
**按用户分文件**的运行索引 ~/.milu/runs/{safe_user_id}.jsonl，供 CLI
`milu trace list/compare/stats` 与 Web 观测面板消费。

为什么按用户分文件（与 memory/scheduler/knowledge 同款 {safe_user_id} 约定）：
- Web 多用户场景只读自己那份，不必扫描他人数据（隔离 + 性能）；
- 每份各自轮转上限（RUNS_INDEX_MAX_LINES），杜绝单个全局文件无限增长——
  这是 milu 中唯一全局、永久追加的 JSONL，长期运行会越读越慢，故加 LRU 式
  截断（每次 append 后若超上限即丢弃最旧行、原子重写）。
旧版单文件 ~/.milu/runs.jsonl 首次访问自动按 user_id 拆分迁移（源文件改名
.migrated 保留可回退），与 scheduler 的迁移同款。

口径说明：
- token 合计 = 全部 generation span 之和（**含子代理**的 LLM 调用；
  judge 判定调用不经 Agent 主循环，其 token 暂不计入——见 guardrail span 时长）；
- 时间分解（llm/tool/judge/wait）只统计**根 span 直接子级**：子代理内部的
  LLM 时间已包含在其 execute_tool 包装 span 时长内，避免重复计入；
- cost 为估算（pricing.py 双轨制），价格缺失 = None，绝不瞎算。
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from milu.observability.pricing import estimate_cost
from milu.observability.span import Span

logger = logging.getLogger(__name__)

# 每个用户的运行索引保留的最近运行条数上限（超出丢弃最旧行）。
# 5000 条 ≈ 数千次对话历史，足够本地观测；文件大小因此封顶在数 MB。
RUNS_INDEX_MAX_LINES = 5000


@dataclass
class RunReport:
    """一次顶层 Agent.run() 的汇总指标（runs.jsonl 一行）。"""

    trace_id: str
    started_at: float
    provider: str
    model: str
    mode: str
    session_id: str | None
    user_id: str | None
    status: str                      # "ok" | "error" | "interrupted"
    error_type: str | None
    duration_ms: float
    turn_count: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cost_estimate: float | None     # 价格缺失 = None（展示层标"未配置价格"）
    cost_currency: str | None
    ttft_ms: float | None           # 首轮首 token 延迟
    llm_time_ms: float              # 时间分解：根 span 直接子级口径
    tool_time_ms: float
    judge_time_ms: float
    wait_time_ms: float             # blocked_on_user 人工审批等待
    tool_calls: int
    tool_errors: int
    judge_denies: int
    judge_confirms: int
    fail_opens: int                 # 判定器失败 fail-open 次数（最需被审计的路径）
    compactions: int
    subagent_runs: int
    trace_path: str | None = None   # trace.jsonl 所在路径（show/下钻定位用）

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _safe_user(user_id: str | None) -> str:
    """user_id 文件系统安全化（与 memory/scheduler 同一规则）。"""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", (user_id or "default").strip() or "default")[:64]


def _runs_dir() -> Path:
    from milu.resources import user_data_dir
    return user_data_dir() / "runs"


def _runs_index_path(user_id: str | None = None) -> Path:
    """某用户的运行索引文件路径（user_id 为 None → default.jsonl，CLI 单人）。"""
    return _runs_dir() / f"{_safe_user(user_id)}.jsonl"


def _atomic_write_text(path: Path, text: str) -> None:
    """原子写文本（同目录临时文件 + fsync + os.replace，仿 knowledge/scheduler）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _trim_runs(path: Path, max_lines: int) -> None:
    """超过上限时只保留最近 max_lines 行（原子重写）。

    在每次 append 之后调用：追加本身是 append-only（崩溃安全），仅当行数
    超限才整读 + 原子重写截断。read 成本被上限封顶（稳态 ≈ max_lines 行），
    且 append 每轮对话才一次（夹在 LLM 调用之间），其耗时可忽略。
    """
    if max_lines <= 0:
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        if len(lines) <= max_lines:
            return
        _atomic_write_text(path, "".join(lines[-max_lines:]))
    except OSError as e:
        logger.warning("运行索引轮转失败（已忽略）: %s", e)


def _migrate_legacy_runs() -> None:
    """旧版单文件 ~/.milu/runs.jsonl → runs/{safe_user_id}.jsonl（按 user_id 拆分，幂等）。

    源文件改名 .migrated 保留（可回退）。损坏行跳过；user_id 缺失归入 default。
    迁移即按上限截断各用户文件，避免迁移后首次 append 读超大文件。
    """
    from milu.resources import user_data_dir
    legacy = user_data_dir() / "runs.jsonl"
    if not legacy.exists():
        return
    buckets: dict[str, list[str]] = {}
    try:
        with open(legacy, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    uid = json.loads(line).get("user_id")
                except ValueError:
                    continue
                buckets.setdefault(_safe_user(uid), []).append(
                    line if line.endswith("\n") else line + "\n")
        runs_dir = _runs_dir()
        runs_dir.mkdir(parents=True, exist_ok=True)
        for safe, lines in buckets.items():
            dest = runs_dir / f"{safe}.jsonl"
            # 追加到（可能已存在的）目标，再截断到上限
            with open(dest, "a", encoding="utf-8") as f:
                f.writelines(lines)
            _trim_runs(dest, RUNS_INDEX_MAX_LINES)
        # 提交点：改名旧文件，后续不再触发迁移（os.replace 原子，并发只一个成功）
        os.replace(legacy, legacy.with_name("runs.jsonl.migrated"))
    except OSError as e:
        logger.warning("旧版运行索引迁移失败（已忽略）: %s", e)


def build_run_report(
    tracer,
    root_span: Span,
    user_id: str | None = None,
    trace_path: "Path | str | None" = None,
) -> RunReport:
    """从已完成的 span 树聚合 RunReport（root_span 须已 finish）。"""
    spans: list[Span] = tracer.finished
    attrs = root_span.attributes
    gens = [s for s in spans if s.kind == "generation"]
    tools = [s for s in spans if s.kind == "tool"]
    guards = [s for s in spans if s.kind == "guardrail"]
    agents = [s for s in spans if s.kind == "agent"]
    top = {k: [s for s in spans if s.parent_id == root_span.span_id and s.kind == k]
           for k in ("generation", "tool", "guardrail", "confirmation")}

    def _sum_tokens(key: str) -> int:
        return sum(int(s.attributes.get(key, 0) or 0) for s in gens)

    input_tokens = _sum_tokens("gen_ai.usage.input_tokens")
    output_tokens = _sum_tokens("gen_ai.usage.output_tokens")
    reasoning_tokens = _sum_tokens("milu.reasoning_tokens")

    # TTFT：最早开始的 generation span 的首 chunk 延迟（秒 → 毫秒）
    ttft_ms = None
    for s in sorted(gens, key=lambda s: s.start_monotonic):
        v = s.attributes.get("gen_ai.response.time_to_first_chunk")
        if v is not None:
            ttft_ms = round(float(v) * 1000, 1)
            break

    def _dur(spans_: list[Span]) -> float:
        return round(sum(s.duration_ms or 0 for s in spans_), 1)

    denies = confirms = fail_opens = 0
    for g in guards:
        if g.attributes.get("milu.judge.fail_open"):
            fail_opens += 1
        for v in g.attributes.get("milu.judge.verdicts", []) or []:
            d = v.get("decision")
            if d == "deny":
                denies += 1
            elif d == "confirm":
                confirms += 1

    model = str(attrs.get("gen_ai.request.model", "") or "")
    cost = attrs.get("milu.cost_estimate")
    currency = attrs.get("milu.cost_currency")
    if cost is None:
        est = estimate_cost(model, input_tokens, output_tokens,
                            getattr(tracer.config, "price_table", None))
        if est is not None:
            cost, currency = est

    return RunReport(
        trace_id=root_span.trace_id,
        started_at=root_span.start_time,
        provider=str(attrs.get("gen_ai.provider.name", "") or ""),
        model=model,
        mode=str(attrs.get("milu.mode", "") or ""),
        session_id=attrs.get("gen_ai.conversation.id"),
        user_id=user_id,
        status=root_span.status,
        error_type=attrs.get("error.type"),
        duration_ms=root_span.duration_ms or 0.0,
        turn_count=int(attrs.get("milu.turn_count", 0) or 0),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cost_estimate=cost,
        cost_currency=currency if cost is not None else None,
        ttft_ms=ttft_ms,
        llm_time_ms=_dur(top["generation"]),
        tool_time_ms=_dur(top["tool"]),
        judge_time_ms=_dur(top["guardrail"]),
        wait_time_ms=_dur(top["confirmation"]),
        tool_calls=len(tools),
        tool_errors=sum(1 for s in tools if s.status == "error"),
        judge_denies=denies,
        judge_confirms=confirms,
        fail_opens=fail_opens,
        compactions=sum(1 for s in spans if s.kind == "compaction"),
        subagent_runs=max(len(agents) - 1, 0),
        trace_path=str(trace_path) if trace_path else None,
    )


def append_run_index(
    report: RunReport,
    path: "Path | str | None" = None,
    max_lines: int = RUNS_INDEX_MAX_LINES,
) -> None:
    """把 RunReport 追加到该用户的运行索引（一行 JSON），并按上限轮转。

    path 显式给定时直接用（测试/程序化）；否则按 report.user_id 落到
    runs/{safe_user_id}.jsonl（user_id 为 None → default.jsonl，CLI 单人）。
    失败由调用方统一 fail-silent。
    """
    if path is not None:
        p = Path(path)
    else:
        _migrate_legacy_runs()
        p = _runs_index_path(report.user_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(report.to_dict(), ensure_ascii=False, default=str) + "\n")
    _trim_runs(p, max_lines)


def _read_runs_file(p: Path) -> list[dict]:
    """读单个运行索引文件的所有行（损坏行跳过）。"""
    if not p.exists():
        return []
    rows: list[dict] = []
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError as e:
        logger.warning("读取运行索引失败: %s", e)
    return rows


def load_run_index(
    limit: int = 50,
    path: "Path | str | None" = None,
    user_id: str | None = None,
    all_users: bool = False,
) -> list[dict]:
    """读取运行索引（最新在前，按 started_at 降序）。损坏行跳过。

    数据源（三选一）：
    - path 显式给定 → 只读该文件（测试/程序化）；
    - all_users=True → 聚合 runs/ 下所有用户文件（CLI：本机即真相，看全部）；
    - user_id 给定 → 只读该用户文件（Web：按 X-User-Id 隔离，不扫他人）；
    - 都没有 → default.jsonl（CLI 单人）。
    user_id 同时给定时再按其过滤行（path/all_users 场景下生效）。
    """
    _migrate_legacy_runs()
    if path is not None:
        # 显式文件：可能含多用户行（旧格式/测试），按 user_id 二次过滤
        rows = _read_runs_file(Path(path))
        if user_id is not None:
            rows = [r for r in rows if r.get("user_id") == user_id]
    elif all_users:
        rows = []
        d = _runs_dir()
        if d.is_dir():
            for f in d.glob("*.jsonl"):
                rows.extend(_read_runs_file(f))
        if user_id is not None:
            rows = [r for r in rows if r.get("user_id") == user_id]
    else:
        # 按用户文件读取：文件本身即该用户分区（含迁移来的 user_id=None 旧行、
        # CLI 的 default 行），不再二次过滤——与 scheduler/memory 读分区同语义
        rows = _read_runs_file(_runs_index_path(user_id))
    rows.sort(key=lambda r: r.get("started_at") or 0, reverse=True)
    return rows[:limit] if limit else rows


def load_trace(trace_path: "Path | str", trace_id: str | None = None) -> list[dict]:
    """读取 trace.jsonl 的 span 行（损坏行跳过）。

    :param trace_id: 给定时只返回该 trace 的 span。**会话级 trace.jsonl 是
        append-only、累积同一会话所有轮次的 span（每轮一个独立 trace_id）**，
        故查看单次运行必须按 trace_id 过滤，否则会把历史轮次的 span 一并渲染。
    """
    p = Path(trace_path)
    if not p.exists():
        return []
    spans: list[dict] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                span = json.loads(line)
            except ValueError:
                continue
            if trace_id is not None and span.get("trace_id") != trace_id:
                continue
            spans.append(span)
    return spans


def cleanup_old_traces(retention_days: int, traces_dir: "Path | str | None" = None) -> int:
    """清理 ~/.milu/traces 下过期散件（无 session 运行的 trace；跟随 session 的
    trace.jsonl 随会话目录生命周期，不在此清理）。返回删除文件数。"""
    import time as _time
    if retention_days <= 0:
        return 0
    if traces_dir is None:
        from milu.resources import user_data_dir
        traces_dir = user_data_dir() / "traces"
    d = Path(traces_dir)
    if not d.is_dir():
        return 0
    cutoff = _time.time() - retention_days * 86400
    removed = 0
    for f in d.glob("*.jsonl"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            continue
    return removed
