"""ScheduleEngine 多用户并发执行测试。

验证：并发执行错误隔离、同用户并发写无丢更新、任务超时、结果投递
（outbox + on_result）、agent_pool 注入路径、start_background 生命周期。
所有引擎实例 notify=False（测试中不弹系统通知）。
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from milu.agent.events import AgentDone
from milu.llm.base.response import TokenUsage
from milu.scheduler.engine import ScheduleEngine, SchedulerConfig
from milu.scheduler.store import ScheduleStore, ScheduleTask


def _past() -> str:
    return (datetime.now() - timedelta(minutes=5)).isoformat()


def _mk(name: str, user: str = "default", trigger: str = "once",
        interval: int = 10) -> ScheduleTask:
    """构造已到期任务（next_run 设为过去时间）。"""
    task = ScheduleTask.create(
        name=name,
        prompt="p",
        trigger_type=trigger,
        run_at=_past() if trigger == "once" else "",
        interval_minutes=interval if trigger == "interval" else 0,
        user_id=user,
    )
    task.next_run = _past()
    return task


class _FakeRunEngine(ScheduleEngine):
    """覆盖 _run_task：不走真实 LLM，可注入延迟与指定失败。"""

    def __init__(self, *args, delay: float = 0.0, fail_names: set | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.ran: list[str] = []
        self._delay = delay
        self._fail_names = fail_names or set()

    async def _run_task(self, task: ScheduleTask) -> str:
        if self._delay:
            await asyncio.sleep(self._delay)
        self.ran.append(task.name)
        if task.name in self._fail_names:
            raise RuntimeError("boom")
        return f"result-{task.name}"


def _engine(tmp_path: Path, store: ScheduleStore, **kwargs) -> _FakeRunEngine:
    config = kwargs.pop("config", None) or SchedulerConfig(notify=False)
    return _FakeRunEngine(
        store,
        config=config,
        outbox_dir=tmp_path / "outbox",
        **kwargs,
    )


def _read_outbox(tmp_path: Path, user: str) -> list[dict]:
    path = tmp_path / "outbox" / f"{user}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ── 并发执行 ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_concurrent_due_tasks_error_isolated(tmp_path: Path):
    """三个到期任务，中间一个失败，其余两个仍正常完成并删除。"""
    store = ScheduleStore(tmp_path)
    for name in ("t1", "t2", "t3"):
        store.add(_mk(name, "alice"))
    engine = _engine(tmp_path, store, fail_names={"t2"})

    await engine._tick()

    assert sorted(engine.ran) == ["t1", "t2", "t3"]
    remaining = {t.name for t in store.list_user("alice")}
    assert remaining == {"t2"}  # 成功的 once 已删除，失败的保留
    records = _read_outbox(tmp_path, "alice")
    assert {r["task"]: r["ok"] for r in records} == {
        "t1": True, "t2": False, "t3": True,
    }


@pytest.mark.asyncio
async def test_same_user_concurrent_update_no_lost_write(tmp_path: Path):
    """同用户两个 interval 任务并发执行，run_count/next_run 都正确落盘
    （per-user 锁防 load-save 丢更新）。"""
    store = ScheduleStore(tmp_path)
    store.add(_mk("i1", "alice", trigger="interval"))
    store.add(_mk("i2", "alice", trigger="interval"))
    engine = _engine(tmp_path, store, delay=0.05)

    await engine._tick()

    tasks = {t.name: t for t in store.list_user("alice")}
    assert len(tasks) == 2
    for name in ("i1", "i2"):
        assert tasks[name].run_count == 1, f"{name} 的更新被并发覆盖丢失"
        assert tasks[name].next_run is not None
        assert datetime.fromisoformat(tasks[name].next_run) > datetime.now()


@pytest.mark.asyncio
async def test_concurrency_capped_by_semaphore(tmp_path: Path):
    """并发任务数受 max_concurrent_tasks 限制。"""
    store = ScheduleStore(tmp_path)
    for i in range(4):
        store.add(_mk(f"t{i}", "alice"))

    peak = 0
    current = 0

    class _PeakEngine(ScheduleEngine):
        async def _run_task(self, task):
            nonlocal peak, current
            current += 1
            peak = max(peak, current)
            await asyncio.sleep(0.05)
            current -= 1
            return "ok"

    engine = _PeakEngine(
        store,
        config=SchedulerConfig(max_concurrent_tasks=2, notify=False),
        outbox_dir=tmp_path / "outbox",
    )
    await engine._tick()
    assert peak <= 2


@pytest.mark.asyncio
async def test_task_timeout(tmp_path: Path):
    """超时任务被 wait_for 中断，记为失败不挂死。"""
    store = ScheduleStore(tmp_path)
    store.add(_mk("slow", "alice"))
    engine = _engine(
        tmp_path, store, delay=0.5,
        config=SchedulerConfig(task_timeout=0.05, notify=False),
    )
    await engine._tick()
    records = _read_outbox(tmp_path, "alice")
    assert len(records) == 1 and records[0]["ok"] is False
    # 失败的 once 任务保留（未删除）
    assert store.get("slow", "alice") is not None


# ── 结果投递 ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_on_result_callback_invoked(tmp_path: Path):
    store = ScheduleStore(tmp_path)
    store.add(_mk("t", "alice"))
    received: list[tuple[str, str]] = []

    async def on_result(task, result):
        received.append((task.user_id, result))

    engine = _engine(tmp_path, store, on_result=on_result)
    await engine._tick()
    assert received == [("alice", "result-t")]


@pytest.mark.asyncio
async def test_on_result_failure_does_not_break(tmp_path: Path):
    """on_result 回调抛异常不影响任务完成与 outbox 投递。"""
    store = ScheduleStore(tmp_path)
    store.add(_mk("t", "alice"))

    async def bad_callback(task, result):
        raise RuntimeError("push failed")

    engine = _engine(tmp_path, store, on_result=bad_callback)
    await engine._tick()
    assert store.get("t", "alice") is None  # once 正常删除
    assert _read_outbox(tmp_path, "alice")[0]["ok"] is True


@pytest.mark.asyncio
async def test_outbox_per_user_isolated(tmp_path: Path):
    store = ScheduleStore(tmp_path)
    store.add(_mk("ta", "alice"))
    store.add(_mk("tb", "bob"))
    engine = _engine(tmp_path, store)
    await engine._tick()
    assert [r["task"] for r in _read_outbox(tmp_path, "alice")] == ["ta"]
    assert [r["task"] for r in _read_outbox(tmp_path, "bob")] == ["tb"]


# ── agent_pool 注入路径 ──────────────────────────────────

@pytest.mark.asyncio
async def test_engine_with_pool_uses_get_or_create(tmp_path: Path):
    """注入 agent_pool 后，任务经 pool.get_or_create_agent(user_id, ...) 执行。"""

    @dataclass
    class _FakeAgent:
        async def run(self, prompt):
            yield AgentDone(final_text="pool-result",
                            total_usage=TokenUsage(1, 1, 2), turn_count=1)

    @dataclass
    class _FakePool:
        calls: list = field(default_factory=list)

        async def get_or_create_agent(self, user_id, session_id):
            self.calls.append((user_id, session_id))
            return _FakeAgent()

    store = ScheduleStore(tmp_path)
    store.add(_mk("t", "alice"))
    pool = _FakePool()
    # 用基类（不覆盖 _run_task），验证真实的池分支
    engine = ScheduleEngine(
        store,
        config=SchedulerConfig(notify=False),
        agent_pool=pool,
        outbox_dir=tmp_path / "outbox",
    )
    await engine._tick()

    assert pool.calls == [("alice", "sched-t")]
    assert _read_outbox(tmp_path, "alice")[0]["result"] == "pool-result"


# ── 生命周期 ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_start_background_cancelable(tmp_path: Path):
    store = ScheduleStore(tmp_path)
    engine = _engine(tmp_path, store)
    task = engine.start_background()
    await asyncio.sleep(0.05)
    assert not task.done()
    engine.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# ── run_task_now 多用户 ──────────────────────────────────

@pytest.mark.asyncio
async def test_run_task_now_scoped_by_user(tmp_path: Path):
    store = ScheduleStore(tmp_path)
    store.add(_mk("t", "alice"))
    engine = _engine(tmp_path, store)
    with pytest.raises(ValueError):
        await engine.run_task_now("t", "bob")  # 跨用户不可见
    result = await engine.run_task_now("t", "alice")
    assert result == "result-t"
    assert store.get("t", "alice") is None  # once 执行后删除
