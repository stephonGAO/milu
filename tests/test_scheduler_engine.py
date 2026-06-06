"""ScheduleEngine 多用户并发执行测试。

验证：并发执行错误隔离、同用户并发写无丢更新、任务超时、结果投递
（outbox + on_result）、agent_pool 注入路径、start_background 生命周期、
韧性防护（tick 异常不杀主循环、echo 编码容错、后台任务死亡可见）。
所有引擎实例 notify=False（测试中不弹系统通知）。
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
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


# ── 韧性与回显（嵌入运行形态）─────────────────────────────

@pytest.mark.asyncio
async def test_safe_tick_swallows_exception_then_recovers(tmp_path: Path):
    """单次 tick 异常被吞掉（主循环不死），后续 tick 正常执行任务。"""
    store = ScheduleStore(tmp_path)
    store.add(_mk("t", "alice"))
    engine = _engine(tmp_path, store)
    orig_tick = engine._tick
    calls = {"n": 0}

    async def flaky_tick():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("tick boom")
        await orig_tick()

    engine._tick = flaky_tick
    await engine._safe_tick()  # 异常被吞，不抛出
    await engine._safe_tick()  # 恢复正常
    assert engine.ran == ["t"]


@pytest.mark.asyncio
async def test_echo_off_silent_but_outbox_delivered(tmp_path: Path, capsys):
    """echo=False（嵌入默认）：不写 stdout，任务仍正常执行并投递 outbox。"""
    store = ScheduleStore(tmp_path)
    store.add(_mk("t", "alice"))
    engine = _engine(tmp_path, store)  # 默认 echo=False
    await engine._tick()
    assert capsys.readouterr().out == ""
    assert _read_outbox(tmp_path, "alice")[0]["ok"] is True


def test_echo_print_tolerates_encode_error(tmp_path: Path, monkeypatch):
    """echo=True 时 stdout 编码失败（如 GBK 重定向遇 '▶'）降级重打，
    不抛异常——曾导致 Web 嵌入引擎任务静默不执行。"""
    store = ScheduleStore(tmp_path)
    engine = _engine(tmp_path, store, echo=True)
    calls: list[str] = []

    def gbk_print(text):
        if not calls:
            calls.append(text)
            raise UnicodeEncodeError(
                "gbk", str(text), 0, 1, "illegal multibyte sequence")
        calls.append(text)

    monkeypatch.setattr("builtins.print", gbk_print)
    engine._echo_print("[调度器] ▶ 执行任务: x")  # 不抛出
    assert len(calls) == 2  # 第一次失败，回退 errors='replace' 再打印


@pytest.mark.asyncio
async def test_invalid_task_disabled_with_visible_failure(tmp_path: Path):
    """配置无效的任务（once 缺 run_at）被自动禁用并投递可见失败。

    回归：曾静默跳过——任务「创建了却永不执行」且毫无线索（用户经 Web 表单
    创建 once 任务但 run_at 为空时实际撞上）。
    """
    store = ScheduleStore(tmp_path)
    bad = ScheduleTask.create(
        name="bad", prompt="p", trigger_type="once", run_at="",  # 永远算不出 next_run
        user_id="alice",
    )
    store.add(bad)
    engine = _engine(tmp_path, store)

    await engine._tick()

    saved = store.get("bad", "alice")
    assert saved is not None and saved.enabled is False  # 已自动禁用
    records = _read_outbox(tmp_path, "alice")
    assert len(records) == 1 and records[0]["ok"] is False
    assert "配置无效" in records[0]["result"]
    assert engine.ran == []  # 未被执行

    await engine._tick()  # 已禁用：不再重复投递
    assert len(_read_outbox(tmp_path, "alice")) == 1


@pytest.mark.asyncio
async def test_start_with_lock_takes_over_after_holder_exit(tmp_path: Path, monkeypatch):
    """锁被他进程占用时等待；持有者退出后自动接管并执行到期任务。

    回归：曾是「拿不到锁就永远不嵌入」——chat 与 serve 同开时，先开方
    退出后另一方无引擎，任务静默不执行。
    """
    from milu.scheduler.lock import SchedulerLock

    (tmp_path / "scheduler.lock").write_text("99999", encoding="utf-8")
    alive = {"v": True}
    monkeypatch.setattr(
        "milu.scheduler.lock.pid_alive",
        lambda pid: alive["v"] if pid == 99999 else True,
    )

    store = ScheduleStore(tmp_path)
    store.add(_mk("t", "alice"))
    engine = _engine(tmp_path, store)
    lock = SchedulerLock(tmp_path)
    task = asyncio.get_running_loop().create_task(
        engine.start_with_lock(lock, retry_seconds=0.05))

    await asyncio.sleep(0.15)
    assert engine.ran == []  # 锁被占用：等待中，不执行

    alive["v"] = False  # 持有者“退出”
    deadline = asyncio.get_running_loop().time() + 3.0
    while not engine.ran and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.05)
    assert engine.ran == ["t"], "持有者退出后未接管执行"

    engine.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert lock.holder_pid() == 0  # finally 中释放了自己的锁


@pytest.mark.asyncio
async def test_start_with_lock_stop_while_waiting(tmp_path: Path, monkeypatch):
    """等锁阶段 stop() 能自然退出循环，且不碰他人的锁。"""
    from milu.scheduler.lock import SchedulerLock

    (tmp_path / "scheduler.lock").write_text("99999", encoding="utf-8")
    monkeypatch.setattr("milu.scheduler.lock.pid_alive", lambda pid: True)

    store = ScheduleStore(tmp_path)
    engine = _engine(tmp_path, store)
    lock = SchedulerLock(tmp_path)
    task = asyncio.get_running_loop().create_task(
        engine.start_with_lock(lock, retry_seconds=0.05))
    await asyncio.sleep(0.12)
    engine.stop()
    await asyncio.wait_for(task, timeout=2.0)  # 无需 cancel，循环自然结束
    assert lock.holder_pid() == 99999  # 他人的锁原样保留


def test_engine_thread_runs_while_main_thread_blocked(tmp_path: Path):
    """模拟 chat 嵌入形态：引擎在独立线程的专属事件循环里运行，主线程
    同步阻塞（如 REPL 停在 input() 提示符）期间任务仍能执行。

    回归：曾用 start_background() 挂 REPL 主循环——input() 冻结事件循环，
    tick 永远不触发，任务静默不执行。
    """
    store = ScheduleStore(tmp_path)
    store.add(_mk("t", "alice"))
    engine = _engine(tmp_path, store)

    threading.Thread(
        target=lambda: asyncio.run(engine.start()), daemon=True
    ).start()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not engine.ran:
        time.sleep(0.05)  # 主线程纯同步阻塞，不跑任何事件循环
    engine.stop()

    assert engine.ran == ["t"], "主线程阻塞期间引擎线程未执行到期任务"
    assert _read_outbox(tmp_path, "alice")[0]["ok"] is True


@pytest.mark.asyncio
async def test_start_background_logs_fatal_exit(tmp_path: Path, caplog):
    """后台任务异常退出时记 error 日志（无人 await 它，必须可见）。"""
    import logging

    class _BoomEngine(ScheduleEngine):
        async def start(self):
            raise RuntimeError("boom")

    store = ScheduleStore(tmp_path)
    engine = _BoomEngine(store, config=SchedulerConfig(notify=False),
                         outbox_dir=tmp_path / "outbox")
    with caplog.at_level(logging.ERROR, logger="milu.scheduler.engine"):
        task = engine.start_background()
        with pytest.raises(RuntimeError):
            await task
        await asyncio.sleep(0)  # 让 done-callback 跑完
    assert any("调度后台任务异常退出" in r.message for r in caplog.records)


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
