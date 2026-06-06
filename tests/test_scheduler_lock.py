"""SchedulerLock 单实例锁测试。

验证：acquire/release 基本流、活进程拒绝、stale 锁覆盖、损坏锁文件容错、
非持有者 release 为 no-op、pid_alive 跨平台检活、build_scheduler_engine 装配。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from milu.scheduler.lock import SchedulerLock, pid_alive


def test_acquire_release_basic(tmp_path: Path):
    lock = SchedulerLock(tmp_path)
    assert lock.try_acquire() is True
    assert lock.path.exists()
    assert lock.path.read_text(encoding="utf-8") == str(os.getpid())
    assert lock.holder_pid() == os.getpid()
    lock.release()
    assert not lock.path.exists()
    assert lock.holder_pid() == 0


def test_acquire_rejected_while_other_holder_alive(tmp_path: Path, monkeypatch):
    """他进程存活持锁时拒绝获取（用假 PID + 检活打桩模拟外部进程）。"""
    (tmp_path / "scheduler.lock").write_text("99999", encoding="utf-8")
    monkeypatch.setattr("milu.scheduler.lock.pid_alive", lambda pid: True)
    lock = SchedulerLock(tmp_path)
    assert lock.try_acquire() is False
    assert lock.holder_pid() == 99999


def test_try_acquire_reentrant_same_process(tmp_path: Path):
    """本进程已持有时重入幂等（探测与正式获取可分离调用，如 chat 线程形态）。"""
    a = SchedulerLock(tmp_path)
    assert a.try_acquire()
    b = SchedulerLock(tmp_path)
    assert b.try_acquire() is True  # 同进程视为已持有
    b.release()
    assert not a.path.exists()


def test_stale_lock_overridden(tmp_path: Path, monkeypatch):
    """持有者已死的 stale 锁可被覆盖。"""
    (tmp_path / "scheduler.lock").write_text("99999", encoding="utf-8")
    monkeypatch.setattr("milu.scheduler.lock.pid_alive", lambda pid: False)
    lock = SchedulerLock(tmp_path)
    assert lock.holder_pid() == 0
    assert lock.try_acquire() is True
    assert lock.path.read_text(encoding="utf-8") == str(os.getpid())


def test_corrupt_lock_file_treated_as_stale(tmp_path: Path):
    (tmp_path / "scheduler.lock").write_text("not-a-pid", encoding="utf-8")
    lock = SchedulerLock(tmp_path)
    assert lock.holder_pid() == 0
    assert lock.try_acquire() is True


def test_non_holder_release_is_noop(tmp_path: Path, monkeypatch):
    """拿不到锁的一方 release 不得误删持有者的锁文件。"""
    (tmp_path / "scheduler.lock").write_text("99999", encoding="utf-8")
    monkeypatch.setattr("milu.scheduler.lock.pid_alive", lambda pid: True)
    loser = SchedulerLock(tmp_path)
    assert loser.try_acquire() is False
    loser.release()  # no-op
    assert loser.path.exists()
    assert loser.holder_pid() == 99999


def test_pid_alive_self_and_dead_process():
    assert pid_alive(os.getpid()) is True
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert pid_alive(proc.pid) is False  # 已退出（PID 立即复用概率极低）


# ── build_scheduler_engine 装配（daemon 与 chat 嵌入共用）──

def test_build_scheduler_engine_assembly(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MILU_HOME", str(tmp_path))
    from milu.cli.builder import build_scheduler_engine

    engine, store, data_dir = build_scheduler_engine(echo=False)
    assert data_dir == tmp_path
    assert engine._echo is False           # chat/web 嵌入：静默
    assert engine._agent_pool is None      # CLI 形态：独立轻量 Agent
    assert engine._log_dir == tmp_path / "scheduler_logs"
    assert store.list_all() == []

    daemon_engine, _, _ = build_scheduler_engine(echo=True)
    assert daemon_engine._echo is True     # daemon 前台：控制台回显
