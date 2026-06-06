"""定时任务调度引擎（asyncio 主循环）。

每分钟检查一次到期任务，多用户任务并发执行（Semaphore 限流 + per-user
写锁），结果投递三通道：outbox JSONL（按用户）→ on_result 回调（服务端
推送）→ 系统弹窗（CLI 本机）。

三种运行形态（共用 lock.py 的 SchedulerLock 单实例锁，杜绝多引擎重复执行）：
  - CLI 守护进程：`milu scheduler start`（前台 start()，echo=True 控制台回显）
  - 嵌入 Web 服务：FastAPI lifespan 中 `engine.start_background()`，配合
    agent_pool 注入获得 per-user 实例复用 / 共享 MCP / memory 派生
  - 嵌入 CLI chat：`run_chat` 起独立 daemon 线程跑 `asyncio.run(start())`，
    对话期间自动执行、退出即停（echo=False 不污染 REPL，结果走
    outbox/弹窗/日志）。⚠️ 不能用 start_background() 挂 REPL 主循环——
    input() 同步阻塞主线程事件循环，tick 永远得不到调度
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable

from milu.scheduler.notify import send as notify_send
from milu.scheduler.store import ScheduleStore, ScheduleTask, _safe_user

logger = logging.getLogger(__name__)


# 跨进程 outbox append 锁（与 session.py 同档取舍）：POSIX 用 flock，
# Windows 优雅降级为 no-op（单进程内 append 本身同步不交错）。
try:
    import fcntl  # POSIX

    def _lock_file(f) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

    def _unlock_file(f) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
except ImportError:  # 非 POSIX（如 Windows）
    def _lock_file(f) -> None:
        pass

    def _unlock_file(f) -> None:
        pass


@dataclass
class SchedulerConfig:
    """调度引擎可调参数（config.json 的 scheduler 分节；比照 AgentPoolConfig 放引擎文件内）。"""
    max_concurrent_tasks: int = 4    # 单次 tick 内并发执行的任务数上限
    task_timeout: float = 300.0      # 单任务执行超时（秒）
    notify: bool = True              # 执行结果是否弹系统通知（服务端建议关闭）


def compute_next_run(task: ScheduleTask, after: datetime) -> datetime | None:
    """计算任务的下次运行时间（纯函数）。"""
    if task.trigger_type == "cron":
        try:
            from croniter import croniter
            return croniter(task.cron, after).get_next(datetime)
        except ImportError:
            logger.error("cron 触发类型需要安装 croniter：pip install croniter")
            return None
        except Exception as e:
            logger.error("cron 表达式 '%s' 解析失败: %s", task.cron, e)
            return None

    if task.trigger_type == "interval":
        if task.interval_minutes <= 0:
            return None
        from datetime import timedelta
        return after + timedelta(minutes=task.interval_minutes)

    if task.trigger_type == "once":
        # run_at 字段即触发时间
        if task.run_at:
            try:
                return datetime.fromisoformat(task.run_at)
            except ValueError:
                return None
        return None

    return None


class ScheduleEngine:
    """定时任务调度引擎：扫描全部用户的任务、按时并发触发、投递结果。"""

    def __init__(
        self,
        store: ScheduleStore,
        log_dir: Path | None = None,
        config: SchedulerConfig | None = None,
        agent_pool=None,
        on_result: "Callable[[ScheduleTask, str], Awaitable[None]] | None" = None,
        outbox_dir: Path | None = None,
        echo: bool = False,
    ):
        """
        :param store: 任务存储（按用户分文件）
        :param log_dir: 执行日志目录（None 不写日志文件）
        :param config: 调度参数（None → 默认值）
        :param agent_pool: AgentPool 实例（可选）。注入后任务经
            pool.get_or_create_agent(user_id, ...) 执行——per-user 实例复用、
            共享 MCP、memory/schedule_user 自动派生，且不占在线并发许可；
            不注入则每任务自建独立轻量 Agent（CLI 守护进程模式）
        :param on_result: 任务完成后的异步回调 (task, result_text)（可选），
            服务端可用于 SSE/WebSocket 推送或写库
        :param outbox_dir: 结果投递目录；None → user_data_dir()/scheduler_outbox
        :param echo: 是否向 stdout 回显进度（仅 CLI 守护进程前台模式开；
            嵌入 chat/Web 服务时保持 False——既不污染 REPL，也避免 stdout
            被重定向为 GBK 等编码时 UnicodeEncodeError 杀死任务执行）。
            运行形态差异由调用方代码决定，故为构造参数而非 SchedulerConfig 项
        """
        self._store = store
        self._log_dir = log_dir
        self._config = config or SchedulerConfig()
        self._agent_pool = agent_pool
        self._on_result = on_result
        self._echo = echo
        if outbox_dir is None:
            from milu.resources import user_data_dir
            outbox_dir = user_data_dir() / "scheduler_outbox"
        self._outbox_dir = Path(outbox_dir)
        self._running = False
        # 同一用户任务文件的写操作串行化（进程内并发防丢更新；
        # 与 AgentPool per-key 锁同思路）
        self._user_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _echo_print(self, text: str) -> None:
        """安全回显：echo 关闭时静默；开启时打印并容错编码错误。

        stdout 被重定向/管道时编码可能是 GBK 等（Windows 常见），'▶'/'✓'
        会抛 UnicodeEncodeError——容错降级为 errors='replace'，绝不让一次
        打印失败杀死任务执行（曾导致 Web 嵌入引擎任务静默不执行）。
        """
        if not self._echo:
            return
        try:
            print(text)
        except UnicodeEncodeError:
            enc = getattr(sys.stdout, "encoding", None) or "utf-8"
            print(text.encode(enc, errors="replace").decode(enc))

    async def _safe_tick(self) -> None:
        """一次 tick 的异常防护：单次失败记日志，不杀主循环。"""
        try:
            await self._tick()
        except Exception:
            logger.exception("调度 tick 异常（主循环继续）")

    async def start(self) -> None:
        """启动调度主循环（前台阻塞，Ctrl+C 停止）。"""
        self._running = True
        self._echo_print("调度器已启动，每分钟检查一次任务... 按 Ctrl+C 停止")
        logger.info("调度器启动")

        # 立刻做一次检查（初始化 next_run）
        await self._safe_tick()

        while self._running:
            # 睡到下一分钟整点（+2s 避免边界竞争）
            now = datetime.now()
            sleep_sec = 60 - now.second + 2
            await asyncio.sleep(sleep_sec)
            if self._running:
                await self._safe_tick()

    async def start_with_lock(self, lock, retry_seconds: float = 30.0) -> None:
        """持单实例锁运行：抢到锁即启动主循环，抢不到则周期重试，持锁者
        退出后**自动接管**（如 chat 与 serve 同开，先开方退出后另一方接管，
        任务不会因锁竞争而静默不执行）。

        stop() 在两个阶段都生效：等锁阶段下一轮检查即退出；运行阶段同 start()。
        锁在本方法 finally 中释放（等锁中被取消不碰他人的锁，release 有持有者守卫）。

        :param lock: SchedulerLock 实例
        :param retry_seconds: 等锁重试间隔（秒）
        """
        self._running = True
        waiting_logged = False
        while self._running:
            if lock.try_acquire():
                logger.info("已获得调度单实例锁，引擎启动")
                try:
                    await self.start()
                finally:
                    lock.release()
                return
            if not waiting_logged:
                logger.info("调度锁被占用（PID %s），任务由该进程执行；本引擎等待接管",
                            lock.holder_pid())
                waiting_logged = True
            await asyncio.sleep(retry_seconds)

    def start_background(self, lock=None) -> "asyncio.Task[None]":
        """以后台任务方式启动（嵌入 FastAPI / CLI chat 等进程）。

        返回的 Task 可 cancel() 停止；调用方负责生命周期（finally 中
        cancel + 吞 CancelledError）。任务异常退出时记 error 日志——
        没人 await 它，不加 done-callback 的话异常完全静默。

        :param lock: 传入 SchedulerLock 时走 start_with_lock（单实例 + 自动接管）
        """
        coro = self.start_with_lock(lock) if lock is not None else self.start()
        task = asyncio.get_running_loop().create_task(coro)

        def _log_death(t: "asyncio.Task[None]") -> None:
            if not t.cancelled() and t.exception() is not None:
                logger.error("调度后台任务异常退出", exc_info=t.exception())

        task.add_done_callback(_log_death)
        return task

    def stop(self) -> None:
        self._running = False

    async def _tick(self) -> None:
        """检查并并发执行所有到期任务（跨全部用户）。"""
        now = datetime.now()
        tasks = self._store.list_all()
        due = []

        for task in tasks:
            if not task.enabled:
                continue

            # 首次运行：初始化 next_run
            if task.next_run is None:
                next_dt = compute_next_run(task, now)
                if next_dt is None:
                    # 配置无效（once 缺 run_at / cron 表达式错误 / interval≤0）永远
                    # 算不出下次运行——静默跳过会让任务「创建了却永不执行」毫无
                    # 线索，故自动禁用并投递可见失败（outbox/前端提醒/系统弹窗）
                    task.enabled = False
                    async with self._user_locks[task.user_id]:
                        self._store.update(task)
                    msg = (f"任务配置无效，无法计算下次运行时间（{task.trigger_desc()}），"
                           f"已自动禁用。请修正触发参数后重新启用或重建任务。")
                    logger.warning("任务 '%s'（用户 %s）%s", task.name, task.user_id, msg)
                    await self._deliver(task, now, ok=False, result=msg)
                    continue
                task.next_run = next_dt.isoformat()
                async with self._user_locks[task.user_id]:
                    self._store.update(task)
                # 初始化后立刻检查是否已到期（如启动时 run_at 已过）
                if next_dt <= now:
                    due.append(task)
                else:
                    self._echo_print(f"  [调度器] '{task.name}' 下次运行: {task.next_run[:16]}")
                continue

            try:
                next_dt = datetime.fromisoformat(task.next_run)
            except ValueError:
                continue

            if next_dt <= now:
                due.append(task)

        if not due:
            return

        # 并发执行（Semaphore 限流；gather 错误隔离，单任务异常不污染整批）
        sem = asyncio.Semaphore(self._config.max_concurrent_tasks)

        async def _guarded(t: ScheduleTask) -> None:
            async with sem:
                await self._execute(t)

        await asyncio.gather(*(_guarded(t) for t in due), return_exceptions=True)

    async def _execute(self, task: ScheduleTask) -> None:
        """执行单个任务，更新 last_run / next_run / run_count，投递结果。"""
        now = datetime.now()
        self._echo_print(f"\n[调度器] ▶ 执行任务: {task.name}（用户: {task.user_id}）")
        logger.info("执行任务: %s（用户: %s）", task.name, task.user_id)
        try:
            result = await asyncio.wait_for(
                self._run_task(task), timeout=self._config.task_timeout
            )
            task.last_run = now.isoformat()
            task.run_count += 1

            # store 写操作按用户加锁（同用户多任务并发时防 load-save 丢更新）
            async with self._user_locks[task.user_id]:
                if task.trigger_type == "once":
                    self._store.remove(task.name, task.user_id)
                    self._echo_print(f"[调度器] ✓ '{task.name}' 执行完成，已删除")
                else:
                    next_dt = compute_next_run(task, now)
                    task.next_run = next_dt.isoformat() if next_dt else None
                    self._store.update(task)
                    self._echo_print(f"[调度器] ✓ '{task.name}' 执行完成（第 {task.run_count} 次）")
                    if task.next_run:
                        self._echo_print(f"[调度器]   下次运行: {task.next_run[:16]}")

            if self._log_dir:
                self._write_log(task, now, result)

            await self._deliver(task, now, ok=True, result=result)

        except Exception as e:
            logger.error("任务 '%s' 执行失败: %s\n%s", task.name, e, traceback.format_exc())
            self._echo_print(f"[调度器] ✗ '{task.name}' 执行失败: {e}")
            self._echo_print(traceback.format_exc())
            await self._deliver(task, now, ok=False, result=str(e))

    async def run_task_now(self, name: str, user_id: str = "default") -> str:
        """立刻同步执行指定任务（CLI 使用），返回最终文本。"""
        task = self._store.get(name, user_id)
        if not task:
            raise ValueError(f"任务 '{name}' 不存在")
        result = await self._run_task(task)
        now = datetime.now()
        task.last_run = now.isoformat()
        task.run_count += 1
        async with self._user_locks[task.user_id]:
            if task.trigger_type == "once":
                self._store.remove(task.name, task.user_id)
            else:
                next_dt = compute_next_run(task, now)
                task.next_run = next_dt.isoformat() if next_dt else None
                self._store.update(task)
        return result

    async def _run_task(self, task: ScheduleTask) -> str:
        """执行 task.prompt，返回最终回答文本。"""
        from milu.agent.events import AgentDone

        # 经池执行：per-user 实例复用、共享 MCP、memory/schedule_user 自动派生；
        # get_or_create_agent 不占在线并发许可（专为后台任务设计，见 pool.py）
        if self._agent_pool is not None:
            agent = await self._agent_pool.get_or_create_agent(
                task.user_id, f"sched-{task.name}"
            )
            final_text = ""
            async for ev in agent.run(task.prompt):
                if isinstance(ev, AgentDone):
                    final_text = ev.final_text
            return final_text

        # 独立轻量 Agent（CLI 守护进程模式）
        from milu._env import ensure_dotenv_loaded
        from milu.cli.builder import build_llm
        from milu.cli.config import resolve_settings
        from milu.config import load_config

        ensure_dotenv_loaded()
        config = load_config()

        # 用 task 指定的 provider/model 覆盖（getattr 兼容 resolve_settings 的 args 接口）
        class _Args:
            provider = task.provider or None
            model = task.model or None
            api_key = None
            mode = "auto"
            no_session = True
            no_mcp = True
            no_subagents = True
            session = None

        settings = resolve_settings(config, _Args())
        llm = build_llm(settings)

        from milu.agent.agent import Agent
        # schedule_user 透传：任务内若再调 schedule 工具，按本用户隔离
        agent = Agent(
            llm,
            session_enabled=False,
            subagents=[],
            schedule_user=task.user_id,
        )

        final_text = ""
        async for ev in agent.run(task.prompt):
            if isinstance(ev, AgentDone):
                final_text = ev.final_text

        return final_text

    # ── 结果投递 ─────────────────────────────────────────

    async def _deliver(
        self, task: ScheduleTask, run_time: datetime, ok: bool, result: str
    ) -> None:
        """三通道投递：outbox JSONL → on_result 回调 → 系统弹窗。各通道互不影响。"""
        self._write_outbox(task, run_time, ok, result)

        if ok and self._on_result is not None:
            try:
                await self._on_result(task, result)
            except Exception as e:
                logger.warning("on_result 回调失败: %s", e)

        if self._config.notify:
            if ok:
                summary = (result[:80] + "…") if len(result) > 80 else result
                notify_send(f"milu · {task.name}", summary or task.description or task.prompt)
            else:
                notify_send(f"milu · {task.name} 执行失败", result[:120])

    def _write_outbox(
        self, task: ScheduleTask, run_time: datetime, ok: bool, result: str
    ) -> None:
        """按用户追加结果到 outbox JSONL（带锁 append，多进程不交错）。"""
        try:
            self._outbox_dir.mkdir(parents=True, exist_ok=True)
            path = self._outbox_dir / f"{_safe_user(task.user_id)}.jsonl"
            line = json.dumps(
                {
                    "task": task.name,
                    "user_id": task.user_id,
                    "time": run_time.isoformat(),
                    "ok": ok,
                    "result": result,
                },
                ensure_ascii=False,
            ) + "\n"
            with open(path, "a", encoding="utf-8") as f:
                _lock_file(f)
                try:
                    f.write(line)
                    f.flush()
                finally:
                    _unlock_file(f)
        except OSError as e:
            logger.warning("写入 outbox 失败: %s", e)

    def _write_log(self, task: ScheduleTask, run_time: datetime, result: str) -> None:
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            ts = run_time.strftime("%Y%m%d_%H%M%S")
            log_path = self._log_dir / f"{_safe_user(task.user_id)}__{task.name}_{ts}.txt"
            log_path.write_text(
                f"任务: {task.name}\n"
                f"用户: {task.user_id}\n"
                f"执行时间: {run_time.isoformat()}\n"
                f"Prompt: {task.prompt}\n\n"
                f"结果:\n{result}",
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("写入执行日志失败: %s", e)
