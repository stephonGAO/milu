"""定时任务调度引擎（asyncio 主循环）。

每分钟检查一次到期任务，异步执行，输出日志到 ~/.milu/scheduler_logs/。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from milu.scheduler.store import ScheduleStore, ScheduleTask

logger = logging.getLogger(__name__)


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
    """定时任务调度引擎：加载任务列表、按时触发、执行 Agent.run()。"""

    def __init__(self, store: ScheduleStore, log_dir: Path | None = None):
        self._store = store
        self._log_dir = log_dir
        self._running = False

    async def start(self) -> None:
        """启动调度主循环（前台阻塞，Ctrl+C 停止）。"""
        self._running = True
        print("调度器已启动，每分钟检查一次任务... 按 Ctrl+C 停止")
        logger.info("调度器启动")

        # 立刻做一次检查（初始化 next_run）
        await self._tick()

        while self._running:
            # 睡到下一分钟整点（+2s 避免边界竞争）
            now = datetime.now()
            sleep_sec = 60 - now.second + 2
            await asyncio.sleep(sleep_sec)
            if self._running:
                await self._tick()

    def stop(self) -> None:
        self._running = False

    async def _tick(self) -> None:
        """检查并执行所有到期任务。"""
        now = datetime.now()
        tasks = self._store.list_all()
        due = []

        for task in tasks:
            if not task.enabled:
                continue

            # 首次运行：初始化 next_run
            if task.next_run is None:
                next_dt = compute_next_run(task, now)
                task.next_run = next_dt.isoformat() if next_dt else None
                self._store.update(task)
                if next_dt is None:
                    continue
                # 初始化后立刻检查是否已到期（如启动时 run_at 已过）
                if next_dt <= now:
                    due.append(task)
                else:
                    print(f"  [调度器] '{task.name}' 下次运行: {task.next_run[:16]}")
                continue

            try:
                next_dt = datetime.fromisoformat(task.next_run)
            except ValueError:
                continue

            if next_dt <= now:
                due.append(task)

        for task in due:
            await self._execute(task)

    async def _execute(self, task: ScheduleTask) -> None:
        """执行单个任务，更新 last_run / next_run / run_count。"""
        now = datetime.now()
        print(f"\n[调度器] ▶ 执行任务: {task.name}")
        logger.info("执行任务: %s", task.name)
        try:
            result = await self._run_task(task)
            task.last_run = now.isoformat()
            task.run_count += 1

            if task.trigger_type == "once":
                self._store.remove(task.name)
                print(f"[调度器] ✓ '{task.name}' 执行完成，已删除")
            else:
                next_dt = compute_next_run(task, now)
                task.next_run = next_dt.isoformat() if next_dt else None
                self._store.update(task)
                print(f"[调度器] ✓ '{task.name}' 执行完成（第 {task.run_count} 次）")
                if task.next_run:
                    print(f"[调度器]   下次运行: {task.next_run[:16]}")

            if self._log_dir:
                self._write_log(task, now, result)

        except Exception as e:
            logger.error("任务 '%s' 执行失败: %s", task.name, e)
            print(f"[调度器] ✗ '{task.name}' 执行失败: {e}")

    async def run_task_now(self, name: str) -> str:
        """立刻同步执行指定任务（CLI 使用），返回最终文本。"""
        task = self._store.get(name)
        if not task:
            raise ValueError(f"任务 '{name}' 不存在")
        result = await self._run_task(task)
        now = datetime.now()
        task.last_run = now.isoformat()
        task.run_count += 1
        if task.trigger_type == "once":
            self._store.remove(task.name)
        else:
            next_dt = compute_next_run(task, now)
            task.next_run = next_dt.isoformat() if next_dt else None
            self._store.update(task)
        return result

    async def _run_task(self, task: ScheduleTask) -> str:
        """构建 Agent，执行 task.prompt，返回最终回答文本。"""
        from milu._env import ensure_dotenv_loaded
        from milu.agent.events import AgentDone
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
        agent = Agent(llm, session_enabled=False, subagents=[])

        final_text = ""
        async for ev in agent.run(task.prompt):
            if isinstance(ev, AgentDone):
                final_text = ev.final_text

        return final_text

    def _write_log(self, task: ScheduleTask, run_time: datetime, result: str) -> None:
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            ts = run_time.strftime("%Y%m%d_%H%M%S")
            log_path = self._log_dir / f"{task.name}_{ts}.txt"
            log_path.write_text(
                f"任务: {task.name}\n"
                f"执行时间: {run_time.isoformat()}\n"
                f"Prompt: {task.prompt}\n\n"
                f"结果:\n{result}",
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("写入执行日志失败: %s", e)
