"""定时任务存储：数据模型 + 按用户分文件的 JSON 持久化。

多用户布局（与 memory 的用户级隔离同构）：
    {base_dir}/schedules/{safe_user_id}.json     # 每用户一个文件
旧版单文件 {base_dir}/schedules.json 首次访问时自动迁移为
schedules/default.json，源文件保留为 schedules.json.migrated 以便回退。

跨进程竞态取舍（与 session.py 的 flock/Windows-no-op 同档）：写盘一律
原子写（mkstemp + fsync + os.replace，永不出现半截文件）；CLI 工具进程
与守护进程同毫秒写同一用户文件的丢更新窗口极小，接受之，不引入跨进程
文件锁新依赖。进程内并发（引擎 gather 多任务）由引擎层 per-user
asyncio.Lock 串行化（见 engine.py）。
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


def _safe_user(user_id: str) -> str:
    """user_id 文件系统安全化（与 memory_file_path 同一规则）。"""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", (user_id or "default").strip() or "default")[:64]


def _atomic_write_json(path: Path, obj: dict) -> None:
    """原子写 JSON：同目录临时文件 + fsync + os.replace，不出现半截文件。

    （仿 file_tool._atomic_write；os.replace 在 Windows/POSIX 上均为原子 rename。）
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            json.dump(obj, tmp, ensure_ascii=False, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


@dataclass
class ScheduleTask:
    """定时任务数据模型。"""
    id: str
    name: str
    prompt: str
    trigger_type: str       # "cron" | "interval" | "once"
    cron: str = ""          # cron 表达式，trigger_type=cron 时有效
    interval_minutes: int = 0   # 间隔分钟，trigger_type=interval 时有效
    run_at: str = ""        # ISO 时间，trigger_type=once 时有效
    description: str = ""
    provider: str = ""
    model: str = ""
    enabled: bool = True
    created_at: str = ""
    last_run: str | None = None
    next_run: str | None = None
    run_count: int = 0
    user_id: str = "default"    # 任务归属用户（多用户隔离维度；旧 JSON 无此字段自动补默认值）

    @classmethod
    def create(
        cls,
        name: str,
        prompt: str,
        trigger_type: str,
        cron: str = "",
        interval_minutes: int = 0,
        run_at: str = "",
        description: str = "",
        provider: str = "",
        model: str = "",
        user_id: str = "default",
    ) -> "ScheduleTask":
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            prompt=prompt,
            trigger_type=trigger_type,
            cron=cron,
            interval_minutes=interval_minutes,
            run_at=run_at,
            description=description,
            provider=provider,
            model=model,
            created_at=datetime.now().isoformat(),
            user_id=user_id,
        )

    def trigger_desc(self) -> str:
        """人类可读的触发方式描述。"""
        if self.trigger_type == "cron":
            return f"Cron: {self.cron}"
        if self.trigger_type == "interval":
            return f"每 {self.interval_minutes} 分钟"
        if self.trigger_type == "once":
            return f"一次性: {self.run_at}"
        return self.trigger_type


class ScheduleStore:
    """定时任务存储（按用户分文件，任务名在同一用户内唯一）。

    base_dir 通常为 user_data_dir()（~/.milu）；任务文件位于
    {base_dir}/schedules/{safe_user_id}.json。
    """

    def __init__(self, base_dir: Path):
        self._base_dir = Path(base_dir)
        self._dir = self._base_dir / "schedules"
        self._migrate_legacy_if_needed()

    def _user_path(self, user_id: str) -> Path:
        return self._dir / f"{_safe_user(user_id)}.json"

    def _migrate_legacy_if_needed(self) -> None:
        """旧版单文件 schedules.json → schedules/default.json（幂等）。

        源文件迁移后改名 .migrated 保留（可回退），不删除；损坏的旧文件
        不迁移不删除，留给人工处理。
        """
        legacy = self._base_dir / "schedules.json"
        target = self._dir / "default.json"
        if not legacy.exists() or target.exists():
            return
        try:
            data = json.loads(legacy.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        _atomic_write_json(target, data)
        try:
            legacy.rename(legacy.with_name("schedules.json.migrated"))
        except OSError:
            pass  # 改名失败不影响迁移结果（target 已存在，下次不再迁移）

    # ── 读写（内部）──────────────────────────────────────

    def _load_user(self, user_id: str) -> list[ScheduleTask]:
        path = self._user_path(user_id)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [ScheduleTask(**t) for t in data.get("tasks", [])]
        except (json.JSONDecodeError, OSError, TypeError):
            return []

    def _save_user(self, user_id: str, tasks: list[ScheduleTask]) -> None:
        _atomic_write_json(
            self._user_path(user_id),
            {"tasks": [asdict(t) for t in tasks]},
        )

    # ── CRUD ─────────────────────────────────────────────

    def add(self, task: ScheduleTask) -> None:
        tasks = self._load_user(task.user_id)
        if any(t.name == task.name for t in tasks):
            raise ValueError(f"任务 '{task.name}' 已存在")
        tasks.append(task)
        self._save_user(task.user_id, tasks)

    def get(self, name: str, user_id: str = "default") -> ScheduleTask | None:
        return next((t for t in self._load_user(user_id) if t.name == name), None)

    def remove(self, name: str, user_id: str = "default") -> bool:
        tasks = self._load_user(user_id)
        new_tasks = [t for t in tasks if t.name != name]
        if len(new_tasks) == len(tasks):
            return False
        self._save_user(user_id, new_tasks)
        return True

    def update(self, task: ScheduleTask) -> bool:
        tasks = self._load_user(task.user_id)
        for i, t in enumerate(tasks):
            if t.name == task.name:
                tasks[i] = task
                self._save_user(task.user_id, tasks)
                return True
        return False

    def list_user(self, user_id: str = "default") -> list[ScheduleTask]:
        return self._load_user(user_id)

    def list_all(self) -> list[ScheduleTask]:
        """汇总所有用户的任务（调度引擎 tick 扫描用，每 tick 全量读目录）。"""
        if not self._dir.exists():
            return []
        tasks: list[ScheduleTask] = []
        for f in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                tasks.extend(ScheduleTask(**t) for t in data.get("tasks", []))
            except (json.JSONDecodeError, OSError, TypeError):
                continue
        return tasks
