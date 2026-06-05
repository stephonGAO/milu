"""定时任务存储：数据模型 + JSON 持久化。"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


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
    """定时任务 JSON 持久化存储（~/.milu/schedules.json）。"""

    def __init__(self, path: Path):
        self._path = path

    def load(self) -> list[ScheduleTask]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return [ScheduleTask(**t) for t in data.get("tasks", [])]
        except (json.JSONDecodeError, OSError, TypeError):
            return []

    def save(self, tasks: list[ScheduleTask]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({"tasks": [asdict(t) for t in tasks]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(self, task: ScheduleTask) -> None:
        tasks = self.load()
        if any(t.name == task.name for t in tasks):
            raise ValueError(f"任务 '{task.name}' 已存在")
        tasks.append(task)
        self.save(tasks)

    def get(self, name: str) -> ScheduleTask | None:
        return next((t for t in self.load() if t.name == name), None)

    def remove(self, name: str) -> bool:
        tasks = self.load()
        new_tasks = [t for t in tasks if t.name != name]
        if len(new_tasks) == len(tasks):
            return False
        self.save(new_tasks)
        return True

    def update(self, task: ScheduleTask) -> bool:
        tasks = self.load()
        for i, t in enumerate(tasks):
            if t.name == task.name:
                tasks[i] = task
                self.save(tasks)
                return True
        return False

    def list_all(self) -> list[ScheduleTask]:
        return self.load()
