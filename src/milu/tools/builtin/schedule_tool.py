"""内置工具：定时任务管理（schedule_create / schedule_list / schedule_delete /
schedule_toggle / schedule_run_now）。

工具不直接执行任务，任务由 `milu scheduler start` 守护进程按时触发。
schedule_run_now 将 next_run 设为过去时间，让调度器下次 tick 立即执行；
CLI `milu schedule run <name>` 则同步执行并返回结果。
"""
from __future__ import annotations

from datetime import datetime

from milu.tools.decorator import tool


def _get_store():
    from milu.resources import user_data_dir
    from milu.scheduler.store import ScheduleStore
    return ScheduleStore(user_data_dir() / "schedules.json")


def _format_next(cron: str) -> str:
    try:
        from croniter import croniter
        next_dt = croniter(cron, datetime.now()).get_next(datetime)
        return next_dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "（无法计算）"


# ── 工具函数 ──────────────────────────────────────────────

@tool(
    name="schedule_create",
    description=(
        "创建定时任务：让 Agent 在指定时间或周期自动执行某段指令（prompt）。\n"
        "trigger_type 三选一：\n"
        "  - cron：标准 cron 表达式（如 '0 9 * * 1-5' = 工作日每天早上9点）\n"
        "  - interval：每隔 N 分钟执行一次（interval_minutes 指定间隔，最小 1）\n"
        "  - once：指定时间执行一次（run_at 指定，ISO 格式如 '2026-06-10T09:00:00'）\n"
        "执行时会创建独立 Agent 实例运行 prompt，结果记录到日志目录。\n"
        "**重要**：任务创建后需运行 `milu scheduler start` 守护进程才能自动触发。\n"
        "**重要**：用户说相对时间（如'5分钟后'、'明天早上9点'）时，必须先用 datetime "
        "工具获取当前时间，再据此计算 run_at——切勿凭记忆猜测日期；run_at 为过去时间会被拒绝。"
    ),
    is_safe=False,
)
async def schedule_create(
    name: str,
    prompt: str,
    trigger_type: str,
    cron: str = "",
    interval_minutes: int = 0,
    run_at: str = "",
    description: str = "",
    provider: str = "",
    model: str = "",
) -> str:
    """:param name: 任务唯一名称（建议用英文字母/数字/下划线/连字符，如 daily-report）
    :param prompt: 调度触发时 Agent 收到的指令（自然语言即可）
    :param trigger_type: 触发类型：cron / interval / once
    :param cron: cron 表达式（trigger_type=cron 必填），如 "0 9 * * *"（每天9点）
    :param interval_minutes: 间隔分钟（trigger_type=interval 必填），如 30
    :param run_at: 执行时间（trigger_type=once 必填），ISO 格式如 "2026-06-10T09:00:00"
    :param description: 任务用途说明（可选，便于人类理解）
    :param provider: 使用的 LLM 厂商（可选，默认用系统配置）
    :param model: 使用的模型名（可选，默认用系统配置）
    """
    from milu.scheduler.store import ScheduleTask

    # 参数校验
    if trigger_type not in ("cron", "interval", "once"):
        return f"错误：trigger_type 必须是 cron/interval/once，收到 '{trigger_type}'"
    if trigger_type == "cron":
        if not cron:
            return "错误：trigger_type=cron 时必须提供 cron 参数（如 '0 9 * * *'）"
        try:
            from croniter import croniter
            if not croniter.is_valid(cron):
                return f"错误：无效的 cron 表达式 '{cron}'（示例：'0 9 * * *' 表示每天9点）"
        except ImportError:
            pass  # croniter 未安装时跳过格式校验
    if trigger_type == "interval" and interval_minutes <= 0:
        return "错误：trigger_type=interval 时 interval_minutes 必须是正整数（分钟数）"
    if trigger_type == "once":
        if not run_at:
            return "错误：trigger_type=once 时必须提供 run_at（ISO 格式如 '2026-06-10T09:00:00'）"
        try:
            run_dt = datetime.fromisoformat(run_at)
        except ValueError:
            return f"错误：run_at 格式无效 '{run_at}'，请用 ISO 格式如 '2026-06-10T09:00:00'"
        now = datetime.now()
        if run_dt <= now:
            return (
                f"错误：run_at '{run_at}' 是过去的时间，任务不会被触发。\n"
                f"当前时间是 {now.strftime('%Y-%m-%d %H:%M:%S')}，"
                f"请基于当前时间重新计算执行时间（如'5分钟后'即 "
                f"{now.replace(microsecond=0).isoformat()} 往后加 5 分钟）。"
            )

    store = _get_store()
    task = ScheduleTask.create(
        name=name,
        prompt=prompt,
        trigger_type=trigger_type,
        cron=cron,
        interval_minutes=interval_minutes,
        run_at=run_at,
        description=description,
        provider=provider,
        model=model,
    )
    try:
        store.add(task)
    except ValueError as e:
        return f"错误：{e}"

    # 构造下次运行提示
    if trigger_type == "cron":
        next_hint = f"\n  下次运行预计: {_format_next(cron)}"
    elif trigger_type == "interval":
        next_hint = f"\n  首次触发: 调度器启动后约 {interval_minutes} 分钟"
    else:
        next_hint = f"\n  执行时间: {run_at}"

    prompt_preview = prompt[:80] + ("..." if len(prompt) > 80 else "")
    return (
        f"定时任务 '{name}' 已创建。{next_hint}\n\n"
        f"  触发方式: {task.trigger_desc()}\n"
        f"  执行指令: {prompt_preview}\n\n"
        f"提示：运行 `milu scheduler start` 启动调度守护进程后，任务将按时自动触发。\n"
        f"也可用 `milu schedule run {name}` 立即同步执行一次查看效果。"
    )


@tool(
    name="schedule_list",
    description="列出所有已创建的定时任务，显示触发方式、启用状态、上次/下次运行时间。",
    is_safe=True,
)
async def schedule_list() -> str:
    """列出全部定时任务。"""
    store = _get_store()
    tasks = store.list_all()
    if not tasks:
        return "暂无定时任务。可用 schedule_create 工具创建任务。"

    lines = [f"共 {len(tasks)} 个定时任务："]
    for t in tasks:
        status = "✓ 启用" if t.enabled else "✗ 禁用"
        last = t.last_run[:16].replace("T", " ") if t.last_run else "从未运行"
        nxt = t.next_run[:16].replace("T", " ") if t.next_run else "待计算"
        desc = t.description or t.prompt[:50]
        lines.append(
            f"\n[{t.name}]  {status}  运行 {t.run_count} 次\n"
            f"  触发: {t.trigger_desc()}\n"
            f"  说明: {desc}\n"
            f"  上次: {last}  |  下次: {nxt}"
        )
    return "\n".join(lines)


@tool(
    name="schedule_delete",
    description="删除指定的定时任务（不可恢复）。",
    is_safe=False,
)
async def schedule_delete(name: str) -> str:
    """:param name: 要删除的任务名称"""
    store = _get_store()
    if store.remove(name):
        return f"任务 '{name}' 已删除。"
    return f"错误：任务 '{name}' 不存在。"


@tool(
    name="schedule_toggle",
    description="启用或禁用指定的定时任务（禁用后调度器不再触发，但任务记录保留）。",
    is_safe=False,
)
async def schedule_toggle(name: str, enabled: bool) -> str:
    """:param name: 任务名称
    :param enabled: True 启用，False 禁用
    """
    store = _get_store()
    task = store.get(name)
    if not task:
        return f"错误：任务 '{name}' 不存在。"
    task.enabled = enabled
    store.update(task)
    return f"任务 '{name}' 已{'启用' if enabled else '禁用'}。"


@tool(
    name="schedule_run_now",
    description=(
        "将指定任务标记为「立即运行」——把 next_run 设为过去时间，"
        "调度器（milu scheduler start）下次检查时会立即触发执行。\n"
        "若需同步执行并查看完整结果，请使用 CLI 命令：milu schedule run <name>"
    ),
    is_safe=False,
)
async def schedule_run_now(name: str) -> str:
    """:param name: 要立即运行的任务名称"""
    store = _get_store()
    task = store.get(name)
    if not task:
        return f"错误：任务 '{name}' 不存在。"

    # 设为历史时间，让调度器下次 tick 立即捡起执行
    task.next_run = "2000-01-01T00:00:00"
    if not task.enabled:
        task.enabled = True
    store.update(task)

    return (
        f"任务 '{name}' 已标记为立即运行。\n"
        f"若调度器（milu scheduler start）正在运行，下次检查时将立即触发。\n\n"
        f"如需同步执行并查看结果：milu schedule run {name}"
    )
