"""内置工具：定时任务管理（schedule_create / schedule_manage 两个工具）。

工具拆分原则（参数正交性）：create 有 9 个专属参数（触发类型/时间/模型等），
管理操作（list/delete/enable/disable/run_now）只需 action+name——若合并为
单工具，LLM 做管理操作时 schema 里全是无关的创建参数（噪音）。因此拆为：
  - schedule_create：专职创建（参数多但全部相关），不安全
  - schedule_manage：查/删/启停/立即运行（schema 极简），safe_check 动态
    判定——action=list 只读安全，其余不安全（走审批/AI 判定）

工具不直接执行任务，任务由 `milu scheduler start` 守护进程按时触发。
action=run_now 将 next_run 设为过去时间，让调度器下次 tick 立即执行；
CLI `milu schedule run <name>` 则同步执行并返回结果。

多用户隔离：当前用户标识由 Agent.run() 入口经 ContextVar 注入
（Agent(schedule_user=...)，AgentPool 默认工厂自动派生为 user_id），
工具按用户操作各自的任务文件，user_id 不暴露为 LLM 参数（防伪造他人身份）。
"""
from __future__ import annotations

import contextvars
from datetime import datetime
from typing import Literal

from milu.tools.decorator import tool

# per-run 用户标识（Agent.run() 入口注入；asyncio 任务级隔离，勿用模块级全局变量）
_current_schedule_user: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_schedule_user", default=None
)


def _resolve_user() -> str:
    """解析当前用户标识。

    注意：与 memory 的「未注入即写拒绝」是**有意差异**——schedule 工具在
    BUILTIN_TOOLS 默认列表中，CLI 单人场景不注入用户标识，未注入时必须
    退化为 "default" 保持旧行为兼容（有测试固化此约定）。
    """
    return _current_schedule_user.get() or "default"


def _get_store():
    from milu.resources import user_data_dir
    from milu.scheduler.store import ScheduleStore
    return ScheduleStore(user_data_dir())


def _format_next(cron: str) -> str:
    try:
        from croniter import croniter
        next_dt = croniter(cron, datetime.now()).get_next(datetime)
        return next_dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "（无法计算）"


# ── 创建（专职工具：参数多但全部相关）─────────────────────

@tool(
    name="schedule_create",
    description=(
        "创建定时任务：让 Agent 在指定时间或周期自动执行某段指令（prompt）。\n"
        "trigger_type 三选一：\n"
        "  - cron：标准 cron 表达式（如 '0 9 * * 1-5' = 工作日每天早上9点）\n"
        "  - interval：每隔 N 分钟执行一次（interval_minutes 指定间隔，最小 1）\n"
        "  - once：指定时间执行一次（run_at 指定，ISO 格式如 '2026-06-10T09:00:00'）\n"
        "执行时会创建独立 Agent 实例运行 prompt，结果记录到日志目录。\n"
        "查看/删除/启停/立即运行已有任务请用 schedule_manage 工具。\n"
        "**重要**：任务创建后需运行 `milu scheduler start` 守护进程才能自动触发。\n"
        "**重要**：用户说相对时间（如'5分钟后'、'明天早上9点'）时，必须先用 datetime "
        "工具获取当前时间，再据此计算 run_at——切勿凭记忆猜测日期；run_at 为过去时间会被拒绝。"
    ),
    is_safe=False,
)
async def schedule_create(
    name: str,
    prompt: str,
    trigger_type: Literal["cron", "interval", "once"],
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
        user_id=_resolve_user(),
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


# ── 管理（schema 极简：action + name）─────────────────────

async def _manage_list() -> str:
    store = _get_store()
    tasks = store.list_user(_resolve_user())
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


async def _manage_delete(name: str) -> str:
    store = _get_store()
    if store.remove(name, _resolve_user()):
        return f"任务 '{name}' 已删除。"
    return f"错误：任务 '{name}' 不存在。"


async def _manage_set_enabled(name: str, enabled: bool) -> str:
    store = _get_store()
    task = store.get(name, _resolve_user())
    if not task:
        return f"错误：任务 '{name}' 不存在。"
    task.enabled = enabled
    store.update(task)
    return f"任务 '{name}' 已{'启用' if enabled else '禁用'}。"


async def _manage_run_now(name: str) -> str:
    store = _get_store()
    task = store.get(name, _resolve_user())
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


@tool(
    name="schedule_manage",
    description=(
        "管理已有的定时任务。通过 action 参数选择操作类型：\n"
        "list - 列出当前用户的全部定时任务\n"
        "delete - 删除指定任务（需 name，不可恢复）\n"
        "enable - 启用指定任务（需 name）\n"
        "disable - 禁用指定任务（需 name，任务记录保留）\n"
        "run_now - 标记任务立即运行（需 name，调度器下次检查时触发）\n"
        "创建新任务请用 schedule_create 工具。"
    ),
    is_safe=False,
    safe_check=lambda args: args.get("action") == "list",  # list 只读安全，其余走审批/判定
)
async def schedule_manage(
    action: Literal["list", "delete", "enable", "disable", "run_now"],
    name: str = "",
) -> str:
    """管理定时任务（查/删/启停/立即运行）。

    :param action: 操作类型：list / delete / enable / disable / run_now
    :param name: 任务名称（list 之外的操作必填）
    """
    if action == "list":
        return await _manage_list()
    if action not in ("delete", "enable", "disable", "run_now"):
        return f"错误：未知 action '{action}'（可用：list/delete/enable/disable/run_now）"
    if not name:
        return f"错误：{action} 操作必须提供 name 参数"
    if action == "delete":
        return await _manage_delete(name)
    if action == "enable":
        return await _manage_set_enabled(name, True)
    if action == "disable":
        return await _manage_set_enabled(name, False)
    return await _manage_run_now(name)
