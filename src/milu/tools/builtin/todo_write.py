"""内置工具：会话计划管理（Todo Write）— 无状态版本

设计原则：
- 工具函数本身无闭包、无实例状态
- 状态完全由 Session 目录下的 plan.json 文件承载（per-user 自动隔离）
- Agent 通过 ContextVar 注入当前 session_dir
- LLM 通过 todo_read 主动拉取当前计划
"""
from __future__ import annotations

import contextvars
import json
import logging
from pathlib import Path
from typing import Any

from milu.tools.decorator import tool

logger = logging.getLogger(__name__)

# 计划最大条目数
_MAX_PLAN_ITEMS = 12

_VALID_STATUSES = {"pending", "in_progress", "completed"}

_STATUS_MARKERS = {
    "pending": "[ ]",
    "in_progress": "[>]",
    "completed": "[x]",
}

# ── 状态注入（asyncio 任务级隔离）──────────────────────
#
# 计划存储分两种后端，由 Agent.run() 在入口注入其一：
#   1. 文件后端：有 session 时，注入 session 目录 → 计划落盘 plan.json（跨轮持久化）
#   2. 内存后端：无 session 时，注入一个进程内列表 → 计划存于内存（同一 Agent 跨轮保留，
#      进程退出即弃）。使 todo 工具不再强依赖 session，session_enabled=False（含子代理、
#      用户自管 history）时也能正常使用，不再抛 RuntimeError。

_current_session_dir: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "current_todo_session_dir", default=None
)

_current_plan_items: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "current_todo_plan_items", default=None
)


def _plan_path() -> Path | None:
    """当前 session 的 plan.json 路径（无 session 文件后端时返回 None）。"""
    d = _current_session_dir.get()
    return d / "plan.json" if d else None


def _read_items() -> list[dict[str, Any]]:
    """读取当前计划条目（文件后端优先，其次内存后端）。

    非 Agent.run() 上下文（两种后端都未注入）时返回空列表——读操作保持宽容，
    与历史行为一致（todo_read 在无计划时返回「暂无会话计划」）。
    """
    path = _plan_path()
    if path is not None:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("读取 plan.json 失败: %s", e)
            return []
        return data.get("items", [])

    mem = _current_plan_items.get()
    if mem is not None:
        return list(mem)
    return []


def _write_items(normalized: list[dict[str, Any]]) -> None:
    """写入计划条目（文件后端优先，其次内存后端）。

    两种后端都未注入时抛 RuntimeError——写操作必须在 Agent.run() 上下文中进行，
    否则计划无处落脚，应明确报错而非静默丢弃。
    """
    path = _plan_path()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"items": normalized}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return

    mem = _current_plan_items.get()
    if mem is not None:
        mem[:] = normalized  # 原地替换，保持 Agent 持有的同一列表引用（跨轮可见）
        return

    raise RuntimeError(
        "todo_write 必须在 Agent.run() 上下文中调用（计划存储未注入）"
    )


# ── 纯函数（无状态）────────────────────────────────────

def _validate_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """校验 + 规范化计划条目。纯函数。"""
    if len(items) > _MAX_PLAN_ITEMS:
        raise ValueError(
            f"计划最多 {_MAX_PLAN_ITEMS} 个条目，当前提供了 {len(items)} 个"
        )

    normalized: list[dict[str, Any]] = []
    in_progress_count = 0
    for index, raw in enumerate(items):
        content = str(raw.get("content", "")).strip()
        status = str(raw.get("status", "pending")).lower()
        active_form = str(raw.get("activeForm", "")).strip()

        if not content:
            raise ValueError(f"条目 {index}: content 不能为空")
        if status not in _VALID_STATUSES:
            raise ValueError(
                f"条目 {index}: 无效状态 '{status}'，可选: {', '.join(sorted(_VALID_STATUSES))}"
            )
        if status == "in_progress":
            in_progress_count += 1

        normalized.append({
            "content": content,
            "status": status,
            "activeForm": active_form,
        })

    if in_progress_count > 1:
        raise ValueError(
            "同时只能有 1 个 in_progress 状态的条目。"
            "请将当前正在执行的任务标记为 in_progress，其余标记为 pending。"
        )

    return normalized


def _render(items: list[dict[str, Any]]) -> str:
    """渲染为可读的 checklist 文本。纯函数。"""
    if not items:
        return "暂无会话计划。"

    lines: list[str] = []
    for item in items:
        marker = _STATUS_MARKERS[item["status"]]
        line = f"{marker} {item['content']}"
        if item["status"] == "in_progress" and item.get("activeForm"):
            line += f"  ({item['activeForm']})"
        lines.append(line)
    completed = sum(1 for i in items if i["status"] == "completed")
    lines.append(f"\n({completed}/{len(items)} 已完成)")
    return "\n".join(lines)


# ── 工具函数 ──────────────────────────────────────────

@tool(
    name="todo_write",
    description=(
        "管理当前会话的任务计划。每次调用都是完整重写整个计划"
        "（而非增量修改），请传入完整的任务列表。"
        "**主动调用时机**：开始复杂多步骤任务时强烈建议建立计划；"
        "完成一项后刷新状态（pending → in_progress → completed）；"
        "发现需要调整计划时整体重写。"
        "计划会自动保存到当前 session 目录的 plan.json。"
        "**注意**：同一时刻最多只能有 1 个 in_progress 状态的条目。"
        "此工具必须单独调用，不可与其他工具（如子代理）放在同一批调用中。"
    ),
)
async def todo_write(items: list[dict]) -> str:
    """:param items: 计划条目列表。每个条目包含：
        - content (str, 必填): 任务描述
        - status (str, 必填): pending / in_progress / completed
        - activeForm (str, 可选): 进行时描述，如 "实现认证模块"
    """
    normalized = _validate_items(items)
    _write_items(normalized)
    return _render(normalized)


@tool(
    name="todo_read",
    description=(
        "查看当前会话计划的完整状态。\n"
        "**必须主动调用的场景**（不会自动提醒，请养成习惯）：\n"
        "  - 开始执行下一步操作前，确认还剩什么待办\n"
        "  - 完成一项任务后，刷新整体进度\n"
        "  - 不确定当前 plan 状态、或感觉已偏离原计划\n"
        "  - 完成全部任务后记得更新计划状态！\n"
        "返回包含所有任务的状态清单（pending / in_progress / completed）。"
    ),
)
async def todo_read() -> str:
    """查看当前计划状态。"""
    return _render(_read_items())
