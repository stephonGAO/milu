"""内置工具：长期记忆（memory_write / memory_read）

让 LLM 主动记录跨轮次、跨重启仍需记住的信息（用户偏好、重要事实、长期约定），
与「对话历史」互补——历史会被压缩/截断，记忆条目始终完整保留。

存储后端（与 todo 工具同一双后端模式，由 Agent.run() 经 ContextVar 注入其一）：
  1. 文件后端：有 session 时落盘 {session_dir}/memory.json（跨重启持久，per-user 隔离）
  2. 内存后端：无 session 时存进程内列表（同一 Agent 跨轮保留，进程退出即弃）

对标 MCP 官方 Memory reference server 的轻量实现（条目列表而非知识图谱，够用且零依赖）。
"""
from __future__ import annotations

import contextvars
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_framework.tools.decorator import tool

logger = logging.getLogger(__name__)

# 记忆条目上限：超出时丢弃最旧条目（防无限膨胀）
_MAX_MEMORY_ITEMS = 200

# ── 状态注入（asyncio 任务级隔离）──────────────────────

_current_session_dir: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "current_memory_session_dir", default=None
)

_current_memory_items: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "current_memory_items", default=None
)


def _memory_path() -> Path | None:
    d = _current_session_dir.get()
    return d / "memory.json" if d else None


def _read_items() -> list[dict[str, Any]]:
    """读取记忆条目（文件后端优先，其次内存后端；都未注入时宽容返回空）。"""
    path = _memory_path()
    if path is not None:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("读取 memory.json 失败: %s", e)
            return []
        return data.get("items", [])

    mem = _current_memory_items.get()
    if mem is not None:
        return list(mem)
    return []


def _write_items(items: list[dict[str, Any]]) -> None:
    """写入记忆条目（文件后端优先，其次内存后端；都未注入时报错）。"""
    path = _memory_path()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"items": items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return

    mem = _current_memory_items.get()
    if mem is not None:
        mem[:] = items
        return

    raise RuntimeError(
        "memory_write 必须在 Agent.run() 上下文中调用（记忆存储未注入）"
    )


def _render(items: list[dict[str, Any]], category: str = "") -> str:
    """渲染记忆清单。纯函数。"""
    if category:
        items = [i for i in items if i.get("category") == category]
    if not items:
        return "暂无记忆条目。" if not category else f"暂无分类为「{category}」的记忆条目。"

    lines = [f"长期记忆（共 {len(items)} 条）："]
    for i, item in enumerate(items, 1):
        cat = item.get("category", "fact")
        created = item.get("created_at", "")[:10]  # 只显示日期部分
        suffix = f"（{created}）" if created else ""
        lines.append(f"{i}. [{cat}] {item.get('content', '')}{suffix}")
    return "\n".join(lines)


# ── 工具函数 ──────────────────────────────────────────


@tool(
    name="memory_write",
    description=(
        "记录一条长期记忆（跨轮次、跨会话重启仍保留）。"
        "**主动调用时机**：用户表达了偏好（如「以后回复简短点」）、"
        "提供了重要的个人/项目背景信息、或确立了长期约定时。"
        "对话历史会被压缩截断，但记忆条目始终完整——值得长期记住的信息都应写入。"
        "不要记录一次性的临时信息。"
    ),
)
async def memory_write(content: str, category: str = "fact") -> str:
    """
    :param content: 要记住的内容（一条简洁的事实/偏好/约定陈述）
    :param category: 分类：preference（用户偏好）/ fact（事实背景）/ agreement（长期约定）
    """
    content = content.strip()
    if not content:
        raise ValueError("记忆内容不能为空")

    items = _read_items()

    # 去重：完全相同的内容不重复记录
    if any(i.get("content") == content for i in items):
        return f"该内容已在记忆中，无需重复记录。当前共 {len(items)} 条记忆。"

    items.append({
        "content": content,
        "category": category.strip() or "fact",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    # 超上限丢弃最旧条目
    if len(items) > _MAX_MEMORY_ITEMS:
        items = items[-_MAX_MEMORY_ITEMS:]

    _write_items(items)
    return f"已记住。当前共 {len(items)} 条记忆。"


@tool(
    name="memory_read",
    description=(
        "查看已记录的长期记忆条目。"
        "**主动调用时机**：开始处理涉及用户偏好/背景的任务前、"
        "或感觉需要回忆之前确立的约定时。"
        "可按分类过滤（preference / fact / agreement）。"
    ),
)
async def memory_read(category: str = "") -> str:
    """
    :param category: 可选分类过滤（preference / fact / agreement），留空返回全部
    """
    return _render(_read_items(), category.strip())
