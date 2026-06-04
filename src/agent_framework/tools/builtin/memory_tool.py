"""内置工具：长期记忆（memory_write / memory_read）

让 LLM 主动记录跨会话、跨重启仍需记住的信息（用户偏好、重要事实、长期约定），
与「对话历史」互补——历史会被压缩/截断，记忆条目始终完整保留。

存储：用户级文件 `{user_data_dir()}/memory/{user_id}.json`（默认
`~/.agent_framework/memory/`，可用环境变量 AGENT_FRAMEWORK_HOME 覆盖）。
**与 session 解耦**——同一用户标识跨 session、跨进程共享同一份记忆。

启用方式（默认关闭）：`Agent(memory=True)`（单用户，身份 "default"）或
`Agent(memory="user-123")`（多用户场景按用户隔离）。启用后 Agent 负责：
  1. 注册 memory_write / memory_read 两个工具；
  2. 每轮把记忆条目渲染进 system prompt 末尾（render_memory_prompt）；
  3. run() 入口经 ContextVar 注入记忆文件路径（asyncio 任务级隔离）。

对标 MCP 官方 Memory reference server 的轻量实现（条目列表而非知识图谱，够用且零依赖）。
"""
from __future__ import annotations

import contextvars
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_framework.tools.decorator import tool

logger = logging.getLogger(__name__)

# 记忆条目上限：超出时丢弃最旧条目（防无限膨胀）
_MAX_MEMORY_ITEMS = 200

# ── 状态注入（asyncio 任务级隔离）──────────────────────
#
# Agent.run() 入口注入当前 Agent 的记忆文件路径（未启用 memory 时注入 None，
# 显式隔离——子代理不会经 Context 继承父的记忆路径而意外写入）。

_current_memory_path: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "current_memory_path", default=None
)


def memory_file_path(user_id: str) -> Path:
    """返回某用户标识对应的长期记忆文件路径。

    `{user_data_dir()}/memory/{user_id}.json`，user_id 做文件系统安全化
    （与 AgentPool._derive_session_id 同一规则）。

    :param user_id: 用户标识；空串视为 "default"。
    """
    from agent_framework.resources import user_data_dir

    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", user_id.strip() or "default")[:64]
    return user_data_dir() / "memory" / f"{safe}.json"


def _load_items(path: Path) -> list[dict[str, Any]]:
    """从指定文件读取记忆条目（文件不存在/损坏时宽容返回空）。"""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("读取记忆文件失败 %s: %s", path, e)
        return []
    return data.get("items", [])


def _read_items() -> list[dict[str, Any]]:
    """读取当前注入路径的记忆条目（未注入时宽容返回空）。"""
    path = _current_memory_path.get()
    if path is None:
        return []
    return _load_items(path)


def _write_items(items: list[dict[str, Any]]) -> None:
    """写入记忆条目到当前注入路径（未注入即 memory 未启用时报错）。"""
    path = _current_memory_path.get()
    if path is None:
        raise RuntimeError(
            "长期记忆未启用：构造 Agent 时传 memory=True（或用户标识字符串）开启"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"items": items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
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


def render_memory_prompt(path: Path) -> str:
    """渲染注入 system prompt 末尾的「长期记忆」段落（启用 memory 时由 Agent 每轮调用）。

    每轮重读文件——同一用户的其他 Agent/进程新写入的记忆即时可见（与提示词热重载同理）。
    包含使用指引 + 当前全部条目；无条目时仅注入指引（让 LLM 知道何时该 memory_write）。
    """
    items = _load_items(path)
    lines = [
        "\n\n## 长期记忆",
        "以下条目跨会话、跨重启始终完整保留（对话历史会被压缩，记忆条目不会）。"
        "处理任务时遵循这些记忆；用户表达偏好（如「以后回复简短点」）、提供重要背景、"
        "确立长期约定时，主动用 memory_write 记录，不要记录一次性临时信息。",
        "",
    ]
    if items:
        for i, item in enumerate(items, 1):
            cat = item.get("category", "fact")
            created = item.get("created_at", "")[:10]
            suffix = f"（{created}）" if created else ""
            lines.append(f"{i}. [{cat}] {item.get('content', '')}{suffix}")
    else:
        lines.append("（暂无记忆条目）")
    return "\n".join(lines)


# ── 工具函数 ──────────────────────────────────────────


@tool(
    name="memory_write",
    description=(
        "记录一条长期记忆（按用户身份持久存储，跨会话、跨重启保留；"
        "条目会自动展示在系统提示词的「长期记忆」一节）。"
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
        "查看已记录的长期记忆条目，可按分类过滤（preference / fact / agreement）。"
        "记忆条目已自动注入系统提示词的「长期记忆」一节，通常无需调用；"
        "仅在需要按分类筛查时使用。"
    ),
)
async def memory_read(category: str = "") -> str:
    """
    :param category: 可选分类过滤（preference / fact / agreement），留空返回全部
    """
    return _render(_read_items(), category.strip())
