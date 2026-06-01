"""内置工具：会话计划管理（Todo Write）

让 LLM 在多步骤任务中维护一份当前会话计划：
- 完整重写计划（每次传入完整的 items 列表）
- 每个 item 有 pending / in_progress / completed 三种状态
- 同时只允许一个 in_progress
- 最多 12 个条目，保持计划精简
- 长期未更新时自动提醒 Agent 刷新计划
- 计划持久化到本地文件，防止上下文截断丢失

使用方式：
    from agent_framework.tools.builtin import create_todo_write_tool

    todo_write_tool, todo_read_tool = create_todo_write_tool()
    agent = Agent(llm=llm, tools=[*BUILTIN_TOOLS, todo_write_tool, todo_read_tool])
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_framework.tools.decorator import tool

logger = logging.getLogger(__name__)

# 计划未更新多少轮后提醒 Agent 刷新
_PLAN_REMINDER_INTERVAL = 3
# 计划最大条目数
_MAX_PLAN_ITEMS = 12

_VALID_STATUSES = {"pending", "in_progress", "completed"}

_STATUS_MARKERS = {
    "pending": "[ ]",
    "in_progress": "[>]",
    "completed": "[x]",
}


# ── 数据模型 ──────────────────────────────────────────────


@dataclass
class PlanItem:
    """计划条目"""
    content: str
    status: str = "pending"
    active_form: str = ""


@dataclass
class PlanningState:
    """计划状态"""
    items: list[PlanItem] = field(default_factory=list)
    rounds_since_update: int = 0


# ── 计划管理器 ────────────────────────────────────────────


class TodoManager:
    """管理会话计划的创建、更新和提醒。

    设计原则：
    - 每次 update 都是完整重写（而非增量修改），LLM 始终看到完整状态
    - 自动持久化到本地文件，防止上下文截断丢失
    - 提醒时直接注入完整计划内容，确保 LLM 始终能看到
    """

    def __init__(self, plan_file: Path | str | None = None):
        """
        :param plan_file: 计划持久化文件路径。
            - 传入路径：自动保存到该路径
            - None：不保存文件（纯内存模式）
        """
        self.state = PlanningState()
        self._plan_file: Path | None = Path(plan_file) if plan_file else None

        # 如果有文件且存在，自动加载
        if self._plan_file and self._plan_file.exists():
            try:
                self.load_from_file()
                logger.info("从 %s 恢复了会话计划（%d 个条目）",
                            self._plan_file, len(self.state.items))
            except Exception as e:
                logger.warning("加载计划文件失败: %s", e)

    @property
    def plan_file(self) -> Path | None:
        return self._plan_file

    def update(self, items: list[dict[str, Any]]) -> str:
        """重写整个计划。

        :param items: 计划条目列表，每个包含 content/status/activeForm
        """
        if len(items) > _MAX_PLAN_ITEMS:
            raise ValueError(f"计划最多 {_MAX_PLAN_ITEMS} 个条目，当前提供了 {len(items)} 个")

        normalized: list[PlanItem] = []
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

            normalized.append(PlanItem(
                content=content,
                status=status,
                active_form=active_form,
            ))

        if in_progress_count > 1:
            raise ValueError(
                "同时只能有 1 个 in_progress 状态的条目。"
                "请将当前正在执行的任务标记为 in_progress，其余标记为 pending。"
            )

        self.state.items = normalized
        self.state.rounds_since_update = 0

        # 持久化到文件
        if self._plan_file:
            self.save_to_file()

        return self.render()

    def set_plan_file(self, path: "Path | str | None") -> None:
        """切换计划文件路径（session 切换时由 Agent 自动调用）。

        重置内存状态并从新路径加载（如果文件存在）。
        """
        self._plan_file = Path(path) if path else None
        self.state.items = []
        self.state.rounds_since_update = 0
        if self._plan_file and self._plan_file.exists():
            try:
                self.load_from_file()
                logger.info("切换计划文件到 %s（%d 个条目）",
                            self._plan_file, len(self.state.items))
            except Exception as e:
                logger.warning("加载计划文件失败: %s", e)

    def clear(self) -> str:
        """清空计划"""
        self.state.items = []
        self.state.rounds_since_update = 0
        if self._plan_file and self._plan_file.exists():
            self._plan_file.unlink()
        return "会话计划已清空。"

    def note_round(self) -> None:
        """记录一轮未更新（由工具包装器自动调用）"""
        self.state.rounds_since_update += 1

    def maybe_reminder(self) -> str | None:
        """超过阈值未更新时返回提醒（包含完整计划内容），否则返回 None"""
        if not self.state.items:
            return None
        if self.state.rounds_since_update < _PLAN_REMINDER_INTERVAL:
            return None

        plan_text = self.render()
        return (
            "<reminder>\n"
            "计划已长时间未更新，请在继续工作前刷新计划状态。以下是当前计划：\n\n"
            f"{plan_text}\n\n"
            "请根据计划进度决定下一步操作，"
            "并调用 todo_write 更新计划状态。\n"
            "</reminder>"
        )

    def render(self) -> str:
        """渲染为可读的 checklist 文本"""
        if not self.state.items:
            return "暂无会话计划。"

        lines: list[str] = []
        for item in self.state.items:
            marker = _STATUS_MARKERS[item.status]
            line = f"{marker} {item.content}"
            if item.status == "in_progress" and item.active_form:
                line += f"  ({item.active_form})"
            lines.append(line)

        completed = sum(1 for i in self.state.items if i.status == "completed")
        lines.append(f"\n({completed}/{len(self.state.items)} 已完成)")
        return "\n".join(lines)

    def save_to_file(self) -> None:
        """保存计划到文件"""
        if not self._plan_file:
            return
        data = {
            "items": [
                {
                    "content": item.content,
                    "status": item.status,
                    "activeForm": item.active_form,
                }
                for item in self.state.items
            ],
        }
        self._plan_file.parent.mkdir(parents=True, exist_ok=True)
        self._plan_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_from_file(self) -> None:
        """从文件加载计划"""
        if not self._plan_file or not self._plan_file.exists():
            return

        text = self._plan_file.read_text(encoding="utf-8")
        data = json.loads(text)
        items = data.get("items", [])

        normalized: list[PlanItem] = []
        for raw in items:
            normalized.append(PlanItem(
                content=str(raw.get("content", "")).strip(),
                status=str(raw.get("status", "pending")).lower(),
                active_form=str(raw.get("activeForm", "")).strip(),
            ))
        self.state.items = normalized
        self.state.rounds_since_update = 0

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "items": [
                {
                    "content": item.content,
                    "status": item.status,
                    "activeForm": item.active_form,
                }
                for item in self.state.items
            ],
        }


# ── 工具工厂 ──────────────────────────────────────────────


def create_todo_write_tool(
    manager: TodoManager | None = None,
    plan_file: Path | str | None = None,
) -> tuple:
    """创建 todo_write + todo_read 两个会话计划工具。

    :param manager: 自定义 TodoManager 实例，默认自动创建
    :param plan_file: 计划持久化文件路径（None 为纯内存模式）

    Returns:
        (todo_write, todo_read) 两个工具函数的元组

    用法：
        # 使用默认管理器，不持久化
        todo_write, todo_read = create_todo_write_tool()

        # 带文件持久化
        todo_write, todo_read = create_todo_write_tool(plan_file=".plan.json")

        # 自定义管理器
        manager = TodoManager(plan_file=".plan.json")
        todo_write, todo_read = create_todo_write_tool(manager=manager)
    """
    mgr = manager or TodoManager(plan_file=plan_file)

    # ── todo_write 工具 ──

    @tool(
        name="todo_write",
        description=(
            "管理当前会话的任务计划。每次调用都是完整重写整个计划"
            "（而非增量修改），请传入完整的任务列表。"
            "多步骤任务时主动使用此工具跟踪进度。"
            "计划会自动保存到本地文件，防止上下文丢失。"
            "**注意**：同一时刻最多只能有 1 个 in_progress 状态的条目。"
            "此工具必须单独调用，不可与其他工具（如子代理）放在同一批调用中。"
        ),
    )
    async def _todo_write(items: list[dict]) -> str:
        """
        :param items: 计划条目列表。每个条目包含：
            - content (str, 必填): 任务描述
            - status (str, 必填): pending / in_progress / completed
            - activeForm (str, 可选): 进行时描述，如 "实现认证模块"
        """
        return mgr.update(items)

    # 包装原函数：每次调用后追踪轮次并注入提醒
    _write_wrapper = _todo_write._tool_wrapper
    _write_original_func = _write_wrapper.func

    async def _wrapped_write(**kwargs):
        result = await _write_original_func(**kwargs)
        # 每次工具调用后重置轮次计数器
        # mgr.state.rounds_since_update = 0  更新时已重置
        # 如果计划长期未被 todo_write 刷新，附加提醒（含完整计划内容）
        reminder = mgr.maybe_reminder()
        if reminder:
            result = f"{result}\n\n{reminder}"
        return result

    _write_wrapper.func = _wrapped_write
    _todo_write._tool_wrapper = _write_wrapper

    # ── todo_read 工具 ──

    @tool(
        name="todo_read",
        description=(
            "查看当前会话计划的完整状态。"
            "当你不确定当前计划内容、或想确认进度时调用此工具。"
            "返回包含所有任务的状态清单。"
            "此工具必须单独调用，不可与其他工具放在同一批调用中。"
        ),
    )
    async def _todo_read() -> str:
        """查看当前计划状态。"""
        return mgr.render()

    # todo_read 包装：每次读取后也重置轮次（避免刚读完又被提醒）
    _read_wrapper = _todo_read._tool_wrapper
    _read_original_func = _read_wrapper.func

    async def _wrapped_read(**kwargs):
        result = await _read_original_func(**kwargs)
        mgr.state.rounds_since_update = 0
        return result

    _read_wrapper.func = _wrapped_read
    _todo_read._tool_wrapper = _read_wrapper

    # 标记 TodoManager，供 Agent 自动发现并绑定 session
    _wrapped_write._todo_manager = mgr
    _wrapped_read._todo_manager = mgr

    return _todo_write, _todo_read
