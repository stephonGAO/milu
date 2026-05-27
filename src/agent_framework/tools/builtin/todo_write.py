"""内置工具：会话计划管理（Todo Write）

让 LLM 在多步骤任务中维护一份当前会话计划：
- 完整重写计划（每次传入完整的 items 列表）
- 每个 item 有 pending / in_progress / completed 三种状态
- 同时只允许一个 in_progress
- 最多 12 个条目，保持计划精简
- 长期未更新时自动提醒 Agent 刷新计划

使用方式：
    from agent_framework.tools.builtin import create_todo_write_tool

    todo_tool = create_todo_write_tool()
    agent = Agent(llm=llm, tools=[*BUILTIN_TOOLS, todo_tool])
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_framework.tools.decorator import tool

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

    设计原则：每次 update 都是完整重写（而非增量修改），
    这样 LLM 始终看到完整的计划状态，避免增量操作的复杂性。
    """

    def __init__(self):
        self.state = PlanningState()

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
            raise ValueError("同时只能有 1 个 in_progress 状态的条目")

        self.state.items = normalized
        self.state.rounds_since_update = 0
        return self.render()

    def note_round(self) -> None:
        """记录一轮未更新（由工具包装器自动调用）"""
        self.state.rounds_since_update += 1

    def maybe_reminder(self) -> str | None:
        """超过阈值未更新时返回提醒，否则返回 None"""
        if not self.state.items:
            return None
        if self.state.rounds_since_update < _PLAN_REMINDER_INTERVAL:
            return None
        return "<reminder>计划已长时间未更新，请在继续工作前刷新当前计划。</reminder>"

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


# ── 工具工厂 ──────────────────────────────────────────────


def create_todo_write_tool(manager: TodoManager | None = None):
    """创建 todo_write 会话计划管理工具。

    :param manager: 自定义 TodoManager 实例，默认自动创建

    用法：
        # 使用默认管理器
        todo_tool = create_todo_write_tool()
        agent = Agent(llm=llm, tools=[*BUILTIN_TOOLS, todo_tool])

        # 自定义管理器（便于测试或共享状态）
        manager = TodoManager()
        todo_tool = create_todo_write_tool(manager)
    """
    mgr = manager or TodoManager()

    @tool(
        name="todo_write",
        description=(
            "管理当前会话的任务计划。每次调用都是完整重写整个计划"
            "（而非增量修改），请传入完整的任务列表。"
            "多步骤任务时主动使用此工具跟踪进度。"
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
    _inner_func = _todo_write
    _wrapper = _inner_func._tool_wrapper
    _original_func = _wrapper.func

    async def _wrapped_func(**kwargs):
        result = await _original_func(**kwargs)
        # 每次工具调用后重置轮次计数器
        mgr.state.rounds_since_update = 0
        # 如果计划长期未被 todo_write 刷新，附加提醒
        reminder = mgr.maybe_reminder()
        if reminder:
            result = f"{result}\n\n{reminder}"
        return result

    # 让 executor 调用包装后的函数（wrapper.func 才是 executor 实际调用的）
    _wrapper.func = _wrapped_func
    _inner_func._tool_wrapper = _wrapper
    return _inner_func
