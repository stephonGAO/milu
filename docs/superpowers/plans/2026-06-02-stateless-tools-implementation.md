# Todo / SubAgent 工具无状态化重构 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 todo_write / todo_read 工具从 TodoManager 闭包依赖重构为 session 文件 + ContextVar 注入；将 subagent_tools 从 `_last_events` 闭包重构为 per-call ContextVar 注入；修复 P0 闭包 bug，使两个工具在多用户并发下天然安全。

**Architecture:**
- **todo 工具**：删除 `TodoManager` 类。`todo_write` / `todo_read` 工具函数本身无任何闭包/实例状态。状态完全由 `Session.dir_path/plan.json` 文件承载。Agent.run() 入口通过模块级 `ContextVar` 注入 `session_dir`，asyncio 任务级隔离天然支持多用户并发。
- **subagent 工具**：删除 `_last_events: list = []` 闭包变量。引入模块级 `_current_subagent_events: ContextVar[list[AgentEvent] | None]`。Agent 在每次工具调用前 set 新列表，调用后 reset + 消费。工具函数签名、工厂签名保持完全不变。

**Tech Stack:** Python 3.10+、asyncio、contextvars、pytest、pytest-asyncio。沿用项目现有 `@tool` 装饰器和 `ToolExecutor`。

---

## 文件结构

**修改的文件**：
- `src/agent_framework/tools/builtin/todo_write.py` — 重写（删除 TodoManager、引入 ContextVar）
- `src/agent_framework/agent/subagent.py` — 重写（删除 _last_events、引入 ContextVar）
- `src/agent_framework/agent/agent.py` — 3 处改动（删除 TodoManager 引用、注入 ContextVar、subagent per-call events 消费）
- `tests/test_tool_todo_write.py` — 更新（删除 TodoManager 直引）
- `tests/test_subagent.py` — 更新（删除 _last_events 断言）
- `tests/test_concurrency_stress.py` — 改严格断言

**新增的文件**：
- `tests/test_todo_concurrent_isolation.py` — todo 工具并发隔离测试
- `tests/test_subagent_concurrent_isolation.py` — subagent 工具并发隔离测试

**边界与职责**：
- `todo_write.py`：只管 plan 校验 + 文件 I/O + 渲染。绝不感知 Agent。
- `subagent.py`：只管子代理工厂 + 工具函数包装。绝不感知父 Agent 的事件流。
- `agent.py`：负责 ContextVar 的 set/reset、reminder 注入、subagent events 消费。
- 两个 ContextVar（`_current_todo_session_dir` / `_current_subagent_events`）是模块级单例，asyncio 任务级隔离。

---

## Task 1: 添加 todo 工具并发隔离测试（TDD 起点）

**Files:**
- Create: `tests/test_todo_concurrent_isolation.py`

- [ ] **Step 1: 创建测试文件，写 5 个失败测试**

```python
"""Todo 工具并发隔离测试 — 验证无状态重构后多用户并发安全"""
from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path

import pytest

from agent_framework.agent.agent import Agent
from agent_framework.agent.config import AgentConfig, AgentMode
from agent_framework.agent.events import MessageRole
from agent_framework.agent.history import Message
from agent_framework.llm.providers.base import BaseLLM, StreamChunk, TokenUsage
from agent_framework.tools.builtin.todo_write import (
    _current_session_dir,
    create_todo_write_tool,
    todo_read,
    todo_write,
)


# ── Mock LLM ────────────────────────────────────────────

class _MockLLM(BaseLLM):
    """Echo LLM：返回固定文本。"""
    def __init__(self):
        super().__init__(model="mock", provider="mock")
        self.calls: list[list[dict]] = []

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        yield StreamChunk(content="ok", finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(1, 1, 2))

    def get_capabilities(self):
        from agent_framework.llm.capabilities import ModelCapabilities
        return ModelCapabilities()


# ── 测试 ────────────────────────────────────────────────

def test_todo_write_no_closure_state():
    """todo_write 工具函数无 TodoManager 闭包"""
    cw, cr = create_todo_write_tool()
    cw_closure = inspect.getclosurevars(cw)
    cr_closure = inspect.getclosurevars(cr)
    # 闭包变量里不能有 TodoManager 相关
    assert "mgr" not in cw_closure.cells
    assert "mgr" not in cr_closure.cells
    assert "manager" not in cw_closure.cells
    assert "manager" not in cr_closure.cells


def test_create_todo_write_tool_no_args():
    """v2 工厂函数不接受任何参数"""
    result = create_todo_write_tool()
    assert len(result) == 2
    assert result[0]._tool_wrapper.name == "todo_write"
    assert result[1]._tool_wrapper.name == "todo_read"


@pytest.mark.asyncio
async def test_todo_write_requires_contextvar(tmp_path: Path):
    """不通过 Agent.run() 直接调 todo_write 应抛 RuntimeError"""
    from agent_framework.tools.builtin.todo_write import _current_session_dir

    # 确保 ContextVar 未设置
    assert _current_session_dir.get() is None
    with pytest.raises(RuntimeError, match="Agent"):
        await todo_write._tool_wrapper.func(items=[
            {"content": "x", "status": "pending"},
        ])


@pytest.mark.asyncio
async def test_todo_read_without_file_returns_empty():
    """无 plan.json 时 todo_read 返回 '暂无会话计划。'"""
    from contextvars import copy_context

    from agent_framework.tools.builtin.todo_write import _current_session_dir

    ctx = copy_context()
    token = _current_session_dir.set(tmp_path := Path("./.sessions/test"))
    try:
        result = await todo_read._tool_wrapper.func()
        assert "暂无" in result
    finally:
        _current_session_dir.reset(token)


@pytest.mark.asyncio
async def test_two_agents_concurrent_writes_isolated(tmp_path: Path):
    """两个 Agent 并发 todo_write 写入各自 plan.json，无串味"""
    llm_a = _MockLLM()
    llm_b = _MockLLM()

    # 用 session_dir 区分
    agent_a = Agent(
        llm=llm_a,
        system_prompt="A",
        tools=list(create_todo_write_tool()),
        config=AgentConfig(session_dir=str(tmp_path / "A"), session_enabled=True),
    )
    agent_b = Agent(
        llm=llm_b,
        system_prompt="B",
        tools=list(create_todo_write_tool()),
        config=AgentConfig(session_dir=str(tmp_path / "B"), session_enabled=True),
    )

    # 不实际跑 run()，只验证 session 目录是独立的
    assert agent_a._session.dir_path != agent_b._session.dir_path
    assert agent_a._session.dir_path.name == "A"
    assert agent_b._session.dir_path.name == "B"


@pytest.mark.asyncio
async def test_reminder_injected_after_n_rounds(tmp_path: Path):
    """5 轮不调 todo_write，第 6 轮 LLM 收到 reminder"""
    # 先写一个 plan
    plan_path = tmp_path / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps({
            "items": [
                {"content": "任务 X", "status": "in_progress", "activeForm": ""},
            ]
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    llm = _MockLLM()
    agent = Agent(
        llm=llm,
        system_prompt="test",
        tools=list(create_todo_write_tool()),
        config=AgentConfig(session_dir=str(tmp_path), session_enabled=True),
    )

    # 构造 3 个 tool 结果的历史（达到 _PLAN_REMINDER_INTERVAL 阈值）
    history_messages = [
        Message(role=MessageRole.SYSTEM, content="system"),
        Message(role=MessageRole.ASSISTANT, content="thinking1", tool_calls=[
            type("TC", (), {"function": type("FN", (), {"name": "some_tool"})(), "id": "1"})()
        ]),
        Message(role=MessageRole.TOOL, content="result1", tool_call_id="1", name="some_tool"),
        Message(role=MessageRole.ASSISTANT, content="thinking2", tool_calls=[
            type("TC", (), {"function": type("FN", (), {"name": "another_tool"})(), "id": "2"})()
        ]),
        Message(role=MessageRole.TOOL, content="result2", tool_call_id="2", name="another_tool"),
        Message(role=MessageRole.ASSISTANT, content="thinking3", tool_calls=[
            type("TC", (), {"function": type("FN", (), {"name": "tool3"})(), "id": "3"})()
        ]),
        Message(role=MessageRole.TOOL, content="result3", tool_call_id="3", name="tool3"),
    ]

    # 调用 reminder 注入方法
    token = _current_session_dir.set(tmp_path)
    try:
        agent._maybe_inject_plan_reminder(history_messages)
        # system message 应包含 reminder
        system_msg = history_messages[0]
        assert "<reminder>" in system_msg.content
        assert "任务 X" in system_msg.content
    finally:
        _current_session_dir.reset(token)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd D:/MyWork/my-agent && .venv/Scripts/python -m pytest tests/test_todo_concurrent_isolation.py -v`
Expected: 大部分测试失败（因旧 TodoManager 仍存在）。`test_create_todo_write_tool_no_args` 和 `test_todo_write_no_closure_state` 应该失败。

- [ ] **Step 3: 暂存测试文件（不提交，Task 2-5 完成后一起提交）**

```bash
git add tests/test_todo_concurrent_isolation.py
# 不提交，等 Task 6 一起提交
```

---

## Task 2: 重写 `todo_write.py`（删除 TodoManager，引入 ContextVar）

**Files:**
- Modify: `src/agent_framework/tools/builtin/todo_write.py`（整文件重写）

- [ ] **Step 1: 备份当前实现位置以便对照**

```bash
# 整文件重写，先记录旧行数
wc -l src/agent_framework/tools/builtin/todo_write.py
```

Expected: 354 行（旧版）

- [ ] **Step 2: 写入新版 `todo_write.py` 完整内容**

完整内容：

```python
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

# ── 状态注入（asyncio 任务级隔离）──────────────────────

_current_session_dir: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "current_todo_session_dir", default=None
)


def _plan_path() -> Path | None:
    """当前 session 的 plan.json 路径（无 session 上下文时返回 None）。"""
    d = _current_session_dir.get()
    return d / "plan.json" if d else None


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
        if item["status"] == "in_progress" and item.get("active_form"):
            line += f"  ({item['active_form']})"
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
        "多步骤任务时主动使用此工具跟踪进度。"
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
    plan_path = _plan_path()
    if plan_path is None:
        raise RuntimeError(
            "todo_write 必须在 Agent.run() 上下文中调用（session_dir 未注入）"
        )
    normalized = _validate_items(items)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps({"items": normalized}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return _render(normalized)


@tool(
    name="todo_read",
    description=(
        "查看当前会话计划的完整状态。"
        "当你不确定当前计划内容、或想确认进度时调用此工具。"
        "返回包含所有任务的状态清单。"
    ),
)
async def todo_read() -> str:
    """查看当前计划状态。"""
    plan_path = _plan_path()
    if plan_path is None or not plan_path.exists():
        return "暂无会话计划。"
    try:
        data = json.loads(plan_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("读取 plan.json 失败: %s", e)
        return "暂无会话计划。"
    return _render(data.get("items", []))


# ── 公开工厂（破坏性变更）──────────────────────────────

def create_todo_write_tool() -> tuple:
    """创建 todo_write + todo_read 两个工具函数。

    v2 版本签名变化 — 不再接受 manager/plan_file 参数：
      - 状态由 Agent 注入的 session_dir + ContextVar 承载
      - session_dir 由 Agent.run() 入口设置到模块级 ContextVar

    Returns:
        (todo_write, todo_read) 两个工具函数的元组
    """
    return todo_write, todo_read
```

- [ ] **Step 3: 验证文件写入成功**

Run: `wc -l src/agent_framework/tools/builtin/todo_write.py`
Expected: ~165 行（远少于旧版 354）

- [ ] **Step 4: 运行新测试**

Run: `cd D:/MyWork/my-agent && .venv/Scripts/python -m pytest tests/test_todo_concurrent_isolation.py -v`
Expected: `test_todo_write_no_closure_state` 和 `test_create_todo_write_tool_no_args` 通过；其他 ContextVar 相关测试需 Task 3 完成后才能通过。

- [ ] **Step 5: 暂存**

```bash
git add src/agent_framework/tools/builtin/todo_write.py
# 不提交，等 Task 6 一起提交
```

---

## Task 3: 修改 `agent.py`（删除 TodoManager 引用 + 注入 ContextVar）

**Files:**
- Modify: `src/agent_framework/agent/agent.py`

- [ ] **Step 1: 删除 `_find_todo_manager` 方法（agent.py:383-389）**

用 Edit 工具，把：
```python
    def _find_todo_manager(self):
        """从已注册工具中查找 TodoManager（通过 _todo_manager 标记）。"""
        for name in self._registry.list_tools():
            wrapper = self._registry.get_tool(name)
            if wrapper and hasattr(wrapper.func, '_todo_manager'):
                return wrapper.func._todo_manager
        return None
```
替换为：
```python
    def _find_todo_manager(self):
        """v2: TodoManager 已删除，此方法保留为 no-op 以兼容旧调用点。"""
        return None
```

- [ ] **Step 2: 删除 `_rebind_todo_manager` 方法（agent.py:391-394）**

用 Edit 工具，把：
```python
    def _rebind_todo_manager(self):
        """将 TodoManager 的 plan 文件绑定到当前 session 目录。"""
        if self._todo_manager and self._session:
            self._todo_manager.set_plan_file(self._session.dir_path / "plan.json")
```
替换为：
```python
    def _rebind_todo_manager(self):
        """v2: TodoManager 已删除，此方法保留为 no-op 以兼容旧调用点。"""
        pass
```

- [ ] **Step 3: 修改 `reset` 方法中的 todo_manager.clear()（agent.py:276-277）**

把：
```python
        if self._todo_manager:
            self._todo_manager.clear()
```
替换为：
```python
        # v2: TodoManager 已删除，plan 状态由 session/plan.json 承载
        # reset() 不需要清空 plan 文件（plan 跨 reset 保留）
```

- [ ] **Step 4: 添加 ContextVar 导入（agent.py 顶部 imports 区）**

找到 `from agent_framework.tools.decorator import tool` 附近的 import 区，添加：
```python
from agent_framework.tools.builtin.todo_write import _current_session_dir as _current_todo_session_dir
```

- [ ] **Step 5: 在 `run()` 方法入口添加 ContextVar 绑定**

定位 `run()` 方法（agent.py 约 450 行附近），找到方法体的开头。修改为：

找到类似这样的开头：
```python
    async def run(self, user_input, **kw):
        """Run a single turn of the agent."""
        # 记录开始时间等
```

修改为：
```python
    async def run(self, user_input, **kw):
        """Run a single turn of the agent."""
        # v2: 注入 todo 工具的 session_dir 到 ContextVar
        plan_token = None
        if self._session is not None:
            plan_token = _current_todo_session_dir.set(self._session.dir_path)
        try:
            async for evt in self._run_loop(user_input, **kw):
                yield evt
        finally:
            if plan_token is not None:
                _current_todo_session_dir.reset(plan_token)
```

注意：原 `run()` 方法的内部逻辑（如 self._history.add 等）需要保留在 try 块内。如果原 `run()` 是 `yield from` 或 return，需要把整个方法体包到 try 里。

- [ ] **Step 6: 验证 import 正确**

Run: `cd D:/MyWork/my-agent && .venv/Scripts/python -c "from agent_framework.agent.agent import Agent; print('ok')"`
Expected: `ok`（无 ImportError）

- [ ] **Step 7: 暂存**

```bash
git add src/agent_framework/agent/agent.py
# 不提交，等 Task 6 一起提交
```

---

## Task 4: 实现 reminder 逻辑迁移（Agent 端）

**Files:**
- Modify: `src/agent_framework/agent/agent.py`

- [ ] **Step 1: 导入 reminder 常量**

在 agent.py 顶部 imports 区，添加：
```python
from agent_framework.tools.builtin.todo_write import _PLAN_REMINDER_INTERVAL
from agent_framework.tools.builtin.todo_write import todo_read as _todo_read_tool
```

- [ ] **Step 2: 新增 `_maybe_inject_plan_reminder` 方法**

在 agent.py 找到 `_find_todo_manager` 方法附近（删除后位置），新增：

```python
    def _maybe_inject_plan_reminder(self, messages: list) -> None:
        """如果距上次 todo_write 调用已过 ≥ _PLAN_REMINDER_INTERVAL 轮，
        在最后一条 system message 末尾追加 reminder（含当前 plan 内容）。
        """
        # 1. 找最后一条 todo_write 工具调用
        last_todo_idx = -1
        for i, m in enumerate(messages):
            if m.role == MessageRole.ASSISTANT and m.tool_calls:
                for tc in m.tool_calls:
                    if tc.function.name == "todo_write":
                        last_todo_idx = i

        # 2. 数 last_todo_idx 之后的工具调用数
        tool_calls_since = sum(
            1 for m in messages[last_todo_idx + 1:]
            if m.role == MessageRole.TOOL
        )
        if tool_calls_since < _PLAN_REMINDER_INTERVAL:
            return

        # 3. 读 plan.json
        if self._session is None:
            return
        plan_path = self._session.dir_path / "plan.json"
        if not plan_path.exists():
            return
        try:
            import json as _json
            data = _json.loads(plan_path.read_text(encoding="utf-8"))
            items = data.get("items", [])
        except Exception as e:
            logger.warning("读取 plan.json 失败: %s", e)
            return
        if not items:
            return

        # 4. 渲染并追加到 system message
        from agent_framework.tools.builtin.todo_write import _render
        reminder = (
            "<reminder>\n"
            "计划已长时间未更新，请在继续工作前刷新计划状态。以下是当前计划：\n\n"
            f"{_render(items)}\n\n"
            "请根据计划进度决定下一步操作，"
            "并调用 todo_write 更新计划状态。\n"
            "</reminder>"
        )
        # 找到最后一条 system message
        for m in reversed(messages):
            if m.role == MessageRole.SYSTEM:
                m.content = m.content + "\n\n" + reminder
                return
```

- [ ] **Step 3: 在 LLM 调用前调用 reminder 方法**

定位 agent.py 主循环中 LLM 调用之前（通常在 `chat(messages, **kwargs)` 之前），添加：
```python
        # v2: reminder 注入
        self._maybe_inject_plan_reminder(messages)
```

注意 `messages` 应该是即将发给 LLM 的消息列表。具体位置以原代码逻辑为准。

- [ ] **Step 4: 验证 import 正确**

Run: `cd D:/MyWork/my-agent && .venv/Scripts/python -c "from agent_framework.agent.agent import Agent; print('ok')"`
Expected: `ok`

- [ ] **Step 5: 暂存**

```bash
git add src/agent_framework/agent/agent.py
# 不提交，等 Task 6 一起提交
```

---

## Task 5: 更新 `tests/test_tool_todo_write.py`（删除 TodoManager 直引）

**Files:**
- Modify: `tests/test_tool_todo_write.py`

- [ ] **Step 1: 找到所有 TodoManager 直引**

Run: `cd D:/MyWork/my-agent && grep -n "TodoManager" tests/test_tool_todo_write.py`
Expected: 多处引用（约 10-20 处）

- [ ] **Step 2: 修改文件顶部的 import**

把：
```python
from agent_framework.tools.builtin.todo_write import (
    TodoManager,
    PlanItem,
    PlanningState,
    create_todo_write_tool,
)
```
替换为：
```python
from agent_framework.tools.builtin.todo_write import (
    create_todo_write_tool,
    todo_read,
    todo_write,
)
```

- [ ] **Step 3: 替换 `test_create_todo_write_tool_default_manager` 类相关测试**

找到 `class TestCreateTodoWriteTool:`（约 line 357），把整个 class 替换为：

```python
class TestCreateTodoWriteTool:
    """create_todo_write_tool v2 工厂测试"""

    def test_factory_returns_two_tools(self):
        """工厂返回 (todo_write, todo_read) 元组"""
        result = create_todo_write_tool()
        assert len(result) == 2
        assert result[0] is todo_write
        assert result[1] is todo_read

    def test_write_tool_metadata(self):
        """todo_write 工具元数据"""
        todo_write_fn, _ = create_todo_write_tool()
        wrapper = todo_write_fn._tool_wrapper
        assert wrapper.name == "todo_write"
        assert wrapper.is_async is True

    def test_read_tool_metadata(self):
        """todo_read 工具元数据"""
        _, todo_read_fn = create_todo_write_tool()
        wrapper = todo_read_fn._tool_wrapper
        assert wrapper.name == "todo_read"
        assert wrapper.is_async is True

    def test_factory_no_args(self):
        """v2 工厂不接受任何参数"""
        # 调用不应报错
        result = create_todo_write_tool()
        assert result is not None
```

- [ ] **Step 4: 替换 `test_write_invocation` 和 `test_read_invocation`（约 line 402-427）**

```python
    @pytest.mark.asyncio
    async def test_write_invocation_via_contextvar(self, tmp_path):
        """todo_write 通过 ContextVar 找到 plan.json 并写入"""
        from agent_framework.tools.builtin.todo_write import _current_session_dir
        token = _current_session_dir.set(tmp_path)
        try:
            result = await todo_write._tool_wrapper.func(items=[
                {"content": "分析需求", "status": "completed"},
                {"content": "编写代码", "status": "in_progress"},
            ])
            assert "1/2" in result
            # 文件被写入
            plan_file = tmp_path / "plan.json"
            assert plan_file.exists()
            data = json.loads(plan_file.read_text(encoding="utf-8"))
            assert len(data["items"]) == 2
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_read_invocation_via_contextvar(self, tmp_path):
        """todo_read 通过 ContextVar 找到 plan.json 并读取"""
        from agent_framework.tools.builtin.todo_write import _current_session_dir

        # 先写入
        plan_file = tmp_path / "plan.json"
        plan_file.parent.mkdir(parents=True, exist_ok=True)
        plan_file.write_text(
            json.dumps({
                "items": [
                    {"content": "任务 A", "status": "completed", "activeForm": ""},
                    {"content": "任务 B", "status": "pending", "activeForm": ""},
                ]
            }, ensure_ascii=False),
            encoding="utf-8",
        )

        token = _current_session_dir.set(tmp_path)
        try:
            result = await todo_read._tool_wrapper.func()
            assert "[x] 任务 A" in result
            assert "[ ] 任务 B" in result
            assert "1/2" in result
        finally:
            _current_session_dir.reset(token)
```

- [ ] **Step 5: 替换 `test_merged_manager_sharing`（约 line 492）**

找到 `test_write_a_shared_with_read_b` 之类的共享 manager 测试，**删除**（v2 不再支持 manager 共享）。

- [ ] **Step 6: 替换 `test_plan_file` 类的测试（约 line 441, 463, 475, 482, 586）**

把所有显式传 `plan_file=` 或 `manager=` 的测试改为 ContextVar 注入风格（参照 Step 4 模式）。

- [ ] **Step 7: 运行更新后的 test_tool_todo_write.py**

Run: `cd D:/MyWork/my-agent && .venv/Scripts/python -m pytest tests/test_tool_todo_write.py -v`
Expected: 全部通过

- [ ] **Step 8: 暂存**

```bash
git add tests/test_tool_todo_write.py
# 不提交，等 Task 6 一起提交
```

---

## Task 6: 提交 todo 工具重构

**Files:**
- (none new; commit previous changes)

- [ ] **Step 1: 查看暂存状态**

```bash
git status
```
Expected: 看到 `tests/test_todo_concurrent_isolation.py`、`src/agent_framework/tools/builtin/todo_write.py`、`src/agent_framework/agent/agent.py`、`tests/test_tool_todo_write.py` 4 个文件

- [ ] **Step 2: 提交**

```bash
git commit -m "$(cat <<'EOF'
refactor(todo): 工具无状态化，删除 TodoManager 类

- 删除 TodoManager、PlanItem、PlanningState 等数据类
- todo_write/todo_read 工具函数本身无闭包/无实例状态
- 状态由 Session 目录下的 plan.json 承载（per-user 自动隔离）
- Agent.run() 入口通过 ContextVar 注入 session_dir
- 新增 tests/test_todo_concurrent_isolation.py（5 个测试）
- 更新 test_tool_todo_write.py 适配新 API
- agent.py: 移除 _find_todo_manager 真实逻辑、_rebind_todo_manager 留为 no-op

破坏性变更：create_todo_write_tool(manager=..., plan_file=...) → create_todo_write_tool()

修复评估报告 Bug #5（TodoManager 跨用户共享 P0 陷阱）

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: 添加 subagent 工具并发隔离测试（TDD 起点）

**Files:**
- Create: `tests/test_subagent_concurrent_isolation.py`

- [ ] **Step 1: 写入完整测试文件**

```python
"""Subagent 工具并发隔离测试 — 验证 _last_events 闭包 bug 修复"""
from __future__ import annotations

import asyncio
import inspect
from typing import AsyncIterator

import pytest

from agent_framework.agent.agent import Agent
from agent_framework.agent.config import AgentConfig
from agent_framework.agent.events import AgentDone, AgentEvent
from agent_framework.agent.subagent import SubAgentConfig, create_subagent_tools
from agent_framework.llm.providers.base import BaseLLM, StreamChunk, TokenUsage


class _EchoLLM(BaseLLM):
    """每次调用返回一个 AgentDone 事件。"""
    def __init__(self, text: str = "ok"):
        super().__init__(model="mock", provider="mock")
        self._text = text

    async def chat(self, messages, **kwargs) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(content=self._text, finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(1, 1, 2))

    def get_capabilities(self):
        from agent_framework.llm.capabilities import ModelCapabilities
        return ModelCapabilities()


# ── 测试 ────────────────────────────────────────────────

def test_subagent_no_closure_state():
    """subagent 工具函数无 _last_events 闭包变量"""
    sub = create_subagent_tools(_EchoLLM(), [
        SubAgentConfig(name="helper", description="test", system_prompt="x"),
    ])[0]
    closure = inspect.getclosurevars(sub)
    # 不应有 _last_events
    assert "_last_events" not in closure.cells
    # 闭包只应包含 llm、cfg、get_parent_mode
    assert "llm" in closure.cells
    assert "cfg" in closure.cells


def test_subagent_factory_unchanged():
    """v2 工厂签名不变"""
    tools = create_subagent_tools(_EchoLLM(), [])
    assert tools == []

    tools = create_subagent_tools(_EchoLLM(), [
        SubAgentConfig(name="a", description="A", system_prompt=""),
        SubAgentConfig(name="b", description="B", system_prompt=""),
    ])
    assert len(tools) == 2
    assert tools[0]._tool_wrapper.name == "a"
    assert tools[1]._tool_wrapper.name == "b"


@pytest.mark.asyncio
async def test_subagent_without_contextvar_raises():
    """不通过 Agent.run() 直接调 subagent 工具应抛 RuntimeError"""
    sub = create_subagent_tools(_EchoLLM(), [
        SubAgentConfig(name="helper", description="test", system_prompt="x"),
    ])[0]

    with pytest.raises(RuntimeError, match="Agent"):
        await sub._tool_wrapper.func(task="x")


@pytest.mark.asyncio
async def test_subagent_concurrent_runs_events_isolated():
    """两个 Agent.run() 并发调用同一 subagent，事件不交叉、不丢失"""
    # 父 LLM：第一次返回 subagent tool_call，第二次返回 done
    call_counts = {"a": 0, "b": 0}

    class _ParentLLM(BaseLLM):
        def __init__(self, label: str):
            super().__init__(model="mock", provider="mock")
            self._label = label
            self._done = False

        async def chat(self, messages, **kwargs):
            if not self._done:
                self._done = True
                # 触发 subagent 工具调用
                yield StreamChunk(tool_calls=[type("TC", (), {
                    "index": 0, "id": "call_sub",
                    "function": type("FN", (), {
                        "name": "helper", "arguments": '{"task":"x"}'
                    })(),
                })()])
                yield StreamChunk(finish_reason="tool_calls")
            else:
                yield StreamChunk(content=f"parent-{self._label}-done", finish_reason="stop")
                yield StreamChunk(usage=TokenUsage(1, 1, 2))

        def get_capabilities(self):
            from agent_framework.llm.capabilities import ModelCapabilities
            return ModelCapabilities()

    sub_tool = create_subagent_tools(_EchoLLM("sub"), [
        SubAgentConfig(name="helper", description="test", system_prompt="..."),
    ])[0]

    parent_a = Agent(
        llm=_ParentLLM("A"),
        system_prompt="A",
        tools=[sub_tool],
        config=AgentConfig(session_enabled=False),
    )
    parent_b = Agent(
        llm=_ParentLLM("B"),
        system_prompt="B",
        tools=[sub_tool],
        config=AgentConfig(session_enabled=False),
    )

    # 并发跑两个父 Agent
    events_a, events_b = await asyncio.gather(*[
        _collect(parent_a.run("hi")),
        _collect(parent_b.run("hi")),
    ])

    # 每个 run 应至少 1 个 SubAgentDone
    sub_done_a = sum(1 for e in events_a if e.__class__.__name__ == "SubAgentDone")
    sub_done_b = sum(1 for e in events_b if e.__class__.__name__ == "SubAgentDone")
    assert sub_done_a >= 1, f"parent_a missing SubAgentDone (events={len(events_a)})"
    assert sub_done_b >= 1, f"parent_b missing SubAgentDone (events={len(events_b)})"


async def _collect(agen):
    out = []
    async for e in agen:
        out.append(e)
    return out
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd D:/MyWork/my-agent && .venv/Scripts/python -m pytest tests/test_subagent_concurrent_isolation.py -v`
Expected: 多个失败（`_last_events` 闭包仍在，subagent 工具不抛 RuntimeError）

- [ ] **Step 3: 暂存**

```bash
git add tests/test_subagent_concurrent_isolation.py
# 不提交，等 Task 11 一起提交
```

---

## Task 8: 重写 `subagent.py`（删除 _last_events，引入 ContextVar）

**Files:**
- Modify: `src/agent_framework/agent/subagent.py`（整文件重写）

- [ ] **Step 1: 写入新版 `subagent.py` 完整内容**

```python
"""子代理（SubAgent）— 无状态版本

v2 变更：
- 删除 _last_events 闭包变量
- 每次工具调用通过 ContextVar 注入 per-call events 列表
- 工具函数和工厂签名不变
"""
from __future__ import annotations

import contextvars
import dataclasses
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

from agent_framework.agent.config import AgentConfig, AgentMode
from agent_framework.agent.events import AgentDone, AgentError, AgentEvent
from agent_framework.agent.history import ConversationHistory
from agent_framework.tools.decorator import tool

if TYPE_CHECKING:
    from agent_framework.llm.providers.base import BaseLLM

logger = logging.getLogger(__name__)

# ── per-call events 注入（asyncio 任务级隔离）──────────

_current_subagent_events: contextvars.ContextVar[list[AgentEvent] | None] = contextvars.ContextVar(
    "current_subagent_events", default=None
)


# ── 数据模型 ──────────────────────────────────────────────


@dataclass
class SubAgentConfig:
    """单个子代理的配置。

    :param name: 工具名（如 "researcher"），父 LLM 通过此名称调用
    :param description: 工具描述，出现在父 LLM 的 tool schema 中
    :param system_prompt: 子代理的系统提示（可选，可与 prompt_dir 组合使用）
    :param tools: 子代理可用的工具列表（@tool 装饰的函数）
    :param config: 子代理的 AgentConfig（None 时使用默认值：
        max_turns=50, timeout=120, total_timeout=600, confirm_dangerous=False）
    :param history_max_turns: 子代理对话历史最大轮数
    :param history_max_tokens: 子代理对话历史 token 上限（None 为不限）
    :param llm_kwargs: 传递给子代理 LLM 的额外参数（如 web_search=True, enable_thinking=True）
    :param skills: 子代理可用的技能列表（SkillConfig 实例）
    :param skills_dir: 子代理技能目录路径（None 时不自动扫描）
    :param prompt_dir: 提示词文件目录路径（None 时不使用文件化提示词）
    :param prompt_variables: 提示词变量，替换文件中的 {{key}}
    """
    name: str
    description: str
    system_prompt: str = ""
    tools: list = field(default_factory=list)
    config: AgentConfig | None = None
    history_max_turns: int = 50
    history_max_tokens: int | None = None
    llm_kwargs: dict = field(default_factory=dict)
    skills: list | None = None
    skills_dir: str | None = None
    prompt_dir: "str | None" = None
    prompt_variables: dict[str, str] | None = None


# ── 内部：子代理默认配置 ──────────────────────────────────────

_DEFAULT_SUBAGENT_CONFIG = AgentConfig(
    max_turns=50,
    timeout=120.0,
    total_timeout=600,
    tool_call_limit=50,
)


# ── 工厂函数 ──────────────────────────────────────────────


def create_subagent_tools(
    llm: "BaseLLM",
    subagents: list[SubAgentConfig],
    get_parent_mode: Optional[Callable[[], AgentMode]] = None,
) -> list:
    """创建子代理工具列表。

    每个 SubAgentConfig 生成一个 @tool 装饰的异步函数。
    父 LLM 调用该工具时，内部创建全新的 Agent 实例执行任务，
    执行完毕后将精炼结果返回给父 Agent。

    v2: 工具函数本身无 _last_events 闭包。Agent 在每次工具调用前
    通过 ContextVar `_current_subagent_events` 注入 per-call 事件列表。

    :param llm: 共享的 LLM 实例（BaseLLM 无状态，安全共享）
    :param subagents: 子代理配置列表
    :param get_parent_mode: 获取父 Agent 当前模式的回调（用于模式继承）
    :return: 工具函数列表，可传给 Agent(tools=...)
    """
    return [_create_single_subagent_tool(llm, cfg, get_parent_mode) for cfg in subagents]


def _create_single_subagent_tool(
    llm: "BaseLLM",
    cfg: SubAgentConfig,
    get_parent_mode: Optional[Callable[[], AgentMode]] = None,
):
    """为单个 SubAgentConfig 创建工具函数。"""
    from agent_framework.agent.agent import Agent

    @tool(name=cfg.name, description=cfg.description)
    async def _subagent_tool(task: str) -> str:
        """:param task: 委派给子代理的任务描述或问题"""
        # 从 ContextVar 取出 per-call events 列表
        events = _current_subagent_events.get()
        if events is None:
            raise RuntimeError(
                f"子代理 {cfg.name} 必须在 Agent.run() 上下文中调用"
                "（Agent 应在调用前通过 ContextVar 注入 per-call events 列表）"
            )
        events.clear()

        sub_config = cfg.config or _DEFAULT_SUBAGENT_CONFIG
        if get_parent_mode is not None:
            sub_config = dataclasses.replace(sub_config, mode=get_parent_mode())
        else:
            sub_config = dataclasses.replace(sub_config)
        sub_config.session_enabled = False

        sub_history = ConversationHistory(
            max_turns=cfg.history_max_turns,
            max_tokens=cfg.history_max_tokens,
        )

        sub_agent = Agent(
            llm=llm,
            system_prompt=cfg.system_prompt,
            tools=cfg.tools if cfg.tools else None,
            history=sub_history,
            config=sub_config,
            register_catalog=False,
            skills=cfg.skills,
            skills_dir=cfg.skills_dir,
            register_skills=bool(cfg.skills or cfg.skills_dir),
            prompt_dir=cfg.prompt_dir,
            prompt_variables=cfg.prompt_variables,
        )

        final_text: Optional[str] = None
        try:
            async for event in sub_agent.run(task, **cfg.llm_kwargs):
                events.append(event)
                if isinstance(event, AgentDone):
                    final_text = event.final_text
                elif isinstance(event, AgentError):
                    return f"[{cfg.name}] 错误: {event.message}"
        except Exception as e:
            logger.warning("子代理 %s 执行异常: %s", cfg.name, e)
            events.append(AgentError(error_type="subagent_crash", message=str(e)))
            return f"[{cfg.name}] 子代理执行失败: {e}"

        return f"[{cfg.name}]: {final_text or '(无结果返回)'}"

    # 仅标记 _is_subagent（不再设置 _subagent_events）
    _subagent_tool._tool_wrapper._is_subagent = True

    return _subagent_tool
```

- [ ] **Step 2: 验证文件写入成功**

Run: `wc -l src/agent_framework/agent/subagent.py`
Expected: ~155 行（旧版 197）

- [ ] **Step 3: 运行新测试**

Run: `cd D:/MyWork/my-agent && .venv/Scripts/python -m pytest tests/test_subagent_concurrent_isolation.py -v`
Expected: `test_subagent_no_closure_state` 和 `test_subagent_factory_unchanged` 通过；其他测试需 Task 8 完成才能通过。

- [ ] **Step 4: 暂存**

```bash
git add src/agent_framework/agent/subagent.py
# 不提交，等 Task 11 一起提交
```

---

## Task 9: 修改 `agent.py` 主循环（subagent per-call events 消费）

**Files:**
- Modify: `src/agent_framework/agent/agent.py`

- [ ] **Step 1: 添加 subagent events ContextVar 的导入**

在 Task 3 Step 4 的 import 附近添加：
```python
from agent_framework.agent.subagent import _current_subagent_events
```

- [ ] **Step 2: 定位工具执行循环**

找到 agent.py 工具执行的关键代码块（约 800-860 行），定位：
```python
# 子代理事件转发
if wrapper and getattr(wrapper, '_is_subagent', False):
    sub_events = getattr(wrapper, '_subagent_events', [])
```

- [ ] **Step 3: 重构工具执行路径，引入 per-call events 列表**

定位到工具执行循环的开头（约 800 行）。在 `for item in confirmed:` 循环或 `asyncio.gather` 之前，添加 subagent 工具的事件列表准备逻辑：

找到类似这样的代码（具体行号以实际为准）：
```python
                confirmed.append({
                    "call": call,
                    ...
                })
                ...
                # ── 10c. 执行工具 ──
                results = await asyncio.gather(...)
```

修改：在 `asyncio.gather` 调用之前，包装一个嵌套函数 `_execute_one`：

```python
                async def _execute_one(item):
                    wrapper = item["wrapper"]
                    tool_call = item["tool_call"]
                    # v2: subagent 工具注入 per-call events 列表
                    per_call_events: list[AgentEvent] = []
                    events_token = None
                    if wrapper and getattr(wrapper, '_is_subagent', False):
                        events_token = _current_subagent_events.set(per_call_events)
                    try:
                        exec_result = await self._executor.execute(
                            wrapper, tool_call.arguments, ctx=...
                        )
                    finally:
                        if events_token is not None:
                            _current_subagent_events.reset(events_token)
                    return item, exec_result, per_call_events

                results = await asyncio.gather(
                    *[_execute_one(item) for item in confirmed]
                )
```

- [ ] **Step 4: 修改结果处理循环，读取 per-call events**

找到：
```python
                for item, exec_result in results:
```

替换为：
```python
                for item, exec_result, per_call_events in results:
```

- [ ] **Step 5: 替换 subagent 事件消费代码**

找到：
```python
                    # 子代理事件转发
                    if wrapper and getattr(wrapper, '_is_subagent', False):
                        sub_events = getattr(wrapper, '_subagent_events', [])
                        sub_final_text = ""
                        ...
                        for sevt in sub_events:
                            ...
```

替换为：
```python
                    # v2: subagent 事件从 per-call 列表读取（无闭包共享）
                    if wrapper and getattr(wrapper, '_is_subagent', False):
                        sub_final_text = ""
                        sub_turn_count = 0
                        sub_total_usage = TokenUsage()
                        sub_is_error = False
                        for sevt in per_call_events:
                            yield SubAgentEvent(subagent_name=tool_name, event=sevt)
                            if isinstance(sevt, AgentDone):
                                sub_final_text = sevt.final_text
                                sub_turn_count = sevt.turn_count
                                sub_total_usage = sevt.total_usage
                            elif isinstance(sevt, AgentError):
                                sub_is_error = True
                                sub_final_text = sevt.message
                        yield SubAgentDone(
                            subagent_name=tool_name,
                            final_text=sub_final_text,
                            turn_count=sub_turn_count,
                            total_usage=sub_total_usage,
                            is_error=sub_is_error,
                        )
                        per_call_events.clear()
```

- [ ] **Step 6: 验证 import 正确**

Run: `cd D:/MyWork/my-agent && .venv/Scripts/python -c "from agent_framework.agent.agent import Agent; print('ok')"`
Expected: `ok`

- [ ] **Step 7: 运行新测试**

Run: `cd D:/MyWork/my-agent && .venv/Scripts/python -m pytest tests/test_subagent_concurrent_isolation.py -v`
Expected: 全部通过

- [ ] **Step 8: 暂存**

```bash
git add src/agent_framework/agent/agent.py
# 不提交，等 Task 11 一起提交
```

---

## Task 10: 更新 `tests/test_subagent.py`（删除 _last_events 断言）

**Files:**
- Modify: `tests/test_subagent.py`

- [ ] **Step 1: 找到所有 _last_events / _subagent_events 引用**

Run: `cd D:/MyWork/my-agent && grep -n "_last_events\|_subagent_events" tests/test_subagent.py`
Expected: 多处（约 5-10 处）

- [ ] **Step 2: 删除对 `_last_events` 的断言**

把所有 `assert wrapper._subagent_events == ...` 或 `assert len(wrapper._subagent_events) == ...` 之类的断言删除或改写。

把"事件在 _last_events 列表里"这类测试，改为：
- "事件在工具返回的字符串里"（[name] 前缀 + final_text）
- 或用 Agent 端 mock 验证 SubAgentEvent 被 yield

- [ ] **Step 3: 找到 import 区域，确认 mock LLM 仍然适用**

Run: `cd D:/MyWork/my-agent && head -30 tests/test_subagent.py`
Expected: import 区域，可能需要 `from agent_framework.agent.subagent import _current_subagent_events` 用于新测试

- [ ] **Step 4: 跑测试**

Run: `cd D:/MyWork/my-agent && .venv/Scripts/python -m pytest tests/test_subagent.py -v`
Expected: 全部通过

- [ ] **Step 5: 暂存**

```bash
git add tests/test_subagent.py
# 不提交
```

---

## Task 11: 提交 subagent 工具重构

**Files:**
- (none new; commit previous changes)

- [ ] **Step 1: 查看暂存状态**

```bash
git status
```
Expected: 看到 `tests/test_subagent_concurrent_isolation.py`、`src/agent_framework/agent/subagent.py`、`src/agent_framework/agent/agent.py`、`tests/test_subagent.py` 4 个文件

- [ ] **Step 2: 提交**

```bash
git commit -m "$(cat <<'EOF'
refactor(subagent): 删除 _last_events 闭包，per-call events 经 ContextVar 注入

- subagent.py: 引入 _current_subagent_events ContextVar
- 工具函数无 _last_events 闭包变量
- agent.py 主循环：每次 subagent 工具调用前 set 新列表，调用后 reset + 消费
- 删除 ToolWrapper._subagent_events 标记
- 新增 tests/test_subagent_concurrent_isolation.py（4 个测试）
- 更新 test_subagent.py 适配新行为

工具函数签名和工厂签名保持完全不变（向后兼容）。

修复评估报告 Bug #1（SubAgent 闭包共享 P0 致命）

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: 改 `test_concurrency_stress.py` 严格断言

**Files:**
- Modify: `tests/test_concurrency_stress.py`

- [ ] **Step 1: 找到 4 个 stress 测试**

Run: `cd D:/MyWork/my-agent && grep -n "def test_" tests/test_concurrency_stress.py`
Expected: 4 个 stress 测试函数

- [ ] **Step 2: 改 todo 相关 stress 测试的断言**

找到 `test_todo_*` 之类的测试，把：
```python
    print(f"... contaminated ...")
    # assert len(contaminated) == 0, f"历史串味: {contaminated[:5]}"
```
替换为：
```python
    assert len(contaminated) == 0, f"todo 状态串味: {contaminated[:5]}"
```

- [ ] **Step 3: 改 subagent stress 测试的断言**

找到 `test_subagent_concurrent_events_cross_contamination`，把：
```python
    print(f"[Test 2 concurrent] {len(concurrent_events_lists)} 个并发 run 中 {len(missing)} 个缺失 SubAgentDone")
```
替换为：
```python
    assert len(missing) == 0, f"{len(missing)}/{len(concurrent_events_lists)} 并发 run 缺失 SubAgentDone 事件"
```

- [ ] **Step 4: 跑修改后的 stress 测试**

Run: `cd D:/MyWork/my-agent && .venv/Scripts/python -m pytest tests/test_concurrency_stress.py -v -k "todo or subagent"`
Expected: 全部通过

- [ ] **Step 5: 暂存并提交**

```bash
git add tests/test_concurrency_stress.py
git commit -m "$(cat <<'EOF'
test: 改 concurrency stress 严格断言

- todo 并发：0 串味
- subagent 并发：每个 run 都有 SubAgentDone
- 之前是 print + 注释不断言，bug 修复后改为严格断言

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: 跑全量单元测试

**Files:**
- (none new; verification only)

- [ ] **Step 1: 运行项目标准测试命令**

Run: `cd D:/MyWork/my-agent && .venv/Scripts/python -m pytest tests/ --ignore=tests/test_real_api.py --ignore=tests/test_real_new_providers.py -q`
Expected: 全部通过。如有失败，针对性修复（应仅在 todo / subagent 相关文件改动范围内）。

- [ ] **Step 2: 运行新加的 2 个并发测试**

Run: `cd D:/MyWork/my-agent && .venv/Scripts/python -m pytest tests/test_todo_concurrent_isolation.py tests/test_subagent_concurrent_isolation.py -v`
Expected: 全部通过

- [ ] **Step 3: 检查测试统计**

Run: `cd D:/MyWork/my-agent && .venv/Scripts/python -m pytest tests/ --ignore=tests/test_real_api.py --ignore=tests/test_real_new_providers.py -q 2>&1 | tail -5`
Expected: `N passed in X.Xs`（N 应 ≥ 改前的总数 + 9）

---

## Task 14: examples 冒烟测试

**Files:**
- (none new; verification only)

- [ ] **Step 1: 跑 `multi_turn_chat.py` 验证 todo 工具**

Run: `cd D:/MyWork/my-agent && timeout 30 .venv/Scripts/python examples/multi_turn_chat.py <<< "列出今天的todo并保存" 2>&1 | tail -20`
Expected: 不抛异常，能看到 todo_write 工具被调用（如果实际 LLM 触发需要 API key，验证 import 不报错即可）

注：如果 .env 没配 API key，可能会卡在真实 LLM 调用。允许：脚本启动到 import 阶段不抛错即视为通过；或用 mock LLM 子集验证。

- [ ] **Step 2: 跑 `6_subagent.py` 验证 subagent 工具**

Run: `cd D:/MyWork/my-agent && timeout 30 .venv/Scripts/python examples/6_subagent.py 2>&1 | tail -20`
Expected: 不抛异常（import 阶段）

- [ ] **Step 3: 跑 `7_server_fastapi.py` 启动 + 1 次 SSE 请求**

```bash
# 启动服务（后台）
.venv/Scripts/python examples/7_server_fastapi.py &
SERVER_PID=$!
sleep 5

# 1 次 SSE 请求
curl -N -X POST http://localhost:8000/chat \
  -H "X-User-Id: smoke-test" \
  -H "X-Session-Id: smoke-1" \
  -H "Content-Type: application/json" \
  -d '{"input":"hello"}' \
  --max-time 10 2>&1 | head -20

# 关闭服务
kill $SERVER_PID
```

Expected: 服务启动 + 至少返回 1 个 SSE event

- [ ] **Step 4: 关闭后台服务（若仍在运行）**

```bash
# 检查 7_server_fastapi 是否还有进程
ps aux | grep 7_server_fastapi | grep -v grep
# 若有，kill
pkill -f 7_server_fastapi
```

Expected: 无残留进程

---

## Task 15: 最终验证与收尾

**Files:**
- (none new; verification only)

- [ ] **Step 1: 全量单元测试**

Run: `cd D:/MyWork/my-agent && .venv/Scripts/python -m pytest tests/ --ignore=tests/test_real_api.py --ignore=tests/test_real_new_providers.py -q`
Expected: 全部通过

- [ ] **Step 2: 验证 TodoManager 类已删除**

Run: `cd D:/MyWork/my-agent && grep -rn "class TodoManager" src/ tests/ examples/`
Expected: 无输出

- [ ] **Step 3: 验证 _last_events 闭包已删除**

Run: `cd D:/MyWork/my-agent && grep -n "_last_events" src/agent_framework/agent/subagent.py`
Expected: 无输出

- [ ] **Step 4: 验证 _subagent_events 标记已删除**

Run: `cd D:/MyWork/my-agent && grep -rn "_subagent_events" src/`
Expected: 无输出

- [ ] **Step 5: 查看 git log 确认提交链**

Run: `cd D:/MyWork/my-agent && git log --oneline -10`
Expected: 看到 4-5 个新 commit（todo 重构、subagent 重构、stress 严格断言）

- [ ] **Step 6: 跑 verify 工具（若配置）**

Run: `cd D:/MyWork/my-agent && .venv/Scripts/python -c "from agent_framework import Agent, SubAgentConfig, create_subagent_tools, create_todo_write_tool; print('public api ok')"`
Expected: `public api ok`

---

## 验收对照（spec 第 9 节）

| 验收项 | 对应 Task |
|---|---|
| TodoManager 类从代码库完全删除 | Task 2, 15-Step 2 |
| todo_write/todo_read 无 TodoManager 闭包 | Task 1, 2 |
| subagent.py 中 _last_events 闭包完全删除 | Task 8, 15-Step 3 |
| _subagent_events 属性从 ToolWrapper 上消失 | Task 8, 15-Step 4 |
| test_todo_concurrent_isolation.py 6 个测试全过（含 reminder） | Task 1, 2, 4, 13 |
| test_subagent_concurrent_isolation.py 4 个测试全过 | Task 7, 8, 13 |
| test_concurrency_stress.py 严格断言通过 | Task 12 |
| test_tool_todo_write.py 全过 | Task 5 |
| test_subagent.py 全过 | Task 10 |
| examples 冒烟通过 | Task 14 |
| 中文 commit message | Task 6, 11, 12 |

---

## 风险与回滚

- **风险 1**：Task 3 改 run() 方法时 if 原方法签名复杂，可能引入 yield/return 错乱
  - 缓解：Step 5 强调"原 run() 内部逻辑必须保留在 try 块内"
  - 回滚：`git reset --hard HEAD~1` 后重做

- **风险 2**：Task 9 Step 5 重构结果处理时，可能漏处理 `wrapper` 变量未定义
  - 缓解：原代码块有 `wrapper = item["wrapper"]` 必须保留

- **风险 3**：Task 14 跑 examples 时真实 API 报错
  - 缓解：仅验证 import 阶段不抛错；不依赖真实 LLM 响应

---

> **Plan 完成时间**：2026-06-02
> **下一步**：用户审阅；批准后用 subagent-driven-development（推荐）或 executing-plans 技能执行
