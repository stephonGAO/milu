# Todo 工具与 SubAgent 工具无状态化重构

> **日期**：2026-06-02
> **范围**：`src/agent_framework/tools/builtin/todo_write.py` + `src/agent_framework/agent/subagent.py` + `src/agent_framework/agent/agent.py`
> **目的**：消除 P0 闭包 bug，让两个工具天然支持多用户并发
> **配套文档**：`docs/superpowers/spikes/2026-06-02-stateless-tools-comparison.md`（方案对比）、`docs/superpowers/specs/2026-06-02-multi-user-concurrency-evaluation.md`（整体评估）

---

## 1. 背景与目标

### 1.1 现状问题

**todo_write（`src/agent_framework/tools/builtin/todo_write.py`）**
- `TodoManager` 类持有 `state.items` / `state.rounds_since_update` 实例状态
- `create_todo_write_tool(manager)` 通过闭包捕获 manager，绑定到工具函数
- Agent 通过 `_todo_manager` 属性 + `_rebind_todo_manager()` 突变式切换 plan_file
- 评估报告 Bug #5：跨用户共享 TodoManager 是 P0 陷阱（即使 AgentPool 已隔离，仍是 API 暴露的隐患）

**subagent_tools（`src/agent_framework/agent/subagent.py:130`）**
```python
_last_events: list[AgentEvent] = []  # 闭包变量！

@tool(...)
async def _subagent_tool(task: str) -> str:
    _last_events.clear()  # 并发时清掉对方的 events
    async for event in sub_agent.run(task):
        _last_events.append(event)  # 事件交叉
```
- 评估报告 Bug #1：P0 致命，stress test 验证 50% 事件丢失
- 同一父 Agent 多并发 run 看到对方的子事件

### 1.2 目标

- **todo_write**：删除 `TodoManager` 类，工具函数无闭包/无实例状态；状态完全由 `Session/plan.json` 文件承载；LLM 通过 `todo_read` 工具主动拉取
- **subagent_tools**：删除 `_last_events` 闭包；每次工具调用通过 `ContextVar` 注入 per-call events 列表；工具函数和工厂函数签名保持不变

### 1.3 非目标

- 不重构 Agent 主循环的 per-run 状态字段（`_work_started` / `_plan_created` / `_conn_retry_count`）— 属 PR-3 范围
- 不修复 Session JSONL 并发写加锁 — 属 PR-4 范围
- 不改变 LLM 看到的工具签名（`todo_write(items)` / `todo_read()` / `subagent(task)`）
- 不改变 subagent 工厂的公开 API

---

## 2. 方案选型

| 工具 | 方案 | 选型理由 |
|---|---|---|
| todo_write | **A（纯无状态 + 外部存储）** | LLM 需主动调用 `todo_read` 拉取数据（用户明确要求），session 文件天然 per-user 隔离，删除 TodoManager 类最彻底 |
| subagent_tools | **C（per-call events 列表）** | 工具函数/工厂签名不变，向后兼容；per-call 列表经 ContextVar 注入；与评估报告 PR-2 思路一致 |

详细方案对比见 spike 文档第 3 节。

---

## 3. 详细设计

### 3.1 todo_write 重构

#### 3.1.1 新文件 `src/agent_framework/tools/builtin/todo_write.py`

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

# ── 常量 ──────────────────────────────────────────────
_PLAN_REMINDER_INTERVAL = 3
_MAX_PLAN_ITEMS = 12
_VALID_STATUSES = {"pending", "in_progress", "completed"}
_STATUS_MARKERS = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}

# ── 状态注入 ──────────────────────────────────────────
# Agent.run() 入口 set，出口 reset。asyncio 任务级隔离。
_current_session_dir: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "current_todo_session_dir", default=None
)


def _plan_path() -> Path | None:
    d = _current_session_dir.get()
    return d / "plan.json" if d else None


# ── 纯函数（无状态）────────────────────────────────────

def _validate_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """校验 + 规范化计划条目。纯函数。"""
    if len(items) > _MAX_PLAN_ITEMS:
        raise ValueError(f"计划最多 {_MAX_PLAN_ITEMS} 个条目，当前提供了 {len(items)} 个")

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
    """:param items: 计划条目列表。每个条目包含 content/status/activeForm"""
    plan_path = _plan_path()
    if plan_path is None:
        raise RuntimeError(
            "todo_write 必须在 Agent 上下文中调用（session_dir 未注入）"
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

    注意：v2 版本签名变化 — 不再接受 manager/plan_file 参数。
    - 状态由 Agent 注入的 session_dir + ContextVar 承载
    - session_dir 必须由 Agent.run() 入口设置

    Returns:
        (todo_write, todo_read) 两个工具函数的元组
    """
    return todo_write, todo_read
```

#### 3.1.2 Agent 端改动 `src/agent_framework/agent/agent.py`

**删除**：
- 第 186-187 行：`self._work_started` / `self._plan_created` 字段（**保留**，本 PR 不动）
- 第 190-192 行：`_todo_manager` 字段初始化 + `_rebind_todo_manager()` 调用
- 第 276-277 行（`reset` 方法）：`self._todo_manager.clear()`
- 第 299 行（`new_session`）：`_rebind_todo_manager()` 调用
- 第 314 行（`load_session`）：`_rebind_todo_manager()` 调用
- 第 383-389 行：`_find_todo_manager()` 方法（整段）
- 第 391-394 行：`_rebind_todo_manager()` 方法（整段）

**新增**：
- 第 195 行附近：记录 session 目录引用
  ```python
  self._plan_session_dir: Path | None = (
      self._session.dir_path if self._session else None
  )
  ```
- `run()` 方法入口：注入 session_dir 到 ContextVar
  ```python
  async def run(self, user_input, **kw):
      # 注意：_RunContext（per-run 状态）属 PR-3 范围，本 PR 暂不引入
      # 本 PR 维持原 run() 签名和内部循环结构
      token = None
      if self._plan_session_dir is not None:
          token = _current_todo_session_dir.set(self._plan_session_dir)
      try:
          async for evt in self._run_loop(user_input, **kw):
              yield evt
      finally:
          if token is not None:
              _current_todo_session_dir.reset(token)
  ```
- 引入新模块级 ContextVar（在文件顶部）：
  ```python
  from agent_framework.tools.builtin.todo_write import (
      _current_session_dir as _current_todo_session_dir,
  )
  ```

**reminder 逻辑迁移**：在 LLM 调用前（`_run_loop` 内部），扫描 history 数距上次 `todo_write` 调用的工具调用数；若 ≥ `_PLAN_REMINDER_INTERVAL` 且 plan.json 存在，向 system prompt 增量部分追加 reminder（rendered 当前 plan）。具体实现：新增方法 `_maybe_inject_plan_reminder(messages: list[Message]) -> None`，把 reminder 拼到当前 system message 末尾。

**实施注意**：因为 `_work_started` / `_plan_created` 由 PR-3 处理（per-run 状态），本 PR 只迁移 TodoManager 相关代码，**不**触碰 `_work_started` / `_plan_created`。这两块的 todo 工具感知可保留（agent.py:864 `if tool_name == "todo_write": self._plan_created = True`）。

### 3.2 subagent_tools 重构

#### 3.2.1 新文件 `src/agent_framework/agent/subagent.py`

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

# ── per-call events 注入 ─────────────────────────────────
_current_subagent_events: contextvars.ContextVar[list[AgentEvent] | None] = contextvars.ContextVar(
    "current_subagent_events", default=None
)


@dataclass
class SubAgentConfig:
    """单个子代理的配置。（与 v1 完全相同）"""
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


_DEFAULT_SUBAGENT_CONFIG = AgentConfig(
    max_turns=50,
    timeout=120.0,
    total_timeout=600,
    tool_call_limit=50,
)


def create_subagent_tools(
    llm: "BaseLLM",
    subagents: list[SubAgentConfig],
    get_parent_mode: Optional[Callable[[], AgentMode]] = None,
) -> list:
    """创建子代理工具列表。（签名与 v1 相同）"""
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
        events.clear()  # 本次调用从空开始

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
                events.append(event)  # 写入 per-call 列表（无闭包）
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

#### 3.2.2 Agent 主循环改动 `src/agent_framework/agent/agent.py`

**导入**：
```python
from agent_framework.agent.subagent import _current_subagent_events
```

**修改工具执行路径**（约第 800-855 行）：

```python
# ── 10c. 执行工具（每个工具独立 per-call 状态）──
async def _execute_tool_item(item, run_context):
    wrapper = item["wrapper"]
    tool_call = item["tool_call"]
    tool_name = item["tool_name"]

    # subagent 工具：注入 per-call events 列表
    per_call_events: list[AgentEvent] = []
    events_token = None
    if wrapper and getattr(wrapper, '_is_subagent', False):
        events_token = _current_subagent_events.set(per_call_events)

    try:
        exec_result = await self._executor.execute(
            wrapper, tool_call.arguments, ctx=run_context
        )
    finally:
        if events_token is not None:
            _current_subagent_events.reset(events_token)

    return exec_result, per_call_events
```

**消费 subagent 事件**（替换原 agent.py:830-853）：

```python
# 工具执行后
for item, exec_result_pair in results:
    exec_result, per_call_events = exec_result_pair
    tool_name = item["tool_name"]
    wrapper = item["wrapper"]

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

    yield ToolResult(...)  # 不变
    # ... 后续不变
```

**删除**：
- 旧 agent.py:830-853 块中 `getattr(wrapper, '_subagent_events', [])` 引用

---

## 4. 数据流示例

### 4.1 todo_write 多用户并发

```
用户 A:  Agent A (session_dir=/sessions/A)
           └─ run() set _current_todo_session_dir = /sessions/A
           └─ LLM → tool_call(todo_write, items=[...])
           └─ 工具读 ContextVar → /sessions/A/plan.json → 写文件
           └─ reset()

用户 B:  Agent B (session_dir=/sessions/B)
           └─ run() set _current_todo_session_dir = /sessions/B
           └─ LLM → tool_call(todo_write, items=[...])
           └─ 工具读 ContextVar → /sessions/B/plan.json → 写文件
           └─ reset()
```

完全独立。ContextVar 任务级隔离 + 文件路径 per-user。

### 4.2 subagent 并发

```
父 Agent.run() #1 (task A)
  └─ tool_call(subagent_name="researcher")
  └─ 工具读 ContextVar → events 列表 #1
  └─ 工具 sub_agent.run() → events 列表 #1.append(...)
  └─ Agent 读 events 列表 #1 → yield SubAgentEvent(*)

父 Agent.run() #2 (task B) [并发]
  └─ tool_call(subagent_name="researcher")  ← 同一 wrapper，但不同 events 列表 #2
  └─ 工具读 ContextVar → events 列表 #2
  └─ 工具 sub_agent.run() → events 列表 #2.append(...)
  └─ Agent 读 events 列表 #2 → yield SubAgentEvent(*)
```

ContextVar 隔离 + 每次调用新列表，零闭包共享。

---

## 5. 错误处理

| 场景 | 当前行为 | 新行为 |
|---|---|---|
| `todo_write` 校验失败 | `TodoManager.update()` 抛 `ValueError` | 工具直接抛 `ValueError`，`ToolExecutor` 捕获并返回 `is_error=True` |
| `todo_write` 无 session 上下文 | N/A（TodoManager 总是存在） | 抛 `RuntimeError`，`ToolExecutor` 捕获 |
| `todo_read` 文件不存在 | N/A | 返回 "暂无会话计划。" |
| `todo_read` 文件 JSON 损坏 | N/A | logger.warning + 返回 "暂无会话计划。" |
| `subagent` 无 Agent 上下文 | N/A（闭包隐式存在） | 抛 `RuntimeError`，`ToolExecutor` 捕获 |
| `subagent` LLM 异常 | catch → 返回 `[name] 子代理执行失败: {e}` | 行为不变 |
| reminder 注入失败 | 不存在 | logger.warning + 跳过（不阻塞 run） |

---

## 6. 测试策略

### 6.1 新增测试

#### `tests/test_todo_concurrent_isolation.py`（新文件）

| 测试方法 | 验证内容 |
|---|---|
| `test_two_agents_concurrent_writes_isolated` | 两个 Agent 并发 todo_write，结果写入各自 plan.json，无串味 |
| `test_todo_read_reflects_latest_write_in_concurrent_context` | 并发场景下，Agent 调 `todo_read` 返回当前 session 的最新 plan |
| `test_todo_tools_have_no_closure_state` | 用 `inspect.getclosurevars(todo_write)` 验证无 TodoManager 闭包 |
| `test_reminder_injected_after_n_rounds` | 模拟 5 轮不调 `todo_write`，验证第 6 轮 LLM 收到 reminder |
| `test_session_dir_injection_via_contextvar` | 验证不调用 `agent.run()` 直接调工具抛 `RuntimeError` |

#### `tests/test_subagent_concurrent_isolation.py`（新文件）

| 测试方法 | 验证内容 |
|---|---|
| `test_subagent_concurrent_runs_events_isolated` | 2 个 Agent.run() 并发调用同一 subagent，子事件归属正确，无丢失 |
| `test_subagent_parallel_in_same_turn` | 同一 turn 内调 2 个不同 subagent 工具，事件不交叉 |
| `test_subagent_no_closure_state` | `inspect.getclosurevars(_subagent_tool)` 不含 `_last_events` |
| `test_subagent_without_contextvar_raises` | 不通过 Agent.run() 直接调 subagent 工具抛 `RuntimeError` |

### 6.2 现有测试更新

#### `tests/test_tool_todo_write.py`

- 删除/重写：所有 `TodoManager` 直接引用（行 381-493 等多处）
- 保留并改写：基本读写逻辑、校验错误、render 输出（改为通过 ContextVar 注入 session_dir）
- 新增：session 文件不存在、JSON 损坏、并发写不同 plan.json

#### `tests/test_subagent.py`

- 删除：`test_subagent_events_closure_storage`（如果存在）
- 保留并改写：所有 `_last_events` 相关断言 → 改为断言 `events` 列表在工具调用结束后被消费
- 保留：基础工厂测试、subagent 配置、嵌套禁止

#### `tests/test_concurrency_stress.py`

- 改严格断言（4 个 stress 测试）：
  - `test_subagent_concurrent_events_cross_contamination`：bug 修复后必须 0 个 `SubAgentDone` 缺失
  - `test_shared_agent_history_pollution`：需要等 PR-3，本 PR 不动
  - `test_session_jsonl_concurrent_writes_corruption`：需要等 PR-4，本 PR 不动
  - `test_todo_*`（如有）：todo 并发场景下无串味

### 6.3 不回归验证

- `examples/multi_turn_chat.py` — 启动后能完成 1 轮对话 + todo_write 工具正常
- `examples/multi_user_chat.py` — 多用户场景下 todo 不串味
- `examples/6_subagent.py` — subagent 工具正常返回结果
- `examples/7_server_fastapi.py` — 启动后 SSE 流式响应正常

---

## 7. 迁移影响

### 7.1 破坏性变更

| 变更 | 影响 | 缓解 |
|---|---|---|
| `create_todo_write_tool(manager=..., plan_file=...)` → `create_todo_write_tool()` | 上游若显式传 `manager` 或 `plan_file` 报 `TypeError` | 删除这些参数后类型即不存在，调用时报错立即发现 |
| `TodoManager` 类从 `agent_framework.tools.builtin.todo_write` 删除 | 上游若 import 直接报 `ImportError` | 同上 |
| 工具函数不再有 `_todo_manager` 属性 | Agent `_find_todo_manager()` 改为不依赖此属性 | 重构同时移除 |
| 工具函数不再有 `_plan_file` / `plan_file` 属性 | 不影响公开 API | 内部细节 |

### 7.2 受影响文件

```
修改：
  src/agent_framework/tools/builtin/todo_write.py        (重写)
  src/agent_framework/agent/agent.py                     (删除 TodoManager 引用)
  src/agent_framework/agent/subagent.py                  (删除 _last_events)

新增：
  tests/test_todo_concurrent_isolation.py
  tests/test_subagent_concurrent_isolation.py

更新：
  tests/test_tool_todo_write.py
  tests/test_subagent.py
  tests/test_concurrency_stress.py  (todo + subagent 部分)
```

### 7.3 不受影响

- `src/agent_framework/serving/pool.py`（AgentPool 与本 PR 无关）
- 9 个 LLM provider
- `src/agent_framework/tools/decorator.py`（@tool 装饰器本身）
- 其他内置工具（file_read/write/python_repl/http_request 等）
- `src/agent_framework/agent/history.py`（ConversationHistory）

---

## 8. 风险与缓解

| 风险 | 严重度 | 缓解 |
|---|---|---|
| reminder 迁移到 Agent 后漏触发 | 中 | 单元测试 `test_reminder_injected_after_n_rounds` 直接覆盖 |
| ContextVar 在某些边缘场景失效 | 低 | Python 3.7+ 标准库，asyncio 任务级隔离有完整保证；FastAPI 兼容 |
| session 文件 I/O 慢（高并发 session 数） | 低 | 仍用本地文件，无锁；可后续优化（不在本 PR 范围） |
| reminder 中的 plan 渲染阻塞 LLM 调用 | 低 | 纯函数调用，纳秒级；plan 文件已在本地 |
| 上游代码 import `TodoManager` 失败 | 中 | 在 commit message 中明确标注；如需 deprecation 期可保留空壳类 1 版本 |

---

## 9. 验收标准

- [ ] `TodoManager` 类从代码库完全删除（grep 无引用）
- [ ] `todo_write` / `todo_read` 工具函数无 TodoManager 闭包（`inspect.getclosurevars` 验证）
- [ ] `subagent.py` 中 `_last_events` 闭包变量完全删除
- [ ] `_subagent_events` 属性从 `ToolWrapper` 上消失
- [ ] `tests/test_todo_concurrent_isolation.py` 5 个新测试全部通过
- [ ] `tests/test_subagent_concurrent_isolation.py` 4 个新测试全部通过
- [ ] `tests/test_concurrency_stress.py` 中 todo + subagent stress 测试改为严格断言并通过
- [ ] `tests/test_tool_todo_write.py` 现有测试全部通过（更新后）
- [ ] `tests/test_subagent.py` 现有测试全部通过（更新后）
- [ ] `examples/multi_turn_chat.py` 启动后完成至少 1 轮 todo 工具调用
- [ ] `examples/multi_user_chat.py` 启动后多用户 todo 不串味
- [ ] `examples/6_subagent.py` 启动后 subagent 工具正常返回
- [ ] `examples/7_server_fastapi.py` 启动后 SSE 流式响应正常
- [ ] 中文 commit message 描述改动
- [ ] commit 前所有改动均通过 `inspect` 验证无状态污染

---

## 10. 实施顺序建议

1. **PR-1** 重写 `todo_write.py`（删除 TodoManager + 引入 ContextVar）
2. **PR-2** 改 `agent.py`：删除 `_find_todo_manager` / `_rebind_todo_manager`，引入 `run()` 入口 ContextVar 绑定
3. **PR-3** 改 `subagent.py`：删除 `_last_events`，引入 `_current_subagent_events` ContextVar
4. **PR-4** 改 `agent.py` 主循环：subagent 工具调用前后管理 ContextVar + per-call 列表
5. **PR-5** 更新测试：test_tool_todo_write.py + test_subagent.py
6. **PR-6** 新增并发隔离测试：test_todo_concurrent_isolation.py + test_subagent_concurrent_isolation.py
7. **PR-7** 改 test_concurrency_stress.py 严格断言
8. **PR-8** examples 冒烟测试

每步独立可测、commit 后跑测试套件。

---

## 附录 A：核心代码 diff 摘要

### A.1 todo_write.py 行数变化

| 指标 | v1 | v2 |
|---|---|---|
| 总行数 | 354 | ~150 |
| 类数 | 1（TodoManager）+ 数据类 2 | 0 |
| 闭包变量 | 1（mgr） | 0 |
| 工具函数装饰器包装 | 2 层（_wrapped_write/_wrapped_read） | 0 层（直接 @tool） |

### A.2 subagent.py 行数变化

| 指标 | v1 | v2 |
|---|---|---|
| 总行数 | 197 | ~150 |
| 闭包变量 | 1（_last_events） | 0 |
| ToolWrapper 标记 | 2（_subagent_events, _is_subagent） | 1（_is_subagent） |

### A.3 agent.py 改动行数

- 删除：~25 行（`_find_todo_manager`, `_rebind_todo_manager`, `_todo_manager` 引用 4 处）
- 新增：~30 行（ContextVar 导入、`_plan_session_dir`、`run()` 入口绑定、`_maybe_inject_plan_reminder` 方法）
- 修改：~20 行（subagent 事件读取路径）

---

> **Spec 完成时间**：2026-06-02
> **下一步**：用户审阅本文档；批准后进入 `superpowers:writing-plans` 技能制定分步实施计划
