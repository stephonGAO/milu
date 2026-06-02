# 工具无状态化重构：3 种方案对比

> **目的**：将 todo_write 和 subagent_tools 重构为无状态形式，使其天然支持多用户并发。对比 3 种方案，便于选型。
>
> **范围**：仅展示关键代码片段和并发隔离原理，省略装饰器/类型/错误处理样板。

---

## 方案 A：纯无状态（工具函数是纯函数，状态经参数/外部存储流转）

**核心思想**：工具函数不持有任何内部状态。状态由调用方通过参数传入，或由外部存储承载。

### A.1 todo_write

```python
# 工具函数本身完全无状态
@tool(name="todo_write", description="...")
async def todo_write(items: list[dict], current_state: dict | None = None) -> str:
    """重写计划。current_state 由 Agent 注入当前 plan 快照。"""
    # 校验（与原版相同）
    normalized = _validate_items(items)
    rendered = _render(normalized)
    return {"items": normalized, "rendered": rendered}
    # 注意：不写文件，不更新任何字段
    # 持久化由 Agent 负责：把 {"items": normalized} 写入 session/state.json

@tool(name="todo_read", description="...")
async def todo_read(current_state: dict | None = None) -> str:
    return _render(current_state.get("items", []) if current_state else [])
```

Agent 侧（每次工具调用前注入 current_state）：

```python
async def _run_tool_call(self, tool_call):
    wrapper = self._registry.get(tool_call.name)
    # 工具需要的"状态"从 session 读出来
    state = await self._session.load_json("plan.json") if self._session else None

    # 注入到工具调用（不修改工具签名——通过闭包或参数注入）
    if tool_call.name in ("todo_write", "todo_read"):
        tool_call.arguments["current_state"] = state  # 或在 call 时绑定

    result = await wrapper.execute(tool_call.arguments, ctx=self._run_context)
    # 工具返回 {"items": [...]}，Agent 写回 session
    if tool_call.name == "todo_write" and not result.is_error:
        await self._session.save_json("plan.json", json.loads(result.output))
    return result
```

**并发隔离原理**：
- 工具函数无状态 → 多协程调用同一函数零干扰
- 状态在 session 文件中（per-user 自动隔离）
- 读/写顺序由 Agent 的工具执行队列保证（同 turn 内串行）

### A.2 subagent_tools

```python
@tool(name="researcher", description="...")
async def researcher(task: str) -> str:
    """纯函数。事件不通过闭包回流，由 Agent 直接订阅。"""
    sub_history = ConversationHistory(max_turns=50)
    sub_agent = Agent(llm=llm, history=sub_history, ..., register_subagent=False)
    final_text = None
    async for evt in sub_agent.run(task):
        if isinstance(evt, AgentDone):
            final_text = evt.final_text
        # 关键：直接 yield 给父 Agent 的事件流（同一协程）
        # 不存到任何共享变量
    return f"[researcher]: {final_text}"
```

Agent 侧：子代理的工具结果本身就是 string（已经是返回值），不需要"事件转发"——除非前端要实时流式展示。

**并发隔离原理**：
- 子代理事件直接走"父 Agent 事件流"（如果父 Agent 想要这些事件，需要在 tool execute 阶段把子事件透传——见方案 B 的 call_id 模式）
- 没有共享变量

### A 方案要点

| 优点 | 缺点 |
|---|---|
| 工具函数零状态，并发零风险 | 工具签名要带 `current_state`（破坏 @tool 的简洁） |
| 测试简单（纯函数） | 状态读写责任转嫁给 Agent |
| 工具可在多 Agent 间任意复用 | 失去 TodoManager 的"自动持久化"封装 |

---

## 方案 B：保留内部数据结构，但消除闭包共享（per-call 隔离 + ContextVar）

**核心思想**：工具内部仍用 TodoManager / events 列表，但用 call_id 隔离或 `contextvars.ContextVar`（asyncio 任务级隔离）防止共享。

### B.1 todo_write

```python
import contextvars

# 每个 asyncio 任务独立的 TodoManager
_current_manager: contextvars.ContextVar[TodoManager | None] = contextvars.ContextVar(
    "current_todo_manager", default=None
)

@tool(name="todo_write", description="...")
async def todo_write(items: list[dict]) -> str:
    mgr = _current_manager.get()
    if mgr is None:
        raise RuntimeError("todo_write must be called within an Agent context")
    return mgr.update(items)

@tool(name="todo_read", description="...")
async def todo_read() -> str:
    mgr = _current_manager.get()
    return mgr.render() if mgr else "暂无计划"


def create_todo_write_tool(plan_file: Path | str | None = None) -> tuple:
    """每个 Agent 创建一个 manager，绑到 ContextVar 生命周期。"""
    mgr = TodoManager(plan_file=plan_file)

    # 用 Token 在每次 run 时绑定，run 结束后还原
    set_token = None

    def bind_to_current_task():
        nonlocal set_token
        set_token = _current_manager.set(mgr)

    def unbind():
        if set_token:
            _current_manager.reset(set_token)

    todo_write._bind_context = bind_to_current_task
    todo_write._unbind_context = unbind
    return todo_write, todo_read
```

Agent 侧（每次 run 开始时绑定）：

```python
async def run(self, user_input, **kw):
    # 为这次 run 绑定 TodoManager
    bind_fn = getattr(self, "_bind_todo_context", None)
    if bind_fn:
        bind_fn()
    try:
        async for evt in self._run_loop(...):
            yield evt
    finally:
        unbind_fn = getattr(self, "_unbind_todo_context", None)
        if unbind_fn:
            unbind_fn()
```

**并发隔离原理**：
- `ContextVar` 是 asyncio 任务隔离的：每个 `agent.run()` 协程有自己的上下文
- 多用户并发调用同一 tool 实例时，各协程的 `mgr` 互不干扰
- 完美匹配 AgentPool 的 per-key Agent 隔离（每用户独立 Agent → 每用户独立 bind）

### B.2 subagent_tools

```python
import contextvars

# 当前 run 的事件收集器
_current_event_collector: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "current_event_collector", default=None
)

def create_subagent_tools(llm, subagents):
    tools = []
    for cfg in subagents:
        @tool(name=cfg.name, description=cfg.description)
        async def sub_tool(task: str) -> str:
            collector = _current_event_collector.get()
            if collector is None:
                raise RuntimeError("subagent must be called within Agent.run()")

            sub_config = cfg.config or _DEFAULT_SUBAGENT_CONFIG
            sub_history = ConversationHistory(...)
            sub_agent = Agent(llm=llm, history=sub_history, ...)

            final_text = "(无结果)"
            try:
                async for event in sub_agent.run(task, **cfg.llm_kwargs):
                    collector.append(event)  # 写到当前协程的 collector
                    if isinstance(event, AgentDone):
                        final_text = event.final_text
                    elif isinstance(event, AgentError):
                        return f"[{cfg.name}] 错误: {event.message}"
            except Exception as e:
                return f"[{cfg.name}] 子代理执行失败: {e}"

            return f"[{cfg.name}]: {final_text}"

        tools.append(sub_tool)

    return tools
```

Agent 侧（每个 run 创建一个 collector）：

```python
async def run(self, user_input, **kw):
    events_collector: list[AgentEvent] = []
    token = _current_event_collector.set(events_collector)
    try:
        async for evt in self._run_loop(events_collector, ...):
            yield evt
    finally:
        _current_event_collector.reset(token)
```

`_run_loop` 在子代理工具执行后：

```python
# 工具执行后
if wrapper and getattr(wrapper, '_is_subagent', False):
    # 不再读 wrapper._subagent_events（已删除）
    # 直接消费传入的 collector
    for sevt in events_collector:
        yield SubAgentEvent(...)
    events_collector.clear()
```

**并发隔离原理**：
- `events_collector` 是 `run()` 局部变量 + `ContextVar` 注入
- 多并发 run 各有独立 collector → 零干扰

### B 方案要点

| 优点 | 缺点 |
|---|---|
| 工具 API 保持不变（仍是 `todo_write(items)`） | 引入 ContextVar 心智负担 |
| 内部数据结构可继续用 TodoManager | 需要在 Agent.run() 入口/出口正确 set/reset |
| 并发安全（ContextVar 任务级隔离） | 工具无法脱离 Agent 上下文独立运行（必须 bind） |
| 改动小：删除 _todo_manager 标记 + _subagent_events 闭包 | — |

---

## 方案 C：状态外移到 Agent 实例（Agent 持有 manager/collector，工具通过参数接收引用）

**核心思想**：工具函数只负责"逻辑"，状态对象（TodoManager、events collector）由 Agent 创建并通过参数或闭包传给工具实例。

### C.1 todo_write

```python
# 工具工厂接收 manager，不再自己创建/闭包
def create_todo_write_tool(manager: TodoManager) -> tuple:
    @tool(name="todo_write", description="...")
    async def todo_write(items: list[dict]) -> str:
        return manager.update(items)

    @tool(name="todo_read", description="...")
    async def todo_read() -> str:
        return manager.render()

    # 标记 manager 归属（但不再闭包共享）
    todo_write._owns_manager = manager
    return todo_write, todo_read


# Agent 构造时创建独立 manager
class Agent:
    def __init__(self, ..., todo_manager: TodoManager | None = None, ...):
        ...
        # 总是创建独立的 TodoManager（即使外部传入也复制，或强制要求外部不传）
        self._todo_manager = todo_manager or TodoManager(
            plan_file=self._session.dir_path / "plan.json" if self._session else None
        )

        # 把 manager 注入到 todo 工具的闭包
        self._tools = self._inject_todo_manager(tools)
```

```python
def _inject_todo_manager(self, tools):
    """为 todo 工具创建绑到 self._todo_manager 的新实例。"""
    result = []
    for t in tools:
        if getattr(t, '_is_todo_tool', False):
            # 重新创建工具，绑到当前 Agent 的 manager
            new_t = _wrap_todo_with_manager(t, self._todo_manager)
            result.append(new_t)
        else:
            result.append(t)
    return result
```

**并发隔离原理**：
- 每 Agent 实例一个 TodoManager
- AgentPool 已经保证 per-user Agent → per-user manager
- 同一 Agent 多并发 run：需要再加 per-run 快照（不是 manager 共享）

### C.2 subagent_tools

```python
def create_subagent_tools(llm, subagents):
    tools = []
    for cfg in subagents:
        @tool(name=cfg.name, description=cfg.description)
        async def sub_tool(task: str, *, _events: list[AgentEvent]) -> str:
            """_events 由 Agent 在调用前注入（per-call 局部列表）。"""
            _events.clear()
            sub_config = cfg.config or _DEFAULT_SUBAGENT_CONFIG
            sub_history = ConversationHistory(...)
            sub_agent = Agent(llm=llm, history=sub_history, ...)

            final_text = "(无结果)"
            try:
                async for event in sub_agent.run(task, **cfg.llm_kwargs):
                    _events.append(event)
                    if isinstance(event, AgentDone):
                        final_text = event.final_text
                    elif isinstance(event, AgentError):
                        return f"[{cfg.name}] 错误: {event.message}"
            except Exception as e:
                return f"[{cfg.name}] 子代理执行失败: {e}"

            return f"[{cfg.name}]: {final_text}"

        # 标记：需要在调用前注入 events 列表
        sub_tool._is_subagent = True
        sub_tool._needs_per_call_events = True
        tools.append(sub_tool)

    return tools
```

Agent 侧（每次工具调用前注入新的 events 列表）：

```python
# 工具调用前
events_for_this_call: list[AgentEvent] = []
result = await tool.execute(
    arguments,
    ctx=self._run_context,
    _extra_kwargs={"_events": events_for_this_call}  # per-call 注入
)

# 工具调用后（同一调用内的事件）
if wrapper._needs_per_call_events:
    for sevt in events_for_this_call:
        yield SubAgentEvent(...)
    events_for_this_call.clear()
```

**并发隔离原理**：
- 每次工具调用前 Agent 创建新的 `events_for_this_call` 列表
- 工具写入这个列表，调用结束后 Agent 消费
- 没有任何共享列表

### C 方案要点

| 优点 | 缺点 |
|---|---|
| 状态归属清晰（Agent 拥有 manager，调用前注入 events） | 工具签名/调用协议变化（需要 _extra_kwargs 注入） |
| 工具代码本身无需感知并发 | 工厂函数签名变了（必须传 manager）—— 破坏向后兼容 |
| 测试容易（manager 和 events 都是 Agent 的可见属性） | 同一 Agent 多并发 run：还需要 per-run manager 副本（否则同问题） |

---

## 三方案对比

| 维度 | A 纯无状态 | B ContextVar 隔离 | C 状态外移到 Agent |
|---|---|---|---|
| 工具函数持有状态 | ❌ 无 | ✅ 有（ContextVar 后挂） | ✅ 有（参数注入） |
| 工具 API 破坏程度 | 大（带 current_state） | 小（API 不变） | 中（工厂签名变） |
| 并发隔离机制 | 函数无状态 + session 文件 | ContextVar 任务级隔离 | per-call 列表注入 |
| 同一 Agent 多 run 并发 | ✅ 安全 | ✅ 安全 | ⚠️ 需要 per-run 副本 |
| 多 Agent 共享 tool 实例 | ✅ 安全 | ✅ 安全 | ⚠️ 需要 per-Agent 注入 |
| TodoManager 自动持久化 | ❌ 转嫁给 Agent | ✅ 保留 | ✅ 保留 |
| SubAgent 事件转发 | ⚠️ 需重新设计事件流 | ✅ 透明 | ✅ 透明 |
| 改动量 | 大 | 小 | 中 |
| 心智负担 | 低（纯函数） | 中（ContextVar） | 中（依赖注入） |
| 与现有评估报告 PR-2/5.1 兼容性 | 重写 | 兼容（替换闭包为 ContextVar） | 兼容（重写工厂） |

---

## 推荐

**方案 B（ContextVar 隔离）** 作为首选：
- 与现有 API 完全兼容，破坏性最小
- 解决 P0 闭包 bug
- 与评估报告 PR-2 / PR-5.1 完全对齐
- 改动集中（删除 `_last_events` 闭包、`_todo_manager` 闭包，替换为 ContextVar）

**方案 A** 作为长期目标：若未来工具越来越多，把状态彻底外推到 session/参数。

**方案 C** 不推荐：破坏向后兼容，且与 AgentPool 的 per-user 隔离有重叠。

---

## 验证策略

无论选哪个方案，都需要：
1. 改 `tests/test_concurrency_stress.py` 中 4 个 stress test 为严格断言
2. 新增 `test_todo_concurrent_isolation`：多 Agent 并发 todo_write / todo_read 不串味
3. 新增 `test_subagent_concurrent_isolation`：同父 Agent 多 run 并发调用 subagent，事件不丢失
4. 现有 `test_todo_write.py` / `test_subagent.py` 全部通过

