# 多用户并发能力 — PR 拆分与实施路线

> 配套文档：本目录下的 `2026-06-01-in-process-agent-scheduler-design.md` 是 scheduler 改造的 spec。
> 本文档聚焦"多用户并发服务化"目标。

## 现状摘要

通过代码分析和并发压测（`tests/test_concurrency_stress.py`）确认：

| 问题 | 严重度 | 压测证据 |
|---|---|---|
| 共享 Agent 并发 run → 历史串味 | P0 致命 | 20/20 用户历史被污染 |
| SubAgent 闭包共享 `_last_events` | P0 致命 | 2 并发时 1 个 run 缺失 SubAgentDone |
| `_agent_busy` 是单布尔而非互斥锁 | P0 设计 | 5 并发 run 全部进入执行 |
| Session JSONL 无锁追加 | P1 理论风险 | Windows + GIL 下未复现，跨平台风险存在 |
| AgentPool 缺失 | P1 架构 | 需新增 `milu.serving` 子包 |
| 缺 Web 服务层示例 | P1 部署 | examples/ 全部是 CLI 脚本 |

好消息：**底层（LLM、provider、@tool）天然协程安全；Agent 实例化边界清晰。** 因此无需大改核心引擎。

## 改造总览

7 个 PR，可串行实施。最少前 5 个 PR 即可让多用户并发工作。

```
PR-1 压测基线 ──► PR-2 SubAgent 修复 ──► PR-3 Agent 内部状态重构 ──► PR-4 Session 加锁
                                                                          │
                            ┌─────────────────────────────────────────────┘
                            ▼
PR-5 AgentPool ──► PR-6 FastAPI 示例 ──► PR-7 (可选) 持久化调度器
```

## PR-1：并发压测基线（已有，可直接合入）

**目标**：把 `tests/test_concurrency_stress.py` 固化为回归基线，**当前测试应**全部打印诊断信息而不失败（标记 bug 存在），修复后改为严格断言。

**改动**：
- 文件：`tests/test_concurrency_stress.py`（新增）
- 内容：4 个并发场景，记录历史串味 / SubAgent 事件错乱 / JSONL 损坏 / `_agent_busy` 误伤
- 标注 `@pytest.mark.stress`，CI 单独跑

**验收**：4 个测试通过（不严格断言），输出打印当前 bug 严重度

**实施状态**：✅ 已完成（已存在）

**预估时间**：0.5h

---

## PR-2：修复 SubAgent 闭包共享

**目标**：消除 `milu/agent/subagent.py:130` 的 `_last_events = []` 闭包共享 bug。

**问题代码**：
```python
def _create_single_subagent_tool(llm, cfg, get_parent_mode=None):
    _last_events: list[AgentEvent] = []  # ← 闭包变量，并发共享！
    @tool(...)
    async def _subagent_tool(task: str) -> str:
        _last_events.clear()  # 清掉上一次的事件
        ...
```

**改动**：
- 文件：`src/milu/agent/subagent.py`
- 把 `_last_events` 改为 `asyncio.Event` + `dict[call_id, list[events]]`，或用 `contextvars.ContextVar`
- 最简方案：把 events 存到 ToolWrapper 的实例属性上，但**给每次调用一个唯一 ID 隔离**：

```python
_call_counter = 0
_call_events: dict[int, list[AgentEvent]] = {}

async def _subagent_tool(task: str) -> str:
    nonlocal _call_counter
    _call_counter += 1
    my_id = _call_counter
    _call_events[my_id] = []
    try:
        ...
        async for event in sub_agent.run(task, **cfg.llm_kwargs):
            _call_events[my_id].append(event)
        ...
    finally:
        # 保留给父 Agent 读，读取后清理
        pass
```

**配套改动**：
- 父 Agent 读取 events 的代码（agent.py:944-967）改为读 `_call_events[call_id]`
- 需要把 `call_id` 传到 SubAgentDone 事件中

**新增测试**：
- `tests/test_subagent.py::test_subagent_concurrent_isolation` — 5 个并发同 subagent，断言每个父调用看到自己的 events

**验收**：
- 现有 `tests/test_subagent.py` 全过
- 新测试通过
- PR-1 压测中 Test 2 从"1/2 缺失"变为"0 缺失"

**风险**：低。SubAgent 事件读取路径较短，改动面可控。

**预估时间**：2h（含测试）

---

## PR-3：Agent 内部 per-run 状态重构

**目标**：把 `_work_started` / `_plan_created` / `_conn_retry_count` 从 `self` 移到 `_RunContext`，消除多 run 之间的状态污染。

**问题代码**（agent.py）：
```python
self._work_started = False   # 流程约束（每 run 独立）
self._plan_created = False   # 流程约束（每 run 独立）
self._conn_retry_count = ... # 重试计数（每 run 独立，但存到 self）
```

**改动**：
- 新增内部 dataclass：

```python
@dataclass
class _RunContext:
    work_started: bool = False
    plan_created: bool = False
    conn_retry_count: int = 0
    start_time: float = 0.0
    turn_count: int = 0
    total_tool_calls: int = 0
    total_usage: TokenUsage = field(default_factory=TokenUsage)
```

- `run()` 入口创建 `_RunContext`，传给 `_run_loop(ctx, user_input)`
- 所有 `self._work_started` / `self._plan_created` 引用改为 `ctx.work_started` / `ctx.plan_created`
- `_conn_retry_count` 用 ctx 字段，不再 `getattr/del` self 属性

**不动**：
- `self._agent_busy`（PR-3 保留，由 PR-5 替换为 AgentPool 隔离）

**新增测试**：
- `tests/test_agent.py::test_concurrent_runs_dont_share_state` — 模拟两个 run 在 turn 2 时同时设 `_work_started = True`，验证互不影响

**验收**：
- 现有 agent 测试全过
- PR-1 Test 1 改为严格断言：每个用户的 history 只含自己的 user 消息
- PR-1 Test 1 通过（0/20 污染）

**风险**：中。`_work_started` 在 agent.py 中 7 处使用，需逐个迁移。建议逐文件 Edit + 跑测试。

**预估时间**：3h

---

## PR-4：Session JSONL 加锁

**目标**：消除 `Session.log_message` 多协程并发追加的风险。

**改动**：
- 文件：`src/milu/agent/session.py`
- 给 `Session` 类加 `self._write_lock = asyncio.Lock()`
- `log_message` 入口 `async with self._write_lock:`
- `log_compaction` 同样
- **同时**考虑把 `open(..., 'a')` 改为 `aiofiles` 异步 IO（避免阻塞事件循环）

**新增测试**：
- `tests/test_session.py::test_concurrent_log_message_no_corruption` — 1000 个并发 log_message，断言文件行数 == 1000 且全部可解析

**验收**：
- 现有 session 测试全过
- 新测试通过
- PR-1 Test 3 改为严格断言：1000 条 valid、0 corrupt

**风险**：低。`log_message` 是同步函数，改为 async 需调用方 `await`，但 agent.py 中只有 2 处调用，改动量小。

**预估时间**：2h

---

## PR-5：AgentPool 资源管理器（核心架构补全）

**目标**：为多用户场景提供 Agent 生命周期管理。

**改动**：
- 新增 `src/milu/serving/__init__.py`
- 新增 `src/milu/serving/pool.py`（含 `AgentPool`, `AgentPoolConfig`, `PooledAgent`）
- 在 `src/milu/__init__.py` re-export 公开 API

**核心 API**：
```python
pool = AgentPool(
    llm_factory=lambda uid, sid: shared_llm,
    config=AgentPoolConfig(max_agents=200, max_concurrent_runs=50, idle_ttl_seconds=300),
)
await pool.start()

async with pool.acquire(user_id, session_id) as h:
    async for evt in h.agent.run(input_):
        ...

await pool.stop()
```

**关键不变量**：
- 每个 (user_id, session_id) 最多 1 个 Agent 实例
- 实例数 ≤ `max_agents`，超出 LRU 淘汰
- 全局并发 ≤ `max_concurrent_runs`（Semaphore）
- 空闲超 `idle_ttl_seconds` 的实例被后台清理

**测试覆盖**（已完成 8 个）：
- 创建 / 复用 / 隔离 / LRU 淘汰 / 限流 / 后台清理 / 并发 run 隔离 / 旁路创建

**验收**：
- 8 个 AgentPool 测试全过
- PR-1 Test 1 通过（0/20 污染）—— 因为每个 user 有独立 Agent

**风险**：中。需考虑：
- per-key 锁的清理（`_entry_locks` 会无限增长，PR-5.1 加定期清理）
- `agent._stop_scheduler` 在 `_close_entry` 中调用，需 scheduler 已实现
- TodoManager 仍是单例，跨 Agent 共享（PR-5 标注为已知，PR-5.1 修）

**预估时间**：5h（含测试）

---

## PR-6：FastAPI 多用户服务示例

**目标**：演示如何用 AgentPool 构建生产级多用户 HTTP 服务。

**改动**：
- 新增 `examples/7_server_fastapi.py`
- 不修改框架代码
- 文档：`docs/server_deployment.md`（部署建议 + nginx 配置 + Gunicorn worker 数计算）

**端点**：
- `POST /chat` — SSE 流式聊天（X-User-Id, X-Session-Id headers）
- `POST /reset` — 清空某用户 history
- `GET /stats` — Pool 监控
- `GET /health` — 健康检查

**测试**：
- `tests/test_server_example.py` — 用 `httpx.AsyncClient` + `TestClient` 模拟两个用户并发请求

**依赖**：
- `fastapi>=0.110`
- `uvicorn[standard]`
- `sse-starlette>=2.0`

**验收**：
- 启动后 `curl /stats` 返回合理数据
- `curl -N -X POST /chat` 返回 SSE 流

**风险**：低。纯示例代码，不影响框架。

**预估时间**：2h

---

## PR-7（可选）：调度器持久化

**目标**：把 `InProcessScheduler` 的 `ScheduledTask` 持久化到 SQLite，进程重启后恢复。

**改动**：
- `scheduler.py` 引入 `TaskStore` 抽象 + `SqliteTaskStore` 默认实现
- `schedule_once` / `schedule_interval` / `schedule_heartbeat` 同步写库
- `_start_scheduler` 时从库加载
- 配合 PR-5 AgentPool：调度器也按 (user_id, session_id) 隔离

**验收**：进程重启后已注册的 interval/heartbeat 任务继续运行

**预估时间**：6h

---

## PR 依赖关系

```
PR-1 ──► PR-2 ──┐
                 ├──► PR-5 ──► PR-6
       PR-3 ─────┤
                 │
       PR-4 ─────┘
                           (独立)
                           PR-7
```

- PR-1（基线）必须先有
- PR-2 / PR-3 / PR-4 可并行
- PR-5 强烈依赖 PR-3（否则 Agent 内部状态还是会污染）
- PR-6 依赖 PR-5
- PR-7 独立

## 实施顺序建议

**最小可用多用户服务**（约 14h 编码）：
1. PR-1（基线）→ 0.5h
2. PR-3（Agent 状态重构）→ 3h
3. PR-4（Session 加锁）→ 2h
4. PR-5（AgentPool）→ 5h
5. PR-6（FastAPI 示例）→ 2h
6. 联调 + 文档 → 1.5h

**修复所有 P0 bug**（再加 PR-2，2h）→ 16h

## 测试策略

| 层级 | 工具 | 覆盖 |
|---|---|---|
| 单元 | pytest | 8 个 AgentPool + 现有 700+ 测试 |
| 并发压测 | pytest + asyncio.gather | 4 个 stress 测试，PR-1 起的基线 |
| 集成 | httpx + TestClient | FastAPI 端到端（PR-6） |
| 真实 API | 现有 `test_real_api.py` | 确认改造不破坏 LLM 调用 |

## 风险登记

| 风险 | 触发条件 | 缓解措施 |
|---|---|---|
| 改造破坏现有 API | PR-3 改了 Agent 内部状态 | 公共 API 不变，行为兼容 |
| 改造破坏性能 | 加锁后吞吐下降 | 仅 Session 加锁；AgentPool 限流可配置 |
| TodoManager 单例导致冲突 | 多用户用 todo_write | PR-5.1 显式 per-Agent 化 |
| `_last_events` 修复后引入 deadlock | SubAgent 工具死锁 | 不用 Lock，改用 ID 隔离 |
| per-key 锁泄漏 | AgentPool _entry_locks 无限增长 | PR-5.1 加定期清理 |
| Scheduler 跨进程调度 | 多副本部署同一服务 | PR-7 持久化 + 分布式锁 |
