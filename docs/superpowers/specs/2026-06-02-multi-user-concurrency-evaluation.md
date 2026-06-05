# 多用户并发架构评估报告

> **范围**：评估 `milu` 项目作为"AI Agent 核心库 + 多用户并发服务"两大用途的可行性，识别当前设计的并发缺陷，给出改造路线。
>
> **数据来源**：源码全量阅读（src/、tests/、examples/、docs/、pyproject.toml）、并发压测基线（`tests/test_concurrency_stress.py`）、既有路线图（`docs/superpowers/plans/2026-06-02-multi-user-concurrency-roadmap.md`）。

---

## 0. TL;DR — 一句话结论

**项目骨架是健康的**——LLM/Provider/@tool 天然协程安全、Agent 实例化边界清晰、已经实现了 `AgentPool` + FastAPI 示例。但 **在 P0 bug（3 个未修）修复之前不可生产化**；此外 LLM 连接池、流控、可观测性、pip 入口都需要补强。最小可用多用户服务还需要约 **14 小时编码**（路线图 7 个 PR 中已完成 3 个、剩余 4 个）。

| 维度 | 现状 | 评级 | 关键差距 |
|---|---|---|---|
| **Agent 实例隔离** | AgentPool 已实现 (PR-5 ✅) | 🟢 良好 | 仅缺 PR-2/3 修复 P0 bug |
| **会话/状态隔离** | PooledAgent 隔离 history + session | 🟢 良好 | Session JSONL 写未加锁（PR-4） |
| **LLM 客户端复用** | 共享 AsyncOpenAI（协程安全） | 🟢 良好 | 缺连接池参数、超时、重试、流控 |
| **多租户 key 隔离** | 工厂可注入 `(uid, sid) → LLM` | 🟡 可行 | 缺示例/文档 |
| **FastAPI 服务层** | `examples/7_server_fastapi.py` 已就位 | 🟢 良好 | 缺鉴权、限速、横向扩展方案 |
| **流式 SSE** | sse-starlette + ping 心跳 | 🟢 良好 | 缺背压、客户端断连清理 |
| **可观测性** | `pool.get_stats()` + logging | 🟡 基础 | 缺 metrics/tracing 钩子 |
| **pip 打包** | hatchling + dev/mcp optional | 🟡 半成品 | 无 `entry_points`、无版本语义化、依赖含 web_search(http) |
| **横向扩展** | 单进程 | 🔴 缺失 | 多副本无状态共享、Session 需共享存储 |
| **TodoManager 跨用户** | 仍为单例 | 🔴 P0 | PR-5.1 待办 |

---

## 1. 现状盘点 — 4 层 + 2 个新增组件

### 1.1 LLM 层（`src/milu/llm/`）— 🟢 协程安全，但缺流控

**已实现能力**：
- `BaseLLM` 抽象统一接口 `chat() -> AsyncIterator[StreamChunk]`
- 9 个 provider 全部用 `openai.AsyncOpenAI` 客户端
- 客户端**懒加载 + 实例级单例**（`base.py:126-133`）：`self._client` 在第一次调用时创建并复用
- `ModelRegistry` 工厂模式（`__init__.py:28`），启动时一次性注册，**实际并发不修改**

**并发安全评估**：
- ✅ `AsyncOpenAI` 官方声明协程安全，多用户共享同一实例无问题
- ✅ `ModelRegistry` 裸 dict 但**只在启动时 register**，运行期不修改 — 实际无风险
- 🟡 客户端无连接池配置（默认 httpx limits：`max_connections=100, max_keepalive_connections=20`）
- 🟡 无 `timeout` / `max_retries` / `rate_limit` 参数透传
- 🟡 `_get_api_key()` 每次调用都 `load_dotenv()`，多用户 key 隔离只能通过工厂传 `api_key` 构造 LLM

**关键代码引用**：
```python
# src/milu/llm/providers/base.py:120-133
def _get_client(self) -> AsyncOpenAI:
    if self._client is None:
        self._client = AsyncOpenAI(
            api_key=self._get_api_key(),
            base_url=self.base_url,
        )
    return self._client
```

### 1.2 Agent 层（`src/milu/agent/`）— 🟡 实例边界清晰，per-run 状态未重构

**已实现能力**：
- `Agent` 构造时**每次创建独立** `_history`、`_registry`、`_executor`、`_skill_registry`、`_session`（`agent.py:125-225`）
- `ConversationHistory` 支持 4 种截断策略（`history.py:140-237`）
- `run() -> AsyncIterator[AgentEvent]` 完整事件流
- 高优先级工具串行、普通工具 `asyncio.gather` 并发（`agent.py:813-821`）

**并发缺陷（未修的 P0 bug）**：

| # | 位置 | 问题 | 风险 |
|---|---|---|---|
| 1 | `history.py:47, 75-79` | `_messages: list` 无锁保护，并发 `add` + 迭代 `get_messages` 会有 race | 高 |
| 2 | `agent.py:466, 474` | `run()` 直接 `self._history.add()`，无 per-run 互斥 | 高 |
| 3 | `agent.py:474` | `_work_started` / `_plan_created` 是实例字段，多 run 共享 | 高 |
| 4 | `agent.py:_conn_retry_count` | 重试计数存在 `self` 上，跨 run 污染 | 中 |
| 5 | `subagent.py:130, 140, 176` | `_last_events: list` 是**闭包变量**，多并发调用互相覆盖 | **P0 致命** |
| 6 | `subagent.py:194` | 事件挂到 `_tool_wrapper._subagent_events` 仍然是闭包引用 | **P0 致命** |
| 7 | `tools/builtin/todo_write.py:79-137` | `TodoManager.state` 无锁，多用户写同一 manager 串味 | 中 |

**关键代码引用**：
```python
# subagent.py:130 - 闭包共享 (P0)
_last_events: list[AgentEvent] = []

@tool(...)
async def _subagent_tool(task: str) -> str:
    _last_events.clear()  # ← 并发时清掉对方的 events
    ...
    async for event in sub_agent.run(task, **cfg.llm_kwargs):
        _last_events.append(event)
```

### 1.3 Tool 层（`src/milu/tools/`）— 🟡 双池设计、缺锁

**已实现能力**：
- `@tool` 装饰器（`decorator.py:1-48`）自动生成 JSON Schema
- `ToolRegistry` 双池（活跃 + 休眠），`catalog.py` 三个元工具支持运行时激活
- `ToolExecutor` 异常捕获 + JSON 参数校验

**并发缺陷**：
- 🟡 `ToolRegistry._tools` / `_dormant` 是 dict 无锁（`registry.py:21-23`）
  - 影响：Agent 创建期和元工具 `activate_tools` 偶尔并发 → 竞态
  - 实际：单 Agent 内部的元工具是串行的，风险较低
- 🟡 `TodoManager` 状态可写但无锁（`todo_write.py:79-137`）
  - 影响：跨 Agent 共享 todo_manager 时不同用户改 plan 会互相覆盖
  - 缓解：AgentPool 当前为每用户独立 Agent，TodoManager 也随实例独立（除非显式共享）
- 🟢 `ToolExecutor.execute` 无锁，但工具函数**理论上应是纯函数或自带并发安全**——大多数内置工具都是

### 1.4 MCP 层（`src/milu/tools/mcp/`）— 🔴 无 per-user 隔离

**已实现能力**：
- 3 种传输：`stdio` / `streamable_http` / `sse`
- `MCPManager.connect_all()` 用 `asyncio.gather` 并行连接
- converter 用 `{server_name}__` 前缀避免命名冲突
- 配置文件 `config/mcp_servers.json`

**并发缺陷**：
- 🔴 `MCPManager` 一份配置 → 一组连接。**没有 user_id 维度**。
  - 影响：所有用户共享同一组 MCP 子进程；MCP server 端看不到用户身份
  - **资源成本**：`pool.py:68-73` 注释明确指出："MCP 才是瓶颈"
    - 无 MCP：~1 MB/Agent，16GB 机器 ~12000 并发
    - 重度 MCP（5+ servers）：~50 MB/Agent，16GB 机器 ~240 用户
- 🟡 `connect_all()` 每次新建 `MCPServerConnection`（`manager.py:37`），AgentPool 调用时如有缓存会重复连接
- 🟢 命名冲突有 warning 检查

**关键代码引用**：
```python
# src/milu/serving/pool.py:65-77 (注释)
# 单 Agent 内存占用：
#   - 无 MCP 场景：~1 MB
#   - 有 MCP 场景：~15-50 MB（MCP 子进程是真正的瓶颈）
# 16 GB 机器（Python 运行时占 4 GB，留 12 GB 给 Agent）：
#   - 无 MCP：~12,000 并发用户 ✅
#   - 重度 MCP（5+ server + HTTP 长连接）：~240 用户 ❌ 需横向扩展
```

### 1.5 serving 层（`src/milu/serving/pool.py`）— 🟢 核心架构已就位

**AgentPool 4 个硬不变量**（`pool.py:92-96`）：
1. 每个 `(user_id, session_id)` 最多 1 个 Agent 实例
2. 实例数 ≤ `config.max_agents`，超出 LRU 淘汰
3. 全局 `run()` 并发 ≤ `config.max_concurrent_runs`（Semaphore 限流）
4. 空闲 ≥ `config.idle_ttl_seconds` 被后台清理

**关键 API**：
```python
pool = AgentPool(
    llm_factory=lambda uid, sid: shared_llm,
    config=AgentPoolConfig(max_agents=200, max_concurrent_runs=50, idle_ttl_seconds=300),
    agent_config=AgentConfig(...),
)
await pool.start()

async with pool.acquire(user_id, session_id) as h:
    async for evt in h.agent.run(input_):
        ...

await pool.stop()
```

**内部实现亮点**（`pool.py:230-237`）：
- `OrderedDict` 兼顾查找 + LRU
- per-key `asyncio.Lock` 避免多协程并发创建同 key
- `_global_lock` 保护淘汰/扫描
- Semaphore 限制全局并发

**已覆盖测试**（`tests/test_agent_pool.py`，8 个）：
- 创建/复用/隔离/LRU/限流/后台清理/并发隔离/旁路创建

### 1.6 examples/7_server_fastapi.py — 🟢 基础可用

**端点**：
- `POST /chat` — SSE 流式聊天（X-User-Id, X-Session-Id headers）
- `POST /reset` — 清空 history
- `GET /stats` — Pool 监控
- `GET /health` — 健康检查

**实现亮点**：
- `lifespan` 启动/关闭 Pool（`7_server_fastapi.py:50-76`）
- LLM 共享（`lambda uid, sid: llm`）
- 15s ping 防代理超时
- 客户端断连检测（`request.is_disconnected()`）

**已验证**：`tests/test_concurrency_with_pool.py` 用 AgentPool 后并发隔离生效。

---

## 2. 关键并发 Bug 详述（未修复的 P0）

### Bug #1：SubAgent `_last_events` 闭包共享

**位置**：`src/milu/agent/subagent.py:130, 140, 176, 194`

**症状**：
- 同一父 Agent 在不同协程中调用同一 subagent 工具
- 第一次调用开始时 `_last_events.clear()` 清空了第二次调用的累积
- 第二次调用结束时 `async for` 累加时与第一次的事件交叉

**压测证据**（`tests/test_concurrency_stress.py:119-173`）：
- 2 并发时 1 个 run 缺失 `SubAgentDone` 事件（**50% 数据丢失**）
- 多用户场景下，父 Agent 读到的子事件是"混合态"，用户 A 看到用户 B 的工具调用过程

**修复方案**（已规划在 PR-2）：
```python
# 用 ID 隔离 + dict 存储
_call_counter = 0
_call_events: dict[int, list[AgentEvent]] = {}

async def _subagent_tool(task: str) -> str:
    nonlocal _call_counter
    _call_counter += 1
    my_id = _call_counter
    _call_events[my_id] = []
    try:
        async for event in sub_agent.run(task, **cfg.llm_kwargs):
            _call_events[my_id].append(event)
        ...
    finally:
        pass  # 父 Agent 读后清理
```
- 风险：低；工作量 2h
- 关联：父 Agent 读取 events 的代码（`agent.py:944-967`）需改读 `_call_events[call_id]`

### Bug #2：Agent 内部状态共享

**位置**：`src/milu/agent/agent.py:466, 474, _conn_retry_count`

**症状**：
- `_work_started` / `_plan_created` 是实例字段，跨 run 共享
- 多并发 run 会互相看到对方的标记状态，导致流程控制错乱（"重复注入工作清单"等）

**压测证据**（`tests/test_concurrency_stress.py:77-111`）：
- 20 个并发 run 用同一 Agent，**20/20 用户 history 被污染**（P0 致命）
- 用户 A 看到用户 B 的 USER 消息

**修复方案**（已规划在 PR-3）：
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

async def run(self, user_input, **kw) -> AsyncIterator[AgentEvent]:
    ctx = _RunContext(start_time=time.monotonic())
    async for evt in self._run_loop(ctx, user_input):
        yield evt
```
- 风险：中；`_work_started` 在 agent.py 中 7 处使用，逐个迁移
- 工作量：3h
- **与 PR-5 关系**：即使有 AgentPool per-user 实例，每用户多并发 run 仍需 per-run 上下文

### Bug #3：Session JSONL 并发写

**位置**：`src/milu/agent/session.py:111-114, 155-157`

**症状**：
- `log_message` 同步 `open(..., 'a')` 写盘
- 两个 Agent 共享同一 Session 时，await 让出时间窗会导致文件行交错 / 半行写入

**压测证据**（`tests/test_concurrency_stress.py:188-243`）：
- Windows + GIL 下未复现（OSError 容易捕获），但 **Linux/macOS 多线程 asyncio 下风险真实存在**
- 跨进程多 worker 时一定损坏

**修复方案**（已规划在 PR-4）：
```python
class Session:
    def __init__(self, ...):
        ...
        self._write_lock = asyncio.Lock()

    def log_message(self, message: Message) -> int:
        # 改为 async 或加锁
        ...
    # 或更彻底：用 aiofiles + Lock
```
- 工作量：2h
- **同时考虑**：`log_message` 改为 async → 调用方 `agent.py:466` 等 2 处需 `await`

### Bug #4：`_agent_busy` 单布尔误判

**位置**：`src/milu/agent/agent.py`（具体行需进一步定位）

**症状**：
- 原设计意图是"防止同一 Agent 并发 run 互踩"
- 实际实现是单个 `bool`，并发时所有 run 都看到 False
- 由 AgentPool 的 per-user 隔离取代，**已自然失效**，但代码仍在

**修复方案**：
- PR-3 中直接删除 `_agent_busy` 字段和检查点
- 工作量：包含在 PR-3 的 3h 内

### Bug #5：TodoManager 跨用户共享

**位置**：`src/milu/tools/builtin/todo_write.py:79-137`

**症状**：
- 如果上游项目显式 `create_todo_write_tool(manager=shared_manager)` 把同一 TodoManager 传给多用户 Agent
- 用户 A 创建的 todo 会出现在用户 B 的上下文里

**当前状态**：
- AgentPool 默认每用户独立 Agent，**实际不会触发**
- 但**API 暴露了"共享 manager"的可能性**——是潜在陷阱

**修复方案**（PR-5.1）：
- 在 `Agent` 构造时强制创建独立 TodoManager（即使上游传入也复制）
- 或在 `create_todo_write_tool` 加 `clone=True` 参数

---

## 3. 4 个评估维度的详细结论

### 3.1 并发隔离（Agent 实例、会话、状态）

**结论**：✅ **架构正确，已实现 per-user 隔离骨架**；🔴 **3 个 P0 bug 修复后即可生产**。

**已隔离的资源**（AgentPool 自动化）：
- `ConversationHistory` — 每用户独立
- `Session` JSONL — 每用户独立（前提是 Pool 创建时传 `session_dir=Path(f".sessions/{uid}")`）
- `ToolRegistry` — 每 Agent 独立
- `SkillRegistry` — 每 Agent 独立
- `MCPManager` — 每 Agent 独立连接
- `TodoManager` — 每 Agent 独立（默认）

**仍需修的**：
1. SubAgent 闭包（Bug #1）
2. Agent per-run 状态（Bug #2）
3. Session JSONL 加锁（Bug #3）
4. TodoManager 共享 API 显式隔离（Bug #5）

**测试策略**：
- 已有 `test_concurrency_stress.py` 4 个 stress 测试（标注 bug 存在）
- `test_concurrency_with_pool.py` 2 个测试验证 AgentPool 隔离生效
- PR-1 → PR-2/3/4 完成后应改为严格断言

### 3.2 LLM 连接池与限流

**结论**：🟡 **基础可用**；🟡 **缺生产级配置**。

**现状**：
- `AsyncOpenAI` 默认 `httpx.Limits(max_connections=100, max_keepalive_connections=20)`
- 100 并发用户可能打满连接池 → 请求排队
- 无 token-bucket 限速
- 无 provider 维度的 RPM/TPM 限流
- 无 graceful degradation（429/429 错误处理）

**需补强**：
1. **BaseLLM 增加连接池/超时配置**（`base.py:126`）
   ```python
   def __init__(self, ..., timeout: float = 60.0,
                max_connections: int = 100,
                max_keepalive: int = 20,
                max_retries: int = 3):
   ```
2. **多租户 key 隔离**：通过 `LLMFactory` 实现（已就位），需文档化
3. **限流**：
   - 全局 RPM/TPM：用 `aiolimiter` 或 `asyncio.Semaphore` 包 BaseLLM.chat
   - 每用户：可基于 AgentPool 扩展
4. **可观测性**：每个 LLM 调用的 token 数、延迟、错误率

**当前权宜**：
- AgentPool 已有 `Semaphore(max_concurrent_runs)`，提供"应用层"并发上限
- 适合小规模（< 200 并发）部署

### 3.3 FastAPI 服务层 + 部署

**结论**：🟢 **基础示例已就位**；🟡 **缺生产化要素**。

**已有能力**（`examples/7_server_fastapi.py`）：
- lifespan 管理 Pool 生命周期
- SSE 流式 + 15s ping
- 客户端断连检测
- Pool stats 端点

**生产化缺口**：

| 缺口 | 严重度 | 建议方案 |
|---|---|---|
| 鉴权 | 🔴 P0 | JWT/OAuth2 中间件，依赖 header 中的 user_id 可被伪造 |
| 用户级限速 | 🟡 P1 | 在 `acquire()` 前检查 Redis 计数器 |
| 横向扩展 | 🔴 P0 | 多副本需要共享 Session 存储（S3/Redis/PostgreSQL） |
| 优雅关闭 | 🟡 P1 | SIGTERM 后 drain in-flight requests（uvicorn 已支持） |
| 健康检查 | 🟢 | `/health` 已就位，可加深度检查 |
| Metrics 端点 | 🟡 P1 | Prometheus `/metrics`：活跃 Agent 数、等待队列长度、p99 延迟 |
| Tracing | 🟡 P2 | OpenTelemetry 注入到 LLM 调用和工具执行 |
| Worker 数计算 | 🟡 P1 | docs 给出公式：`(2 * CPU) + 1` 起步，按 AgentPool 并发上限调 |
| 进程隔离 | 🟡 P1 | 危险工具（shell/file）建议 Docker 沙箱 |

**单进程容量估算**（来自 `pool.py:65-77`）：

| 配置 | 内存/Agent | 16GB 机器可承载 |
|---|---|---|
| 无 MCP，纯 LLM | ~1 MB | ~12,000 用户 |
| 轻度 MCP（3 server） | ~15 MB | ~800 用户 |
| 重度 MCP（5+ server） | ~50 MB | ~240 用户 |

**横向扩展方案**（路线图未涵盖）：
- L7 LB（nginx/envoy）轮询多副本
- 共享 Session 存储：S3 / Redis / PostgreSQL
- 共享 LLM 客户端：**不推荐**（HTTP 客户端连接无法跨进程）
- 每个进程独立 AgentPool

### 3.4 pip 打包与上游项目集成

**结论**：🟡 **半成品**；需补 `entry_points`、依赖分组、API 公开面。

**当前 `pyproject.toml`**：
```toml
[project]
name = "milu"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "openai>=1.66.0",
    "python-dotenv>=1.0.0",
    "httpx>=0.27.0",
    "jsonschema>=4.20.0",
    "ddgs>=8.0.0",     # ⚠️ web_search 工具依赖，所有上游都要装
    "pyyaml>=6.0",
]
[project.optional-dependencies]
dev = ["pytest>=8.0.0", "pytest-asyncio>=0.23.0"]
mcp = ["mcp>=1.0.0"]
```

**问题清单**：

| 问题 | 严重度 | 修复 |
|---|---|---|
| 无 `entry_points` | 🟡 | 加 `agent-server = "milu.cli:main"` 启动多用户服务 |
| 无 README / 长描述 | 🔴 | 加 `readme = "README.md"` 和项目文档 |
| 无 `license` | 🔴 | 加 `license = {text = "MIT"}` |
| 无 `authors` | 🔴 | 补全 |
| `ddgs` 写死 | 🟡 | 移入 `[search]` 可选依赖 |
| `httpx` 在主依赖 | 🟡 | 改可选（被 `openai` 间接依赖，重复声明） |
| 无版本号语义化 | 🟡 | 仍 0.1.0，但应进入 0.2.0（加 PR-2/3/4/5 后） |
| 公开 API 无声明 | 🟡 | 哪些是公开 API、哪些是 internal？建议加 `__all__` 和 `milu.public_api` 标记 |
| `AgentPool` 已在 `__init__.py` | ✅ | `serving/__init__.py` re-export 已就位 |
| 上游项目集成示例 | 🟡 | 缺 `examples/integration_basic/`、`examples/integration_with_pool/` |

**建议改造的 `pyproject.toml` 草案**：
```toml
[project]
name = "milu"
version = "0.2.0"
description = "Unified AI model abstraction + multi-user Agent orchestration"
readme = "README.md"
license = {text = "MIT"}
authors = [{name = "...", email = "..."}]
requires-python = ">=3.10"
keywords = ["ai", "agent", "llm", "mcp"]
classifiers = [
    "Programming Language :: Python :: 3.10",
    "Framework :: AsyncIO",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
dependencies = [
    "openai>=1.66.0",
    "python-dotenv>=1.0.0",
    "jsonschema>=4.20.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
search = ["ddgs>=8.0.0"]           # web_search 工具
mcp = ["mcp>=1.0.0"]               # MCP 协议
server = ["fastapi>=0.110", "uvicorn[standard]>=0.27", "sse-starlette>=2.0"]  # 多用户服务
dev = ["pytest>=8.0.0", "pytest-asyncio>=0.23.0", "httpx>=0.27.0"]

[project.urls]
Homepage = "https://..."
Repository = "https://..."

[project.scripts]
agent-server = "milu.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**上游项目集成建议**：
```python
# 上游 A：仅用 LLM
from milu.llm import ModelRegistry
llm = ModelRegistry.create("qwen", model="qwen3.7-max")

# 上游 B：用 Agent
from milu import Agent, AgentConfig
agent = Agent(llm=llm, config=AgentConfig(mode="auto"))

# 上游 C：多用户服务
from milu.serving import AgentPool, AgentPoolConfig
pool = AgentPool(llm_factory=lambda uid, sid: shared_llm, config=AgentPoolConfig(...))
```

---

## 4. 改造路线图（基于既有 7-PR 计划）

### 4.1 现状（已完成 3 / 7）

| PR | 状态 | 交付物 |
|---|---|---|
| **PR-1** 并发压测基线 | ✅ 已完成 | `tests/test_concurrency_stress.py`（4 个 stress test） |
| **PR-2** SubAgent 闭包修复 | ❌ 未做 | `subagent.py` 改用 ID 隔离 + 新增 `test_subagent_concurrent_isolation` |
| **PR-3** Agent per-run 状态重构 | ❌ 未做 | 新增 `_RunContext` dataclass + 7 处状态迁移 |
| **PR-4** Session JSONL 加锁 | ❌ 未做 | `session.py` 加 `asyncio.Lock`（或改 aiofiles） |
| **PR-5** AgentPool | ✅ 已完成 | `serving/pool.py` + 8 个单元测试 |
| **PR-6** FastAPI 示例 | ✅ 已完成 | `examples/7_server_fastapi.py` |
| **PR-7** 调度器持久化 | ❌ 可选 | SQLite TaskStore |

### 4.2 推荐实施顺序

**最小可用多用户服务**（约 14h）：
1. PR-3（Agent 状态重构，3h）
2. PR-4（Session 加锁，2h）
3. PR-5.1（TodoManager 显式 per-Agent，1h）
4. 联调 + 文档（2h）

**修复所有 P0 bug**（再加 4h）：
- PR-2（SubAgent 闭包，2h）
- 把 `test_concurrency_stress.py` 改为严格断言（2h）

**生产化要素**（额外 8h）：
- PR-8（新增）：LLM 连接池/限流配置
- PR-9（新增）：FastAPI 鉴权 + 用户级限速
- PR-10（新增）：横向扩展方案文档 + Session 共享存储
- PR-11（新增）：Prometheus metrics + OpenTelemetry tracing

**打包与发布**（额外 4h）：
- PR-12（新增）：pyproject.toml 补 `entry_points`、可选依赖分组、CLI
- PR-13（新增）：README + 上游集成示例

### 4.3 总时间估算

| 目标 | 工作量 | 投入 |
|---|---|---|
| 多用户并发可用 | ~14h | 1.5 人天 |
| 修完所有 P0 bug | ~18h | 2.5 人天 |
| 生产级 + 可发布 | ~30h | 4 人天 |

---

## 5. 风险登记

| # | 风险 | 触发条件 | 严重度 | 缓解措施 |
|---|---|---|---|---|
| R1 | 上游项目共享 LLM client 但接不同 key | 多租户场景 | 中 | `LLMFactory` 文档化，禁止显式共享 |
| R2 | MCP 子进程内存爆炸 | 重度 MCP 配置 | 高 | 文档给出容量估算；Pool `max_agents` 限制 |
| R3 | Session JSONL 损坏 | 多进程横向扩展 | 高 | PR-4 加锁；横向扩展前必须改造为共享存储 |
| R4 | SubAgent 闭包 bug 未修 | 启用 subagent 工具 | 高 | 路线图 PR-2；上游项目升级后启用 |
| R5 | AgentPool per-key lock 泄漏 | 长时间运行 | 中 | PR-5.1 加定期清理 `_entry_locks` |
| R6 | 危险工具无沙箱 | 启用 shell/python_repl | 高 | 文档建议 Docker 沙箱；上游自行隔离 |
| R7 | LLM provider 限速被忽略 | 高并发 | 中 | PR-8 加 token bucket |
| R8 | 横向扩展后无 Session 共享 | 部署 ≥ 2 副本 | 高 | 部署文档明确说明 |
| R9 | `_agent_busy` 残留 | 历史代码 | 低 | PR-3 顺便删除 |
| R10 | 公开 API 边界不清 | 上游项目升级 | 中 | 加 `__all__` 和 deprecation policy |
| R11 | 无 SLO/监控 | 生产部署 | 中 | PR-11 加 metrics 端点 |
| R12 | `ddgs` web_search 写死 | 上游不需要搜索 | 低 | 移入可选依赖 |

---

## 6. 实施建议

### 6.1 立即可做（不改代码）
- ✅ 用 AgentPool 的项目已经可以"基本可用"（demo / 内部工具 / 小规模生产）
- ✅ 上游项目通过 `pip install -e ".[mcp,server]"` 即可使用

### 6.2 优先级 P0：必须修
1. **PR-3**：Agent 内部状态重构（3h）
2. **PR-4**：Session JSONL 加锁（2h）
3. **PR-2**：SubAgent 闭包 bug（2h）

> 修完这 3 个 PR 后，`tests/test_concurrency_stress.py` 4 个测试可改为严格断言。

### 6.3 优先级 P1：建议修
4. **PR-8**：LLM 连接池 + 超时 + 重试配置
5. **PR-9**：FastAPI 鉴权中间件
6. **PR-12**：pyproject 补全 + CLI 入口
7. **TodoManager 显式隔离**

### 6.4 优先级 P2：可选
8. PR-7 调度器持久化
9. PR-11 metrics + tracing
10. 横向扩展 + Session 共享存储

### 6.5 不建议做
- 重写 Agent 核心为 Actor 模式（投入产出比低）
- 引入消息队列（Celery/RQ）做任务调度（与 in-process Agent 模式冲突）
- 引入 Redis 做 session 存储（应在项目外做）

---

## 7. 总结

**可行性判断**：

| 用途 | 可行性 | 备注 |
|---|---|---|
| **单进程多用户 Agent 服务**（< 200 并发） | ✅ 可行 | AgentPool + FastAPI 已就位，修 PR-2/3/4 后稳定 |
| **单进程高并发 Agent 服务**（200-2000） | 🟡 需改造 | 需 PR-8（连接池/限流）+ 调优 |
| **多进程横向扩展** | 🔴 需重大改造 | Session 存储、共享 LLM 客户端需重新设计 |
| **作为 pip 库供上游集成** | 🟡 需补强 | 需 PR-12（entry_points/可选依赖） |
| **作为多用户 SaaS 服务** | 🔴 需 30+ h 改造 | 鉴权、限速、监控、扩展、运维全链路 |

**最终建议**：
- **现在**：作为单进程多用户 demo / 内部工具可立即使用
- **3 周内**：完成 PR-2/3/4/5.1/8/12，进入"小规模生产"状态
- **3 个月内**：补 PR-9/10/11/13，进入"可对外服务"状态

---

## 附录 A：关键文件清单

```
src/milu/
├── __init__.py                    # 公开 API 出口
├── agent/
│   ├── agent.py                   # ⚠️ PR-2/3 改造点（per-run 状态）
│   ├── history.py                 # 🟡 截断 + 潜在锁
│   ├── session.py                 # ⚠️ PR-4 改造点（JSONL 加锁）
│   ├── subagent.py                # 🔴 PR-2 改造点（闭包 bug）
│   └── ...
├── llm/
│   ├── providers/
│   │   ├── base.py                # 🟡 PR-8 改造点（连接池配置）
│   │   └── ...                    # 9 个 provider
│   └── ...
├── serving/
│   ├── pool.py                    # ✅ AgentPool 核心
│   └── __init__.py                # ✅ 公开 API
└── tools/
    ├── registry.py                # 🟡 双池无锁
    ├── builtin/
    │   ├── todo_write.py          # 🔴 PR-5.1 改造点
    │   └── ...
    └── mcp/
        ├── manager.py             # 🟡 共享 MCP 连接
        └── ...

examples/
└── 7_server_fastapi.py            # ✅ 多用户服务示例

docs/superpowers/plans/
└── 2026-06-02-multi-user-concurrency-roadmap.md  # 既有路线图（7 PR）

pyproject.toml                     # 🟡 PR-12 改造点
```

## 附录 B：测试矩阵

| 测试文件 | 状态 | 覆盖场景 |
|---|---|---|
| `tests/test_agent_pool.py` | ✅ 通过（8 cases） | Pool 创建/复用/隔离/LRU/限流/清理 |
| `tests/test_concurrency_stress.py` | 🟡 4 个 stress | P0 bug 暴露（未严格断言） |
| `tests/test_concurrency_with_pool.py` | ✅ 通过（2 cases） | AgentPool 修复验证 |
| 单元测试（`test_*.py`） | ✅ 700+ | 各模块功能 |
| 真实 API 集成（`test_real_*.py`） | ⚠️ 需 API key | 端到端冒烟 |

## 附录 C：术语表

- **Per-user Agent**：每个用户独占一个 Agent 实例，互不共享状态
- **Per-run Context**：单次 `agent.run()` 的私有状态（流程标记、计数器）
- **AgentPool**：维护 (user_id, session_id) → Agent 的多用户资源池
- **PooledAgent**：`async with pool.acquire(...)` 返回的句柄
- **Hard Invariant**：代码改动的硬约束（Pool 4 个、Provider 协程安全等）
- **SSE**：Server-Sent Events，HTTP 长连接单向流
- **TTL**：Time To Live，空闲超时自动清理
- **LRU**：Least Recently Used，淘汰最久未用

---

> **报告完成时间**：2026-06-02
> **配套路线图**：`docs/superpowers/plans/2026-06-02-multi-user-concurrency-roadmap.md`
> **下一步**：用户审阅本报告后，决定具体实施哪些 PR。
