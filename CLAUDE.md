# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

agent-framework：统一的 AI 模型抽象层 + Agent 编排引擎。支持 9 个 LLM 提供商（通义千问、Kimi、GLM、DeepSeek、MiniMax、豆包、ChatGPT、Gemini、Claude），提供工具系统（含 MCP 协议）、子 Agent、内置工具、技能（Skills）、文件化系统提示词、会话持久化、上下文自动压缩、以及多用户并发资源池等能力。

技术栈：Python 3.10+、openai SDK（作为统一 HTTP 客户端）、hatchling 构建、pytest + pytest-asyncio。

## 常用命令

```bash
# 安装（开发模式）
pip install -e ".[dev,mcp]"

# 运行全部单元测试（跳过真实 API 调用）
.venv/Scripts/python -m pytest tests/ --ignore=tests/test_real_api.py --ignore=tests/test_real_new_providers.py -q

# 运行单个测试文件
.venv/Scripts/python -m pytest tests/test_agent.py -v

# 运行单个测试方法
.venv/Scripts/python -m pytest tests/test_agent.py::TestAgent::test_method_name -v

# 真实 API 集成测试（需要 .env 中配置对应 API Key）
.venv/Scripts/python tests/run_tests.py

# 运行示例（examples/ 下编号示例，从 LLM → Agent → 工具 → MCP → 子Agent → 服务）
.venv/Scripts/python "examples/1. basic_llm.py"
.venv/Scripts/python examples/7_server_fastapi.py      # FastAPI 单 Agent 服务
.venv/Scripts/python examples/multi_user_chat.py       # AgentPool 多用户并发
```

> Windows 环境，虚拟环境解释器路径为 `.venv/Scripts/python`（非 `.venv/bin/python`）。

## 架构概览

六层架构，全部 async-first。核心数据流：`AgentPool`（可选）→ `Agent.run()` 循环 → `LLM.chat()` 流式 → 解析工具调用 → `ToolExecutor` 执行 → 回传 → 重复。

### 1. LLM 层 (`src/agent_framework/llm/`)

- `BaseLLM`（抽象基类）定义统一接口：`chat(messages, **kwargs) -> AsyncIterator[StreamChunk]`
- `ModelCapabilities`（frozen dataclass）声明提供商能力（function_calling, streaming, reasoning, `max_context_window` 等）
- `ModelRegistry` 工厂模式：每个 provider 文件末尾自注册 `ModelRegistry.register("name", Class)`
- **关键设计**：所有 9 个 provider 统一使用 `openai.AsyncOpenAI` 客户端，通过不同 `base_url` + `extra_body` 适配各提供商。`AsyncOpenAI` 协程安全，**LLM 实例可在多用户间安全共享**
- `ModelConfig` 基类 + 扩展配置（`WebSearchConfig`, `ThinkingConfig`, `FunctionCallingConfig` 等）

### 2. Agent 层 (`src/agent_framework/agent/`)

- `Agent.run()` 核心循环（`agent.py`，~880 行）：每轮 `重建 system prompt → 自动压缩 → LLM 流式调用 → 解析文本/工具调用 → 安全检查 → 执行工具 → 回传结果`，返回 `AsyncIterator[AgentEvent]`
- 事件类型（`events.py`）：`TextDelta`, `ReasoningDelta`, `ToolCallStart`, `ToolConfirmRequired`, `ToolResult`, `AgentDone`, `AgentError`, `SubAgentEvent`, `SubAgentDone`, `HistoryCompacted`, `SessionLoaded`
- **Agent 构造（全配默认，开箱即用）**：`Agent(llm)` 即得完整体——顶层 Agent 各参数为 `None`（默认）时自动注入：内置 `main` 角色提示词（`prompt_dir`，传了 `system_prompt` 则只用它）、内置技能（`skills_dir`）、**全套内置工具 `BUILTIN_TOOLS`（`tools`，显式 `[]` 即无工具）、内置子代理三件套（`subagents`，显式 `[]` 即关闭）**。统一约定：`None`→内置默认、`[]`/显式值→覆盖；子代理 `register_catalog=False`，所有默认注入对其不生效（保持精简 + 结构性不嵌套）。**「全配默认」只在 Agent 一处实现——直接构造与经 AgentPool 构造拿到同规格实例**。能力参数 `mode` / `session_enabled` / `session_dir` / `mcp_tools_active_by_default` / `subagents` 是 `Agent.__init__` 的直接参数（`AgentConfig` 仅含运行限额 `max_turns`/`timeout`/`total_timeout`/`max_total_tokens`/`tool_call_limit`）。`session_dir` 默认 `~/.agent_framework/sessions`（与 CWD 解耦，可用环境变量 `AGENT_FRAMEWORK_HOME` 覆盖；见 `resources.user_data_dir()`）
- **操作模式 `AgentMode`（枚举定义于 `config.py`，作为 `Agent(mode=...)` 直接参数）** —— 安全模型的核心，运行时可 `agent.set_mode(...)` 切换（写实例字段 `self._mode`，天然无跨用户串扰）：
  - `talk`：只读，调用前用 `_is_safe_call()` 拦截所有不安全工具
  - `manual`：人工审批，安全工具直接执行、不安全工具产出 `ToolConfirmRequired` 等待审批
  - `auto`（默认）：自主决策（类 Claude Code），不安全工具自动执行；可选配 AI 安全判定器兜底（见下）
  - `superwork`：全权限，跳过所有安全检查（含 AI 判定）
- **AI 安全判定器（`judge.py`，对齐 Claude Code auto mode 分类器）**：**默认启用**——`judge_llm` 参数 `None`（默认）→ 复用主 llm；`False` → 关闭；`BaseLLM 实例` → 指定判定模型（建议便宜快速模型）；`judge_rules` 追加自定义规则文本。仅 auto 模式生效：`is_safe`/`safe_check` 快路径不经 AI（零开销），其余不安全调用**批量一次**交判定器三态裁决——`allow`→执行 / `confirm`→转 `on_confirm` 人工审批（无回调则拒绝）/ `deny`→拒绝并把理由回传 LLM。判定调用以 `temperature=0` 发起（确定性输出；不支持该参数的 provider 自动过滤）。**判定失败一律 fail-open**（回退为直接执行+警告日志，判定器是增强安全网而非硬门槛）。经 ContextVar `_current_parent_judge` 继承给子代理，父关闭时子也关闭（委派不旁路判定）；`judge_llm` 为 AsyncOpenAI 封装可跨 Agent 共享
- **工具执行顺序**：todo 计划工具（`todo_write`/`todo_read`）必须单独成批调用（不可与其他工具混在一批）；普通工具一批内通过 `asyncio.gather` 并发执行。另有「跨轮次顺序守卫」：一旦开始执行非计划工具（`_work_started`），禁止再创建计划
- `ConversationHistory`（`history.py`）：消息列表 + 4 种截断策略（`none`, `sliding_window`, `token_limit`, `head_tail`），内部持有 `Compactor`
- `Session`（`session.py`）：会话持久化，每会话一个 `{session_dir}/{id}/` 目录（默认 `~/.agent_framework/sessions/`），`conversation.jsonl`（append-only 消息日志，SYSTEM 不记录）+ `session.json`（元数据）。支持 compaction 快照点恢复
- `Compactor`（`compactor.py`）：上下文自动压缩流水线，**0/1 次 API 调用分层**：L1 消息数裁剪 → 轮次分层工具结果压缩（旧轮→占位符、中间轮→截断、近期轮→保留，均带 session 文件指针）→ L4 超 `trigger_ratio` 时 LLM 摘要。阈值随 `max_context_window` 动态计算。另提供 `compact` 元工具供 LLM 主动触发
- `SubAgent`（`subagent.py`）：`create_subagent_tools()` 工厂为每个 `SubAgentConfig` 生成一个 `@tool` 闭包；调用时**每次新建独立 Agent + 干净历史**，`register_catalog=False`/不嵌套子 Agent（结构性保证）。**模式与确认回调均通过 ContextVar 继承**（`_current_parent_mode` / `_current_parent_confirm`，`Agent.run()` 入口注入）——无需 `get_parent_mode` 回调、子代理工具可在 Agent 之前创建且跨 Agent 共享；AUTO 模式下子代理内的不安全工具同样走父的人工确认（**委派不构成安全旁路**）。`SubAgentConfig.role`（main/coder/researcher/reader/reviewer）便利字段自动套用对应内置角色提示词
- **内置子代理三件套**：`builtin_subagent_configs()` 返回生产级预设——`researcher` 调研员（web_search/web_fetch/datetime + deep-research 技能，只读）、`reader` 长内容阅读员（file_read/web_fetch，定向提取，只读）、`coder` 编码执行员（python_repl/file_read/file_write）；可选 `reviewer` 审查员（include 加入）。选型标准：上下文隔离 / 权限收窄 / 可并行

### 3. 工具层 (`src/agent_framework/tools/`)

- `@tool(name, description, is_safe=True, safe_check=None)` 装饰器 → 生成 `ToolWrapper`（含自动 JSON Schema、`is_safe`、`safe_check` 动态判定、`meta` 元工具标记）。**注意：旧版的 `dangerous`/`priority` 参数已移除**
- `ToolRegistry` 双池设计（`registry.py`）：active pool（schema 注入 LLM）+ dormant pool（MCP 工具待激活）
- `catalog.py` 提供三个元工具（`list_catalog`, `search_tools`, `activate_tools`）供 LLM 自主发现/激活 dormant 工具
- `ToolExecutor`（`executor.py`）安全执行：JSON 参数解析、async/sync 兼容、异常捕获
- 内置工具（`builtin/`）：`file_tool`（read/write 分开）、`shell_command`、`python_repl`、`http_request`（API/JSON 场景）、`web_fetch`（网页→Markdown 正文提取，阅读场景优先，省 token）、`web_search`（**可插拔后端**：环境变量 `WEB_SEARCH_PROVIDER`=ddg 默认/tavily/bocha + 对应 Key；DDG 国内不可用，国内部署配 bocha 或用 LLM 自带搜索）、`datetime_tool`、`structured_output`、`todo_write`

### 4. MCP 层 (`src/agent_framework/tools/mcp/`)

- 支持三种传输：`stdio`, `streamable_http`, `sse`
- `MCPManager` 多服务器编排：`asyncio.gather` 并行连接，错误隔离
- `converter.py` 将 MCP Tool 转为 `ToolWrapper`，工具名加 `{server_name}__` 前缀避免冲突
- 配置文件搜索顺序（未显式传 `mcp_config_path` 时）：`./config/mcp_servers.json`（项目级，相对 CWD）→ `~/.agent_framework/mcp_servers.json`（用户级，可用 `AGENT_FRAMEWORK_HOME` 覆盖）；亦可用环境变量 `MCP_CONFIG_PATH` 指定绝对路径

### 5. 提示词 & 技能层 (`src/agent_framework/prompts/`, `skills/`)

- `PromptBuilder`（`prompts/builder.py`）：从 Markdown 目录**分层拼装** system prompt。每个 `.md` 是一个片段，YAML frontmatter 控制 `section`(safeguard/soul/agent/memory/custom) + `order` + `enabled`，`{{key}}` 变量插值，**每次 `build()` 重读文件支持热重载**。预置角色提示词随包分发在 `src/agent_framework/templates/prompts/{main,coder,researcher,reviewer}/`，通过 `agent_framework.builtin_prompts_dir(role)` 定位（内置技能同理用 `builtin_skills_dir()`）
- `SkillRegistry`（`skills/registry.py`）：技能**元数据（name/description/triggers）始终注入 system prompt**，正文按需通过 `load_skill` 元工具拉取。无激活/卸载生命周期。支持平铺 `skills/x.md` 或子目录 `skills/x/SKILL.md`

### 6. 服务层 (`src/agent_framework/serving/`)

- `AgentPool`（`pool.py`）：多用户并发资源池。**按 `(user_id, session_id)` 缓存独立 Agent 实例**，`async with pool.acquire(uid, sid) as h: h.agent.run(...)`。LRU + idle TTL 淘汰、全局 `Semaphore` 并发限流、后台 sweep 清理、`get_stats()` 监控（含 hit_rate）
- 便利构造：**`AgentPool.from_llm(llm)`** 一行起池（共享同一 LLM 实例，AsyncOpenAI 协程安全）。「全配默认」由 Agent 自身提供（见 Agent 层），默认工厂只叠加服务层语义：确定性 session_id 派生 + 共享 MCP 注入；运行限额经 `agent_config=AgentConfig(...)`、其余 Agent 参数经 `agent_kwargs={"mode": ..., "tools": [...], "subagents": [...]}` 原样透传
- 四个硬不变量：每个 `(user_id, session_id)` ≤1 实例；实例数 ≤ `max_agents`；并发 run ≤ `max_concurrent_runs`；空闲超 `idle_ttl_seconds` 被清理

## 关键设计约束（多用户并发 / 无状态化）

这是近期重构的核心，改动相关代码前必读 `serving/pool.py` 顶部的长注释：

- **Agent 含实例级共享状态**（`history`、`session`、`_work_started`、`_mcp_manager`、`tools` 等），多用户**不能共享同一个 Agent**。唯一安全方案是 **per-user Agent**（`AgentPool` 即为此而生）。瓶颈是 MCP 子进程内存（每 Agent 3-5 个 server 占 15-50 MB），不是 Agent 本身
- **todo 工具与 subagent 已无状态化**：不再用模块级单例/闭包变量，而是 Agent 在 `run()` 入口通过 **ContextVar** 注入 per-call 状态（`todo_write._current_session_dir` 注入 session 目录、`todo_write._current_plan_items` 注入内存计划、`subagent._current_subagent_events` 注入事件列表、`subagent._current_parent_mode` 注入父模式），实现 asyncio 任务级隔离。新增任何"跨调用共享"的工具状态时，沿用 ContextVar 模式，**切勿用模块级全局变量**
- **todo 计划存储双后端**（已与 session 解耦）：有 session → 文件后端 `{session_dir}/plan.json`（持久化、per-user 天然隔离）；无 session → 内存后端（`_current_plan_items` ContextVar，同一 Agent 跨轮保留、进程退出即弃）。因此 `session_enabled=False`（含子代理、用户自管 history）时 todo 也能用，不再抛 `RuntimeError`。LLM 通过 `todo_read` 主动拉取

## 代码风格约定

- 中文 docstring 和注释，中文 git commit message
- 使用 Python 3.10+ 语法：`str | None` 联合类型、`match/case`
- 数据模型全部用 `@dataclass`，不可变模型加 `frozen=True`
- 环境变量命名：`{PROVIDER}_API_KEY`（如 `QWEN_API_KEY`, `ANTHROPIC_API_KEY`）
- 无 lint/type-check 工具配置（未启用 ruff、mypy、black）

## 测试模式

- 单元测试使用 `unittest.mock.AsyncMock` mock LLM 响应；`conftest.py` 提供 `mock_openai_client` fixture
- 每个 provider 有独立测试文件 `test_<provider>.py`；MCP 测试在 `tests/test_mcp/`
- 并发/隔离专项测试：`test_concurrency_with_pool.py`、`test_concurrency_stress.py`、`test_agent_pool.py`、`test_subagent_concurrent_isolation.py`、`test_todo_concurrent_isolation.py`（验证无状态化的核心保证）
- 真实 API 测试在 `test_real_api.py` / `test_real_new_providers.py`，需要 `.env` 配置
- pytest 配置：`asyncio_mode = "auto"`（`pyproject.toml`），所有 async 测试无需 `@pytest.mark.asyncio`
