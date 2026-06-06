# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

milu：统一的 AI 模型抽象层 + Agent 编排引擎。支持 9 个 LLM 提供商（通义千问、Kimi、GLM、DeepSeek、MiniMax、豆包、ChatGPT、Gemini、Claude），提供工具系统（含 MCP 协议）、子 Agent、内置工具、技能（Skills）、文件化系统提示词、会话持久化、上下文自动压缩、以及多用户并发资源池等能力。

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

# CLI 命令行（pip install 后注册入口点 milu）
milu                       # 无子命令 → 进入交互式对话（chat；首次无 Key 时自动询问进入 setup 引导）
milu setup                 # 初始化引导：选厂商/模型 → API Key → 搜索工具（密钥写 ~/.milu/.env）
milu chat -p deepseek      # 指定厂商进入对话（默认嵌入调度引擎，定时任务对话期间自动执行；--no-scheduler 关）
milu run "你好" -q          # 一次性执行，-q 只输出最终回答（可管道；不嵌入调度）
echo "总结这段话" | milu run # 从 stdin 读取指令
milu providers             # 列出 9 个厂商及 Key 配置状态
milu config set provider qwen   # 写入 ~/.milu/config.json
milu sessions list         # 查看历史会话
milu serve                 # 启动内置 Web 服务（多用户对话 + 全功能演示前端，默认 127.0.0.1:8000）
milu serve --port 9000 --no-scheduler   # 自定义端口、不嵌入定时任务调度
# 开发期未重装时也可：.venv/Scripts/python -m milu.cli <args>
```

> Windows 环境，虚拟环境解释器路径为 `.venv/Scripts/python`（非 `.venv/bin/python`）。
> 新增/修改 `[project.scripts]` 入口点后需 `pip install -e .` 重装才能注册控制台脚本。

## 架构概览

六层架构，全部 async-first。核心数据流：`AgentPool`（可选）→ `Agent.run()` 循环 → `LLM.chat()` 流式 → 解析工具调用 → `ToolExecutor` 执行 → 回传 → 重复。

### 1. LLM 层 (`src/milu/llm/`)

- `BaseLLM`（抽象基类）定义统一接口：`chat(messages, **kwargs) -> AsyncIterator[StreamChunk]`
- `ModelCapabilities`（frozen dataclass）声明提供商能力（function_calling, streaming, reasoning, `max_context_window` 等）
- `ModelRegistry` 工厂模式：每个 provider 文件末尾自注册 `ModelRegistry.register("name", Class)`
- **关键设计**：所有 9 个 provider 统一使用 `openai.AsyncOpenAI` 客户端，通过不同 `base_url` + `extra_body` 适配各提供商。`AsyncOpenAI` 协程安全，**LLM 实例可在多用户间安全共享**
- `ModelConfig` 基类 + 扩展配置（`WebSearchConfig`, `ThinkingConfig`, `FunctionCallingConfig` 等）

### 2. Agent 层 (`src/milu/agent/`)

- `Agent.run()` 核心循环（`agent.py`，~880 行）：每轮 `重建 system prompt → 自动压缩 → LLM 流式调用 → 解析文本/工具调用 → 安全检查 → 执行工具 → 回传结果`，返回 `AsyncIterator[AgentEvent]`
- 事件类型（`events.py`）：`TextDelta`, `ReasoningDelta`, `ToolCallStart`, `ToolConfirmRequired`, `ToolResult`, `AgentDone`, `AgentError`, `SubAgentEvent`, `SubAgentDone`, `HistoryCompacted`, `SessionLoaded`
- **Agent 构造（全配默认，开箱即用）**：`Agent(llm)` 即得完整体——顶层 Agent 各参数为 `None`（默认）时自动注入：内置 `main` 角色提示词（`prompt_dir`，传了 `system_prompt` 则只用它）、内置技能（`skills_dir`）、**全套内置工具 `BUILTIN_TOOLS`（`tools`，显式 `[]` 即无工具）、内置子代理三件套（`subagents`，显式 `[]` 即关闭）**。统一约定：`None`→内置默认、`[]`/显式值→覆盖；子代理 `register_catalog=False`，所有默认注入对其不生效（保持精简 + 结构性不嵌套）。**「全配默认」只在 Agent 一处实现——直接构造与经 AgentPool 构造拿到同规格实例**。能力参数 `mode` / `session_enabled` / `session_dir` / `mcp_tools_active_by_default` / `subagents` 是 `Agent.__init__` 的直接参数（`AgentConfig` 仅含运行限额 `max_turns`/`timeout`/`total_timeout`/`max_total_tokens`/`tool_call_limit`）。`session_dir` 默认 `~/.milu/sessions`（与 CWD 解耦，可用环境变量 `MILU_HOME` 覆盖；见 `resources.user_data_dir()`）
- **操作模式 `AgentMode`（枚举定义于 `config.py`，作为 `Agent(mode=...)` 直接参数）** —— 安全模型的核心，运行时可 `agent.set_mode(...)` 切换（写实例字段 `self._mode`，天然无跨用户串扰）：
  - `talk`：只读，调用前用 `_is_safe_call()` 拦截所有不安全工具
  - `manual`：人工审批，安全工具直接执行、不安全工具产出 `ToolConfirmRequired` 等待审批
  - `auto`（默认）：自主决策（类 Claude Code），不安全工具自动执行；可选配 AI 安全判定器兜底（见下）
  - `superwork`：全权限，跳过所有安全检查（含 AI 判定）
- **AI 安全判定器（`judge.py`，对齐 Claude Code auto mode 分类器）**：**默认启用**——`judge_llm` 参数 `None`（默认）→ 复用主 llm；`False` → 关闭；`BaseLLM 实例` → 指定判定模型（建议便宜快速模型）；`judge_rules` 追加自定义规则文本。仅 auto 模式生效：`is_safe`/`safe_check` 快路径不经 AI（零开销），其余不安全调用**批量一次**交判定器三态裁决——`allow`→执行 / `confirm`→转 `on_confirm` 人工审批（无回调则拒绝）/ `deny`→拒绝并把理由回传 LLM。判定调用以 `temperature=0` 发起（确定性输出；不支持该参数的 provider 自动过滤）。**判定失败一律 fail-open**（回退为直接执行+警告日志，判定器是增强安全网而非硬门槛）。经 ContextVar `_current_parent_judge` 继承给子代理，父关闭时子也关闭（委派不旁路判定）；`judge_llm` 为 AsyncOpenAI 封装可跨 Agent 共享
- **工具执行顺序**：todo 计划工具（`todo_write`/`todo_read`）必须单独成批调用（不可与其他工具混在一批）；普通工具一批内通过 `asyncio.gather` 并发执行。另有「跨轮次顺序守卫」：一旦开始执行非计划工具（`_work_started`），禁止再创建计划
- `ConversationHistory`（`history.py`）：消息列表 + 4 种截断策略（`none`, `sliding_window`, `token_limit`, `head_tail`），内部持有 `Compactor`
- `Session`（`session.py`）：会话持久化，每会话一个 `{session_dir}/{id}/` 目录（默认 `~/.milu/sessions/`），`conversation.jsonl`（append-only 消息日志，SYSTEM 不记录）+ `session.json`（元数据）。支持 compaction 快照点恢复。**日志分段**：`Agent.reset()` 在同一会话目录内切换到新空日志段 `conversation.N.jsonl`（旧段保留归档），活动段 = 目录中段号最大者（磁盘即真相），load/计数只针对活动段 → reset 后加载不含旧历史
- `Compactor`（`compactor.py`）：上下文自动压缩流水线，**0/1 次 API 调用分层**：L1 消息数裁剪 → 轮次分层工具结果压缩（旧轮→占位符、中间轮→截断、近期轮→保留，均带 session 文件指针）→ L4 超 `trigger_ratio` 时 LLM 摘要。阈值随 `max_context_window` 动态计算。另提供 `compact` 元工具供 LLM 主动触发
- `SubAgent`（`subagent.py`）：`create_subagent_tools()` 工厂为每个 `SubAgentConfig` 生成一个 `@tool` 闭包；调用时**每次新建独立 Agent + 干净历史**，`register_catalog=False`/不嵌套子 Agent（结构性保证）。**模式与确认回调均通过 ContextVar 继承**（`_current_parent_mode` / `_current_parent_confirm`，`Agent.run()` 入口注入）——无需 `get_parent_mode` 回调、子代理工具可在 Agent 之前创建且跨 Agent 共享；AUTO 模式下子代理内的不安全工具同样走父的人工确认（**委派不构成安全旁路**）。`SubAgentConfig.role`（main/coder/researcher/reader/reviewer）便利字段自动套用对应内置角色提示词
- **内置子代理三件套**：`builtin_subagent_configs()` 返回生产级预设——`researcher` 调研员（web_search/web_fetch/datetime + deep-research 技能，只读）、`reader` 长内容阅读员（file_read/web_fetch/image_read/doc_read，定向提取，只读）、`coder` 编码执行员（python_repl/file_read/file_write）；可选 `reviewer` 审查员（include 加入）。选型标准：上下文隔离 / 权限收窄 / 可并行

### 3. 工具层 (`src/milu/tools/`)

- `@tool(name, description, is_safe=True, safe_check=None)` 装饰器 → 生成 `ToolWrapper`（含自动 JSON Schema、`is_safe`、`safe_check` 动态判定、`meta` 元工具标记）。**注意：旧版的 `dangerous`/`priority` 参数已移除**
- `ToolRegistry` 双池设计（`registry.py`）：active pool（schema 注入 LLM）+ dormant pool（MCP 工具待激活）
- `catalog.py` 提供三个元工具（`list_catalog`, `search_tools`, `activate_tools`）供 LLM 自主发现/激活 dormant 工具
- `ToolExecutor`（`executor.py`）安全执行：JSON 参数解析、async/sync 兼容、异常捕获
- 内置工具（`builtin/`）：`file_tool`（read/write 分开；对二进制文档/图片扩展名返回 doc_read/image_read 引导）、`image_read`（**图片视觉输入**，见下）、`doc_read`（**Office/PDF 文档提取**：docx 段落+表格转 Markdown、xlsx/xlsm/xls 按 sheet 读取上限 500 行、pdf 按页码范围默认前 20 页、pptx 逐页；.doc/.ppt 老格式返回转换指引；解析库 python-docx/openpyxl/pypdf/python-pptx/xlrd 为**核心硬依赖**）、`shell_command`、`python_repl`、`http_request`（API/JSON 场景）、`web_fetch`（网页→Markdown 正文提取，阅读场景优先，省 token）、`web_search`（**可插拔后端**：环境变量 `WEB_SEARCH_PROVIDER`=ddg 默认/tavily/bocha + 对应 Key；DDG 国内不可用，国内部署配 bocha 或用 LLM 自带搜索）、`datetime_tool`、`structured_output`、`todo_write`、`memory_write/memory_read`（长期记忆，**不在 BUILTIN_TOOLS 默认列表**，由 `Agent(memory=...)` 开关启用时自动注册，见下方设计约束）
- **图片视觉输入（`image_tool.py` + `llm/base/vision.py`）**：让视觉模型"看到"本地图片。两条入口：① LLM 自主调 `image_read(path)`（校验存在/格式 png/jpg/jpeg/webp/gif/bmp/≤10MB/视觉能力），路径登记到 per-run ContextVar `_current_pending_images`，Agent 在该批工具结果回填后**追加注入一条多模态 user 消息**；② 程序化 `Agent.run(user_input, images=[...])`。**轻量引用块设计**：历史/会话日志只存 `{"type":"image_path","path":...}`（不含 base64，JSONL 不膨胀、token 估算不失真、会话可恢复），发送 API 时才在 `BaseLLM._messages_to_dicts()` 单点物化为 base64 data URL（chatgpt Responses API 经 `_convert_user_content` 转 input_image）；物化失败降级为 text 占位块不中断对话。能力声明是 provider 级（仅 deepseek 无视觉），实际需配视觉模型（如 qwen-vl-plus）

### 4. MCP 层 (`src/milu/tools/mcp/`)

- 支持三种传输：`stdio`, `streamable_http`, `sse`
- `MCPManager` 多服务器编排：`asyncio.gather` 并行连接，错误隔离
- `converter.py` 将 MCP Tool 转为 `ToolWrapper`，工具名加 `{server_name}__` 前缀避免冲突
- 配置文件搜索顺序（未显式传 `mcp_config_path` 时）：`{project_dir()}/config/mcp_servers.json`（项目级「读配置」，默认 CWD，可用 `MILU_PROJECT_DIR` 覆盖）→ `~/.milu/mcp_servers.json`（用户级兜底，可用 `MILU_HOME` 覆盖）；亦可用环境变量 `MCP_CONFIG_PATH` 指定绝对路径

### 5. 提示词 & 技能层 (`src/milu/prompts/`, `skills/`)

- `PromptBuilder`（`prompts/builder.py`）：从 Markdown 目录**分层拼装** system prompt。每个 `.md` 是一个片段，YAML frontmatter 控制 `section`(safeguard/soul/agent/memory/custom) + `order` + `enabled`，`{{key}}` 变量插值，**每次 `build()` 重读文件支持热重载**。预置角色提示词随包分发在 `src/milu/templates/prompts/{main,coder,researcher,reviewer}/`，通过 `milu.builtin_prompts_dir(role)` 定位（内置技能同理用 `builtin_skills_dir()`）
- `SkillRegistry`（`skills/registry.py`）：技能**元数据（name/description/triggers）始终注入 system prompt**，正文按需通过 `load_skill` 元工具拉取。无激活/卸载生命周期。支持平铺 `skills/x.md` 或子目录 `skills/x/SKILL.md`；**多文件技能**（目录内含 examples/reference/scripts 等附属文件）在 load_skill 返回时自动注明技能目录绝对路径，供 file_read 访问附属资源
- **内置技能 9 个**：自研 4 个（skill-creator/deep-research/content-writing/doc-formatting）+ 移植 5 个（官方 anthropics/skills Apache-2.0：frontend-design/internal-comms/mcp-builder；社区 obra/superpowers MIT：systematic-debugging/test-driven-development）。来源与许可见 `templates/skills/THIRD_PARTY_NOTICES.txt`。⚠️ 官方 docx/pdf/pptx/xlsx 四件套为专有许可禁止再分发，**不可移植**。（translator/code-review 曾内置，已删）

### 6. 服务层 (`src/milu/serving/`)

- `AgentPool`（`pool.py`）：多用户并发资源池。**按 `(user_id, session_id)` 缓存独立 Agent 实例**，`async with pool.acquire(uid, sid) as h: h.agent.run(...)`。LRU + idle TTL 淘汰、全局 `Semaphore` 并发限流、后台 sweep 清理、`get_stats()` 监控（含 hit_rate）
- 便利构造：**`AgentPool.from_llm(llm)`** 一行起池（共享同一 LLM 实例，AsyncOpenAI 协程安全）。「全配默认」由 Agent 自身提供（见 Agent 层），默认工厂只叠加服务层语义：确定性 session_id 派生 + 共享 MCP 注入；运行限额经 `agent_config=AgentConfig(...)`、其余 Agent 参数经 `agent_kwargs={"mode": ..., "tools": [...], "subagents": [...]}` 原样透传
- 四个硬不变量：每个 `(user_id, session_id)` ≤1 实例；实例数 ≤ `max_agents`；并发 run ≤ `max_concurrent_runs`；空闲超 `idle_ttl_seconds` 被清理
- `pool.remove(user_id, session_id)`：主动驱逐并关闭指定实例（`_global_lock` 下复用 `_close_entry`），供「运行时切换设置后按新配置重建」用；运行中也可移除（运行协程持自身 agent 引用不受影响，共享 MCP 由池拥有不被 `disconnect_mcp` 断开）

### 6.5 内置 Web 服务 (`src/milu/serving/web/`，CLI `milu serve` 一键启动)

把 AgentPool（多用户 SSE 流式对话）+ 可选嵌入式 ScheduleEngine（定时任务，`--no-scheduler` 关）+ 共享 MCP，沉淀为**包内置的正规服务**，配一个**全功能单页演示前端**（`static/index.html`，纯 vanilla 无构建链/无外部 CDN，随 wheel 分发）。

- **应用工厂 `create_app(...)`**（`app.py`）+ 入口 `run_server(host, port, reload, **opts)`；CLI `_cmd_serve` 经 `resolve_settings` 取默认厂商/模型/模式后启动 uvicorn
- **依赖隔离**：fastapi/uvicorn/sse-starlette 为 `[serve]` 可选依赖（`pip install "milu[serve]"`）；子包 `__init__` 延迟导入，`import milu.serving.web` 不强依赖它们，未装时 `milu serve` 给中文提示。⚠️ `app.py` 顶层必须导入 `Request` 等 FastAPI 类型——配合 `from __future__ import annotations`，函数内局部导入会让 FastAPI 无法从模块全局解析字符串注解（误判为 query 参数 → 422）
- **运行时切换厂商/模型/模式**：`app.state.prefs[(user,session)]` 存偏好覆盖、`app.state.llm_cache[(provider,model,...)]` 按型号缓存共享 LLM（AsyncOpenAI 协程安全）；自定义 `agent_factory` 读偏好取缓存 LLM 构造 Agent（memory/schedule_user 按 user_id 派生隔离）。`POST /api/settings` 写偏好 + `pool.remove()` 驱逐 → 下条消息按新设置重建；`POST /api/mode` 即时 `set_mode` 并存偏好
- **确认流队列桥接**：危险工具确认时 `on_confirm` 会**阻塞** `agent.run()` 生成器，故后台任务跑 `agent.run()` 喂 `asyncio.Queue`，`on_confirm` 在 `await future` 前把「ConfirmationRequest」推进队列，SSE 协程独立排空队列——保证弹窗在阻塞期间即时显示；`POST /api/confirm`（按 user+session 定位 Future）解析放行，120s 超时自动拒绝
- **端点**：`/`（前端）、`/api/chat`（SSE，`/` 命令复用 `_exec_command`）、`/api/confirm`、`/api/providers`、`/api/settings`、`/api/mode`、`/api/sessions` + `/api/session/action`（new/load/save/reset）、`/api/schedule/{tasks,create,action,results}`、`/api/tools|skills|memory|stats`
- **定时任务结果提醒**：服务端 `notify=False`（无桌面弹窗），前端全局轮询 `/api/schedule/results`（15s），新结果插入聊天区系统行 + toast、任务面板自动刷新（首轮只记游标不回放历史）；run_at 输入用 `datetime-local` 控件防格式错
- 前端四面板：设置（厂商下拉含 Key 状态/模型/模式/开关）、会话、定时任务（创建表单 + 启停/删除/立即运行 + 结果轮询）、信息（工具/技能/记忆/统计）；主区流式渲染（文本/思考/工具/子代理 + 轻量 Markdown）+ 危险工具确认弹窗
- 示例 `examples/multi_user_chat.py` / `examples/scheduler_server.py` 保留作教学（本服务正是其能力的内置化整合）

### 7. CLI 层 (`src/milu/cli/`)

- 入口点：`pyproject.toml` 的 `[project.scripts]` 注册 `milu`，解析到 `milu.cli:main`；亦支持 `python -m milu.cli`
- `app.py`：argparse 命令面 + `main()`。子命令 `chat`（无子命令时的默认）/ `run`（一次性，支持 stdin 管道、`-q` 只输出最终文本）/ `setup`（初始化引导）/ `config`（show/path/get/set/init）/ `sessions`（list/show）/ `providers` / `version`。全局选项（`-p/--provider`、`-m/--model`、`--api-key`、`--mode`、`--no-session/--no-mcp/--no-subagents`）写在子命令**之后**。`config get/set` 用**点号路径**操作分层配置（如 `milu config set agent.max_turns 50` 写用户级），`config init` 在项目生成 `config/milu.json` 全量模板。`main()` 统一捕获 `AuthenticationError`/`ValueError`/`MiluError` 给中文友好提示与退出码
- `config.py`（`src/milu/config.py`，**核心分层配置**，见下「分层配置体系」）+ `cli/config.py`（CLI 参数层：`Settings` + `resolve_settings()`，把文件配置叠加 CLI 参数 + 解析 API Key）。**解析优先级**：provider/model/mode 等 = CLI 参数 > 用户 `~/.milu/config.json` > 项目 `config/milu.json` > 内置默认；api_key = CLI 参数 > 环境变量 `{PROVIDER}_API_KEY`（**密钥不再落 config.json**）。注意：配置里的 `model` 只在「未切换厂商」时沿用，避免给 deepseek 套上 qwen 的模型名
- `setup_wizard.py`（`milu setup` **初始化引导**，pip 安装后零手工配置）：4 步交互——选厂商（带中文名/Key 状态/默认模型）→ 选模型 → API Key（带各厂商申请地址，已有 Key 回车保留）→ 搜索后端（bocha/tavily/ddg + 对应 Key）；可选发一次最小请求验证 Key。**密钥写用户级 `~/.milu/.env`**（`update_env_file` 合并写：已有键原位更新、注释保留），厂商/模型经 `set_user_value` 写 `~/.milu/config.json`；写入后同步 `os.environ` 当前进程即时生效。配套 `_env.py` 的 `ensure_dotenv_loaded()` 在项目级 .env（CWD 向上）之后**兜底加载用户级 `~/.milu/.env`**（override=False，进程环境变量 > 项目级 > 用户级），任意目录运行 CLI 均生效。`milu chat` 入口（TTY 下）检测不到当前厂商 Key 时经 `offer_first_run_setup` 自动询问进入引导，完成后重新加载配置继续对话。输入统一剥离 BOM/零宽字符（Windows 管道喂入首行带 BOM）
- `builder.py`：`build_llm()` / `build_agent()`——**最简创建**，依托 Agent「全配默认」：tools/skills/prompt 不传 → 自动注入全套内置工具、内置技能、内置 main 提示词；`subagents=None` → 内置三件套（`--no-subagents` 传 `[]` 关闭）；运行限额/压缩来自分层配置（`settings.agent` 构造 `AgentConfig`、`settings.compact` 经 `ConversationHistory` 传 `CompactConfig`）
- `render.py` / `repl.py`：终端渲染（ANSI 颜色、不安全工具确认回调、事件流渲染）与交互式 REPL（全部 `/命令`），**迁移并整理自 `examples/multi_turn_chat.py`**
- Windows 注意：`main()` 启动时 `os.system("")` 启用 ANSI；stdin 管道按 `sys.stdin.buffer` 显式 UTF-8 解码（规避控制台代码页 + surrogateescape 把中文解成孤立代理字符）

### 8. 调度层 (`src/milu/scheduler/`)

- **多用户定时任务**：`ScheduleStore`（`store.py`）按用户分文件 `~/.milu/schedules/{safe_user_id}.json`（与 memory 同一 safe 化规则；旧单文件 `schedules.json` 首次访问自动迁移为 `schedules/default.json`，源文件保留 `.migrated`）。任务名在**同一用户内**唯一；写盘一律原子写（mkstemp+fsync+os.replace）；跨进程丢更新窗口与 session.py 同档取舍（POSIX flock / Windows 接受低概率）
- `ScheduleEngine`（`engine.py`）：asyncio 主循环每分钟 tick，到期任务 `gather + Semaphore(max_concurrent_tasks)` 并发执行，store 写操作按 `task.user_id` 取 `asyncio.Lock` 防进程内丢更新；单任务 `wait_for(task_timeout)` 超时保护。`SchedulerConfig`（engine.py 顶部，比照 AgentPoolConfig 不单独建文件）进 config.json `scheduler` 分节
- **三种运行形态**（共用单实例锁，见下）：① CLI 守护进程 `milu scheduler start`（前台 `start()`，`echo=True` 控制台回显）；② 嵌入 Web 服务 `engine.start_background()`（FastAPI lifespan 中启动，见 `serving/web/app.py`）；③ **嵌入 CLI chat**——`run_chat`（repl.py）启动时起**独立 daemon 线程**跑 `asyncio.run(engine.start())`，对话期间定时任务自动执行、退出即停（`--no-scheduler` 关，`Settings.use_scheduler`）；⚠️ 不能用 `start_background()` 挂 REPL 主循环——REPL 的 `input()` 同步阻塞主线程事件循环（停在 You> 提示符时即冻结），tick 永远得不到调度；`milu run` 一次性命令不嵌入
- **单实例锁 `SchedulerLock`（`lock.py`，已从 CLI 层下沉）**：PID 文件 `~/.milu/scheduler.lock` + 跨平台进程检活 `pid_alive`（Windows 切勿 os.kill 探测）。多引擎并存会重复执行任务（tick 全量扫盘、无任务级防重），故 daemon/chat 嵌入/web 嵌入三方都走锁；`try_acquire()` 同进程重入幂等（探测与正式获取可分离）；`release()` 仅持有者生效（非持有者 no-op，防误删他人锁）；stale 锁（持有者已死）自动覆盖
- **锁守望/自动接管 `engine.start_with_lock(lock, retry_seconds=30)`**：嵌入方（chat 线程 / web `start_background(lock)`）抢到锁即运行，被占用则周期重试、**持锁者退出后自动接管**——chat 与 serve 同开时先开方退出，另一方接管执行，任务不会因锁竞争静默不执行（曾是「拿不到锁就永远不嵌入」的坑）；等锁阶段 `stop()` 下一轮检查即退出且不碰他人的锁；daemon `milu scheduler start` 仍是「拿不到锁直接拒绝启动」（前台进程语义）
- **引擎韧性防护（曾修复 Web 嵌入任务静默不执行的 bug）**：`echo` 是 `ScheduleEngine.__init__` 参数（非 SchedulerConfig 项——运行形态差异由调用方代码决定，不进 config.json），`echo=False`（嵌入默认）完全不写 stdout，`echo=True` 经 `_echo_print` 容错编码（stdout 重定向为 GBK 时 `▶`/`✓` 曾抛 UnicodeEncodeError 被 gather 静默吞掉、任务永不执行）；`_safe_tick` 单次 tick 异常不杀主循环；`start_background()` 带 done-callback，后台任务异常退出记 error 日志（无人 await 它）
- **执行器二选一**：注入 `agent_pool` → 任务经 `pool.get_or_create_agent(task.user_id, ...)` 执行（per-user 实例复用/共享 MCP/memory 派生，**不占在线并发许可**）；不注入 → 每任务自建独立轻量 Agent（CLI 模式）
- **结果投递三通道**：outbox JSONL `~/.milu/scheduler_outbox/{user_id}.jsonl`（带 flock append）→ `on_result` 异步回调（服务端推送）→ 系统弹窗（`notify.py`，Windows ctypes MessageBoxW / macOS osascript / Linux notify-send；`SchedulerConfig.notify` 可关）
- **工具层用户上下文**（`schedule_tool.py`）：`_current_schedule_user` ContextVar 由 `Agent.run()` 入口注入（`Agent(schedule_user=...)`，与 `memory` 平级的能力参数）；**未注入时退化为 "default"（与 memory 的写拒绝是有意差异**，因 schedule 工具在 BUILTIN_TOOLS 默认列表，CLI 单人无注入须兼容）；user_id 不暴露为 LLM 参数（防伪造他人身份）。AgentPool 默认工厂**默认**按 user_id 派生 `schedule_user`（不派生则全部用户共用 default 任务空间，跨用户可见/可删）
- **工具拆分（参数正交性原则）**：LLM 侧暴露两个工具——`schedule_create`（专职创建，9 个参数全部相关，整体不安全）+ `schedule_manage`（action=list/delete/enable/disable/run_now，schema 极简仅 action+name；`safe_check` 动态判定 list 只读安全，其余走审批/AI 判定）。不合并为单工具的原因：create 专属参数占比过高，管理操作的 schema 会充满无关参数噪音；也不再拆细——delete/enable/disable/run_now 形态一致（都只要 name）

## 关键设计约束（多用户并发 / 无状态化）

这是近期重构的核心，改动相关代码前必读 `serving/pool.py` 顶部的长注释：

- **Agent 含实例级共享状态**（`history`、`session`、`_work_started`、`_mcp_manager`、`tools` 等），多用户**不能共享同一个 Agent**。唯一安全方案是 **per-user Agent**（`AgentPool` 即为此而生）。瓶颈是 MCP 子进程内存（每 Agent 3-5 个 server 占 15-50 MB），不是 Agent 本身
- **todo 工具与 subagent 已无状态化**：不再用模块级单例/闭包变量，而是 Agent 在 `run()` 入口通过 **ContextVar** 注入 per-call 状态（`todo_write._current_session_dir` 注入 session 目录、`todo_write._current_plan_items` 注入内存计划、`subagent._current_subagent_events` 注入事件列表、`subagent._current_parent_mode` 注入父模式），实现 asyncio 任务级隔离。新增任何"跨调用共享"的工具状态时，沿用 ContextVar 模式，**切勿用模块级全局变量**
- **todo 计划存储双后端**（已与 session 解耦）：有 session → 文件后端 `{session_dir}/plan.json`（持久化、per-user 天然隔离）；无 session → 内存后端（`_current_plan_items` ContextVar，同一 Agent 跨轮保留、进程退出即弃）。因此 `session_enabled=False`（含子代理、用户自管 history）时 todo 也能用，不再抛 `RuntimeError`。LLM 通过 `todo_read` 主动拉取
- **分层配置体系**（`src/milu/config.py` 为单一真相源）：可调参数统一走「**CLI 参数 > 用户 `~/.milu/config.json` > 项目 `config/milu.json` > 代码内 dataclass 默认值**」四级优先级。`MiluConfig` 嵌套分节 `agent`（含 `mode`/`session_enabled`/`llm` 模型对象/运行限额）/`compact`/`pool`/`default_models`（仅查看参考）；`_builtin_defaults()` **从现有 dataclass 派生**基线（`AgentConfig`/`CompactConfig`/`AgentPoolConfig`），**默认值在代码里只有一份**，config.json 只承载覆盖。`load_config()` 做分层深合并。**库纯净性**：配置只在应用/CLI 入口（`build_agent` / `AgentPool.from_llm`）显式加载下传，**不侵入** `Agent` / 裸 `AgentPool(...)` 构造——直接 `Agent(llm)` 仍走 dataclass 默认，单测 hermetic。**职责分离**：`.env` 只放密钥（`{PROVIDER}_API_KEY`、搜索后端 Key）与进程级开关（`MILU_HOME`/`MILU_PROJECT_DIR`/`MCP_CONFIG_PATH` 等），可调参数全部在 config.json；config.json 旧 `api_keys` 字段已废弃（加载时忽略 + 一次性告警）。`config set agent.max_turns 50` 按当前值类型转换、稀疏写入用户级文件
- **目录策略：写数据 vs 读配置分离，均与裸 CWD 解耦**（`resources.py` 顶部注释为单一真相源）：**写数据**（会话日志、记忆、CLI 配置）锚定 `user_data_dir()`（默认 `~/.milu`，`MILU_HOME` 覆盖）；**读配置**（`mcp_servers.json`、`.env` 等项目自带配置）锚定 `project_dir()`（默认 CWD，`MILU_PROJECT_DIR` 覆盖），项目级找不到再回退用户级。新增「写状态」目录走 `user_data_dir()`、新增「读配置」走 `project_dir()`，**切勿在代码里写裸相对路径 `./xxx`**（作为库被集成时 CWD 漂移）
- **memory 长期记忆为用户级存储、单开关启用**（`memory_tool.py`）：**默认关闭**——`Agent(memory=False)`（默认）不注册工具不注入提示词；`memory=True` 启用（身份 `"default"`）；`memory="user_id"` 启用并按用户隔离。存储与 session **解耦**：`~/.milu/memory/{user_id}.json`（`MILU_HOME` 可覆盖），同一标识跨 session、跨进程共享。启用时记忆条目**每轮渲染进 system prompt 末尾**（`render_memory_prompt`，每轮重读文件），记忆文件路径在 `run()` 入口经 ContextVar `_current_memory_path` 注入（未启用注入 None，子代理不会继承父路径误写）。AgentPool 默认工厂把 `agent_kwargs={"memory": True}` 派生为 `memory=user_id`（防全部用户共享一份 default 记忆）。条目 {content, category, created_at}，上限 200 条丢弃最旧，内容级去重。与对话历史互补——历史会被压缩截断，记忆条目始终完整

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
