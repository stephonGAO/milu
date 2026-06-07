# 变更日志

本项目重要变更记录。日期格式 YYYY-MM-DD。

## [0.1.0] - 2026-06-07

🎉 **首个公开发布版本**。核心能力一览：

- **统一 LLM 抽象层**：一套接口接入 9 个提供商（通义千问、Kimi、GLM、DeepSeek、MiniMax、豆包、ChatGPT、Gemini、Claude），全部 async-first 流式
- **Agent 编排引擎**：工具调用循环、四种操作模式（talk/manual/auto/superwork）、AI 安全判定器、子代理三件套（researcher/reader/coder）
- **工具系统**：20+ 内置工具（文件/Shell/Python/网页抓取/可插拔搜索/Office 文档/图片视觉输入等）+ MCP 协议（stdio/HTTP/SSE）+ 技能系统（内置 9 个技能）
- **会话与上下文**：会话持久化、上下文自动压缩、长期记忆（memory）、向量知识库 RAG（knowledge，含前置自动检索）
- **多用户服务**：AgentPool 并发资源池（LRU/TTL/限流）、内置 Web 服务 `milu serve`（SSE 流式 + 演示前端）、多用户定时任务调度
- **CLI**：`milu chat/run/setup/serve/scheduler/config/sessions/providers`，零配置开箱即用

以下为发布前开发期的主要变更记录（保留供参考）：

### 项目改名 milu（2026-06-04）

#### 变更（破坏性）

- **项目正式更名为 `milu`（麋鹿）**：原名 `agent-framework` 与微软 PyPI 官方包冲突，
  发布前必须更换；新名呼应「四不像」集多家之长，PyPI 占位包 milu 0.0.1 已锁定名称。
- **全面迁移**：
  - 包名/导入：`agent_framework` → `milu`（`src/agent_framework/` → `src/milu/`）
  - PyPI 包名与 CLI 入口点：`agent-framework` → `milu`；**短别名 `afx` 移除**（milu 已足够短）
  - 异常基类：`AgentFrameworkError` → `MiluError`
  - 环境变量：`AGENT_FRAMEWORK_HOME` → `MILU_HOME`、`AGENT_FRAMEWORK_NO_DOTENV` → `MILU_NO_DOTENV`
  - 用户数据目录：`~/.agent_framework/` → `~/.milu/`（会话、配置、记忆、MCP 用户级配置）
- 升级提示：已安装旧包的环境需 `pip uninstall agent-framework` 后重装；
  本地已有 `~/.agent_framework/` 数据的用户手动迁移到 `~/.milu/` 即可继续使用

### memory 长期记忆升级为用户级存储（2026-06-04）

#### 变更（破坏性）

- **存储位置**：`{session_dir}/memory.json` → `~/.milu/memory/{user_id}.json`
  （`MILU_HOME` 可覆盖）。与 session 彻底解耦——同一用户标识跨 session、
  跨进程共享同一份记忆，解决「记忆锁在单个会话里，开新对话就丢」的问题。
  旧的 session 内 memory.json 与无 session 内存后端**均已移除**。
- **默认关闭，单开关启用**：`Agent(memory=False)`（默认）不注册 memory 工具；
  `memory=True` 启用（身份 "default"）；`memory="user-123"` 启用并按用户隔离。
  `memory_write` / `memory_read` 相应移出 `BUILTIN_TOOLS` 默认列表。
- **记忆条目注入 system prompt 末尾**：启用时每轮把全部条目渲染进系统提示词最后的
  「长期记忆」一节（每轮重读文件，跨进程新写入即时可见），LLM 无需调用 memory_read
  即可看到；main 角色提示词中的静态记忆指引随之移除（指引并入动态注入段）。
- **AgentPool 多用户安全**：默认工厂把 `agent_kwargs={"memory": True}` 自动派生为
  `memory=user_id`，避免所有用户共享同一份 "default" 记忆；显式传字符串则尊重调用方
  （团队共享记忆场景）。

### 移植 5 个热门开源技能（2026-06-05）

#### 新增

- **从官方与社区仓库原样移植 5 个热门技能**（内置技能 6 → 11 个）：
  - anthropics/skills（Apache-2.0 @ da20c92）：`frontend-design` 前端设计质量、
    `internal-comms` 职场内部沟通写作（附 examples/）、`mcp-builder` MCP server 构建指南
    （附 reference/ 与 scripts/）
  - obra/superpowers（MIT @ 6fd4507）：`systematic-debugging` 系统化调试方法论、
    `test-driven-development` TDD 纪律
  - 各技能目录保留上游 LICENSE.txt；来源与移植说明见 `templates/skills/THIRD_PARTY_NOTICES.txt`
- **load_skill 多文件技能增强**：目录式技能（含附属文件）在返回正文时自动注明
  技能目录绝对路径，LLM 可据此用 file_read 读取 examples/reference/scripts 等附属资源

#### 说明

- 官方 docx/pdf/pptx/xlsx 文档四件套为**专有许可**（禁止提取/复制/衍生/再分发），
  依法不可移植进本包；未选 superpowers 的 brainstorming（其 "MUST use before any
  creative work" 描述会劫持通用助手的创作类请求）

### 内置技能扩充 + MCP 推荐配置模板（2026-06-04）

#### 新增

- **内置技能 `content-writing`**（中文内容创作）：公众号/小红书/邮件/文案/报告多体裁，
  含平台风格速查与写作原则。C 端最高频需求，纯指令零依赖。
- **内置技能 `doc-formatting`**（文档排版整理）：零散内容 → 规范 Markdown
  （标题层级/表格/列表/目录），只重组不改写。轻量替代 office 文档类重依赖技能。
- **MCP 推荐配置模板 `config/mcp_servers.example.json`**：按生态热度选型
  （Playwright 浏览器自动化、Context7 文档检索、官方 Git/Filesystem/Sequential-Thinking），
  重依赖能力走 MCP 而非内置工具，多用户场景配合 shared_mcp 整池共享。

#### 变更

- **全部内置技能 description 按官方最佳实践重写**：第三人称、「做什么 + 何时用」、
  嵌入用户会说的触发词（description 决定技能能否被正确触发）。
  涉及 translator / code-review / deep-research。内置技能增至 6 个。

### memory 长期记忆工具（2026-06-04）

#### 新增

- **内置工具 `memory_write` / `memory_read`**：LLM 主动记录跨轮次、跨重启的长期信息
  （用户偏好 preference / 事实背景 fact / 长期约定 agreement）。与对话历史互补——
  历史会被压缩截断，记忆条目始终完整保留。对标 MCP 官方 Memory reference server 的轻量实现。
- 存储复用 todo 同款双后端模式：有 session → `{session_dir}/memory.json`（持久化、
  per-user 隔离）；无 session → 内存后端（同一 Agent 跨轮保留）。
- 上限 200 条自动丢弃最旧；内容级去重；main 角色提示词补充主动记忆指引。

### web_search 可插拔后端（2026-06-04）

#### 变更

- **`web_search` 后端可插拔**（解决 DuckDuckGo 中国大陆不可用的生产问题）：
  环境变量 `WEB_SEARCH_PROVIDER` 选择后端——`ddg`（默认，无需 Key）/
  `tavily`（`TAVILY_API_KEY`，为 LLM 设计）/ `bocha`（`BOCHA_API_KEY`，国内可直连）。
  配置了 provider 却缺 Key 时明确报错（不静默回退）。
  自定义通用 API（`SEARCH_API_URL`+`SEARCH_API_KEY`）保持向后兼容且优先级最高。
- 国内部署亦可直接使用 LLM 自带联网搜索（如 Qwen `web_search=True`），不依赖本工具。
- 各后端结果统一格式化（标题/摘要/链接编号列表）。

### 提示词体系完善（2026-06-04）

#### 变更

- **重写内置 main 角色四件套**（对齐业界最佳实践）：补齐语气与风格、工具使用策略
  （并行调用 / 专用优先 / 参数真实 / 时效用 datetime_tool / 失败如实）、主动性边界、
  防提示注入等板块；修正过时内容（子代理名单 reviewer→reader、废弃的 dangerous 标记
  表述、memory.md 仓库私货）；子代理委派四要素（目标/返回契约/信息指引/边界）写入提示词
- **soul 语气定位**：跟随用户语言（不限中文）；对话交流自然有温度、可提供情绪价值，
  执行任务简洁利落——区分两种场景而非一刀切。后续可扩展为多风格 soul 变体切换
- **环境变量机制为可选项**（缓存友好）：`_default_prompt_variables()` 提供
  `{{current_date}}/{{platform}}/{{cwd}}` 供**自定义模板**按需引用（动态值会使提示词
  前缀缓存失效，谁引用谁承担）；**内置模板不引用**，时效判断走 datetime_tool 按需获取

### web_fetch 网页正文抓取工具（2026-06-04）

#### 新增

- **内置工具 `web_fetch`**（网页 → Markdown 正文提取）：自动剔除脚本/样式/导航等噪声标签，
  优先提取 article/main 正文容器，实测可节省 20-30% token。对标 MCP 官方 Fetch server 与
  Claude Code 的 WebFetch。与 `http_request` 分工：读网页用 web_fetch，调 API 用 http_request。
  支持 max_chars 截断（标注全文长度）、JSON 美化、转换失败降级原文。
- 新依赖 `markdownify`（含 beautifulsoup4，轻量纯 Python）。

#### 变更

- 内置子代理换装：`researcher` 工具集 → web_search + **web_fetch** + datetime_tool；
  `reader` → file_read + **web_fetch**（原 http_request 移出，页面阅读场景由 web_fetch 接管）。
- researcher/reader 角色提示词同步更新。

### 全配默认下沉进 Agent（2026-06-04）

「开箱即用」策略从 AgentPool 默认工厂下沉到 `Agent.__init__` 本身——**直接构造 Agent
与经池构造拿到同规格实例**，默认策略只有一份。Pool 仍只用于多用户服务场景；
单用户/脚本/嵌入场景直接 `Agent(llm)` 即全配，两个入口都是一等公民。

#### 变更（含行为变更）

- **`Agent(tools=None)`（顶层）→ 默认注入全套 `BUILTIN_TOOLS`**；显式 `[]` = 无工具，显式列表 = 该列表。
- **新增 `Agent(subagents=...)` 参数**：`None`（顶层）→ 默认注册内置子代理三件套；`[]` = 关闭；`list[SubAgentConfig]` = 自定义。子代理 `register_catalog=False` 不触发默认 → 结构上保证不嵌套。
- AgentPool 默认工厂随之**简化**：不再自带 tools/subagents 默认逻辑，仅注入 session_id 派生与共享 MCP；`agent_kwargs` 的 `"subagents"` 从特殊键转为正式 Agent 参数（外部用法不变）。
- `examples/multi_turn_chat.py` 的 `build_agent()` 由 ~100 行手工组装简化为零配置构造；`examples/multi_user_chat.py` 同步精简。

### 内置子代理三件套 + 确认回调透传（2026-06-04）

#### 新增

- **内置子代理预设 `builtin_subagent_configs()`**（按「上下文隔离/权限收窄/可并行」选型）：
  - `researcher` 调研员：web_search + http_request + datetime_tool（只读），配 `deep-research` 内置技能；
  - `reader` 长内容阅读员（**新增角色**，含内置提示词 `templates/prompts/reader/`）：file_read + http_request（只读），长文档/日志/网页定向提取；
  - `coder` 编码执行员：python_repl + file_read + file_write；
  - 可选 `reviewer` 审查员：file_read + python_repl，配 `code-review` 技能（`include` 加入，不在默认集）。
- **确认回调透传**：`Agent.run()` 经 ContextVar `_current_parent_confirm` 把 `on_confirm` 注入子代理——AUTO 模式下子代理内的不安全工具（file_write 等）同样走父 Agent 的人工确认，**委派不再是安全审批的旁路**。
- **AgentPool 默认工厂自动注册三件套子代理**；`agent_kwargs={"subagents": []}` 关闭、传 `list[SubAgentConfig]` 自定义（特殊键，非 Agent 参数）。
- 新增内置技能 `deep-research`；`PROMPT_ROLES` 加入 `reader`。

#### 变更

- `researcher`/`coder` 角色提示词补充**返回契约**（最终回复即交付物：只回传蒸馏结论/产物路径，不回传过程）；researcher 流程修正为使用 `web_search`（原文误写为 http_request）。
- `examples/multi_user_chat.py` 改为消费 `builtin_subagent_configs()`，子代理工具创建一次、整池共享（无状态化后 `get_parent_mode` 闭包限制已不存在）。

### 默认极简化 / 能力参数归位（2026-06-03）

围绕「C 端开箱即用 + 技术人员可覆盖」的规范化整理。**含破坏性 API 变更**，迁移见下。

#### 新增

- **开箱即用默认**：`Agent(llm)` 即可用——顶层 Agent 默认套用内置 `main` 角色提示词 + 内置技能（无需再手写 `prompt_dir` / `skills_dir`）。
- **用户级数据目录**：`resources.user_data_dir()` / `default_session_dir()` / `default_mcp_config_path()`，默认锚定 `~/.milu`，可用环境变量 `MILU_HOME` 覆盖（与 CWD 解耦，适合部署）。
- **todo 计划内存后端**：无 session 时 todo 计划落内存（run 级注入、同一 Agent 跨轮保留），`session_enabled=False`（含子代理、用户自管 history）时 todo 不再抛 `RuntimeError`。
- **`AgentPool.from_llm(llm)`**：单 LLM 实例一行起池；默认工厂给每个 per-user Agent 配齐 `BUILTIN_TOOLS` + 内置提示词/技能。
- **`AgentPool(..., agent_kwargs={...})`**：向默认工厂透传 Agent 能力参数（`mode` / `session_enabled` / `session_dir` / `tools` / ...）。
- **`SubAgentConfig.role`**：传 `main`/`coder`/`researcher`/`reviewer` 自动套用内置角色提示词。
- **子代理模式继承改用 ContextVar**（`_current_parent_mode`）：不再需要 `create_subagent_tools(..., get_parent_mode=...)` 回调，子代理工具可在 Agent 之前创建（`get_parent_mode` 仍保留为可选显式覆盖）。

#### 破坏性变更

1. **`mode` / `session_enabled` / `session_dir` / `mcp_tools_active_by_default` 从 `AgentConfig` 移到 `Agent.__init__` 直接参数。** `AgentConfig` 现仅含运行限额（`max_turns` / `timeout` / `total_timeout` / `max_total_tokens` / `tool_call_limit`）。

   ```python
   # 旧
   Agent(llm, config=AgentConfig(mode="talk", session_enabled=False, session_dir="/data", max_turns=20))
   # 新
   Agent(llm, mode="talk", session_enabled=False, session_dir="/data", config=AgentConfig(max_turns=20))
   ```

   `AgentPool` 侧：把这些从 `agent_config` 改为 `agent_kwargs`：

   ```python
   # 旧
   AgentPool(llm_factory=..., agent_config=AgentConfig(mode="auto", session_enabled=False))
   # 新
   AgentPool(llm_factory=..., agent_kwargs={"mode": "auto", "session_enabled": False})
   ```

2. **默认会话目录从 CWD `./.sessions` 改为 `~/.milu/sessions`**（与 CWD 解耦）。需要旧行为者显式传 `session_dir="./.sessions"`。

3. **裸 `Agent(llm)` 的默认 system prompt 行为变更**：以前为空，现默认加载内置 `main` 角色提示词 + 内置技能。需要「空提示词」者显式传 `prompt_dir=None`（注意：传 `system_prompt` 时只用 `system_prompt`，不叠加内置）。

4. **移除 CWD 相对的技能自动扫描**（旧 `_SKILL_SEARCH_PATHS` 扫描宿主 CWD 的 `skills/`）。改为默认内置技能目录；自定义技能显式传 `skills_dir=`。

#### 内部 / 设计

- `Agent.set_mode()` 改写实例字段 `self._mode`，不再原地修改共享 `AgentConfig`——从根上消除多用户 mode 跨用户串扰（越权）隐患。
- `prompt_dir` / `skills_dir` 的「`None` → 内置默认」仅对顶层 Agent 生效（以 `register_catalog` 区分），子代理保持精简不变量。
- MCP 配置搜索改为运行期解析用户级路径（尊重 `MILU_HOME`），保留 `./config/mcp_servers.json` 项目级兜底。
- 测试 `conftest.py` autouse fixture 重定向 `MILU_HOME` 到 `tmp_path`，保持单测 hermetic。
- 已验证构建 wheel 包含 `templates/`（内置提示词/技能），pip 安装后开箱可用。
