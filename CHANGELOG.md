# 变更日志

本项目重要变更记录。日期格式 YYYY-MM-DD。

## [Unreleased] — 提示词体系完善（2026-06-04）

### 变更

- **重写内置 main 角色四件套**（对齐业界最佳实践）：补齐语气与风格、工具使用策略
  （并行调用 / 专用优先 / 参数真实 / 时效用 datetime_tool / 失败如实）、主动性边界、
  防提示注入等板块；修正过时内容（子代理名单 reviewer→reader、废弃的 dangerous 标记
  表述、memory.md 仓库私货）；子代理委派四要素（目标/返回契约/信息指引/边界）写入提示词
- **soul 语气定位**：跟随用户语言（不限中文）；对话交流自然有温度、可提供情绪价值，
  执行任务简洁利落——区分两种场景而非一刀切。后续可扩展为多风格 soul 变体切换
- **环境变量机制为可选项**（缓存友好）：`_default_prompt_variables()` 提供
  `{{current_date}}/{{platform}}/{{cwd}}` 供**自定义模板**按需引用（动态值会使提示词
  前缀缓存失效，谁引用谁承担）；**内置模板不引用**，时效判断走 datetime_tool 按需获取

## [Unreleased] — web_fetch 网页正文抓取工具（2026-06-04）

### 新增

- **内置工具 `web_fetch`**（网页 → Markdown 正文提取）：自动剔除脚本/样式/导航等噪声标签，
  优先提取 article/main 正文容器，实测可节省 20-30% token。对标 MCP 官方 Fetch server 与
  Claude Code 的 WebFetch。与 `http_request` 分工：读网页用 web_fetch，调 API 用 http_request。
  支持 max_chars 截断（标注全文长度）、JSON 美化、转换失败降级原文。
- 新依赖 `markdownify`（含 beautifulsoup4，轻量纯 Python）。

### 变更

- 内置子代理换装：`researcher` 工具集 → web_search + **web_fetch** + datetime_tool；
  `reader` → file_read + **web_fetch**（原 http_request 移出，页面阅读场景由 web_fetch 接管）。
- researcher/reader 角色提示词同步更新。

## [Unreleased] — 全配默认下沉进 Agent（2026-06-04）

「开箱即用」策略从 AgentPool 默认工厂下沉到 `Agent.__init__` 本身——**直接构造 Agent
与经池构造拿到同规格实例**，默认策略只有一份。Pool 仍只用于多用户服务场景；
单用户/脚本/嵌入场景直接 `Agent(llm)` 即全配，两个入口都是一等公民。

### 变更（含行为变更）

- **`Agent(tools=None)`（顶层）→ 默认注入全套 `BUILTIN_TOOLS`**；显式 `[]` = 无工具，显式列表 = 该列表。
- **新增 `Agent(subagents=...)` 参数**：`None`（顶层）→ 默认注册内置子代理三件套；`[]` = 关闭；`list[SubAgentConfig]` = 自定义。子代理 `register_catalog=False` 不触发默认 → 结构上保证不嵌套。
- AgentPool 默认工厂随之**简化**：不再自带 tools/subagents 默认逻辑，仅注入 session_id 派生与共享 MCP；`agent_kwargs` 的 `"subagents"` 从特殊键转为正式 Agent 参数（外部用法不变）。
- `examples/multi_turn_chat.py` 的 `build_agent()` 由 ~100 行手工组装简化为零配置构造；`examples/multi_user_chat.py` 同步精简。

## [Unreleased] — 内置子代理三件套 + 确认回调透传（2026-06-04）

### 新增

- **内置子代理预设 `builtin_subagent_configs()`**（按「上下文隔离/权限收窄/可并行」选型）：
  - `researcher` 调研员：web_search + http_request + datetime_tool（只读），配 `deep-research` 内置技能；
  - `reader` 长内容阅读员（**新增角色**，含内置提示词 `templates/prompts/reader/`）：file_read + http_request（只读），长文档/日志/网页定向提取；
  - `coder` 编码执行员：python_repl + file_read + file_write；
  - 可选 `reviewer` 审查员：file_read + python_repl，配 `code-review` 技能（`include` 加入，不在默认集）。
- **确认回调透传**：`Agent.run()` 经 ContextVar `_current_parent_confirm` 把 `on_confirm` 注入子代理——AUTO 模式下子代理内的不安全工具（file_write 等）同样走父 Agent 的人工确认，**委派不再是安全审批的旁路**。
- **AgentPool 默认工厂自动注册三件套子代理**；`agent_kwargs={"subagents": []}` 关闭、传 `list[SubAgentConfig]` 自定义（特殊键，非 Agent 参数）。
- 新增内置技能 `deep-research`；`PROMPT_ROLES` 加入 `reader`。

### 变更

- `researcher`/`coder` 角色提示词补充**返回契约**（最终回复即交付物：只回传蒸馏结论/产物路径，不回传过程）；researcher 流程修正为使用 `web_search`（原文误写为 http_request）。
- `examples/multi_user_chat.py` 改为消费 `builtin_subagent_configs()`，子代理工具创建一次、整池共享（无状态化后 `get_parent_mode` 闭包限制已不存在）。

## [Unreleased] — 默认极简化 / 能力参数归位（2026-06-03）

围绕「C 端开箱即用 + 技术人员可覆盖」的规范化整理。**含破坏性 API 变更**，迁移见下。

### 新增

- **开箱即用默认**：`Agent(llm)` 即可用——顶层 Agent 默认套用内置 `main` 角色提示词 + 内置技能（无需再手写 `prompt_dir` / `skills_dir`）。
- **用户级数据目录**：`resources.user_data_dir()` / `default_session_dir()` / `default_mcp_config_path()`，默认锚定 `~/.agent_framework`，可用环境变量 `AGENT_FRAMEWORK_HOME` 覆盖（与 CWD 解耦，适合部署）。
- **todo 计划内存后端**：无 session 时 todo 计划落内存（run 级注入、同一 Agent 跨轮保留），`session_enabled=False`（含子代理、用户自管 history）时 todo 不再抛 `RuntimeError`。
- **`AgentPool.from_llm(llm)`**：单 LLM 实例一行起池；默认工厂给每个 per-user Agent 配齐 `BUILTIN_TOOLS` + 内置提示词/技能。
- **`AgentPool(..., agent_kwargs={...})`**：向默认工厂透传 Agent 能力参数（`mode` / `session_enabled` / `session_dir` / `tools` / ...）。
- **`SubAgentConfig.role`**：传 `main`/`coder`/`researcher`/`reviewer` 自动套用内置角色提示词。
- **子代理模式继承改用 ContextVar**（`_current_parent_mode`）：不再需要 `create_subagent_tools(..., get_parent_mode=...)` 回调，子代理工具可在 Agent 之前创建（`get_parent_mode` 仍保留为可选显式覆盖）。

### 破坏性变更

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

2. **默认会话目录从 CWD `./.sessions` 改为 `~/.agent_framework/sessions`**（与 CWD 解耦）。需要旧行为者显式传 `session_dir="./.sessions"`。

3. **裸 `Agent(llm)` 的默认 system prompt 行为变更**：以前为空，现默认加载内置 `main` 角色提示词 + 内置技能。需要「空提示词」者显式传 `prompt_dir=None`（注意：传 `system_prompt` 时只用 `system_prompt`，不叠加内置）。

4. **移除 CWD 相对的技能自动扫描**（旧 `_SKILL_SEARCH_PATHS` 扫描宿主 CWD 的 `skills/`）。改为默认内置技能目录；自定义技能显式传 `skills_dir=`。

### 内部 / 设计

- `Agent.set_mode()` 改写实例字段 `self._mode`，不再原地修改共享 `AgentConfig`——从根上消除多用户 mode 跨用户串扰（越权）隐患。
- `prompt_dir` / `skills_dir` 的「`None` → 内置默认」仅对顶层 Agent 生效（以 `register_catalog` 区分），子代理保持精简不变量。
- MCP 配置搜索改为运行期解析用户级路径（尊重 `AGENT_FRAMEWORK_HOME`），保留 `./config/mcp_servers.json` 项目级兜底。
- 测试 `conftest.py` autouse fixture 重定向 `AGENT_FRAMEWORK_HOME` 到 `tmp_path`，保持单测 hermetic。
- 已验证构建 wheel 包含 `templates/`（内置提示词/技能），pip 安装后开箱可用。
