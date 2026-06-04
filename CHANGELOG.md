# 变更日志

本项目重要变更记录。日期格式 YYYY-MM-DD。

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
