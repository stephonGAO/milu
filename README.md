# milu

统一的 AI 模型抽象层 + Agent 编排引擎。一套接口接入 9 个 LLM 提供商（通义千问、Kimi、GLM、DeepSeek、MiniMax、豆包、ChatGPT、Gemini、Claude），内置工具系统（含 MCP 协议）、子 Agent、技能、文件化系统提示词、会话持久化、上下文自动压缩，以及**多用户并发资源池**。

- Python 3.10+ · async-first · 以 `openai` SDK 作为统一 HTTP 客户端
- 设计为「核心库」：其它项目 `pip install` 后即可快速获得 Agent 能力，并可部署为多用户并发服务

## 安装

```bash
# 开发模式（含测试与 MCP 依赖）
pip install -e ".[dev,mcp]"
```

配置 API Key：在项目根目录放一个 `.env`，按 `{PROVIDER}_API_KEY` 命名：

```dotenv
QWEN_API_KEY=sk-xxx
DEEPSEEK_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-xxx
```

> 作为库被集成时，可设 `MILU_NO_DOTENV=1` 关闭对 .env 的自动加载，改由宿主应用通过 `milu.load_env(path)` 或自身机制管理环境变量。

## 命令行（CLI）

`pip install` 后注册命令 `milu`，无需写代码即可使用：

```bash
milu                      # 无子命令 → 进入交互式多轮对话
milu chat -p deepseek     # 指定厂商进入对话（默认厂商 qwen）
milu run "用一句话介绍你自己"   # 一次性执行
milu run "总结" -q          # -q 只输出最终回答（便于管道）
echo "翻译成英文：你好" | milu run   # 从 stdin 读取指令
milu providers            # 列出 9 个厂商及 API Key 配置状态
milu config set provider deepseek   # 持久化默认厂商
milu config set-key qwen sk-xxx     # 保存某厂商的 Key 到配置文件
milu sessions list        # 查看历史会话
```

- **配置**：`~/.milu/config.json`（`config` 子命令管理）。解析优先级 **CLI 参数 > 环境变量 `{PROVIDER}_API_KEY` > 配置文件 > 内置默认**。
- **能力**：交互式 `chat` 支持流式输出、工具调用可视化、内置子代理（researcher/reader/coder）、会话持久化、`/mode` 切换操作模式、MCP 自动接入，以及 `/help` 列出的全部 `/命令`。
- 全局选项写在子命令之后：`-p/--provider`、`-m/--model`、`--api-key`、`--mode {talk,manual,auto,superwork}`、`--no-session`、`--no-mcp`、`--no-subagents`。

## 快速开始

### 1. 直接调用 LLM（流式）

```python
import asyncio
from milu import ModelRegistry, Message, MessageRole

async def main():
    llm = ModelRegistry.create("qwen", model="qwen-plus")
    async for chunk in llm.chat([Message(role=MessageRole.USER, content="你好")]):
        if chunk.content:
            print(chunk.content, end="", flush=True)

asyncio.run(main())
```

### 2. 跑一个带工具的 Agent

```python
import asyncio
from milu import Agent, AgentConfig, ModelRegistry, AgentDone
from milu.tools.builtin import BUILTIN_TOOLS

async def main():
    llm = ModelRegistry.create("qwen", model="qwen-plus")
    agent = Agent(llm=llm, tools=list(BUILTIN_TOOLS),
                  config=AgentConfig(session_enabled=False))
    async for evt in agent.run("现在几点？用工具查一下"):
        if hasattr(evt, "text") and evt.text:
            print(evt.text, end="", flush=True)
        if isinstance(evt, AgentDone):
            print(f"\n[turns={evt.turn_count}]")

asyncio.run(main())
```

### 3. 自定义工具

```python
from milu import tool

@tool(name="add", description="两数相加", is_safe=True)
async def add(a: int, b: int) -> int:
    """:param a: 加数\n:param b: 被加数"""
    return a + b

agent = Agent(llm=llm, tools=[add])
```

### 4. 多用户并发（AgentPool）

按 `(user_id, session_id)` 隔离 Agent 实例，自带 LRU/TTL 淘汰、全局并发限流、同会话串行：

```python
from milu import AgentPool, AgentPoolConfig, AgentConfig, ModelRegistry

llm = ModelRegistry.create("qwen", model="qwen-plus")  # LLM 可安全共享
pool = AgentPool(
    llm_factory=lambda user_id, session_id: llm,
    agent_config=AgentConfig(session_enabled=True),     # 历史按 (user,session) 自动持久化/恢复
    config=AgentPoolConfig(max_agents=200, max_concurrent_runs=50, idle_ttl_seconds=300),
)
await pool.start()

async with pool.acquire("user-1", "session-A") as h:
    async for evt in h.agent.run("你好"):
        ...

await pool.stop()
```

完整的 FastAPI + Web UI 示例见 `examples/multi_user_chat.py`。

### 5. 内置角色提示词与技能

预置提示词（main/coder/researcher/reviewer）与技能随包分发，pip 安装后即可用：

```python
from milu import Agent, builtin_prompts_dir, builtin_skills_dir

agent = Agent(
    llm=llm,
    prompt_dir=builtin_prompts_dir("main"),
    skills_dir=str(builtin_skills_dir()),
)
```

## 部署建议（多 worker / 高并发）

- **粘性路由**：按 `user_id` 做一致性哈希（如 nginx `ip_hash`），让同一用户恒定落到同一 worker。同会话由进程内 `AgentPool` 的 entry 锁串行，**无需分布式锁**。
- **会话持久化**：`AgentConfig(session_enabled=True)` 时历史按 `(user, session)` 落盘，淘汰/重启后自动恢复；多 worker 故障转移可叠加共享存储（NFS/EFS）。
- **MCP 是内存大头**：默认每个 Agent 自带 MCP 子进程（约 15–50MB）。高并发下建议开启**共享 MCP**——整池只连一组 MCP 进程、由所有 Agent 复用，内存不再随用户数增长：
  - 默认 factory：`AgentPoolConfig(shared_mcp=True, mcp_config_path=...)`；
  - 自定义 factory：自建一个已连接的 `MCPManager`，透传 `Agent(mcp_manager=...)`（见 `examples/multi_user_chat.py`）。
  - ⚠️ 共享 = MCP server 端看不到 user_id、无法按用户隔离，**仅适合无状态 MCP 工具**（抓取/搜索/查询）；有服务端「每用户状态」的 MCP 仍应用默认的 per-agent 模式。
- **多租户 API Key 隔离**：不同租户/用户需用各自的 API Key 时，用内置的 `KeyedLLMProvider` 作为 `llm_factory`——它按 Key 缓存 LLM 实例：同 Key 复用同一连接池、不同 Key 隔离，总数受 `max_clients` 约束（超出 LRU 淘汰并关闭连接池）：

  ```python
  from milu import AgentPool, AgentPoolConfig, KeyedLLMProvider

  provider = KeyedLLMProvider(
      "qwen", model="qwen-plus",
      resolve_api_key=lambda user_id, session_id: lookup_tenant_key(user_id),  # 你的 租户→Key 映射
      max_clients=256,
  )
  pool = AgentPool(llm_factory=provider, config=AgentPoolConfig(max_agents=200))
  await pool.start()
  # ... 服务请求 ...
  await pool.stop(); await provider.aclose()   # aclose 关闭所有缓存的连接池
  ```

  适用于「Key 映射到租户/组织」（distinct Key 数远小于用户数）的常见形态；`resolve_api_key` 返回 `None` 则回退到默认 Key/环境变量。**Key 切勿明文硬编码在源码中**——应从环境变量/密钥管理服务（Vault、云 KMS）读取后在 `resolve_api_key` 中返回。完整可运行示例见 `examples/multi_tenant_keys.py`。
- **服务端确认**：高并发下避免交互式逐工具确认（会拖累吞吐）；如需确认，`AgentPool` 已保证等待确认期间不占用全局并发名额。

## 示例

`examples/` 下按编号递进：`1. basic_llm` → `2. agent_basic` → `3. builtin_tools` → `5_mcp_tools` → `6_subagent` → `7_server_fastapi` → `multi_turn_chat` / `multi_user_chat` / `multi_tenant_keys`。

```bash
.venv/Scripts/python "examples/1. basic_llm.py"
.venv/Scripts/python examples/multi_user_chat.py    # 多用户并发 + Web UI
```

## 测试

```bash
# 单元测试（跳过真实 API 调用）
.venv/Scripts/python -m pytest tests/ --ignore=tests/test_real_api.py --ignore=tests/test_real_new_providers.py -q
```

## 进一步阅读

- 架构与开发约定：[`CLAUDE.md`](CLAUDE.md)
- 多用户高并发能力评估与加固记录：[`docs/并发能力评估报告.md`](docs/并发能力评估报告.md)
