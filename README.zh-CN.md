<div align="center">

# 🦌 milu（米鹿）

**生产级多用户 Agent 框架 —— 国产大模型一等公民。**

多用户并发池 · 一套接口接入 9 家大模型（国产一等公民） · 内置工具与 MCP · 子代理 · 技能 · RAG 知识库 · 定时任务

[![PyPI](https://img.shields.io/pypi/v/milu)](https://pypi.org/project/milu/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/milu/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1100%2B%20passed-brightgreen)](tests/)
[![GitHub stars](https://img.shields.io/github/stars/stephonGAO/milu?style=flat&logo=github&label=stars)](https://github.com/stephonGAO/milu/stargazers)

[English](README.md) | **简体中文**

<!-- TODO(launch): 演示 GIF —— milu serve 前端或 CLI 对话，≤30 秒 -->

</div>

---

## 为什么是 milu？

多数 agent 框架止步于单用户 demo，又把国产模型当二等公民。milu 从它们止步的地方开始：

- 🏭 **一个库走完 demo 到生产** —— `AgentPool` 提供 per-user 实例隔离、LRU/TTL 淘汰、全局并发限流、共享 MCP 进程。每个框架都留给你自己解决的问题——*「demo 跑通了，怎么让 100 个用户同时用且会话互不串扰？」*——这里有现成答案，1100+ 测试兜底。同一个池还能把租户映射到各自的 API Key（`KeyedLLMProvider`），从个人项目平滑长到多租户 SaaS。
- 🇨🇳 **国产模型一等公民** —— 通义千问、DeepSeek、Kimi、GLM、MiniMax、豆包原生支持，与 OpenAI、Gemini、Claude 同级。不用自己维护 base_url 路由表，各家的思考模式、自带联网搜索、参数差异已经适配好，搜索工具自带国内可直连的后端（博查）。
- 🔋 **真·开箱即用** —— 20+ 内置工具（文件、Shell、Python、网页抓取/搜索、Office/PDF 文档读取、图片视觉输入）、MCP 协议（stdio/HTTP/SSE）、子代理、技能、会话持久化、上下文自动压缩、长期记忆、RAG 知识库、定时任务，外加一个内置的多用户 Web 服务。
- 🛡️ **真正的安全模型** —— 四种操作模式（`talk` / `manual` / `auto` / `superwork`）、不安全工具调用的 AI 安全判定器（对齐 Claude Code）、人工确认流，且子代理委派永不旁路审批。
- 🪶 **薄封装，不造概念** —— 直接以 `openai` SDK 为统一 HTTP 客户端，事件以普通 dataclass 流式输出。没有链、没有图、没有需要学习的 DSL。

## 两种用法

milu 既是开箱即用的完整 Agent，也是可二次开发的框架——即开即用，需要时再嵌入：

- 🚀 **直接用**：`milu` 进入对话、`milu serve` 起多用户服务，零代码即得完整能力。
- 🧩 **嵌进你的产品**：`from milu import Agent`，把 Agent 能力嵌入自有后端，再用 AgentPool 扩成多用户/多租户服务——**数据与技术栈完全归你**。

## 安装

**最简单 —— 无需预装 Python。** 用 [uv](https://github.com/astral-sh/uv)，它会顺带把 Python 装好：

```bash
# 1. 安装 uv（一行命令，不需要 Python）
curl -LsSf https://astral.sh/uv/install.sh | sh            # macOS / Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex" # Windows

# 2. 安装 milu（缺 Python 时 uv 会自动下载一个）
uv tool install milu
```

**用 pip** —— 如果你已经有 Python 3.10+：

```bash
pip install milu              # 全功能：CLI、Web 服务、RAG 知识库、MCP 全含
```

**Docker** —— 宿主机完全不用装 Python：

```bash
cp .env.example .env          # 填入至少一家厂商的 API Key
docker compose up -d
```

<details>
<summary><b>没装过 Python？小白分步指引</b></summary>

1. 到 [python.org/downloads](https://www.python.org/downloads/) 下载 Python 3.10+。Windows 安装时**务必勾选 “Add Python to PATH”**。
2. 打开终端（Windows 用 PowerShell · macOS 用 终端）验证：`python --version` 应显示 3.10 或更高。
3. 执行 `pip install milu`
4. 输入 `milu` 开始对话。
</details>

## 快速开始

**命令行** —— 零配置到第一次对话：

```bash
milu                # 首次运行自动进入引导：选厂商 → 填 API Key → 开聊
```

**代码** —— 3 行得到一个全功能 Agent：

```python
from milu import Agent, ModelRegistry

agent = Agent(ModelRegistry.create("deepseek", model="deepseek-v4-flash"))
async for event in agent.run("现在几点了？用工具查一下"):
    ...
```

`Agent(llm)` 默认即完整体：内置系统提示词、20+ 工具、技能、三个子代理、会话持久化与上下文压缩——只在需要覆盖时才传参数。

**多用户 Web 服务** —— 一条命令：

```bash
milu serve          # 多用户对话 + 全功能演示前端，http://127.0.0.1:8000
```

## 横向对比

| | milu | LangChain | CrewAI | smolagents | Qwen-Agent |
|---|---|---|---|---|---|
| 国产模型原生支持（6 家） | ✅ | 社区包零散 | 经 LiteLLM | 经 LiteLLM | 仅 Qwen 系 |
| 多用户并发池（库内置） | ✅ AgentPool | 平台（收费） | 平台 | — | — |
| MCP 协议 | ✅ 三种传输 | ✅ | ✅ | ✅ | ✅ |
| 内置工具（文件/文档/视觉/搜索） | ✅ 20+ | 按集成单装 | 部分 | 少量 | 部分 |
| 工具安全模式 + AI 判定 | ✅ | — | — | 仅沙箱 | — |
| RAG 知识库（库内置） | ✅ | 自行组装 | 部分 | — | ✅ |
| 定时任务（多用户） | ✅ | — | — | — | — |
| CLI + Web 服务开箱即用 | ✅ | — | 部分 | — | 演示 UI |

> ✅ = 库内直接提供；「—」= 未内置（通常可经外部平台或少量自研代码补齐）。数据截至 2026 年 6 月，竞品迭代很快，欢迎经 issue/PR 指正。
>
> **何时选 milu：** 你要基于国产大模型构建、需要的是生产级多用户 / 多租户服务（而不止单用户 demo）、且想要开箱即用——既能直接跑，也能当库嵌入。或作为一个纯净灵活的开发核心与智能底座。
>
> **何时该选别的框架：** 要最大的第三方生态 / 集成，选 [LangChain](https://github.com/langchain-ai/langchain)；纯多 agent 编排，选 [CrewAI](https://github.com/crewAIInc/crewAI) / [AutoGen](https://github.com/microsoft/autogen)；要极简代码优先的 agent，选 [smolagents](https://github.com/huggingface/smolagents)。

## 你可以用它做什么

- **个人 AI 助理** —— `milu` 一条命令进入对话；长期记忆记住你的偏好与习惯、定时任务帮你提醒与发每日简报、内置联网搜索 / 文件 / 文档 / 视觉等工具随手可用——全程本地运行，数据归你自己。
- **企业知识库助手** —— 把手册 / FAQ / 规章制度灌入 RAG 知识库，每轮自动检索、回答区分「内部资料 vs 网络来源」，杜绝大模型张口就来的幻觉；每个员工独立会话与记忆，互不干扰。
- **智能客服 / 工单机器人** —— 高频重复咨询、工单分类与初步处理，AgentPool 扛高并发多用户，安全模式管住可执行动作。
- **行业垂直助手（工业 / 医疗 / 法务等）** —— 子代理分工 + 文档 / 图片视觉读取 + MCP 接入行业系统与数据库，把行业知识和真实数据接进来。
- **团队「AI 同事」** —— 从对话里抓任务、按节奏催进度、定时生成复盘小结（定时任务 + 多用户 + 工具组合），把人机协同变成团队每天能感知的一两件事。
- **私有化部署的内部助手** —— `docker compose up -d` 起多用户服务，国产模型直连、数据全部落本地，不依赖任何外部平台。
- **多租户 SaaS / 给应用服务商做底座** —— `KeyedLLMProvider` 按租户映射各自 API Key，资源池保证实例与并发隔离，从个人项目平滑长到多租户产品。

## 渐进式示例

<details>
<summary><b>1 · 直接调用 LLM（流式）</b></summary>

```python
import asyncio
from milu import ModelRegistry, Message, MessageRole

async def main():
    llm = ModelRegistry.create("qwen", model="qwen3.6-plus")
    async for chunk in llm.chat([Message(role=MessageRole.USER, content="你好")]):
        if chunk.content:
            print(chunk.content, end="", flush=True)

asyncio.run(main())
```

把 `"qwen"` 换成 `"deepseek"`、`"kimi"`、`"glm"`、`"minimax"`、`"doubao"`、`"openai"`、`"gemini"`、`"anthropic"` ——同一套接口，API Key 从环境变量 `{PROVIDER}_API_KEY` 读取。
</details>

<details>
<summary><b>2 · 带工具的 Agent 与事件流</b></summary>

```python
import asyncio
from milu import Agent, ModelRegistry, AgentDone, TextDelta

async def main():
    agent = Agent(ModelRegistry.create("deepseek", model="deepseek-v4-flash"))
    async for evt in agent.run("总结一下 ./report.pdf 的内容"):
        if isinstance(evt, TextDelta):
            print(evt.text, end="", flush=True)
        elif isinstance(evt, AgentDone):
            print(f"\n[共 {evt.turn_count} 轮]")

asyncio.run(main())
```

Agent 以类型化事件流式输出——正文增量、思考过程、工具调用、确认请求、子代理进度——按需消费，其余忽略。
</details>

<details>
<summary><b>3 · 自定义工具</b></summary>

```python
from milu import Agent, tool

@tool(name="add", description="两数相加", is_safe=True)
async def add(a: int, b: int) -> int:
    """:param a: 加数\n:param b: 被加数"""
    return a + b

agent = Agent(llm, tools=[add])        # 显式列表会替换内置工具
```

`is_safe=False` 的工具调用会进入当前安全模式的管控：AI 自动判定、人工确认或直接拦截——取决于模式。
</details>

<details>
<summary><b>4 · 安全模式</b></summary>

```python
agent = Agent(llm, mode="manual")   # 不安全工具等待人工审批
agent.set_mode("talk")              # 只读：不安全工具一律拦截
```

| 模式 | 行为 |
|---|---|
| `talk` | 只读——所有不安全工具调用被拦截 |
| `manual` | 安全工具直接执行；不安全工具产出确认事件并等待 |
| `auto`（默认） | 自主决策；不安全调用交 **AI 安全判定器**三态裁决（放行/转确认/拒绝） |
| `superwork` | 全权限，跳过所有检查 |

子代理继承父 Agent 的模式与确认回调——**委派不构成安全旁路**。
</details>

<details>
<summary><b>5 · 长期记忆与 RAG 知识库</b></summary>

```python
agent = Agent(llm, memory="user-42", knowledge="user-42")
```

- **记忆（memory）**：少量持久事实，每轮渲染进系统提示词，跨会话、跨进程留存。
- **知识库（knowledge）**：文档（pdf/docx/xlsx/pptx/md/txt）分块向量化 + 余弦检索，来源目录常驻提示词做检索路由，可选每轮前置自动检索，配 `kb_search` / `kb_ingest` / `kb_manage` 三工具。按用户隔离存储。
</details>

<details>
<summary><b>6 · 多用户并发（AgentPool）</b></summary>

```python
from milu import AgentPool, ModelRegistry

llm = ModelRegistry.create("qwen", model="qwen3.6-plus")   # 协程安全，可共享
pool = AgentPool.from_llm(llm)
await pool.start()

async with pool.acquire("user-1", "session-A") as h:
    async for evt in h.agent.run("你好"):
        ...

await pool.stop()
```

四个硬不变量：每 `(user, session)` 至多 1 个实例 · 实例总数有界 · 并发 run 有界 · 空闲实例自动淘汰。会话、记忆、知识库自动按用户派生隔离。
</details>

<details>
<summary><b>7 · MCP 服务器</b></summary>

```jsonc
// config/mcp_servers.json
{
  "mcpServers": {
    "playwright": { "command": "npx", "args": ["@playwright/mcp@latest"] },
    "my-http":    { "type": "streamable_http", "url": "http://localhost:3000/mcp" }
  }
}
```

支持 stdio / streamable HTTP / SSE 三种传输，并行连接、错误隔离。双池设计：MCP 工具 schema 不挤占上下文——Agent 按需发现并激活休眠工具。高并发部署可整池共享一组 MCP 进程。
</details>

<details>
<summary><b>8 · 定时任务</b></summary>

```bash
milu chat
> 每个工作日早上 9 点提醒我总结昨天的 AI 新闻    # Agent 自动创建定时任务
```

按用户隔离的 cron 任务，在 `milu chat` / `milu serve` 内嵌执行（或独立守护进程 `milu scheduler start`），单实例锁 + 自动接管。结果投递到 outbox 文件、服务端推送或桌面通知。
</details>

## 命令行

```text
milu                 交互式对话（首次运行自动进入初始化引导）
milu setup           厂商 / API Key / 搜索后端引导
milu chat -p glm     指定厂商对话
milu run "..." -q    一次性执行，可管道
milu serve           多用户 Web 服务 + 演示前端
milu providers       列出 9 家厂商与 Key 配置状态
milu config ...      分层配置（CLI 参数 > 用户级 > 项目级 > 内置默认）
milu sessions list   查看历史会话
milu schedule ...    定时任务管理
```

## 架构

```text
AgentPool（多用户，可选）
  └─ Agent.run() 循环 ── 重建系统提示词 → 自动压缩
       ├─ LLM 层        9 家厂商，一套 AsyncOpenAI 统一接口
       ├─ 工具层        内置工具 · 自定义 @tool · MCP（活跃/休眠双池）
       ├─ 安全层        操作模式 · AI 判定器 · 确认流
       ├─ 子代理        researcher / reader / coder（上下文隔离）
       ├─ 提示词与技能  分层 Markdown 提示词 · 技能按需加载
       └─ 会话          JSONL 持久化 · 压缩快照恢复
```

Python 3.10+ · 全链路 async · 所有厂商统一走 `openai.AsyncOpenAI` 客户端，LLM 实例协程安全、可跨用户共享。

## 生产部署要点

- **横向扩容**：按 `user_id` 粘性路由（如 nginx `ip_hash`）；同会话由进程内 entry 锁串行，**无需分布式锁**。会话落盘，淘汰/重启后自动恢复。
- **内存预算**：MCP 子进程是大头（每 Agent 15–50 MB）。开 `AgentPoolConfig(shared_mcp=True)` 让整池共享一组 MCP 进程。
- **多租户 Key 隔离**：`KeyedLLMProvider` 按 Key 缓存 LLM 客户端、LRU 淘汰——见 `examples/multi_tenant_keys.py`。
- **Docker**：见 [docs/Docker部署.md](docs/Docker部署.md) —— 健康检查、数据卷、SSE 反代配置、调度器单实例行为。

## 路线图

- [ ] 可观测性：OpenTelemetry tracing 钩子
- [ ] `python_repl` / `shell_command` 的沙箱执行后端
- [ ] 知识库可插拔 ANN 后端（sqlite-vec），超越暴力余弦
- [ ] 英文文档集（架构与指南，当前为中文）
- [ ] 容器镜像发布到镜像仓库
- [ ] 一键安装器 / 独立可执行文件（无需预装 Python）

## 贡献

欢迎 Issue 与 PR。运行测试：

```bash
pip install -e ".[dev]"
python -m pytest tests/ --ignore=tests/test_real_api.py --ignore=tests/test_real_new_providers.py -q
```

更多内部设计文档见 [`CLAUDE.md`](CLAUDE.md) 与 [`docs/`](docs/)（并发能力评估、压测报告、知识库评测等）。

## 许可

[MIT](LICENSE)。5 个内置技能移植自 [anthropics/skills](https://github.com/anthropics/skills)（Apache-2.0）与 [obra/superpowers](https://github.com/obra/superpowers)（MIT）——详见 [THIRD_PARTY_NOTICES](src/milu/templates/skills/THIRD_PARTY_NOTICES.txt)。

---

<div align="center">

**milu（米鹿/麋鹿）** —— 取自「四不像」：角似鹿非鹿、脸似马非马、蹄似牛非牛、尾似驴非驴——一身集多家之长。

</div>
