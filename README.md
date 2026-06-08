<div align="center">

# 🦌 milu

**Production-ready multi-user AI agents — with Chinese LLMs as first-class citizens.**

One unified interface for 9 LLM providers · Built-in tools & MCP · Sub-agents · Skills · RAG · Scheduler · Multi-user agent pool

[![PyPI](https://img.shields.io/pypi/v/milu)](https://pypi.org/project/milu/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/milu/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1100%2B%20passed-brightgreen)](tests/)

**English** | [简体中文](README.zh-CN.md)

<!-- TODO(launch): demo GIF here — `milu serve` web UI or CLI session, ≤30s -->

</div>

---

## Why milu?

Most agent frameworks treat Chinese LLM providers as an afterthought, and stop at single-user demos. milu starts where they stop:

- 🇨🇳 **Chinese LLMs as first-class citizens** — Qwen, DeepSeek, Kimi, GLM, MiniMax, Doubao natively supported alongside OpenAI, Gemini and Claude. No `base_url` juggling, provider quirks (thinking mode, built-in web search, parameter differences) pre-adapted, plus a China-reachable search backend out of the box.
- 🏭 **From demo to production in one library** — `AgentPool` gives you per-user agent isolation, LRU/TTL eviction, global concurrency limits and shared MCP processes. The question every framework leaves as "an exercise for the reader" — *"my demo works, how do I serve 100 concurrent users without sessions bleeding into each other?"* — is answered here, backed by 1100+ tests.
- 🔋 **Batteries actually included** — 20+ built-in tools (files, shell, Python, web fetch/search, Office/PDF reading, vision input), MCP protocol (stdio/HTTP/SSE), sub-agents, skills, session persistence, automatic context compaction, long-term memory, RAG knowledge base, scheduled tasks, and a built-in multi-user web service.
- 🛡️ **A real safety model** — four operation modes (`talk` / `manual` / `auto` / `superwork`), an AI safety judge for unsafe tool calls (Claude-Code-style), human confirmation flows, and delegation that never bypasses approval.
- 🪶 **Thin by design** — built directly on the `openai` SDK as the unified HTTP client. Events stream out as plain dataclasses. No chains, no graphs, no DSL to learn.

## Quick start

```bash
pip install milu              # everything included: CLI, web service, RAG, MCP
```

**CLI** — zero config to first conversation:

```bash
milu                # first run guides you through provider + API key setup
```

**Code** — a full-featured agent in 3 lines:

```python
from milu import Agent, ModelRegistry

agent = Agent(ModelRegistry.create("deepseek", model="deepseek-v4-flash"))
async for event in agent.run("What time is it? Use a tool to check."):
    ...
```

`Agent(llm)` is the complete package by default: built-in system prompt, 20+ tools, skills, three sub-agents, session persistence and context compaction — pass explicit arguments only to override.

**Multi-user web service** — one command:

```bash
milu serve          # multi-user chat + full-featured demo UI at http://127.0.0.1:8000
```

**Docker**:

```bash
cp .env.example .env          # fill in at least one provider API key
docker compose up -d
```

## How it compares

| | milu | LangChain | CrewAI | smolagents | Qwen-Agent |
|---|---|---|---|---|---|
| Chinese providers native (6) | ✅ | community pkgs | via LiteLLM | via LiteLLM | Qwen family |
| Multi-user pool, in-library | ✅ AgentPool | platform (paid) | platform | ❌ | ❌ |
| MCP protocol | ✅ 3 transports | ✅ | ✅ | ✅ | ✅ |
| Built-in tools (files/docs/vision/search) | ✅ 20+ | install per-integration | partial | minimal | partial |
| Tool-safety modes + AI judge | ✅ | ❌ | ❌ | sandbox only | ❌ |
| RAG knowledge base, in-library | ✅ | assemble yourself | partial | ❌ | ✅ |
| Scheduled tasks (multi-user) | ✅ | ❌ | ❌ | ❌ | ❌ |
| CLI + web service out of the box | ✅ | ❌ | partial | ❌ | demo UI |

> Each ✅/❌ reflects what ships *inside the library* as of June 2026 — most gaps can be filled with external platforms or custom code.

## What you can build

- **Customer-support bot over your docs** — RAG knowledge base with auto-retrieval, source-aware answers, multi-user isolation per visitor.
- **An internal multi-user assistant for your team** — `docker compose up -d`, everyone gets isolated sessions, memory and knowledge bases under one roof.
- **A morning-report agent** — scheduled tasks with cron expressions; results land in the web UI, an outbox file, or desktop notifications.
- **Multi-tenant SaaS agents** — `KeyedLLMProvider` maps tenants to their own API keys; the pool enforces per-user instance and concurrency invariants.

## Examples

<details>
<summary><b>1 · Call any LLM directly (streaming)</b></summary>

```python
import asyncio
from milu import ModelRegistry, Message, MessageRole

async def main():
    llm = ModelRegistry.create("qwen", model="qwen3.6-plus")
    async for chunk in llm.chat([Message(role=MessageRole.USER, content="Hello!")]):
        if chunk.content:
            print(chunk.content, end="", flush=True)

asyncio.run(main())
```

Swap `"qwen"` for `"deepseek"`, `"kimi"`, `"glm"`, `"minimax"`, `"doubao"`, `"openai"`, `"gemini"` or `"anthropic"` — same interface, API keys read from `{PROVIDER}_API_KEY` environment variables.
</details>

<details>
<summary><b>2 · Agent with tools and events</b></summary>

```python
import asyncio
from milu import Agent, ModelRegistry, AgentDone, TextDelta

async def main():
    agent = Agent(ModelRegistry.create("deepseek", model="deepseek-v4-flash"))
    async for evt in agent.run("Summarize the contents of ./report.pdf"):
        if isinstance(evt, TextDelta):
            print(evt.text, end="", flush=True)
        elif isinstance(evt, AgentDone):
            print(f"\n[done in {evt.turn_count} turns]")

asyncio.run(main())
```

The agent streams typed events — text deltas, reasoning, tool calls, confirmations, sub-agent progress — consume what you need, ignore the rest.
</details>

<details>
<summary><b>3 · Custom tools</b></summary>

```python
from milu import Agent, tool

@tool(name="add", description="Add two numbers", is_safe=True)
async def add(a: int, b: int) -> int:
    """:param a: first number\n:param b: second number"""
    return a + b

agent = Agent(llm, tools=[add])        # explicit list replaces built-ins
```

`is_safe=False` routes the call through the active safety mode: auto-judged by AI, confirmed by a human, or blocked — depending on the mode.
</details>

<details>
<summary><b>4 · Safety modes</b></summary>

```python
agent = Agent(llm, mode="manual")   # unsafe tools wait for human approval
agent.set_mode("talk")              # read-only: unsafe tools blocked
```

| Mode | Behavior |
|---|---|
| `talk` | read-only — every unsafe tool call is blocked |
| `manual` | safe tools run; unsafe tools emit a confirmation event and wait |
| `auto` (default) | autonomous; unsafe calls are screened by an **AI safety judge** (allow / confirm / deny) |
| `superwork` | full permissions, no checks |

Sub-agents inherit the parent's mode and confirmation callback — delegation is never a bypass.
</details>

<details>
<summary><b>5 · Long-term memory & RAG knowledge base</b></summary>

```python
agent = Agent(llm, memory="user-42", knowledge="user-42")
```

- **Memory**: small set of durable facts, rendered into the system prompt every turn, survives across sessions and processes.
- **Knowledge**: chunked + embedded documents (pdf/docx/xlsx/pptx/md/txt) with cosine retrieval, source-catalog routing in the prompt, optional per-turn auto-retrieval, and `kb_search` / `kb_ingest` / `kb_manage` tools. Per-user isolated storage.
</details>

<details>
<summary><b>6 · Multi-user concurrency (AgentPool)</b></summary>

```python
from milu import AgentPool, ModelRegistry

llm = ModelRegistry.create("qwen", model="qwen3.6-plus")   # coroutine-safe, shareable
pool = AgentPool.from_llm(llm)
await pool.start()

async with pool.acquire("user-1", "session-A") as h:
    async for evt in h.agent.run("Hello!"):
        ...

await pool.stop()
```

Four hard invariants: ≤1 agent per `(user, session)` · bounded instance count · bounded concurrent runs · idle agents evicted. Sessions, memory and knowledge are derived per-user automatically.
</details>

<details>
<summary><b>7 · MCP servers</b></summary>

```jsonc
// config/mcp_servers.json
{
  "mcpServers": {
    "playwright": { "command": "npx", "args": ["@playwright/mcp@latest"] },
    "my-http":    { "type": "streamable_http", "url": "http://localhost:3000/mcp" }
  }
}
```

stdio / streamable HTTP / SSE transports, parallel connection with error isolation, and a dormant-pool design: MCP tool schemas don't bloat the context — the agent discovers and activates them on demand. For high-concurrency deployments, one shared set of MCP processes can serve the entire pool.
</details>

<details>
<summary><b>8 · Scheduled tasks</b></summary>

```bash
milu chat
> Remind me every weekday at 9am to summarize yesterday's AI news   # agent creates the task
```

Cron-style scheduling per user, executed inside `milu chat` / `milu serve` (or a standalone `milu scheduler start` daemon) with a single-instance lock and automatic takeover. Results are delivered to an outbox file, server push, or desktop notification.
</details>

## CLI

```text
milu                 interactive chat (first run launches setup wizard)
milu setup           provider / API key / search backend wizard
milu chat -p glm     chat with a specific provider
milu run "..." -q    one-shot execution, pipe-friendly
milu serve           multi-user web service + demo UI
milu providers       list 9 providers and key status
milu config ...      layered config (CLI > user > project > defaults)
milu sessions list   browse saved sessions
milu schedule ...    manage scheduled tasks
```

## Architecture

```text
AgentPool (multi-user, optional)
  └─ Agent.run() loop ── system prompt rebuild → auto-compaction
       ├─ LLM layer        9 providers, one AsyncOpenAI-based interface
       ├─ Tool layer       built-ins · custom @tool · MCP (active/dormant pools)
       ├─ Safety layer     modes · AI judge · confirmation flow
       ├─ Sub-agents       researcher / reader / coder (isolated context)
       ├─ Prompts & skills layered markdown prompts · on-demand skill loading
       └─ Session          JSONL persistence · compaction snapshots
```

Python 3.10+ · fully async · every provider speaks through one `openai.AsyncOpenAI` client, so LLM instances are coroutine-safe and shareable across users.

## Production notes

- **Scaling out**: route by `user_id` (e.g. nginx `ip_hash`); per-session serialization is handled by in-process entry locks — no distributed locks needed. Sessions persist to disk and recover after eviction or restart.
- **Memory budget**: MCP subprocesses are the dominant cost (15–50 MB per agent). Enable shared MCP (`AgentPoolConfig(shared_mcp=True)`) to keep one set of MCP processes for the whole pool.
- **Multi-tenant keys**: `KeyedLLMProvider` caches one LLM client per distinct API key with LRU eviction — see `examples/multi_tenant_keys.py`.
- **Docker**: see [docs/Docker部署.md](docs/Docker部署.md) — health checks, data volumes, SSE reverse-proxy settings, scheduler single-instance behavior.

## Roadmap

- [ ] Observability: OpenTelemetry tracing hooks
- [ ] Sandboxed code execution backends for `python_repl` / `shell_command`
- [ ] Pluggable ANN backends for the knowledge store (sqlite-vec) beyond brute-force cosine
- [ ] English documentation set (architecture & guides — currently Chinese)
- [ ] Prebuilt images on a container registry

## Contributing

Issues and PRs welcome. Run the test suite with:

```bash
pip install -e ".[dev]"
python -m pytest tests/ --ignore=tests/test_real_api.py --ignore=tests/test_real_new_providers.py -q
```

## License

[MIT](LICENSE). Five built-in skills are ported from [anthropics/skills](https://github.com/anthropics/skills) (Apache-2.0) and [obra/superpowers](https://github.com/obra/superpowers) (MIT) — see [THIRD_PARTY_NOTICES](src/milu/templates/skills/THIRD_PARTY_NOTICES.txt).

---

<div align="center">

**milu (米鹿 / 麋鹿)** — named after Père David's deer, the legendary Chinese animal that "resembles four creatures yet is none of them" — one body, the strengths of many.

</div>
