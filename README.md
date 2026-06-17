<div align="center">

# 🦌 milu

**Production-ready multi-user AI agents — with Chinese LLMs as first-class citizens.**

Multi-user agent pool · One interface for 9 LLM providers (Chinese-first) · Built-in tools & MCP · Sub-agents · Skills · RAG · Scheduler

[![PyPI](https://img.shields.io/pypi/v/milu)](https://pypi.org/project/milu/)
[![CI](https://github.com/stephonGAO/milu/actions/workflows/ci.yml/badge.svg)](https://github.com/stephonGAO/milu/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/milu/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1100%2B%20passed-brightgreen)](tests/)

**English** | [简体中文](README.zh-CN.md)

<img src="https://raw.githubusercontent.com/stephonGAO/milu/main/assets/demo-hero.gif" alt="milu demo" width="820">

</div>

---

## Why milu?

Most agent frameworks stop at single-user demos, and treat Chinese LLM providers as an afterthought. milu starts where they stop:

- 🏭 **From demo to production in one library**<br>
  `AgentPool` gives you per-user agent isolation, LRU/TTL eviction, global concurrency limits and shared MCP processes. The question every framework leaves as "an exercise for the reader" — *"my demo works, how do I serve 100 concurrent users without sessions bleeding into each other?"* — is answered here, backed by 1100+ tests. The same pool maps tenants to their own API keys (`KeyedLLMProvider`), so it scales from a side project to multi-tenant SaaS.
- 🇨🇳 **Chinese LLMs as first-class citizens**<br>
  Qwen, DeepSeek, Kimi, GLM, MiniMax, Doubao natively supported alongside OpenAI, Gemini and Claude. No `base_url` juggling, provider quirks (thinking mode, built-in web search, parameter differences) pre-adapted, plus a China-reachable search backend out of the box.
- 🔋 **Batteries actually included**<br>
  20+ built-in tools (files, shell, Python, web fetch/search, Office/PDF reading, vision input), MCP protocol (stdio/HTTP/SSE), sub-agents, skills, session persistence, automatic context compaction, long-term memory, RAG knowledge base, scheduled tasks, and a built-in multi-user web service.
- 🛡️ **A real safety model**<br>
  Four operation modes (`talk` / `manual` / `auto` / `superwork`), an AI safety judge for unsafe tool calls (Claude-Code-style), human confirmation flows, and delegation that never bypasses approval.
- 🔭 **Observability built in**<br>
  Every `Agent.run()` is a span tree (attribute names aligned with the OpenTelemetry GenAI semantic conventions): per-run tokens / cost / latency, TTFT, time split across LLM vs tools vs safety-judge vs approval-wait, sub-agent nesting, and fail-open audit. Inspect runs from the CLI with `milu trace list/show/compare/stats`, or the per-run waterfall in the web "Observability" panel.
- 🪶 **Thin by design**<br>
  Built directly on the `openai` SDK as the unified HTTP client. Events stream out as plain dataclasses. No chains, no graphs, no DSL to learn.

## Two ways to use it

milu is both a ready-to-run agent and a framework to build on — start instantly, embed when you need to:

- 🚀 **Run it**<br>
  `milu` for chat, `milu serve` for a multi-user service — full capabilities, zero code. Both the CLI and the web UI ship in **English and 中文**.
- 🧩 **Build on it**<br>
  `from milu import Agent` to embed agents in your own backend, then scale to multi-user / multi-tenant with AgentPool — **you own your data and stack**.

---

## Install

> [!TIP]
> **One `pip install milu` gets everything** — CLI, web service, RAG knowledge base and MCP are all included. You only need at least one provider API key to start.

**With pip** — if you already have Python 3.10+:

```bash
pip install -U milu           # everything included: CLI, web service, RAG, MCP
                              # -U installs fresh, and upgrades an existing install to the latest
pip install -U "milu[ddg]"    # optional: adds the key-free DuckDuckGo backend for web_search
                              # (Rust-based dep; kept optional so milu installs on Termux/Alpine)
```

<details>
<summary><b>New to Python? Beginner step-by-step</b></summary>

1. Download Python 3.10+ from [python.org/downloads](https://www.python.org/downloads/). On Windows, tick **"Add Python to PATH"** during setup.
2. Open a terminal (Windows: PowerShell · macOS: Terminal) and check: `python --version` should print 3.10 or higher.
3. `pip install -U milu`
4. `milu` to start chatting.
</details>

**No existing Python? — the easiest one-liner.** [uv](https://github.com/astral-sh/uv) installs Python and milu for you:

```bash
# 1. install uv (one line, needs no Python)
curl -LsSf https://astral.sh/uv/install.sh | sh            # macOS / Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex" # Windows

# 2. install milu (uv fetches a Python automatically if missing)
uv tool install milu
```

**Docker** — no Python on the host at all:

```bash
cp .env.example .env          # fill in at least one provider API key
docker compose up -d
```

## Quick start

> [!NOTE]
> First run launches an interactive setup wizard — pick a provider, paste an API key, and you're chatting. Zero config to first conversation.

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

---

## How it compares

| Capability | milu | LangChain | CrewAI | smolagents | Qwen-Agent |
|---|---|---|---|---|---|
| Chinese providers native (6) | ✅ | community pkgs | via LiteLLM | via LiteLLM | Qwen family |
| Multi-user pool, in-library | ✅ AgentPool | platform (paid) | platform | — | — |
| MCP protocol | ✅ 3 transports | ✅ | ✅ | ✅ | ✅ |
| Built-in tools (files/docs/vision/search) | ✅ 20+ | install per-integration | partial | minimal | partial |
| Tool-safety modes + AI judge | ✅ | — | — | sandbox only | — |
| RAG knowledge base, in-library | ✅ | assemble yourself | partial | — | ✅ |
| Scheduled tasks (multi-user) | ✅ | — | — | — | — |
| CLI + web service out of the box | ✅ | — | partial | — | demo UI |

> ✅ = built-in; "—" = not built-in (often available via an external platform or a few lines of your own code). Reflects each library as of June 2026 — these move fast, so corrections are welcome via issue/PR.
>
> **When milu is the right fit:** you're building on Chinese LLMs, you need a production multi-user / multi-tenant service (not just a single-user demo), and you want batteries included — runnable as-is or embeddable as a library, Or as a strong, bold and flexible development core and intelligent base.
>
> **When to choose something else:** for the largest integration ecosystem, [LangChain](https://github.com/langchain-ai/langchain); for pure multi-agent orchestration, [CrewAI](https://github.com/crewAIInc/crewAI) or [AutoGen](https://github.com/microsoft/autogen); for a tiny, barebones core with almost nothing built in, [smolagents](https://github.com/huggingface/smolagents).

## What you can build

- **Personal AI assistant**<br>
  `milu` drops you into a chat in one command; long-term memory remembers your preferences, scheduled tasks handle reminders and daily digests, and built-in tools (web search, files, docs, vision) are ready to use — all running locally, your data stays yours.
- **Enterprise knowledge assistant**<br>
  Load manuals / FAQs / policies into the RAG knowledge base; auto-retrieval each turn, source-aware answers that separate "internal docs vs web", no hallucinated guesses. Per-user isolated sessions and memory.
- **Customer-support / ticket bot**<br>
  High-volume repetitive queries and ticket triage; AgentPool handles many concurrent users, safety modes gate what actions run.
- **Vertical / industry assistant**<br>
  Sub-agents + document & vision reading + MCP to plug into your own systems and databases, bringing domain knowledge and real data in.
- **An "AI coworker" for your team**<br>
  Pull tasks from chat, nudge progress on a schedule, auto-generate recap summaries (scheduled tasks + multi-user + tools).
- **Private / on-prem deployment**<br>
  `docker compose up -d`; runs entirely in your environment with Chinese (or any) LLMs, data never leaves.
- **Multi-tenant SaaS / a base for AI app vendors**<br>
  `KeyedLLMProvider` maps tenants to their own API keys; the pool enforces per-user instance and concurrency isolation — scale from a side project to a multi-tenant product.

---

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

> [!WARNING]
> `superwork` skips every safety check (including the AI judge). Use it only for fully trusted tasks.

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

<!-- demo-cli.gif (optional) goes here. After recording (see assets/RECORDING.md), uncomment the line below and delete this note. -->
<!-- <img src="https://raw.githubusercontent.com/stephonGAO/milu/main/assets/demo-cli.gif" alt="milu CLI demo" width="760"> -->

```text
milu                 interactive chat (first run launches setup wizard)
milu setup           provider / API key / search backend wizard
milu chat -p glm     chat with a specific provider
milu run "..." -q    one-shot execution, pipe-friendly
milu serve           multi-user web service + demo UI
milu providers       list 9 providers and key status
milu trace ...        inspect agent runs (list / show / compare / stats)
milu config ...      layered config (CLI > user > project > defaults)
milu sessions list   browse saved sessions
milu schedule ...    manage scheduled tasks
milu --lang en ...   switch UI language for one run (zh / en)
```

**Language (中文 / English).** Both the CLI and the web UI are fully bilingual. Pick the interface language in any of these ways:

```bash
milu --lang en providers        # one-off override (also accepts --lang zh)
$env:MILU_LANG="en"; milu chat   # per-session via env var (PowerShell; bash: MILU_LANG=en)
milu config set lang en          # persist to ~/.milu/config.json
milu setup                       # the wizard asks for language as its first step
```

Priority: `--lang` > `MILU_LANG` > `config.json` `lang` > default `zh`. In the web UI, use the **EN / 中文** toggle in the top bar.

## Architecture

<div align="center">
<img src="https://raw.githubusercontent.com/stephonGAO/milu/main/assets/architecture.svg" alt="milu architecture" width="860">
</div>

<details>
<summary>Text version</summary>

```text
AgentPool (multi-user, optional)
  └─ Agent.run() loop ── system prompt rebuild → auto-compaction
       ├─ LLM layer        9 providers, one AsyncOpenAI-based interface
       ├─ Tool layer       built-ins · custom @tool · MCP (active/dormant pools)
       ├─ Safety layer     modes · AI judge · confirmation flow
       ├─ Sub-agents       researcher / reader / coder (isolated context)
       ├─ Prompts & skills layered markdown prompts · on-demand skill loading
       ├─ Session          JSONL persistence · compaction snapshots
       └─ Observability    span-tree tracing · per-run cost / latency · milu trace
```
</details>

Python 3.10+ · fully async · every provider speaks through one `openai.AsyncOpenAI` client, so LLM instances are coroutine-safe and shareable across users.

---

## Production notes

- **Scaling out**: route by `user_id` (e.g. nginx `ip_hash`); per-session serialization is handled by in-process entry locks — no distributed locks needed. Sessions persist to disk and recover after eviction or restart.
- **Memory budget**: MCP subprocesses are the dominant cost (15–50 MB per agent). Enable shared MCP (`AgentPoolConfig(shared_mcp=True)`) to keep one set of MCP processes for the whole pool.
- **Multi-tenant keys**: `KeyedLLMProvider` caches one LLM client per distinct API key with LRU eviction — see `examples/multi_tenant_keys.py`.
- **Docker**: see [docs/Docker部署.md](docs/Docker部署.md) — health checks, data volumes, SSE reverse-proxy settings, scheduler single-instance behavior.

## Roadmap

- [x] Observability: span-tree tracing (OTel GenAI semconv-aligned) + CLI `milu trace`
- [ ] Multi-user observability dashboard + OTLP exporter
- [ ] Sandboxed code execution backends for `python_repl` / `shell_command`
- [ ] Pluggable ANN backends for the knowledge store (sqlite-vec) beyond brute-force cosine
- [ ] English documentation set (architecture & guides — currently Chinese)
- [ ] Prebuilt images on a container registry
- [ ] One-click installers / standalone binaries (no Python required)

---

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

**milu (麋鹿)** — named after Père David's deer, the legendary Chinese animal that "resembles four creatures yet is none of them" — one body, the strengths of many.

</div>
