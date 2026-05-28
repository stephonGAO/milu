# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

agent-framework：统一的 AI 模型抽象层 + Agent 编排引擎。支持 9 个 LLM 提供商（通义千问、Kimi、GLM、DeepSeek、MiniMax、豆包、ChatGPT、Gemini、Claude），提供工具系统（含 MCP 协议）、子 Agent、内置工具等能力。

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

# 运行示例
.venv/Scripts/python examples/1.\ basic_llm.py
```

## 架构概览

四层架构，全部 async-first：

### 1. LLM 层 (`src/agent_framework/llm/`)

- `BaseLLM`（抽象基类）定义统一接口：`chat(messages, **kwargs) -> AsyncIterator[StreamChunk]`
- `ModelCapabilities`（frozen dataclass）声明提供商能力（function_calling, streaming, reasoning 等）
- `ModelRegistry` 工厂模式：每个 provider 文件末尾自注册 `ModelRegistry.register("name", Class)`
- **关键设计**：所有 9 个 provider 统一使用 `openai.AsyncOpenAI` 客户端，通过不同 `base_url` + `extra_body` 适配各提供商
- `ModelConfig` 基类 + 扩展配置（`WebSearchConfig`, `ThinkingConfig`, `FunctionCallingConfig` 等）

### 2. Agent 层 (`src/agent_framework/agent/`)

- `Agent` 核心循环：LLM 调用 → 解析文本/工具调用 → 执行工具 → 回传结果 → 重复
- `run()` 返回 `AsyncIterator[AgentEvent]`，事件类型包括：`TextDelta`, `ToolCallStart`, `ToolResult`, `AgentDone`, `SubAgentEvent` 等
- 高优先级工具（`priority>0`，如 `todo_write`）先顺序执行，普通优先级工具再并发执行（`asyncio.gather`）
- `ConversationHistory` 支持 4 种截断策略：`none`, `sliding_window`, `token_limit`, `head_tail`
- `SubAgent`：通过 `create_subagent_tools()` 工厂创建闭包工具，每次调用使用独立 Agent 实例和干净历史，不允许嵌套

### 3. 工具层 (`src/agent_framework/tools/`)

- `@tool(name, description, dangerous, priority)` 装饰器 → 自动生成 `ToolWrapper`（含 JSON Schema）
- `ToolRegistry` 双池设计：active pool（schema 注入 LLM）+ dormant pool（MCP 工具待激活）
- `catalog.py` 提供三个元工具（`list_catalog`, `search_tools`, `activate_tools`）供 LLM 自主发现/激活 dormant 工具
- `ToolExecutor` 安全执行：JSON 参数解析、async/sync 兼容、异常捕获

### 4. MCP 层 (`src/agent_framework/tools/mcp/`)

- 支持三种传输：`stdio`, `streamable_http`, `sse`
- `MCPManager` 多服务器编排：`asyncio.gather` 并行连接，错误隔离
- `converter.py` 将 MCP Tool 转为 `ToolWrapper`，工具名加 `{server_name}__` 前缀避免冲突
- 配置文件：`config/mcp_servers.json`

## 代码风格约定

- 中文 docstring 和注释，中文 git commit message
- 使用 Python 3.10+ 语法：`str | None` 联合类型、`match/case`
- 数据模型全部用 `@dataclass`，不可变模型加 `frozen=True`
- 环境变量命名：`{PROVIDER}_API_KEY`（如 `QWEN_API_KEY`, `ANTHROPIC_API_KEY`）
- 无 lint/type-check 工具配置（未启用 ruff、mypy、black）

## 测试模式

- 单元测试使用 `unittest.mock.AsyncMock` mock LLM 响应
- `conftest.py` 提供 `mock_openai_client` fixture
- 每个 provider 有独立测试文件 `test_<provider>.py`
- 真实 API 测试在 `test_real_api.py` / `test_real_new_providers.py`，需要 `.env` 配置
- pytest 配置：`asyncio_mode = "auto"`（`pyproject.toml`）
