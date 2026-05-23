# AI Agent Framework - Agent 循环核心引擎设计

> 日期：2026-05-23
> 状态：已批准
> 阶段：第二步 - Agent 循环核心引擎

## 一、概述

在第一步（多厂商模型兼容层）的基础上，构建 Agent 循环核心引擎。Agent 编排器组合 LLM 调用、工具执行、对话历史管理，通过事件流向调用方输出结果，并内置多层安全机制。

### 目标

- 实现 ReAct 风格的 Agent 循环：推理 → 行动 → 观察 → 继续推理
- 支持基础工具调用能力（@tool 装饰器定义工具，自动 schema 生成）
- 提供可配置的对话历史管理和截断策略
- 事件流（AsyncIterator[AgentEvent]）输出，调用方可实时消费
- 多层安全控制防止无限循环和资源滥用

### 不在本步范围

- MCP 协议适配（第三步）
- RAG 检索增强（第四步）
- 复杂记忆系统（第五步）
- Langfuse 监控集成（第六步）

## 二、架构设计

### 2.1 组合式组件架构

```
Agent (编排器)
├── system_prompt: str              # 系统提示词，始终作为第一条消息
├── llm: BaseLLM                    # 第一步的模型层（已有）
├── history: ConversationHistory    # 对话历史管理
├── registry: ToolRegistry          # 工具注册表
├── executor: ToolExecutor          # 工具执行器（含安全控制）
└── config: AgentConfig             # Agent 配置（轮次、超时等）
```

### 2.2 核心数据流

```
用户调用 agent.run("北京天气")
        ↓
  构造 user Message，追加到 history
        ↓
  ┌─── Agent 循环开始 ───┐
  │                       │
  │  history.get_messages() ← 获取截断后的消息列表
  │       ↓               │
  │  llm.chat(messages,   │
  │    tools=registry     │
  │      .get_schemas())  │
  │       ↓               │
  │  流式收集 StreamChunk  │
  │  同时 yield 事件       │
  │       ↓               │
  │  有 tool_calls?        │
  │   ├─ 是 → executor    │
  │   │   .execute()      │
  │   │       ↓           │
  │   │  构造 tool Message │
  │   │  追加到 history    │
  │   │  → 继续循环 ──┐   │
  │   │              │   │
  │   └─ 否 → yield AgentDone，退出循环
  │                       │
  └───────────────────────┘
```

### 2.3 新增文件结构

```
src/agent_framework/
├── __init__.py              # 更新：导出 Agent、AgentConfig、AgentEvent 等
├── exceptions.py            # 更新：新增 Agent 相关异常
├── models/
│   ├── __init__.py          # 更新：导出 AgentEvent
│   ├── message.py           # 已有
│   ├── response.py          # 已有
│   ├── config.py            # 已有
│   └── events.py            # 新增：AgentEvent 及其子类
├── providers/               # 已有，不改动
│   └── ...
├── agent/                   # 新增：Agent 核心模块
│   ├── __init__.py          # 导出 Agent、AgentConfig
│   ├── agent.py             # Agent 编排器（核心循环）
│   ├── config.py            # AgentConfig 数据类
│   ├── history.py           # ConversationHistory
│   └── executor.py          # ToolExecutor
└── tools/                   # 新增：工具系统
    ├── __init__.py          # 导出 @tool、ToolRegistry
    ├── decorator.py         # @tool 装饰器实现
    ├── registry.py          # ToolRegistry
    └── schema.py            # Python 签名 → JSON Schema 转换
```

## 三、AgentEvent 事件体系

### 3.1 事件类型定义

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class AgentEvent:
    """Agent 事件基类"""
    pass

@dataclass(frozen=True)
class TextDelta(AgentEvent):
    """LLM 正文输出片段"""
    text: str

@dataclass(frozen=True)
class ReasoningDelta(AgentEvent):
    """LLM 思考过程输出片段"""
    text: str

@dataclass(frozen=True)
class ToolCallStart(AgentEvent):
    """LLM 决定调用工具"""
    tool_name: str
    tool_call_id: str
    arguments: str           # JSON 字符串

@dataclass(frozen=True)
class ToolResult(AgentEvent):
    """工具执行完成"""
    tool_name: str
    tool_call_id: str
    output: str
    is_error: bool

@dataclass(frozen=True)
class AgentDone(AgentEvent):
    """Agent 循环正常结束"""
    final_text: str
    total_usage: TokenUsage
    turn_count: int

@dataclass(frozen=True)
class AgentError(AgentEvent):
    """Agent 异常终止"""
    error_type: str          # "max_turns" | "call_timeout" | "total_timeout" | "token_limit" | "tool_limit"
    message: str
```

### 3.2 事件流使用示例

```python
async for event in agent.run("北京天气怎么样？"):
    match event:
        case TextDelta(text=t):
            print(t, end="", flush=True)
        case ReasoningDelta(text=t):
            pass  # 可选：显示思考过程
        case ToolCallStart(tool_name=n):
            print(f"\n🔧 调用: {n}")
        case ToolResult(output=o, is_error=False):
            print(f"📋 结果: {o}")
        case AgentDone(usage=u):
            print(f"\n✅ 完成 (tokens: {u.total_tokens})")
        case AgentError(error_type=t, message=m):
            print(f"\n❌ {t}: {m}")
```

## 四、工具系统

### 4.1 @tool 装饰器

```python
from agent_framework.tools import tool

@tool(name="get_weather", description="获取指定城市的天气信息")
async def get_weather(city: str, unit: str = "celsius") -> str:
    """获取天气（docstring 可通过 :param 语法提供参数描述）"""
    return f"{city}：晴，25°C"
```

装饰器内部通过 `inspect` 模块自动提取：
- 函数签名 → parameters schema
- 类型注解 → 参数类型
- docstring → 参数描述（可选，通过 `:param` 语法）
- 生成 OpenAI function calling schema

### 4.2 类型映射规则

| Python 类型 | JSON Schema 类型 |
|------------|-----------------|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| `list` / `list[X]` | `"array"` |
| `dict` | `"object"` |
| `Optional[X]` | 同 X，但不加入 required |
| `Literal["a", "b"]` | `"string"` + `enum: ["a", "b"]` |

### 4.3 自动生成的 Schema 示例

```python
# get_weather 生成的 schema：
{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "city"},
                "unit": {"type": "string", "description": "unit", "default": "celsius"}
            },
            "required": ["city"]
        }
    }
}
```

### 4.4 参数描述增强（可选）

通过 docstring 的 `:param` 语法为参数添加描述：

```python
@tool(name="search", description="搜索信息")
async def search(query: str, max_results: int = 10) -> str:
    """
    搜索互联网信息。
    :param query: 搜索关键词
    :param max_results: 最大返回结果数
    """
    ...
# 生成: {"query": {"type": "string", "description": "搜索关键词"}, ...}
```

### 4.5 危险操作标记

```python
@tool(name="delete_file", description="删除文件", dangerous=True)
async def delete_file(path: str) -> str:
    """删除指定文件"""
    os.remove(path)
    return f"已删除: {path}"
```

标记 `dangerous=True` 的工具预留确认钩子接口。本步默认 `confirm_dangerous=False`，不实现交互式确认。

### 4.6 ToolRegistry

```python
class ToolRegistry:
    """工具注册表 - 管理 @tool 装饰过的函数"""

    def register(self, func) -> None:
        """注册一个 @tool 装饰的函数"""

    def register_many(self, funcs: list) -> None:
        """批量注册"""

    def get_schemas(self) -> list[dict]:
        """返回所有工具的 OpenAI function calling schema 列表，
        可直接传给 llm.chat(tools=...) 使用"""

    def get_tool(self, name: str) -> ToolWrapper | None:
        """根据名称获取工具"""

    def list_tools(self) -> list[str]:
        """列出所有已注册的工具名"""
```

### 4.7 ToolWrapper（内部类）

```python
@dataclass
class ToolWrapper:
    """包装一个 @tool 函数，持有其元数据和 schema"""
    name: str
    description: str
    parameters_schema: dict
    func: Callable          # 原始 async/sync 函数
    is_async: bool          # 是否异步函数
    dangerous: bool         # 是否标记为危险操作
```

## 五、对话历史管理

### 5.1 ConversationHistory

```python
class ConversationHistory:
    """对话历史管理 - 支持多种截断策略"""

    def __init__(
        self,
        strategy: str = "sliding_window",
        max_turns: int = 50,
        max_tokens: int | None = None,
        preserve_system: bool = True,
        head_turns: int = 4,       # head_tail 策略：保留前 N 条
        tail_turns: int = 10,      # head_tail 策略：保留后 M 条
    ):
        self._messages: list[Message] = []
        self._strategy = strategy
        ...

    def set_system(self, message: Message) -> None:
        """设置 system 消息（始终作为第一条）"""

    def add(self, message: Message) -> None:
        """追加一条消息"""

    def get_messages(self) -> list[Message]:
        """获取截断后的消息列表（用于传给 LLM）"""

    def clear(self) -> None:
        """清空历史（保留 system 消息）"""

    @property
    def all_messages(self) -> list[Message]:
        """获取完整未截断的历史（用于调试/日志）"""
```

### 5.2 截断策略

| 策略名 | 行为 | 适用场景 |
|--------|------|----------|
| `"none"` | 不截断，保留全部消息 | 短对话、测试 |
| `"sliding_window"` | 保留最近 N 条消息（system 消息始终保留） | 通用场景 |
| `"token_limit"` | 按 token 数截断，超出部分从最旧的非 system 消息开始移除 | 长对话、大模型 |
| `"head_tail"` | 保留前 `head_turns` 条 + 后 `tail_turns` 条，中间截断（插入一条 "..." 提示消息） | 超长文档分析 |

### 5.3 Token 估算

使用粗略估算，不依赖外部 tokenizer 库：

```python
def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数：中文约 1.5 字/token，英文约 4 字符/token"""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)
```

### 5.4 截断示例

```python
# 滑动窗口：max_turns=4
history.add(Message(role="system", content="你是助手"))   # 始终保留
history.add(Message(role="user", content="问题1"))        # ← 被截断
history.add(Message(role="assistant", content="回答1"))   # ← 被截断
history.add(Message(role="user", content="问题2"))        # 保留
history.add(Message(role="assistant", content="回答2"))   # 保留
history.add(Message(role="user", content="问题3"))        # 保留
history.add(Message(role="assistant", content="回答3"))   # 保留

# get_messages() 返回 5 条：system + 后4条
```

## 六、Agent 编排器

### 6.1 AgentConfig

```python
@dataclass
class AgentConfig:
    """Agent 运行配置"""
    max_turns: int = 10            # 最大循环轮次（防无限循环）
    timeout: float = 120.0         # 单次 LLM 调用超时（秒）
    total_timeout: float = 300.0   # 整个 run() 的总超时（秒）
    max_total_tokens: int | None = None   # 总 token 上限（None=不限制）
    tool_call_limit: int = 20      # 单次 run() 中最大工具调用次数
    confirm_dangerous: bool = False # 危险工具调用前需确认（预留接口）
```

### 6.2 Agent 类

```python
class Agent:
    """Agent 编排器 - 组合 LLM、工具、对话历史，驱动循环"""

    def __init__(
        self,
        llm: BaseLLM,
        system_prompt: str = "你是一个智能助手。",
        tools: list | None = None,
        history: ConversationHistory | None = None,
        config: AgentConfig | None = None,
    ):
        self._llm = llm
        self._system_prompt = system_prompt
        self._registry = ToolRegistry()
        if tools:
            self._registry.register_many(tools)
        self._history = history or ConversationHistory()
        self._config = config or AgentConfig()
        self._executor = ToolExecutor(self._registry, self._config)
        self._history.set_system(
            Message(role=MessageRole.SYSTEM, content=system_prompt)
        )

    async def run(self, user_input: str, **llm_kwargs) -> AsyncIterator[AgentEvent]:
        """
        执行一次完整的 Agent 对话。

        参数:
            user_input: 用户输入文本
            **llm_kwargs: 传递给 LLM 的额外参数（temperature 等）

        Yields:
            AgentEvent 事件流（TextDelta、ReasoningDelta、ToolCallStart、
            ToolResult、AgentDone、AgentError）
        """

    async def reset(self) -> None:
        """重置对话历史（保留 system prompt）"""

    @property
    def history(self) -> ConversationHistory:
        """访问对话历史"""

    @property
    def tools(self) -> ToolRegistry:
        """访问工具注册表"""
```

### 6.3 Agent.run() 核心循环

```python
async def run(self, user_input: str, **llm_kwargs) -> AsyncIterator[AgentEvent]:
    # 1. 追加用户消息
    self._history.add(Message(role=MessageRole.USER, content=user_input))

    turn_count = 0
    total_usage = TokenUsage()
    tool_call_count = 0
    start_time = time.monotonic()

    while turn_count < self._config.max_turns:
        # 安全检查：总超时
        if time.monotonic() - start_time > self._config.total_timeout:
            yield AgentError("total_timeout", "总超时")
            return

        turn_count += 1

        # 2. 构建请求
        messages = self._history.get_messages()
        tools_schema = self._registry.get_schemas() or None

        # 3. 流式调用 LLM，边收集边 yield 事件
        content = ""
        reasoning = ""
        tool_calls = []
        usage = None

        try:
            async with asyncio.timeout(self._config.timeout):
                async for chunk in self._llm.chat(
                    messages, tools=tools_schema, **llm_kwargs
                ):
                    if chunk.content:
                        content += chunk.content
                        yield TextDelta(text=chunk.content)
                    if chunk.reasoning_content:
                        reasoning += chunk.reasoning_content
                        yield ReasoningDelta(text=chunk.reasoning_content)
                    if chunk.tool_calls:
                        _merge_tool_calls(tool_calls, chunk.tool_calls)
                    if chunk.usage:
                        usage = chunk.usage
        except asyncio.TimeoutError:
            yield AgentError("call_timeout",
                f"LLM调用超时({self._config.timeout}s)")
            return

        # 4. 累计 usage
        if usage:
            total_usage = _merge_usage(total_usage, usage)

        # 安全检查：token 上限
        if (self._config.max_total_tokens
                and total_usage.total_tokens > self._config.max_total_tokens):
            yield AgentError("token_limit", "超出token上限")
            return

        # 5. 记录 assistant 消息到 history
        assistant_msg = Message(
            role=MessageRole.ASSISTANT,
            content=content or None,
            tool_calls=tool_calls if tool_calls else None,
        )
        self._history.add(assistant_msg)

        # 6. 如果没有工具调用 → 结束
        if not tool_calls:
            yield AgentDone(
                final_text=content,
                total_usage=total_usage,
                turn_count=turn_count,
            )
            return

        # 7. 执行每个工具调用
        for tc in tool_calls:
            tool_call_count += 1
            if tool_call_count > self._config.tool_call_limit:
                yield AgentError("tool_limit", "工具调用次数超限")
                return

            tc_dict = _tool_call_to_dict(tc)
            func_name = tc_dict["function"]["name"]
            tc_id = tc_dict.get("id", "")
            arguments = tc_dict["function"]["arguments"]

            yield ToolCallStart(
                tool_name=func_name,
                tool_call_id=tc_id,
                arguments=arguments,
            )

            result = await self._executor.execute(tc_dict)

            yield ToolResult(
                tool_name=func_name,
                tool_call_id=tc_id,
                output=result.output,
                is_error=result.is_error,
            )

            # 工具结果追加到 history
            self._history.add(Message(
                role=MessageRole.TOOL,
                content=result.output,
                tool_call_id=tc_id,
                name=func_name,
            ))

        # → 继续循环（带着工具结果再调 LLM）

    # 循环超出 max_turns
    yield AgentError("max_turns", f"超出最大轮次({self._config.max_turns})")
```

### 6.4 流式工具调用合并

OpenAI 流式 API 中，tool_calls 可能分散在多个 chunk 中到达（尤其是 arguments 字段）。需要在内部合并：

```python
def _merge_tool_calls(
    buffer: list[dict],
    new_calls: list,
) -> None:
    """合并流式到达的 tool_call 片段"""
    for tc in new_calls:
        idx = tc.index if hasattr(tc, 'index') else 0
        while len(buffer) <= idx:
            buffer.append({
                "id": "",
                "function": {"name": "", "arguments": ""},
            })
        if hasattr(tc, 'id') and tc.id:
            buffer[idx]["id"] = tc.id
        if hasattr(tc, 'function'):
            if tc.function.name:
                buffer[idx]["function"]["name"] += tc.function.name
            if tc.function.arguments:
                buffer[idx]["function"]["arguments"] += tc.function.arguments
```

## 七、ToolExecutor

### 7.1 类定义

```python
@dataclass
class ToolExecutionResult:
    """工具执行结果"""
    output: str          # 工具输出（始终为字符串，回填给 LLM）
    is_error: bool       # 是否为错误结果


class ToolExecutor:
    """工具执行器 - 负责安全地执行工具并处理异常"""

    def __init__(self, registry: ToolRegistry, config: AgentConfig):
        self._registry = registry
        self._config = config

    async def execute(self, tool_call: dict) -> ToolExecutionResult:
        """执行一个工具调用，异常不中断循环"""
```

### 7.2 执行流程

```
tool_call 到达
     ↓
1. 从 registry 查找工具
     ↓ 找不到 → 返回 is_error=True, output="工具不存在: xxx"
2. 解析 arguments JSON
     ↓ 解析失败 → 返回 is_error=True, output="参数解析失败: ..."
3. 参数校验（检查必需参数是否存在）
     ↓ 校验失败 → 返回 is_error=True, output="参数错误: ..."
4. 执行工具函数（async 或 sync 都支持）
     ↓ 执行异常 → 返回 is_error=True, output="执行异常: {error_message}"
5. 返回值转为 str
     ↓
ToolExecutionResult(output="结果", is_error=False)
```

**关键设计：工具执行异常不中断 Agent 循环**，而是作为错误结果回填给 LLM，让 LLM 自行决定如何处理（重试、换方案、告知用户）。

## 八、多层安全机制

| 层级 | 机制 | 默认值 | 触发行为 |
|------|------|--------|----------|
| 1 | `max_turns` | 10 | 超出 → yield AgentError 并终止 |
| 2 | `timeout` | 120s | 单次 LLM 调用超时 → yield AgentError |
| 3 | `total_timeout` | 300s | 整个 run() 超时 → yield AgentError |
| 4 | `max_total_tokens` | None | 累计 token 超限 → yield AgentError |
| 5 | `tool_call_limit` | 20 | 工具调用次数超限 → yield AgentError |
| 6 | `dangerous` 标记 | False | 危险工具需确认（预留接口，本步不实现交互式确认） |
| 7 | 工具异常回填 | 始终启用 | 异常不中断循环，回填给 LLM 自我修正 |

## 九、异常体系（新增）

在已有 `exceptions.py` 基础上追加：

```python
class AgentLoopError(AgentFrameworkError):
    """Agent 循环异常基类"""

class MaxTurnsExceeded(AgentLoopError):
    """超出最大循环轮次"""

class AgentTimeout(AgentLoopError):
    """Agent 调用超时"""

class TokenLimitExceeded(AgentLoopError):
    """超出 token 上限"""

class ToolCallLimitExceeded(AgentLoopError):
    """工具调用次数超限"""

class ToolExecutionError(AgentLoopError):
    """工具执行异常（内部使用，不抛出，仅回填）"""
```

注意：这些异常类主要用于框架内部标识，实际运行时通过 `AgentError` 事件通知调用方，不直接抛出（避免破坏 async for 循环）。

## 十、测试策略

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/test_tool_decorator.py` | @tool 装饰器、schema 生成、类型映射、参数描述提取 |
| `tests/test_tool_registry.py` | 注册、查找、批量注册、get_schemas |
| `tests/test_history.py` | 4 种截断策略、system 保留、token 估算、clear |
| `tests/test_executor.py` | 工具执行、异常回填、参数校验、sync/async 兼容 |
| `tests/test_agent.py` | 核心循环（mock LLM）、事件流验证、安全机制触发 |
| `tests/test_agent_integration.py` | 真实 LLM + 工具的端到端集成测试 |

**测试方法：**
- Agent 核心循环测试使用 mock LLM（模拟返回 tool_calls 和文本），不依赖真实 API
- 集成测试使用真实 API（Qwen），验证端到端流程
- 安全机制测试：构造 mock LLM 返回无限工具调用，验证 max_turns/tool_call_limit 触发

## 十一、使用示例

```python
from agent_framework import Agent, AgentConfig
from agent_framework.providers import ModelRegistry
from agent_framework.tools import tool

# 1. 定义工具
@tool(name="get_weather", description="获取天气")
async def get_weather(city: str) -> str:
    return f"{city}：晴，25°C"

@tool(name="calculator", description="数学计算")
async def calculator(expression: str) -> str:
    return str(eval(expression))

# 2. 创建 Agent
llm = ModelRegistry.create("qwen", model="qwen-max")
agent = Agent(
    llm=llm,
    system_prompt="你是一个智能助手，可以查天气和做计算。",
    tools=[get_weather, calculator],
    config=AgentConfig(max_turns=5, timeout=60),
)

# 3. 运行（事件流）
async for event in agent.run("北京天气怎么样？顺便算一下 17*23+45"):
    match event:
        case TextDelta(text=t):
            print(t, end="", flush=True)
        case ToolCallStart(tool_name=n):
            print(f"\n🔧 调用: {n}")
        case ToolResult(output=o, is_error=False):
            print(f"📋 结果: {o}")
        case AgentDone(usage=u):
            print(f"\n✅ 完成 (tokens: {u.total_tokens})")

# 4. 继续对话（history 自动保留）
async for event in agent.run("上海呢？"):
    ...

# 5. 重置对话
await agent.reset()
```

## 十二、后续扩展预留

- **第三步（MCP）**：ToolRegistry 扩展 `register_mcp_server()` 方法，MCP 工具自动转为 @tool 格式
- **第四步（RAG）**：Agent.run() 前插入检索增强步骤，将检索结果注入 system_prompt 或 user message
- **第五步（记忆）**：ConversationHistory 替换为更复杂的 MemoryManager，支持向量化检索和摘要
- **第六步（Langfuse）**：Agent 循环各阶段埋点，上报 trace/span 到 Langfuse
