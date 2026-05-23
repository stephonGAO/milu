# Agent 循环核心引擎 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 Agent 循环核心引擎，支持工具调用、对话历史管理和多层安全机制

**Architecture:** 组合式组件架构，Agent 编排器组合 ConversationHistory、ToolRegistry、ToolExecutor 三个独立组件，通过事件流（AsyncIterator[AgentEvent]）输出

**Tech Stack:** Python 3.10+, asyncio, inspect 模块（schema 生成）, 无新增外部依赖

---

## 文件结构

```
src/agent_framework/
├── exceptions.py                    # 修改：追加 Agent 相关异常
├── __init__.py                      # 修改：导出新组件
├── models/
│   ├── __init__.py                  # 修改：导出 AgentEvent
│   └── events.py                    # 新增：AgentEvent 及子类
├── agent/
│   ├── __init__.py                  # 新增：导出 Agent、AgentConfig
│   ├── config.py                    # 新增：AgentConfig
│   ├── history.py                   # 新增：ConversationHistory
│   ├── executor.py                  # 新增：ToolExecutor
│   └── agent.py                     # 新增：Agent 编排器
└── tools/
    ├── __init__.py                  # 新增：导出 @tool、ToolRegistry
    ├── decorator.py                 # 新增：@tool 装饰器
    ├── registry.py                  # 新增：ToolRegistry
    └── schema.py                    # 新增：签名转 schema

tests/
├── test_tool_decorator.py           # 新增
├── test_tool_registry.py            # 新增
├── test_history.py                  # 新增
├── test_executor.py                 # 新增
├── test_agent.py                    # 新增
└── test_agent_integration.py        # 新增
```

---

## Task 1: 异常类

**Files:**
- Modify: `src/agent_framework/exceptions.py`
- Test: `tests/test_exceptions.py` (新建)

- [ ] **Step 1: 写失败测试**

```python
# tests/test_exceptions.py
"""测试 Agent 相关异常类"""
import pytest
from agent_framework.exceptions import (
    AgentFrameworkError,
    AgentLoopError,
    MaxTurnsExceeded,
    AgentTimeout,
    TokenLimitExceeded,
    ToolCallLimitExceeded,
    ToolExecutionError,
)


def test_agent_loop_error_inherits_from_base():
    """AgentLoopError 应继承 AgentFrameworkError"""
    assert issubclass(AgentLoopError, AgentFrameworkError)
    
    error = AgentLoopError("测试错误")
    assert isinstance(error, AgentFrameworkError)
    assert str(error) == "测试错误"


def test_max_turns_exceeded():
    """MaxTurnsExceeded 应继承 AgentLoopError"""
    assert issubclass(MaxTurnsExceeded, AgentLoopError)
    
    error = MaxTurnsExceeded("超出最大轮次")
    assert isinstance(error, AgentLoopError)


def test_agent_timeout():
    """AgentTimeout 应继承 AgentLoopError"""
    assert issubclass(AgentTimeout, AgentLoopError)


def test_token_limit_exceeded():
    """TokenLimitExceeded 应继承 AgentLoopError"""
    assert issubclass(TokenLimitExceeded, AgentLoopError)


def test_tool_call_limit_exceeded():
    """ToolCallLimitExceeded 应继承 AgentLoopError"""
    assert issubclass(ToolCallLimitExceeded, AgentLoopError)


def test_tool_execution_error():
    """ToolExecutionError 应继承 AgentLoopError"""
    assert issubclass(ToolExecutionError, AgentLoopError)
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_exceptions.py -v
```

预期：ImportError，异常类不存在

- [ ] **Step 3: 实现异常类**

在 `src/agent_framework/exceptions.py` 末尾追加：

```python
# Agent 循环相关异常
class AgentLoopError(AgentFrameworkError):
    """Agent 循环异常基类"""
    pass


class MaxTurnsExceeded(AgentLoopError):
    """超出最大循环轮次"""
    pass


class AgentTimeout(AgentLoopError):
    """Agent 调用超时"""
    pass


class TokenLimitExceeded(AgentLoopError):
    """超出 token 上限"""
    pass


class ToolCallLimitExceeded(AgentLoopError):
    """工具调用次数超限"""
    pass


class ToolExecutionError(AgentLoopError):
    """工具执行异常"""
    pass
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_exceptions.py -v
```

预期：6 个测试全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/agent_framework/exceptions.py tests/test_exceptions.py
git commit -m "feat: add Agent loop exception classes

- AgentLoopError: base exception for agent loop
- MaxTurnsExceeded: max turns limit exceeded
- AgentTimeout: call or total timeout
- TokenLimitExceeded: token limit exceeded
- ToolCallLimitExceeded: tool call count limit
- ToolExecutionError: tool execution failure"
```

---

## Task 2: AgentEvent 事件类型

**Files:**
- Create: `src/agent_framework/models/events.py`
- Modify: `src/agent_framework/models/__init__.py`
- Test: `tests/test_events.py` (新建)

- [ ] **Step 1: 写失败测试**

```python
# tests/test_events.py
"""测试 AgentEvent 事件类型"""
import pytest
from agent_framework.models.events import (
    AgentEvent,
    TextDelta,
    ReasoningDelta,
    ToolCallStart,
    ToolResult,
    AgentDone,
    AgentError,
)
from agent_framework.models.response import TokenUsage


def test_text_delta():
    """TextDelta 应存储文本片段"""
    event = TextDelta(text="你好")
    assert event.text == "你好"
    assert isinstance(event, AgentEvent)


def test_reasoning_delta():
    """ReasoningDelta 应存储思考片段"""
    event = ReasoningDelta(text="让我想想")
    assert event.text == "让我想想"
    assert isinstance(event, AgentEvent)


def test_tool_call_start():
    """ToolCallStart 应存储工具调用信息"""
    event = ToolCallStart(
        tool_name="get_weather",
        tool_call_id="call_123",
        arguments='{"city": "北京"}'
    )
    assert event.tool_name == "get_weather"
    assert event.tool_call_id == "call_123"
    assert event.arguments == '{"city": "北京"}'


def test_tool_result():
    """ToolResult 应存储工具执行结果"""
    event = ToolResult(
        tool_name="get_weather",
        tool_call_id="call_123",
        output="北京：晴，25°C",
        is_error=False
    )
    assert event.tool_name == "get_weather"
    assert event.output == "北京：晴，25°C"
    assert event.is_error is False


def test_tool_result_error():
    """ToolResult 应能表示错误结果"""
    event = ToolResult(
        tool_name="get_weather",
        tool_call_id="call_123",
        output="工具不存在",
        is_error=True
    )
    assert event.is_error is True


def test_agent_done():
    """AgentDone 应存储完成信息"""
    usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    event = AgentDone(
        final_text="回答完成",
        total_usage=usage,
        turn_count=3
    )
    assert event.final_text == "回答完成"
    assert event.total_usage.total_tokens == 30
    assert event.turn_count == 3


def test_agent_error():
    """AgentError 应存储错误信息"""
    event = AgentError(
        error_type="max_turns",
        message="超出最大轮次(10)"
    )
    assert event.error_type == "max_turns"
    assert "超出最大轮次" in event.message
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_events.py -v
```

预期：ImportError

- [ ] **Step 3: 实现事件类**

```python
# src/agent_framework/models/events.py
"""Agent 事件类型定义"""
from __future__ import annotations

from dataclasses import dataclass

from agent_framework.models.response import TokenUsage


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
    arguments: str  # JSON 字符串


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
    error_type: str  # "max_turns" | "call_timeout" | "total_timeout" | "token_limit" | "tool_limit"
    message: str
```

- [ ] **Step 4: 更新 models/__init__.py**

在 `src/agent_framework/models/__init__.py` 追加：

```python
from agent_framework.models.events import (
    AgentEvent,
    TextDelta,
    ReasoningDelta,
    ToolCallStart,
    ToolResult,
    AgentDone,
    AgentError,
)

__all__.extend([
    "AgentEvent",
    "TextDelta",
    "ReasoningDelta",
    "ToolCallStart",
    "ToolResult",
    "AgentDone",
    "AgentError",
])
```

- [ ] **Step 5: 运行测试验证通过**

```bash
pytest tests/test_events.py -v
```

预期：7 个测试全部 PASS

- [ ] **Step 6: 提交**

```bash
git add src/agent_framework/models/events.py src/agent_framework/models/__init__.py tests/test_events.py
git commit -m "feat: add AgentEvent types

- TextDelta: streaming text output
- ReasoningDelta: streaming reasoning output
- ToolCallStart: tool invocation event
- ToolResult: tool execution result
- AgentDone: agent completion event
- AgentError: agent error event"
```

---

## Task 3: @tool 装饰器和 Schema 生成

**Files:**
- Create: `src/agent_framework/tools/schema.py`
- Create: `src/agent_framework/tools/decorator.py`
- Create: `src/agent_framework/tools/__init__.py`
- Test: `tests/test_tool_decorator.py` (新建)

- [ ] **Step 1: 写失败测试**

```python
# tests/test_tool_decorator.py
"""测试 @tool 装饰器和 schema 生成"""
import pytest
from agent_framework.tools import tool
from agent_framework.tools.decorator import ToolWrapper


def test_basic_tool_decorator():
    """@tool 应装饰函数并添加元数据"""
    @tool(name="get_weather", description="获取天气")
    async def get_weather(city: str) -> str:
        return f"{city}：晴"
    
    assert hasattr(get_weather, "_tool_wrapper")
    wrapper: ToolWrapper = get_weather._tool_wrapper
    assert wrapper.name == "get_weather"
    assert wrapper.description == "获取天气"
    assert wrapper.is_async is True


def test_schema_generation_basic_types():
    """应从类型注解生成 JSON schema"""
    @tool(name="test_func", description="测试函数")
    async def test_func(
        text: str,
        number: int,
        flag: bool,
        value: float = 3.14
    ) -> str:
        return "ok"
    
    wrapper: ToolWrapper = test_func._tool_wrapper
    schema = wrapper.parameters_schema
    
    assert schema["type"] == "object"
    assert "text" in schema["properties"]
    assert schema["properties"]["text"]["type"] == "string"
    assert schema["properties"]["number"]["type"] == "integer"
    assert schema["properties"]["flag"]["type"] == "boolean"
    assert schema["properties"]["value"]["type"] == "number"
    assert "text" in schema["required"]
    assert "number" in schema["required"]
    assert "flag" in schema["required"]
    assert "value" not in schema["required"]  # 有默认值，不是必需


def test_schema_optional_types():
    """Optional 类型不应加入 required"""
    from typing import Optional
    
    @tool(name="test_optional", description="测试 Optional")
    async def test_optional(
        required: str,
        optional: Optional[str] = None
    ) -> str:
        return "ok"
    
    wrapper: ToolWrapper = test_optional._tool_wrapper
    schema = wrapper.parameters_schema
    
    assert "required" in schema["required"]
    assert "optional" not in schema["required"]


def test_schema_literal_type():
    """Literal 类型应生成 enum"""
    from typing import Literal
    
    @tool(name="test_literal", description="测试 Literal")
    async def test_literal(mode: Literal["fast", "slow"]) -> str:
        return "ok"
    
    wrapper: ToolWrapper = test_literal._tool_wrapper
    schema = wrapper.parameters_schema
    
    assert schema["properties"]["mode"]["type"] == "string"
    assert schema["properties"]["mode"]["enum"] == ["fast", "slow"]


def test_schema_list_type():
    """list 类型应生成 array"""
    @tool(name="test_list", description="测试 list")
    async def test_list(items: list[str]) -> str:
        return "ok"
    
    wrapper: ToolWrapper = test_list._tool_wrapper
    schema = wrapper.parameters_schema
    
    assert schema["properties"]["items"]["type"] == "array"


def test_docstring_param_extraction():
    """应从 docstring 提取参数描述"""
    @tool(name="test_doc", description="测试函数")
    async def test_doc(query: str, limit: int = 10) -> str:
        """
        搜索信息。
        :param query: 搜索关键词
        :param limit: 最大返回数
        """
        return "ok"
    
    wrapper: ToolWrapper = test_doc._tool_wrapper
    schema = wrapper.parameters_schema
    
    assert schema["properties"]["query"]["description"] == "搜索关键词"
    assert schema["properties"]["limit"]["description"] == "最大返回数"


def test_sync_function():
    """应支持同步函数"""
    @tool(name="sync_tool", description="同步工具")
    def sync_tool(x: int) -> str:
        return str(x)
    
    wrapper: ToolWrapper = sync_tool._tool_wrapper
    assert wrapper.is_async is False


def test_dangerous_flag():
    """应支持 dangerous 标记"""
    @tool(name="dangerous_tool", description="危险操作", dangerous=True)
    async def dangerous_tool(path: str) -> str:
        return "deleted"
    
    wrapper: ToolWrapper = dangerous_tool._tool_wrapper
    assert wrapper.dangerous is True
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_tool_decorator.py -v
```

预期：ImportError

- [ ] **Step 3: 实现 schema 生成**

```python
# src/agent_framework/tools/schema.py
"""Python 函数签名 → JSON Schema 转换"""
from __future__ import annotations

import inspect
import re
from typing import get_type_hints, get_origin, get_args, Literal, Optional


def python_type_to_json_schema(py_type) -> dict:
    """将 Python 类型注解转为 JSON Schema 类型"""
    # 基础类型
    if py_type == str:
        return {"type": "string"}
    elif py_type == int:
        return {"type": "integer"}
    elif py_type == float:
        return {"type": "number"}
    elif py_type == bool:
        return {"type": "boolean"}
    
    # 泛型类型
    origin = get_origin(py_type)
    args = get_args(py_type)
    
    # Optional[X] -> Union[X, None]
    if origin is type(None) or (origin and str(origin) == "typing.Union" and type(None) in args):
        # 取非 None 的那个类型
        non_none_args = [a for a in args if a is not type(None)]
        if non_none_args:
            return python_type_to_json_schema(non_none_args[0])
        return {"type": "string"}
    
    # Literal["a", "b"]
    if origin is Literal:
        return {"type": "string", "enum": list(args)}
    
    # list[X]
    if origin is list:
        return {"type": "array"}
    
    # dict
    if origin is dict or py_type == dict:
        return {"type": "object"}
    
    # 默认
    return {"type": "string"}


def extract_param_descriptions(docstring: str | None) -> dict[str, str]:
    """从 docstring 提取参数描述（:param name: description 格式）"""
    if not docstring:
        return {}
    
    descriptions = {}
    # 匹配 :param name: description
    pattern = r":param\s+(\w+):\s*(.+?)(?=\n\s*:|\n\s*$|$)"
    matches = re.findall(pattern, docstring, re.DOTALL)
    
    for name, desc in matches:
        descriptions[name] = desc.strip()
    
    return descriptions


def generate_schema_from_function(func) -> dict:
    """从函数签名生成 OpenAI function calling schema"""
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    param_descriptions = extract_param_descriptions(func.__doc__)
    
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        # 跳过 self/cls
        if param_name in ("self", "cls"):
            continue
        
        # 获取类型
        py_type = type_hints.get(param_name, str)
        json_schema = python_type_to_json_schema(py_type)
        
        # 添加描述
        if param_name in param_descriptions:
            json_schema["description"] = param_descriptions[param_name]
        else:
            json_schema["description"] = param_name
        
        # 添加默认值
        if param.default != inspect.Parameter.empty:
            json_schema["default"] = param.default
        else:
            # 没有默认值且不是 Optional，加入 required
            is_optional = get_origin(py_type) and type(None) in get_args(py_type)
            if not is_optional:
                required.append(param_name)
        
        properties[param_name] = json_schema
    
    schema = {
        "type": "object",
        "properties": properties,
    }
    
    if required:
        schema["required"] = required
    
    return schema
```

- [ ] **Step 4: 实现 @tool 装饰器**

```python
# src/agent_framework/tools/decorator.py
"""@tool 装饰器实现"""
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Callable

from agent_framework.tools.schema import generate_schema_from_function


@dataclass
class ToolWrapper:
    """包装一个 @tool 函数，持有其元数据和 schema"""
    name: str
    description: str
    parameters_schema: dict
    func: Callable  # 原始 async/sync 函数
    is_async: bool  # 是否异步函数
    dangerous: bool  # 是否标记为危险操作


def tool(name: str, description: str, dangerous: bool = False):
    """
    装饰器：将函数标记为 Agent 可调用的工具。
    
    Args:
        name: 工具名称（OpenAI function name）
        description: 工具描述
        dangerous: 是否标记为危险操作（预留）
    
    Returns:
        装饰后的函数，附加 _tool_wrapper 属性
    """
    def decorator(func: Callable) -> Callable:
        # 生成 schema
        parameters_schema = generate_schema_from_function(func)
        
        # 判断是否异步
        is_async = asyncio.iscoroutinefunction(func)
        
        # 创建 wrapper
        wrapper = ToolWrapper(
            name=name,
            description=description,
            parameters_schema=parameters_schema,
            func=func,
            is_async=is_async,
            dangerous=dangerous,
        )
        
        # 附加到函数
        func._tool_wrapper = wrapper
        
        return func
    
    return decorator
```

- [ ] **Step 5: 创建 tools/__init__.py**

```python
# src/agent_framework/tools/__init__.py
"""工具系统 - @tool 装饰器和 ToolRegistry"""
from agent_framework.tools.decorator import tool, ToolWrapper

__all__ = ["tool", "ToolWrapper"]
```

- [ ] **Step 6: 运行测试验证通过**

```bash
pytest tests/test_tool_decorator.py -v
```

预期：9 个测试全部 PASS

- [ ] **Step 7: 提交**

```bash
git add src/agent_framework/tools/ tests/test_tool_decorator.py
git commit -m "feat: add @tool decorator and schema generation

- @tool decorator for marking functions as tools
- Automatic JSON schema generation from type hints
- Support for Optional, Literal, list types
- Docstring :param extraction for descriptions
- Sync and async function support
- dangerous flag for marking risky operations"
```

---

## Task 4: ToolRegistry

**Files:**
- Create: `src/agent_framework/tools/registry.py`
- Modify: `src/agent_framework/tools/__init__.py`
- Test: `tests/test_tool_registry.py` (新建)

- [ ] **Step 1: 写失败测试**

```python
# tests/test_tool_registry.py
"""测试 ToolRegistry"""
import pytest
from agent_framework.tools import tool, ToolRegistry


def test_register_single_tool():
    """应能注册单个工具"""
    @tool(name="get_weather", description="获取天气")
    async def get_weather(city: str) -> str:
        return "晴"
    
    registry = ToolRegistry()
    registry.register(get_weather)
    
    assert "get_weather" in registry.list_tools()


def test_register_many_tools():
    """应能批量注册工具"""
    @tool(name="tool1", description="工具1")
    async def tool1() -> str:
        return "1"
    
    @tool(name="tool2", description="工具2")
    async def tool2() -> str:
        return "2"
    
    registry = ToolRegistry()
    registry.register_many([tool1, tool2])
    
    assert "tool1" in registry.list_tools()
    assert "tool2" in registry.list_tools()


def test_get_tool():
    """应能根据名称获取工具"""
    @tool(name="my_tool", description="我的工具")
    async def my_tool() -> str:
        return "ok"
    
    registry = ToolRegistry()
    registry.register(my_tool)
    
    wrapper = registry.get_tool("my_tool")
    assert wrapper is not None
    assert wrapper.name == "my_tool"
    
    # 不存在的工具
    assert registry.get_tool("nonexistent") is None


def test_get_schemas():
    """应返回所有工具的 OpenAI schema"""
    @tool(name="get_weather", description="获取天气")
    async def get_weather(city: str) -> str:
        return "晴"
    
    @tool(name="calculator", description="计算器")
    async def calculator(expression: str) -> str:
        return "result"
    
    registry = ToolRegistry()
    registry.register_many([get_weather, calculator])
    
    schemas = registry.get_schemas()
    
    assert len(schemas) == 2
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "get_weather"
    assert schemas[1]["function"]["name"] == "calculator"


def test_duplicate_registration():
    """重复注册应覆盖"""
    @tool(name="my_tool", description="版本1")
    async def my_tool_v1() -> str:
        return "v1"
    
    @tool(name="my_tool", description="版本2")
    async def my_tool_v2() -> str:
        return "v2"
    
    registry = ToolRegistry()
    registry.register(my_tool_v1)
    registry.register(my_tool_v2)
    
    wrapper = registry.get_tool("my_tool")
    assert wrapper.description == "版本2"
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_tool_registry.py -v
```

预期：ImportError

- [ ] **Step 3: 实现 ToolRegistry**

```python
# src/agent_framework/tools/registry.py
"""工具注册表"""
from __future__ import annotations

from agent_framework.tools.decorator import ToolWrapper


class ToolRegistry:
    """工具注册表 - 管理 @tool 装饰过的函数"""
    
    def __init__(self):
        self._tools: dict[str, ToolWrapper] = {}
    
    def register(self, func) -> None:
        """注册一个 @tool 装饰的函数"""
        if not hasattr(func, "_tool_wrapper"):
            raise ValueError(f"函数 {func.__name__} 未被 @tool 装饰")
        
        wrapper: ToolWrapper = func._tool_wrapper
        self._tools[wrapper.name] = wrapper
    
    def register_many(self, funcs: list) -> None:
        """批量注册"""
        for func in funcs:
            self.register(func)
    
    def get_schemas(self) -> list[dict]:
        """
        返回所有工具的 OpenAI function calling schema 列表。
        可直接传给 llm.chat(tools=...) 使用。
        """
        schemas = []
        for wrapper in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": wrapper.name,
                    "description": wrapper.description,
                    "parameters": wrapper.parameters_schema,
                },
            })
        return schemas
    
    def get_tool(self, name: str) -> ToolWrapper | None:
        """根据名称获取工具"""
        return self._tools.get(name)
    
    def list_tools(self) -> list[str]:
        """列出所有已注册的工具名"""
        return list(self._tools.keys())
```

- [ ] **Step 4: 更新 tools/__init__.py**

```python
# src/agent_framework/tools/__init__.py
"""工具系统 - @tool 装饰器和 ToolRegistry"""
from agent_framework.tools.decorator import tool, ToolWrapper
from agent_framework.tools.registry import ToolRegistry

__all__ = ["tool", "ToolWrapper", "ToolRegistry"]
```

- [ ] **Step 5: 运行测试验证通过**

```bash
pytest tests/test_tool_registry.py -v
```

预期：5 个测试全部 PASS

- [ ] **Step 6: 提交**

```bash
git add src/agent_framework/tools/registry.py src/agent_framework/tools/__init__.py tests/test_tool_registry.py
git commit -m "feat: add ToolRegistry

- register/register_many for adding tools
- get_schemas for OpenAI function calling format
- get_tool/list_tools for querying
- Duplicate registration overwrites previous"
```

---

## Task 5: ConversationHistory

**Files:**
- Create: `src/agent_framework/agent/history.py`
- Create: `src/agent_framework/agent/__init__.py`
- Test: `tests/test_history.py` (新建)

- [ ] **Step 1: 写失败测试**

```python
# tests/test_history.py
"""测试 ConversationHistory"""
import pytest
from agent_framework.models.message import Message, MessageRole
from agent_framework.agent.history import ConversationHistory


def test_add_and_get_messages():
    """应能添加和获取消息"""
    history = ConversationHistory()
    history.set_system(Message(role=MessageRole.SYSTEM, content="你是助手"))
    history.add(Message(role=MessageRole.USER, content="你好"))
    
    messages = history.get_messages()
    assert len(messages) == 2
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[1].content == "你好"


def test_sliding_window_strategy():
    """滑动窗口应保留最近 N 条消息"""
    history = ConversationHistory(strategy="sliding_window", max_turns=4)
    history.set_system(Message(role=MessageRole.SYSTEM, content="系统"))
    
    # 添加 6 条消息（超过 max_turns=4）
    for i in range(6):
        history.add(Message(role=MessageRole.USER, content=f"消息{i}"))
    
    messages = history.get_messages()
    # 应保留 system + 最近 4 条
    assert len(messages) == 5
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[1].content == "消息2"  # 前 2 条被截断
    assert messages[4].content == "消息5"


def test_none_strategy():
    """none 策略应保留全部消息"""
    history = ConversationHistory(strategy="none")
    history.set_system(Message(role=MessageRole.SYSTEM, content="系统"))
    
    for i in range(20):
        history.add(Message(role=MessageRole.USER, content=f"消息{i}"))
    
    messages = history.get_messages()
    assert len(messages) == 21  # system + 20 条


def test_token_limit_strategy():
    """token_limit 应按 token 数截断"""
    history = ConversationHistory(strategy="token_limit", max_tokens=50)
    history.set_system(Message(role=MessageRole.SYSTEM, content="系统"))
    
    # 添加长消息
    history.add(Message(role=MessageRole.USER, content="这是一条很长的消息" * 10))
    history.add(Message(role=MessageRole.ASSISTANT, content="回复" * 10))
    history.add(Message(role=MessageRole.USER, content="短消息"))
    
    messages = history.get_messages()
    # 应保留 system + 最近的短消息
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[-1].content == "短消息"


def test_head_tail_strategy():
    """head_tail 应保留头尾，中间截断"""
    history = ConversationHistory(
        strategy="head_tail",
        head_turns=2,
        tail_turns=2
    )
    history.set_system(Message(role=MessageRole.SYSTEM, content="系统"))
    
    for i in range(10):
        history.add(Message(role=MessageRole.USER, content=f"消息{i}"))
    
    messages = history.get_messages()
    # 应保留 system + 前 2 条 + "..." + 后 2 条
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[1].content == "消息0"
    assert messages[2].content == "消息1"
    assert "..." in messages[3].content  # 截断提示
    assert messages[-2].content == "消息8"
    assert messages[-1].content == "消息9"


def test_clear():
    """clear 应清空历史但保留 system"""
    history = ConversationHistory()
    history.set_system(Message(role=MessageRole.SYSTEM, content="系统"))
    history.add(Message(role=MessageRole.USER, content="你好"))
    
    history.clear()
    
    messages = history.get_messages()
    assert len(messages) == 1
    assert messages[0].role == MessageRole.SYSTEM


def test_all_messages_property():
    """all_messages 应返回完整未截断的历史"""
    history = ConversationHistory(strategy="sliding_window", max_turns=2)
    history.set_system(Message(role=MessageRole.SYSTEM, content="系统"))
    
    for i in range(5):
        history.add(Message(role=MessageRole.USER, content=f"消息{i}"))
    
    # get_messages 返回截断后的
    truncated = history.get_messages()
    assert len(truncated) == 3  # system + 2
    
    # all_messages 返回完整的
    full = history.all_messages
    assert len(full) == 6  # system + 5
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_history.py -v
```

预期：ImportError

- [ ] **Step 3: 实现 ConversationHistory**

```python
# src/agent_framework/agent/history.py
"""对话历史管理"""
from __future__ import annotations

import re

from agent_framework.models.message import Message, MessageRole


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数：中文约 1.5 字/token，英文约 4 字符/token"""
    if not isinstance(text, str):
        return 0
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


class ConversationHistory:
    """对话历史管理 - 支持多种截断策略"""
    
    def __init__(
        self,
        strategy: str = "sliding_window",
        max_turns: int = 50,
        max_tokens: int | None = None,
        preserve_system: bool = True,
        head_turns: int = 4,
        tail_turns: int = 10,
    ):
        self._messages: list[Message] = []
        self._strategy = strategy
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._preserve_system = preserve_system
        self._head_turns = head_turns
        self._tail_turns = tail_turns
    
    def set_system(self, message: Message) -> None:
        """设置 system 消息（始终作为第一条）"""
        if self._messages and self._messages[0].role == MessageRole.SYSTEM:
            # 替换现有的 system 消息
            self._messages[0] = message
        else:
            # 插入到开头
            self._messages.insert(0, message)
    
    def add(self, message: Message) -> None:
        """追加一条消息"""
        self._messages.append(message)
    
    def get_messages(self) -> list[Message]:
        """获取截断后的消息列表（用于传给 LLM）"""
        if self._strategy == "none":
            return list(self._messages)
        elif self._strategy == "sliding_window":
            return self._apply_sliding_window()
        elif self._strategy == "token_limit":
            return self._apply_token_limit()
        elif self._strategy == "head_tail":
            return self._apply_head_tail()
        else:
            return list(self._messages)
    
    def clear(self) -> None:
        """清空历史（保留 system 消息）"""
        if self._preserve_system and self._messages and self._messages[0].role == MessageRole.SYSTEM:
            self._messages = [self._messages[0]]
        else:
            self._messages = []
    
    @property
    def all_messages(self) -> list[Message]:
        """获取完整未截断的历史（用于调试/日志）"""
        return list(self._messages)
    
    def _apply_sliding_window(self) -> list[Message]:
        """滑动窗口截断"""
        if len(self._messages) <= self._max_turns:
            return list(self._messages)
        
        # 保留 system 消息
        if self._preserve_system and self._messages and self._messages[0].role == MessageRole.SYSTEM:
            system = self._messages[0]
            non_system = self._messages[1:]
            kept = non_system[-(self._max_turns - 1):]
            return [system] + kept
        else:
            return self._messages[-self._max_turns:]
    
    def _apply_token_limit(self) -> list[Message]:
        """按 token 数截断"""
        if not self._max_tokens:
            return list(self._messages)
        
        # 从后往前累加，直到达到上限
        total_tokens = 0
        keep_from_idx = len(self._messages)
        
        for i in range(len(self._messages) - 1, -1, -1):
            msg = self._messages[i]
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            tokens = _estimate_tokens(content)
            
            if total_tokens + tokens > self._max_tokens:
                break
            
            total_tokens += tokens
            keep_from_idx = i
        
        # 保留 system 消息
        if self._preserve_system and self._messages and self._messages[0].role == MessageRole.SYSTEM:
            if keep_from_idx == 0:
                return list(self._messages)
            else:
                return [self._messages[0]] + self._messages[max(1, keep_from_idx):]
        else:
            return self._messages[keep_from_idx:]
    
    def _apply_head_tail(self) -> list[Message]:
        """头尾截断"""
        if len(self._messages) <= self._head_turns + self._tail_turns:
            return list(self._messages)
        
        head = self._messages[:self._head_turns]
        tail = self._messages[-self._tail_turns:]
        
        # 插入截断提示
        separator = Message(
            role=MessageRole.SYSTEM,
            content="... [中间消息已省略] ..."
        )
        
        return head + [separator] + tail
```

- [ ] **Step 4: 创建 agent/__init__.py**

```python
# src/agent_framework/agent/__init__.py
"""Agent 核心模块"""
from agent_framework.agent.history import ConversationHistory

__all__ = ["ConversationHistory"]
```

- [ ] **Step 5: 运行测试验证通过**

```bash
pytest tests/test_history.py -v
```

预期：7 个测试全部 PASS

- [ ] **Step 6: 提交**

```bash
git add src/agent_framework/agent/ tests/test_history.py
git commit -m "feat: add ConversationHistory with truncation strategies

- 4 strategies: none, sliding_window, token_limit, head_tail
- System message preservation
- Rough token estimation (Chinese + English)
- clear() to reset history
- all_messages property for full history access"
```

---

## Task 6: AgentConfig

**Files:**
- Create: `src/agent_framework/agent/config.py`
- Test: `tests/test_agent_config.py` (新建)

- [ ] **Step 1: 写失败测试**

```python
# tests/test_agent_config.py
"""测试 AgentConfig"""
import pytest
from agent_framework.agent.config import AgentConfig


def test_default_config():
    """应有合理的默认值"""
    config = AgentConfig()
    assert config.max_turns == 10
    assert config.timeout == 120.0
    assert config.total_timeout == 300.0
    assert config.max_total_tokens is None
    assert config.tool_call_limit == 20
    assert config.confirm_dangerous is False


def test_custom_config():
    """应能自定义配置"""
    config = AgentConfig(
        max_turns=5,
        timeout=60.0,
        total_timeout=180.0,
        max_total_tokens=10000,
        tool_call_limit=10,
        confirm_dangerous=True,
    )
    assert config.max_turns == 5
    assert config.timeout == 60.0
    assert config.total_timeout == 180.0
    assert config.max_total_tokens == 10000
    assert config.tool_call_limit == 10
    assert config.confirm_dangerous is True
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_agent_config.py -v
```

预期：ImportError

- [ ] **Step 3: 实现 AgentConfig**

```python
# src/agent_framework/agent/config.py
"""Agent 配置"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Agent 运行配置"""
    max_turns: int = 10  # 最大循环轮次（防无限循环）
    timeout: float = 120.0  # 单次 LLM 调用超时（秒）
    total_timeout: float = 300.0  # 整个 run() 的总超时（秒）
    max_total_tokens: int | None = None  # 总 token 上限（None=不限制）
    tool_call_limit: int = 20  # 单次 run() 中最大工具调用次数
    confirm_dangerous: bool = False  # 危险工具调用前需确认（预留接口）
```

- [ ] **Step 4: 更新 agent/__init__.py**

```python
# src/agent_framework/agent/__init__.py
"""Agent 核心模块"""
from agent_framework.agent.config import AgentConfig
from agent_framework.agent.history import ConversationHistory

__all__ = ["AgentConfig", "ConversationHistory"]
```

- [ ] **Step 5: 运行测试验证通过**

```bash
pytest tests/test_agent_config.py -v
```

预期：2 个测试全部 PASS

- [ ] **Step 6: 提交**

```bash
git add src/agent_framework/agent/config.py src/agent_framework/agent/__init__.py tests/test_agent_config.py
git commit -m "feat: add AgentConfig

- max_turns: loop iteration limit
- timeout: per-call timeout
- total_timeout: total run timeout
- max_total_tokens: token usage limit
- tool_call_limit: tool invocation limit
- confirm_dangerous: dangerous tool confirmation flag"
```

---

## Task 7: ToolExecutor

**Files:**
- Create: `src/agent_framework/agent/executor.py`
- Test: `tests/test_executor.py` (新建)

- [ ] **Step 1: 写失败测试**

```python
# tests/test_executor.py
"""测试 ToolExecutor"""
import pytest
from agent_framework.tools import tool, ToolRegistry
from agent_framework.agent.config import AgentConfig
from agent_framework.agent.executor import ToolExecutor, ToolExecutionResult


@pytest.fixture
def registry():
    """创建带测试工具的注册表"""
    @tool(name="add", description="加法")
    async def add(a: int, b: int) -> str:
        return str(a + b)
    
    @tool(name="fail", description="会失败的工具")
    async def fail() -> str:
        raise ValueError("故意失败")
    
    @tool(name="sync_tool", description="同步工具")
    def sync_tool(x: int) -> str:
        return str(x * 2)
    
    reg = ToolRegistry()
    reg.register_many([add, fail, sync_tool])
    return reg


@pytest.mark.asyncio
async def test_execute_success(registry):
    """应成功执行工具"""
    executor = ToolExecutor(registry, AgentConfig())
    
    tool_call = {
        "id": "call_1",
        "function": {
            "name": "add",
            "arguments": '{"a": 5, "b": 3}'
        }
    }
    
    result = await executor.execute(tool_call)
    assert result.output == "8"
    assert result.is_error is False


@pytest.mark.asyncio
async def test_execute_sync_function(registry):
    """应支持同步函数"""
    executor = ToolExecutor(registry, AgentConfig())
    
    tool_call = {
        "id": "call_2",
        "function": {
            "name": "sync_tool",
            "arguments": '{"x": 10}'
        }
    }
    
    result = await executor.execute(tool_call)
    assert result.output == "20"
    assert result.is_error is False


@pytest.mark.asyncio
async def test_execute_tool_not_found(registry):
    """工具不存在应返回错误"""
    executor = ToolExecutor(registry, AgentConfig())
    
    tool_call = {
        "id": "call_3",
        "function": {
            "name": "nonexistent",
            "arguments": "{}"
        }
    }
    
    result = await executor.execute(tool_call)
    assert result.is_error is True
    assert "工具不存在" in result.output


@pytest.mark.asyncio
async def test_execute_invalid_json(registry):
    """参数 JSON 解析失败应返回错误"""
    executor = ToolExecutor(registry, AgentConfig())
    
    tool_call = {
        "id": "call_4",
        "function": {
            "name": "add",
            "arguments": "invalid json"
        }
    }
    
    result = await executor.execute(tool_call)
    assert result.is_error is True
    assert "参数解析失败" in result.output


@pytest.mark.asyncio
async def test_execute_exception_handling(registry):
    """工具执行异常应返回错误而非抛出"""
    executor = ToolExecutor(registry, AgentConfig())
    
    tool_call = {
        "id": "call_5",
        "function": {
            "name": "fail",
            "arguments": "{}"
        }
    }
    
    result = await executor.execute(tool_call)
    assert result.is_error is True
    assert "执行异常" in result.output
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_executor.py -v
```

预期：ImportError

- [ ] **Step 3: 实现 ToolExecutor**

```python
# src/agent_framework/agent/executor.py
"""工具执行器"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

from agent_framework.tools.registry import ToolRegistry
from agent_framework.agent.config import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionResult:
    """工具执行结果"""
    output: str  # 工具输出（始终为字符串，回填给 LLM）
    is_error: bool  # 是否为错误结果


class ToolExecutor:
    """工具执行器 - 负责安全地执行工具并处理异常"""
    
    def __init__(self, registry: ToolRegistry, config: AgentConfig):
        self._registry = registry
        self._config = config
    
    async def execute(self, tool_call: dict) -> ToolExecutionResult:
        """
        执行一个工具调用，异常不中断循环。
        
        Args:
            tool_call: OpenAI 格式的工具调用字典
                {"id": "call_xxx", "function": {"name": "...", "arguments": "..."}}
        
        Returns:
            ToolExecutionResult(output=结果文本, is_error=是否出错)
        """
        # 1. 提取工具名和参数
        func_name = tool_call.get("function", {}).get("name", "")
        arguments_str = tool_call.get("function", {}).get("arguments", "{}")
        
        # 2. 查找工具
        wrapper = self._registry.get_tool(func_name)
        if not wrapper:
            return ToolExecutionResult(
                output=f"工具不存在: {func_name}",
                is_error=True
            )
        
        # 3. 解析参数 JSON
        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError as e:
            return ToolExecutionResult(
                output=f"参数解析失败: {e}",
                is_error=True
            )
        
        # 4. 执行工具函数
        try:
            if wrapper.is_async:
                result = await wrapper.func(**arguments)
            else:
                # 同步函数在线程池中执行，避免阻塞事件循环
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: wrapper.func(**arguments)
                )
            
            # 5. 转换为字符串
            return ToolExecutionResult(
                output=str(result),
                is_error=False
            )
        
        except Exception as e:
            logger.warning(f"工具 {func_name} 执行异常: {e}")
            return ToolExecutionResult(
                output=f"执行异常: {e}",
                is_error=True
            )
```

- [ ] **Step 4: 更新 agent/__init__.py**

```python
# src/agent_framework/agent/__init__.py
"""Agent 核心模块"""
from agent_framework.agent.config import AgentConfig
from agent_framework.agent.executor import ToolExecutor, ToolExecutionResult
from agent_framework.agent.history import ConversationHistory

__all__ = [
    "AgentConfig",
    "ConversationHistory",
    "ToolExecutor",
    "ToolExecutionResult",
]
```

- [ ] **Step 5: 运行测试验证通过**

```bash
pytest tests/test_executor.py -v
```

预期：5 个测试全部 PASS

- [ ] **Step 6: 提交**

```bash
git add src/agent_framework/agent/executor.py src/agent_framework/agent/__init__.py tests/test_executor.py
git commit -m "feat: add ToolExecutor

- Safe tool execution with error handling
- Async and sync function support
- JSON argument parsing with error recovery
- Exception handling returns error result instead of raising
- Sync functions run in thread pool to avoid blocking"
```

---

## Task 8: Agent 编排器（核心循环）

**Files:**
- Create: `src/agent_framework/agent/agent.py`
- Modify: `src/agent_framework/agent/__init__.py`
- Test: `tests/test_agent.py` (新建)

- [ ] **Step 1: 写失败测试**

```python
# tests/test_agent.py
"""测试 Agent 编排器"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from agent_framework.agent import Agent, AgentConfig
from agent_framework.models.events import TextDelta, ToolCallStart, ToolResult, AgentDone
from agent_framework.models.message import MessageRole
from agent_framework.models.response import StreamChunk, TokenUsage
from agent_framework.tools import tool


@pytest.fixture
def mock_llm():
    """创建 mock LLM"""
    llm = AsyncMock()
    llm.chat = AsyncMock()
    return llm


@pytest.fixture
def simple_tool():
    """创建简单测试工具"""
    @tool(name="get_time", description="获取时间")
    async def get_time() -> str:
        return "2026-05-23 10:00"
    return get_time


@pytest.mark.asyncio
async def test_simple_text_response(mock_llm):
    """无工具调用时应直接返回文本"""
    # Mock LLM 返回文本
    async def mock_chat(*args, **kwargs):
        yield StreamChunk(content="你好", finish_reason="stop")
        yield StreamChunk(
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        )
    
    mock_llm.chat = mock_chat
    
    agent = Agent(llm=mock_llm, system_prompt="你是助手")
    
    events = []
    async for event in agent.run("你好"):
        events.append(event)
    
    # 应有 TextDelta 和 AgentDone
    assert any(isinstance(e, TextDelta) for e in events)
    assert any(isinstance(e, AgentDone) for e in events)
    
    done_event = next(e for e in events if isinstance(e, AgentDone))
    assert done_event.final_text == "你好"
    assert done_event.turn_count == 1


@pytest.mark.asyncio
async def test_tool_call_flow(mock_llm, simple_tool):
    """应正确处理工具调用流程"""
    call_count = 0
    
    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        
        if call_count == 1:
            # 第一次调用：返回工具调用
            yield StreamChunk(
                tool_calls=[
                    type('obj', (), {
                        'index': 0,
                        'id': 'call_1',
                        'function': type('obj', (), {
                            'name': 'get_time',
                            'arguments': '{}'
                        })()
                    })()
                ]
            )
            yield StreamChunk(finish_reason="tool_calls")
        else:
            # 第二次调用：返回文本
            yield StreamChunk(content="当前时间是 2026-05-23 10:00", finish_reason="stop")
            yield StreamChunk(
                usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30)
            )
    
    mock_llm.chat = mock_chat
    
    agent = Agent(
        llm=mock_llm,
        system_prompt="你是助手",
        tools=[simple_tool]
    )
    
    events = []
    async for event in agent.run("现在几点了？"):
        events.append(event)
    
    # 应有 ToolCallStart、ToolResult、TextDelta、AgentDone
    assert any(isinstance(e, ToolCallStart) for e in events)
    assert any(isinstance(e, ToolResult) for e in events)
    assert any(isinstance(e, TextDelta) for e in events)
    assert any(isinstance(e, AgentDone) for e in events)
    
    tool_start = next(e for e in events if isinstance(e, ToolCallStart))
    assert tool_start.tool_name == "get_time"
    
    tool_result = next(e for e in events if isinstance(e, ToolResult))
    assert tool_result.output == "2026-05-23 10:00"
    assert tool_result.is_error is False


@pytest.mark.asyncio
async def test_max_turns_limit(mock_llm, simple_tool):
    """超出最大轮次应返回 AgentError"""
    from agent_framework.models.events import AgentError
    
    async def mock_chat(*args, **kwargs):
        # 始终返回工具调用，触发无限循环
        yield StreamChunk(
            tool_calls=[
                type('obj', (), {
                    'index': 0,
                    'id': 'call_x',
                    'function': type('obj', (), {
                        'name': 'get_time',
                        'arguments': '{}'
                    })()
                })()
            ]
        )
    
    mock_llm.chat = mock_chat
    
    agent = Agent(
        llm=mock_llm,
        tools=[simple_tool],
        config=AgentConfig(max_turns=3)
    )
    
    events = []
    async for event in agent.run("测试"):
        events.append(event)
    
    # 应有 AgentError
    assert any(isinstance(e, AgentError) for e in events)
    error = next(e for e in events if isinstance(e, AgentError))
    assert error.error_type == "max_turns"


@pytest.mark.asyncio
async def test_conversation_history():
    """多轮对话应保留历史"""
    call_count = 0
    
    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        yield StreamChunk(content=f"回复{call_count}", finish_reason="stop")
    
    llm = AsyncMock()
    llm.chat = mock_chat
    
    agent = Agent(llm=llm, system_prompt="你是助手")
    
    # 第一轮
    async for event in agent.run("问题1"):
        pass
    
    # 第二轮
    async for event in agent.run("问题2"):
        pass
    
    # 第三轮
    async for event in agent.run("问题3"):
        pass
    
    # history 应有 system + 3 轮（每轮 user + assistant）
    messages = agent.history.all_messages
    assert messages[0].role == MessageRole.SYSTEM
    assert len(messages) == 7  # system + 3*(user + assistant)


@pytest.mark.asyncio
async def test_reset():
    """reset 应清空历史但保留 system"""
    async def mock_chat(*args, **kwargs):
        yield StreamChunk(content="回复", finish_reason="stop")
    
    llm = AsyncMock()
    llm.chat = mock_chat
    
    agent = Agent(llm=llm, system_prompt="你是助手")
    
    async for event in agent.run("你好"):
        pass
    
    assert len(agent.history.all_messages) == 3  # system + user + assistant
    
    await agent.reset()
    
    assert len(agent.history.all_messages) == 1  # 仅 system
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_agent.py -v
```

预期：ImportError

- [ ] **Step 3: 实现 Agent 编排器**

由于代码较长，分两部分实现。先创建基础结构：

```python
# src/agent_framework/agent/agent.py
"""Agent 编排器"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator

from agent_framework.agent.config import AgentConfig
from agent_framework.agent.executor import ToolExecutor
from agent_framework.agent.history import ConversationHistory
from agent_framework.models.events import (
    AgentDone,
    AgentError,
    AgentEvent,
    ReasoningDelta,
    TextDelta,
    ToolCallStart,
    ToolResult,
)
from agent_framework.models.message import Message, MessageRole
from agent_framework.models.response import TokenUsage
from agent_framework.providers.base import BaseLLM
from agent_framework.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _merge_tool_calls(buffer: list[dict], new_calls: list) -> None:
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


def _merge_usage(total: TokenUsage, new: TokenUsage) -> TokenUsage:
    """累加 token 使用量"""
    return TokenUsage(
        prompt_tokens=total.prompt_tokens + new.prompt_tokens,
        completion_tokens=total.completion_tokens + new.completion_tokens,
        total_tokens=total.total_tokens + new.total_tokens,
        reasoning_tokens=total.reasoning_tokens + new.reasoning_tokens,
    )


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
        
        # 初始化 system 消息
        self._history.set_system(
            Message(role=MessageRole.SYSTEM, content=system_prompt)
        )
    
    async def run(self, user_input: str, **llm_kwargs) -> AsyncIterator[AgentEvent]:
        """
        执行一次完整的 Agent 对话。
        
        Args:
            user_input: 用户输入文本
            **llm_kwargs: 传递给 LLM 的额外参数（temperature 等）
        
        Yields:
            AgentEvent 事件流
        """
        # 1. 追加用户消息
        self._history.add(Message(role=MessageRole.USER, content=user_input))
        
        turn_count = 0
        total_usage = TokenUsage()
        tool_call_count = 0
        start_time = time.monotonic()
        
        while turn_count < self._config.max_turns:
            # 安全检查：总超时
            if time.monotonic() - start_time > self._config.total_timeout:
                yield AgentError("total_timeout", f"总超时({self._config.total_timeout}s)")
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
                yield AgentError("call_timeout", f"LLM调用超时({self._config.timeout}s)")
                return
            
            # 4. 累计 usage
            if usage:
                total_usage = _merge_usage(total_usage, usage)
            
            # 安全检查：token 上限
            if (self._config.max_total_tokens
                    and total_usage.total_tokens > self._config.max_total_tokens):
                yield AgentError("token_limit", f"超出token上限({self._config.max_total_tokens})")
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
                    yield AgentError("tool_limit", f"工具调用次数超限({self._config.tool_call_limit})")
                    return
                
                func_name = tc["function"]["name"]
                tc_id = tc.get("id", "")
                arguments = tc["function"]["arguments"]
                
                yield ToolCallStart(
                    tool_name=func_name,
                    tool_call_id=tc_id,
                    arguments=arguments,
                )
                
                result = await self._executor.execute(tc)
                
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
    
    async def reset(self) -> None:
        """重置对话历史（保留 system prompt）"""
        self._history.clear()
    
    @property
    def history(self) -> ConversationHistory:
        """访问对话历史"""
        return self._history
    
    @property
    def tools(self) -> ToolRegistry:
        """访问工具注册表"""
        return self._registry
```

- [ ] **Step 4: 更新 agent/__init__.py**

```python
# src/agent_framework/agent/__init__.py
"""Agent 核心模块"""
from agent_framework.agent.agent import Agent
from agent_framework.agent.config import AgentConfig
from agent_framework.agent.executor import ToolExecutor, ToolExecutionResult
from agent_framework.agent.history import ConversationHistory

__all__ = [
    "Agent",
    "AgentConfig",
    "ConversationHistory",
    "ToolExecutor",
    "ToolExecutionResult",
]
```

- [ ] **Step 5: 运行测试验证通过**

```bash
pytest tests/test_agent.py -v
```

预期：5 个测试全部 PASS

- [ ] **Step 6: 提交**

```bash
git add src/agent_framework/agent/ tests/test_agent.py
git commit -m "feat: add Agent orchestrator with core loop

- ReAct-style loop: reason → act → observe → continue
- Streaming event output (TextDelta, ToolCallStart, ToolResult, AgentDone)
- Multi-turn conversation with automatic history management
- Tool call merging for streaming API
- Safety checks: max_turns, timeout, token_limit, tool_call_limit
- reset() to clear conversation history"
```

---

## Task 9: 更新顶层导出

**Files:**
- Modify: `src/agent_framework/__init__.py`

- [ ] **Step 1: 更新 __init__.py**

在 `src/agent_framework/__init__.py` 追加导出：

```python
# Agent 核心
from agent_framework.agent import Agent, AgentConfig
from agent_framework.models.events import (
    AgentEvent,
    TextDelta,
    ReasoningDelta,
    ToolCallStart,
    ToolResult,
    AgentDone,
    AgentError,
)

__all__.extend([
    "Agent",
    "AgentConfig",
    "AgentEvent",
    "TextDelta",
    "ReasoningDelta",
    "ToolCallStart",
    "ToolResult",
    "AgentDone",
    "AgentError",
])
```

- [ ] **Step 2: 验证导入**

```bash
python -c "from agent_framework import Agent, AgentConfig, TextDelta, ToolCallStart; print('OK')"
```

预期：OK

- [ ] **Step 3: 提交**

```bash
git add src/agent_framework/__init__.py
git commit -m "feat: export Agent core components at top level

- Agent, AgentConfig
- AgentEvent types (TextDelta, ToolCallStart, etc.)"
```

---

## Task 10: 集成测试

**Files:**
- Create: `tests/test_agent_integration.py`

- [ ] **Step 1: 写集成测试**

```python
# tests/test_agent_integration.py
"""Agent 端到端集成测试（需要真实 API Key）"""
import asyncio
import os
import pytest

from agent_framework import Agent, AgentConfig, TextDelta, ToolCallStart, ToolResult, AgentDone
from agent_framework.providers import ModelRegistry
from agent_framework.tools import tool


# 跳过条件：没有 API Key
pytestmark = pytest.mark.skipif(
    not os.environ.get("QWEN_API_KEY"),
    reason="需要 QWEN_API_KEY 环境变量"
)


@tool(name="get_weather", description="获取指定城市的天气")
async def get_weather(city: str) -> str:
    """模拟天气工具"""
    weather_data = {
        "北京": "晴，25°C",
        "上海": "多云，23°C",
        "广州": "阵雨，28°C",
    }
    return weather_data.get(city, f"{city}：未知")


@tool(name="calculator", description="数学计算")
async def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


@pytest.mark.asyncio
async def test_agent_with_tools():
    """测试 Agent + 工具调用端到端流程"""
    llm = ModelRegistry.create("qwen", model="qwen-max")
    
    agent = Agent(
        llm=llm,
        system_prompt="你是一个智能助手，可以查天气和做计算。",
        tools=[get_weather, calculator],
        config=AgentConfig(max_turns=5, timeout=60),
    )
    
    events = []
    async for event in agent.run("北京天气怎么样？顺便算一下 17*23+45"):
        events.append(event)
    
    # 应有工具调用
    tool_starts = [e for e in events if isinstance(e, ToolCallStart)]
    assert len(tool_starts) >= 1
    
    # 应有天气工具调用
    weather_call = next((e for e in tool_starts if e.tool_name == "get_weather"), None)
    assert weather_call is not None
    assert "北京" in weather_call.arguments
    
    # 应有计算工具调用
    calc_call = next((e for e in tool_starts if e.tool_name == "calculator"), None)
    assert calc_call is not None
    
    # 应有最终回复
    done = next((e for e in events if isinstance(e, AgentDone)), None)
    assert done is not None
    assert len(done.final_text) > 0


@pytest.mark.asyncio
async def test_multi_turn_conversation():
    """测试多轮对话历史保留"""
    llm = ModelRegistry.create("qwen", model="qwen-max")
    
    agent = Agent(
        llm=llm,
        system_prompt="你是一个简洁的助手。",
        config=AgentConfig(max_turns=5, timeout=60),
    )
    
    # 第一轮
    events1 = []
    async for event in agent.run("我叫张三"):
        events1.append(event)
    
    # 第二轮
    events2 = []
    async for event in agent.run("我叫什么名字？"):
        events2.append(event)
    
    # 第二轮应能记住第一轮的信息
    done = next((e for e in events2 if isinstance(e, AgentDone)), None)
    assert done is not None
    # 模型应该能回答出"张三"
    assert "张三" in done.final_text or "您" in done.final_text


@pytest.mark.asyncio
async def test_tool_error_handling():
    """测试工具执行异常不中断循环"""
    @tool(name="failing_tool", description="会失败的工具")
    async def failing_tool() -> str:
        raise ValueError("故意失败")
    
    llm = ModelRegistry.create("qwen", model="qwen-max")
    
    agent = Agent(
        llm=llm,
        system_prompt="你是助手。",
        tools=[failing_tool],
        config=AgentConfig(max_turns=3, timeout=60),
    )
    
    events = []
    async for event in agent.run("请调用 failing_tool"):
        events.append(event)
    
    # 应有 ToolResult 且 is_error=True
    tool_results = [e for e in events if isinstance(e, ToolResult)]
    assert len(tool_results) >= 1
    assert any(r.is_error for r in tool_results)
    
    # 应有最终回复（模型处理错误后继续）
    done = next((e for e in events if isinstance(e, AgentDone)), None)
    assert done is not None
```

- [ ] **Step 2: 运行集成测试**

```bash
# 设置 API Key 后运行
pytest tests/test_agent_integration.py -v -s
```

预期：3 个测试全部 PASS（需要真实 API Key）

- [ ] **Step 3: 提交**

```bash
git add tests/test_agent_integration.py
git commit -m "test: add Agent end-to-end integration tests

- Test Agent + tools workflow
- Test multi-turn conversation memory
- Test tool error handling and recovery
- Requires QWEN_API_KEY for real API calls"
```

---

## Task 11: 运行全部测试

**Files:** 无

- [ ] **Step 1: 运行完整测试套件**

```bash
pytest tests/ -v
```

预期：所有测试 PASS（集成测试可能需要 API Key）

- [ ] **Step 2: 检查导入**

```bash
python -c "
from agent_framework import Agent, AgentConfig, TextDelta, ToolCallStart, ToolResult, AgentDone
from agent_framework.tools import tool, ToolRegistry
from agent_framework.agent import ConversationHistory, ToolExecutor
print('所有导入成功')
"
```

预期：所有导入成功

- [ ] **Step 3: 创建使用示例**

```python
# examples/agent_basic.py
"""基础 Agent 使用示例"""
import asyncio
import os

from agent_framework import Agent, AgentConfig, TextDelta, ToolCallStart, ToolResult, AgentDone
from agent_framework.providers import ModelRegistry
from agent_framework.tools import tool


# 定义工具
@tool(name="get_weather", description="获取天气")
async def get_weather(city: str) -> str:
    return f"{city}：晴，25°C"


@tool(name="calculator", description="数学计算")
async def calculator(expression: str) -> str:
    return str(eval(expression))


async def main():
    # 创建 Agent
    llm = ModelRegistry.create("qwen", model="qwen-max")
    agent = Agent(
        llm=llm,
        system_prompt="你是一个智能助手，可以查天气和做计算。",
        tools=[get_weather, calculator],
        config=AgentConfig(max_turns=5, timeout=60),
    )
    
    # 运行
    print("用户: 北京天气怎么样？顺便算一下 17*23+45")
    print("助手: ", end="", flush=True)
    
    async for event in agent.run("北京天气怎么样？顺便算一下 17*23+45"):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, ToolCallStart):
            print(f"\n🔧 调用: {event.tool_name}", flush=True)
        elif isinstance(event, ToolResult):
            print(f"📋 结果: {event.output}", flush=True)
        elif isinstance(event, AgentDone):
            print(f"\n✅ 完成 (tokens: {event.total_usage.total_tokens})")


if __name__ == "__main__":
    # 需要设置 QWEN_API_KEY 环境变量
    asyncio.run(main())
```

- [ ] **Step 4: 提交示例**

```bash
git add examples/agent_basic.py
git commit -m "docs: add basic Agent usage example"
```

---

## 总结

**完成标准：**
- ✅ 11 个 Task 全部完成
- ✅ 所有单元测试 PASS（Task 1-8）
- ✅ 集成测试 PASS（Task 10，需要 API Key）
- ✅ 完整导出可用（Task 9）
- ✅ 示例代码可运行（Task 11）

**预计工作量：** 每个 Task 约 10-15 分钟，总计约 2-3 小时

**下一步：** 第三步 - 工具调用与 MCP 集成（扩展 ToolRegistry 支持 MCP 协议）
