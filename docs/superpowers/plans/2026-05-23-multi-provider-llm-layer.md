# 多厂商模型兼容层实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建统一抽象层，兼容 Qwen/Kimi/GLM/DeepSeek/MiniMax/Doubao 六家厂商的模型调用，支持按能力动态暴露参数。

**Architecture:** 抽象基类 BaseLLM + ModelCapabilities 能力描述符模式。每个厂商继承 BaseLLM，声明自身能力，通过 OpenAI SDK 统一调用。ModelRegistry 工厂管理注册和实例化。

**Tech Stack:** Python 3.10+, openai SDK, pytest, pytest-asyncio, uv

---

## 文件结构

| 文件路径 | 职责 |
|---------|------|
| `pyproject.toml` | 项目配置，依赖管理 |
| `.env.example` | 环境变量模板 |
| `src/milu/__init__.py` | 顶层公共导出 |
| `src/milu/exceptions.py` | 统一异常体系 |
| `src/milu/models/__init__.py` | 数据模型导出 |
| `src/milu/models/message.py` | Message, MessageRole |
| `src/milu/models/response.py` | StreamChunk, TokenUsage |
| `src/milu/models/config.py` | ModelConfig 及各扩展配置类 |
| `src/milu/providers/__init__.py` | ModelRegistry + 厂商导出 |
| `src/milu/providers/base.py` | BaseLLM + ModelCapabilities |
| `src/milu/providers/qwen.py` | 通义千问实现 |
| `src/milu/providers/kimi.py` | Kimi 实现 |
| `src/milu/providers/glm.py` | 智谱GLM 实现 |
| `src/milu/providers/deepseek.py` | DeepSeek 实现 |
| `src/milu/providers/minimax.py` | MiniMax 实现 |
| `src/milu/providers/doubao.py` | 豆包 实现 |
| `tests/conftest.py` | pytest 公共 fixtures |
| `tests/test_models.py` | 数据模型和配置类测试 |
| `tests/test_base_provider.py` | BaseLLM 和 ModelCapabilities 测试 |
| `tests/test_registry.py` | ModelRegistry 测试 |
| `tests/test_qwen.py` | Qwen 实现测试 |
| `tests/test_kimi.py` | Kimi 实现测试 |
| `tests/test_glm.py` | GLM 实现测试 |
| `tests/test_deepseek.py` | DeepSeek 实现测试 |
| `tests/test_minimax.py` | MiniMax 实现测试 |
| `tests/test_doubao.py` | Doubao 实现测试 |
| `examples/basic_usage.py` | 基础使用示例 |

---

## Task 1: 项目脚手架与异常体系

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `src/milu/__init__.py`
- Create: `src/milu/exceptions.py`

- [ ] **Step 1: 创建项目目录结构**

```bash
mkdir -p src/milu/providers src/milu/models tests examples
```

- [ ] **Step 2: 创建 pyproject.toml**

```toml
[project]
name = "milu"
version = "0.1.0"
description = "统一AI模型抽象层，兼容多家厂商API"
requires-python = ">=3.10"
dependencies = [
    "openai>=1.30.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/milu"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 3: 创建 .env.example**

```bash
# 各厂商API Key配置
QWEN_API_KEY=sk-xxx
KIMI_API_KEY=sk-xxx
GLM_API_KEY=xxx
DEEPSEEK_API_KEY=sk-xxx
MINIMAX_API_KEY=xxx
DOUBAO_API_KEY=ark-cn-beijing-xxx
```

- [ ] **Step 4: 创建 src/milu/exceptions.py**

```python
"""统一异常体系 - 所有自定义异常的基类和派生类"""


class MiluError(Exception):
    """框架基础异常，所有其他异常的父类"""
    pass


class ModelConfigError(MiluError):
    """模型配置错误，如传入了不支持的参数"""
    pass


class AuthenticationError(MiluError):
    """API Key 无效或缺失"""
    pass


class RateLimitError(MiluError):
    """请求频率超限"""
    pass


class ModelNotAvailableError(MiluError):
    """指定的模型不可用"""
    pass


class StreamError(MiluError):
    """流式输出过程中发生异常"""
    pass


class FeatureNotSupportedError(MiluError):
    """请求的功能该模型不支持"""
    pass
```

- [ ] **Step 5: 创建顶层 __init__.py（占位，后续完善导出）**

```python
"""milu - 统一AI模型抽象层"""

from milu.exceptions import (
    MiluError,
    AuthenticationError,
    FeatureNotSupportedError,
    ModelConfigError,
    ModelNotAvailableError,
    RateLimitError,
    StreamError,
)

__all__ = [
    "MiluError",
    "ModelConfigError",
    "AuthenticationError",
    "RateLimitError",
    "ModelNotAvailableError",
    "StreamError",
    "FeatureNotSupportedError",
]
```

- [ ] **Step 6: 安装项目依赖**

```bash
uv venv
uv pip install -e ".[dev]"
```

- [ ] **Step 7: 验证异常可导入**

```bash
uv run python -c "from milu.exceptions import MiluError, AuthenticationError; print('OK')"
```

Expected: `OK`

- [ ] **Step 8: 提交**

```bash
git add pyproject.toml .env.example src/milu/
git commit -m "feat: 初始化项目脚手架和异常体系"
```

---

## Task 2: 数据模型（Message, StreamChunk, TokenUsage）

**Files:**
- Create: `src/milu/models/message.py`
- Create: `src/milu/models/response.py`
- Create: `src/milu/models/__init__.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: 编写失败测试 tests/test_models.py**

```python
"""测试数据模型：Message, StreamChunk, TokenUsage"""

import pytest
from milu.models.message import Message, MessageRole
from milu.models.response import StreamChunk, TokenUsage


class TestMessageRole:
    """测试消息角色枚举"""

    def test_role_values(self):
        """验证四种角色值"""
        assert MessageRole.SYSTEM == "system"
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.TOOL == "tool"

    def test_role_is_string(self):
        """验证角色枚举可作为字符串使用"""
        role = MessageRole.USER
        assert isinstance(role, str)
        assert role == "user"


class TestMessage:
    """测试消息结构"""

    def test_basic_text_message(self):
        """基础文本消息"""
        msg = Message(role=MessageRole.USER, content="你好")
        assert msg.role == MessageRole.USER
        assert msg.content == "你好"
        assert msg.tool_calls is None
        assert msg.tool_call_id is None
        assert msg.name is None

    def test_system_message(self):
        """系统消息"""
        msg = Message(role=MessageRole.SYSTEM, content="你是一个助手")
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "你是一个助手"

    def test_multimodal_message(self):
        """多模态消息，content为列表"""
        content = [
            {"type": "text", "text": "这张图是什么？"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        msg = Message(role=MessageRole.USER, content=content)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert msg.content[0]["type"] == "text"
        assert msg.content[1]["type"] == "image_url"

    def test_assistant_message_with_tool_calls(self):
        """助手消息带工具调用"""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "北京"}'},
            }
        ]
        msg = Message(role=MessageRole.ASSISTANT, tool_calls=tool_calls)
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["function"]["name"] == "get_weather"

    def test_tool_message(self):
        """工具返回消息"""
        msg = Message(
            role=MessageRole.TOOL,
            content="晴天，25°C",
            tool_call_id="call_1",
            name="get_weather",
        )
        assert msg.role == MessageRole.TOOL
        assert msg.tool_call_id == "call_1"
        assert msg.name == "get_weather"

    def test_message_to_dict_basic(self):
        """消息转换为字典格式（用于API调用）"""
        msg = Message(role=MessageRole.USER, content="你好")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "你好"}

    def test_message_to_dict_excludes_none(self):
        """转换为字典时排除None字段"""
        msg = Message(role=MessageRole.USER, content="你好")
        d = msg.to_dict()
        assert "tool_calls" not in d
        assert "tool_call_id" not in d
        assert "name" not in d

    def test_message_to_dict_with_tool_calls(self):
        """带工具调用的消息转字典"""
        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "test"}}]
        msg = Message(role=MessageRole.ASSISTANT, tool_calls=tool_calls)
        d = msg.to_dict()
        assert "tool_calls" in d
        assert d["tool_calls"] == tool_calls

    def test_message_to_dict_with_tool_result(self):
        """工具返回消息转字典"""
        msg = Message(
            role=MessageRole.TOOL,
            content="结果",
            tool_call_id="call_1",
            name="func",
        )
        d = msg.to_dict()
        assert d["tool_call_id"] == "call_1"
        assert d["name"] == "func"

    def test_message_to_dict_multimodal(self):
        """多模态消息转字典"""
        content = [
            {"type": "text", "text": "描述图片"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        msg = Message(role=MessageRole.USER, content=content)
        d = msg.to_dict()
        assert d["content"] == content


class TestTokenUsage:
    """测试Token用量统计"""

    def test_default_values(self):
        """默认值全为0"""
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.reasoning_tokens == 0

    def test_custom_values(self):
        """自定义值"""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            reasoning_tokens=30,
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.reasoning_tokens == 30


class TestStreamChunk:
    """测试流式输出数据块"""

    def test_content_chunk(self):
        """正文内容片段"""
        chunk = StreamChunk(content="你好")
        assert chunk.content == "你好"
        assert chunk.reasoning_content is None
        assert chunk.finish_reason is None

    def test_reasoning_chunk(self):
        """思考过程片段"""
        chunk = StreamChunk(reasoning_content="让我想想...")
        assert chunk.reasoning_content == "让我想想..."
        assert chunk.content is None

    def test_finish_chunk_with_usage(self):
        """结束块带token统计"""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunk = StreamChunk(finish_reason="stop", usage=usage)
        assert chunk.finish_reason == "stop"
        assert chunk.usage is not None
        assert chunk.usage.total_tokens == 15

    def test_tool_calls_chunk(self):
        """工具调用片段"""
        chunk = StreamChunk(tool_calls=[{"id": "call_1", "function": {"name": "test"}}])
        assert chunk.tool_calls is not None
        assert len(chunk.tool_calls) == 1
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'milu.models'`

- [ ] **Step 3: 创建 src/milu/models/message.py**

```python
"""统一消息类型定义 - 兼容所有厂商的消息格式"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MessageRole(str, Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """
    统一消息结构，兼容所有厂商的消息格式。

    属性:
        role: 消息角色（system/user/assistant/tool）
        content: 消息内容，str为纯文本，list为多模态内容
        tool_calls: 工具调用信息（仅assistant消息使用）
        tool_call_id: 工具调用ID（仅tool消息使用）
        name: 工具/函数名称（仅tool消息使用）
    """
    role: MessageRole
    content: str | list | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict:
        """
        转换为OpenAI API兼容的字典格式。
        自动排除值为None的字段。
        """
        result: dict = {"role": self.role.value}

        # content 始终包含（即使为None，某些API需要）
        if self.content is not None:
            result["content"] = self.content

        # 可选字段：仅在非None时包含
        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            result["name"] = self.name

        return result
```

- [ ] **Step 4: 创建 src/milu/models/response.py**

```python
"""统一响应模型 - 流式输出数据块和Token用量统计"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TokenUsage:
    """
    统一的Token用量统计。

    属性:
        prompt_tokens: 输入token数量
        completion_tokens: 输出token数量
        total_tokens: 总token数量
        reasoning_tokens: 思考过程消耗的token数量（仅支持思考模式的厂商）
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass
class StreamChunk:
    """
    流式输出的统一数据块。

    每次从流中接收一个StreamChunk，可能包含以下信息之一或组合：
    - content: 正文内容片段
    - reasoning_content: 思考过程片段
    - tool_calls: 工具调用信息
    - finish_reason + usage: 结束标志和统计信息

    属性:
        content: 正文内容片段
        reasoning_content: 思考过程片段（支持思考模式时）
        tool_calls: 工具调用信息列表
        finish_reason: 结束原因（stop/length/tool_calls等）
        usage: Token用量统计（仅在最后一个chunk中）
    """
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None
```

- [ ] **Step 5: 创建 src/milu/models/__init__.py**

```python
"""数据模型包 - 统一的消息、响应和配置结构"""

from milu.models.message import Message, MessageRole
from milu.models.response import StreamChunk, TokenUsage

__all__ = [
    "Message",
    "MessageRole",
    "StreamChunk",
    "TokenUsage",
]
```

- [ ] **Step 6: 运行测试确认通过**

```bash
uv run pytest tests/test_models.py -v
```

Expected: 全部 PASS

- [ ] **Step 7: 提交**

```bash
git add src/milu/models/ tests/test_models.py
git commit -m "feat: 添加统一数据模型（Message, StreamChunk, TokenUsage）"
```

---

## Task 3: 配置参数体系

**Files:**
- Create: `src/milu/models/config.py`
- Modify: `src/milu/models/__init__.py`
- Test: `tests/test_models.py`（追加测试）

- [ ] **Step 1: 追加配置测试到 tests/test_models.py**

在文件末尾追加：

```python
from milu.models.config import (
    ModelConfig,
    WebSearchConfig,
    ThinkingConfig,
    FunctionCallingConfig,
    ImageGenerationConfig,
    AudioGenerationConfig,
)


class TestModelConfig:
    """测试基础模型配置"""

    def test_required_model_field(self):
        """model字段必填"""
        config = ModelConfig(model="qwen-max")
        assert config.model == "qwen-max"

    def test_default_values(self):
        """默认值验证"""
        config = ModelConfig(model="test")
        assert config.temperature is None
        assert config.top_p is None
        assert config.max_tokens is None
        assert config.stop is None
        assert config.stream is True
        assert config.frequency_penalty is None
        assert config.presence_penalty is None

    def test_custom_values(self):
        """自定义值"""
        config = ModelConfig(
            model="test",
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            stop=["\n"],
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.max_tokens == 2048

    def test_to_dict_excludes_none(self):
        """转字典排除None值"""
        config = ModelConfig(model="test", temperature=0.7)
        d = config.to_dict()
        assert d["model"] == "test"
        assert d["temperature"] == 0.7
        assert "top_p" not in d
        assert "max_tokens" not in d

    def test_stream_always_true(self):
        """stream字段始终为True"""
        config = ModelConfig(model="test")
        d = config.to_dict()
        assert d["stream"] is True


class TestWebSearchConfig:
    """测试联网搜索配置"""

    def test_defaults(self):
        config = WebSearchConfig(model="test")
        assert config.web_search is False
        assert config.web_search_strategy == "auto"

    def test_enabled(self):
        config = WebSearchConfig(model="test", web_search=True, web_search_strategy="force")
        assert config.web_search is True
        d = config.to_dict()
        assert d["web_search"] is True
        assert d["web_search_strategy"] == "force"


class TestThinkingConfig:
    """测试思考模式配置"""

    def test_defaults(self):
        config = ThinkingConfig(model="test")
        assert config.enable_thinking is False
        assert config.thinking_level == "medium"

    def test_enabled_with_level(self):
        config = ThinkingConfig(model="test", enable_thinking=True, thinking_level="high")
        assert config.enable_thinking is True
        assert config.thinking_level == "high"

    def test_level_values(self):
        """验证三个级别都可设置"""
        for level in ("low", "medium", "high"):
            config = ThinkingConfig(model="test", thinking_level=level)
            assert config.thinking_level == level


class TestFunctionCallingConfig:
    """测试函数调用配置"""

    def test_defaults(self):
        config = FunctionCallingConfig(model="test")
        assert config.tools is None
        assert config.tool_choice == "auto"

    def test_with_tools(self):
        tools = [{"type": "function", "function": {"name": "test_func"}}]
        config = FunctionCallingConfig(model="test", tools=tools, tool_choice="required")
        d = config.to_dict()
        assert d["tools"] == tools
        assert d["tool_choice"] == "required"


class TestImageGenerationConfig:
    """测试图片生成配置"""

    def test_defaults(self):
        config = ImageGenerationConfig(model="test")
        assert config.image_size == "1024x1024"
        assert config.image_quality == "standard"
        assert config.num_images == 1

    def test_custom(self):
        config = ImageGenerationConfig(
            model="test", image_size="512x512", image_quality="hd", num_images=2
        )
        d = config.to_dict()
        assert d["image_size"] == "512x512"
        assert d["image_quality"] == "hd"
        assert d["num_images"] == 2


class TestAudioGenerationConfig:
    """测试音频生成配置"""

    def test_defaults(self):
        config = AudioGenerationConfig(model="test")
        assert config.voice is None
        assert config.audio_format == "mp3"
        assert config.speed == 1.0

    def test_custom(self):
        config = AudioGenerationConfig(
            model="test", voice="alloy", audio_format="wav", speed=1.5
        )
        d = config.to_dict()
        assert d["voice"] == "alloy"
        assert d["audio_format"] == "wav"
        assert d["speed"] == 1.5
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_models.py -v -k "Config"
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: 创建 src/milu/models/config.py**

```python
"""模型配置参数定义 - 基础配置和按能力扩展配置"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    """
    模型调用的基础参数配置（所有厂商通用）。

    属性:
        model: 模型名称，如 "qwen-max", "kimi-k2.5"
        temperature: 温度参数，控制生成随机性（0.0~2.0）
        top_p: 核采样概率，控制词汇多样性（0.0~1.0）
        max_tokens: 最大输出token数
        stop: 停止词列表，遇到任一停止词则停止生成
        stream: 是否流式输出，本框架固定为True
        frequency_penalty: 频率惩罚，降低重复内容（-2.0~2.0）
        presence_penalty: 存在惩罚，鼓励新话题（-2.0~2.0）
    """
    model: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    stream: bool = True
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

    def to_dict(self) -> dict:
        """转换为字典，排除None值，始终包含stream=True"""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
        # stream 始终包含
        result["stream"] = True
        return result


@dataclass
class WebSearchConfig(ModelConfig):
    """
    联网搜索扩展参数。
    适用于支持 web_search 能力的厂商（Qwen, Kimi, GLM, Doubao）。
    """
    web_search: bool = False
    web_search_strategy: str = "auto"


@dataclass
class ThinkingConfig(ModelConfig):
    """
    思考/推理模式扩展参数。
    适用于支持 thinking 能力的厂商（Qwen, Kimi, GLM, DeepSeek）。

    thinking_level 统一为 low/medium/high 三级，默认 medium。
    各厂商内部自行映射为实际参数，不支持的厂商静默忽略。
    """
    enable_thinking: bool = False
    thinking_level: str = "medium"


@dataclass
class FunctionCallingConfig(ModelConfig):
    """
    函数调用扩展参数。
    适用于支持 function_calling 能力的厂商。
    """
    tools: list[dict] | None = None
    tool_choice: str | dict = "auto"


@dataclass
class ImageGenerationConfig(ModelConfig):
    """
    图片生成扩展参数。
    适用于支持 image_generation 能力的厂商（Qwen, MiniMax, Doubao）。
    """
    image_size: str = "1024x1024"
    image_quality: str = "standard"
    num_images: int = 1


@dataclass
class AudioGenerationConfig(ModelConfig):
    """
    音频生成扩展参数（TTS等）。
    适用于支持 audio_generation 能力的厂商（MiniMax）。
    """
    voice: str | None = None
    audio_format: str = "mp3"
    speed: float = 1.0
```

- [ ] **Step 4: 更新 src/milu/models/__init__.py 导出配置类**

在现有文件中追加导出：

```python
from milu.models.config import (
    ModelConfig,
    WebSearchConfig,
    ThinkingConfig,
    FunctionCallingConfig,
    ImageGenerationConfig,
    AudioGenerationConfig,
)
```

并在 `__all__` 中添加这些名称。

- [ ] **Step 5: 运行测试确认通过**

```bash
uv run pytest tests/test_models.py -v
```

Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add src/milu/models/config.py src/milu/models/__init__.py tests/test_models.py
git commit -m "feat: 添加模型配置参数体系（基础+扩展）"
```

---

## Task 4: BaseLLM 抽象基类与 ModelCapabilities

**Files:**
- Create: `src/milu/providers/base.py`
- Create: `tests/test_base_provider.py`

- [ ] **Step 1: 编写失败测试 tests/test_base_provider.py**

```python
"""测试 BaseLLM 抽象基类和 ModelCapabilities"""

import pytest
from milu.providers.base import BaseLLM, ModelCapabilities
from milu.models.message import Message, MessageRole
from milu.models.response import StreamChunk


class TestModelCapabilities:
    """测试能力描述符"""

    def test_default_capabilities(self):
        """默认能力：仅流式输出"""
        caps = ModelCapabilities()
        assert caps.supports_streaming is True
        assert caps.supports_function_calling is False
        assert caps.supports_json_mode is False
        assert caps.supports_web_search is False
        assert caps.supports_thinking is False
        assert caps.supports_embedding is False
        assert caps.supports_vision is False
        assert caps.supports_audio_understand is False
        assert caps.supports_video is False
        assert caps.supports_document is False
        assert caps.supports_image_generation is False
        assert caps.supports_audio_generation is False
        assert caps.max_context_window == 8192
        assert caps.supported_output_formats == ("text",)

    def test_frozen(self):
        """能力描述符不可变"""
        caps = ModelCapabilities()
        with pytest.raises(AttributeError):
            caps.supports_vision = True

    def test_custom_capabilities(self):
        """自定义能力"""
        caps = ModelCapabilities(
            supports_function_calling=True,
            supports_thinking=True,
            max_context_window=131072,
        )
        assert caps.supports_function_calling is True
        assert caps.supports_thinking is True
        assert caps.max_context_window == 131072


class ConcreteLLM(BaseLLM):
    """用于测试的具体LLM实现"""

    def __init__(self, api_key: str = "test-key", model: str = "test-model", **kwargs):
        super().__init__(api_key=api_key, model=model, **kwargs)

    @property
    def provider_name(self) -> str:
        return "test_provider"

    @property
    def base_url(self) -> str:
        return "https://api.test.com/v1"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_function_calling=True,
            supports_web_search=True,
            supports_thinking=True,
            max_context_window=32768,
        )

    def _get_available_param_names(self) -> set[str]:
        """返回该厂商支持的参数名集合"""
        return {
            "temperature", "top_p", "max_tokens", "stop",
            "frequency_penalty", "presence_penalty",
            "web_search", "enable_thinking", "thinking_level",
        }


class TestBaseLLM:
    """测试BaseLLM抽象基类"""

    def test_provider_name(self):
        llm = ConcreteLLM()
        assert llm.provider_name == "test_provider"

    def test_capabilities(self):
        llm = ConcreteLLM()
        caps = llm.capabilities
        assert caps.supports_function_calling is True
        assert caps.supports_web_search is True
        assert caps.max_context_window == 32768

    def test_model_name(self):
        llm = ConcreteLLM(model="my-model")
        assert llm.model == "my-model"

    def test_get_available_params(self):
        """验证返回当前可用参数字典"""
        llm = ConcreteLLM()
        params = llm.get_available_params()
        assert "temperature" in params
        assert "web_search" in params
        assert "enable_thinking" in params
        assert "thinking_level" in params

    def test_validate_params_keeps_valid(self):
        """验证合法参数被保留"""
        llm = ConcreteLLM()
        result = llm._validate_params({"temperature": 0.7, "web_search": True})
        assert result["temperature"] == 0.7
        assert result["web_search"] is True

    def test_validate_params_filters_unsupported(self):
        """验证不支持的参数被过滤"""
        llm = ConcreteLLM()
        result = llm._validate_params({
            "temperature": 0.7,
            "image_size": "1024x1024",  # 不在可用参数中
        })
        assert result["temperature"] == 0.7
        assert "image_size" not in result

    def test_validate_params_warns_on_filter(self, caplog):
        """过滤参数时输出警告日志"""
        import logging
        with caplog.at_level(logging.WARNING):
            llm = ConcreteLLM()
            llm._validate_params({"unknown_param": "value"})
        assert "unknown_param" in caplog.text

    def test_get_api_key_from_constructor(self):
        """构造函数直接传入api_key"""
        llm = ConcreteLLM(api_key="my-key")
        assert llm._get_api_key() == "my-key"

    def test_get_api_key_from_env(self, monkeypatch):
        """从环境变量读取api_key"""
        monkeypatch.setenv("TEST_PROVIDER_API_KEY", "env-key")
        llm = ConcreteLLM(api_key=None)
        assert llm._get_api_key() == "env-key"

    def test_get_api_key_missing_raises(self, monkeypatch):
        """api_key缺失时抛出AuthenticationError"""
        monkeypatch.delenv("TEST_PROVIDER_API_KEY", raising=False)
        llm = ConcreteLLM(api_key=None)
        from milu.exceptions import AuthenticationError
        with pytest.raises(AuthenticationError):
            llm._get_api_key()

    def test_cannot_instantiate_base_directly(self):
        """不能直接实例化抽象基类"""
        with pytest.raises(TypeError):
            BaseLLM(api_key="key", model="model")
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_base_provider.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 创建 src/milu/providers/base.py**

```python
"""BaseLLM 抽象基类和 ModelCapabilities 能力描述符"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from openai import AsyncOpenAI

from milu.exceptions import AuthenticationError
from milu.models.message import Message
from milu.models.response import StreamChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelCapabilities:
    """
    模型能力描述符 - 声明某个厂商/模型支持的功能集合。

    每个厂商实例化一个 ModelCapabilities，明确声明自身支持和不支持的功能。
    调用方可通过 model.capabilities 查询能力，实现运行时动态UI展示。
    使用 frozen=True 确保创建后不可修改。
    """
    # 基础能力
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_json_mode: bool = False
    supports_web_search: bool = False
    supports_thinking: bool = False
    supports_embedding: bool = False

    # 多模态理解能力
    supports_vision: bool = False
    supports_audio_understand: bool = False
    supports_video: bool = False
    supports_document: bool = False

    # 多模态生成能力
    supports_image_generation: bool = False
    supports_audio_generation: bool = False

    # 模型规格
    max_context_window: int = 8192
    supported_output_formats: tuple = ("text",)


class BaseLLM(ABC):
    """
    所有厂商模型的抽象基类。

    子类需要实现:
        - provider_name: 厂商名称（用于环境变量查找等）
        - base_url: API基础URL
        - capabilities: 该厂商的能力声明
        - _get_available_param_names(): 该厂商支持的参数名集合
        - chat(): 流式聊天接口

    基类提供:
        - OpenAI客户端管理
        - API Key获取（构造函数 > 环境变量）
        - 参数校验和过滤
        - 可用参数查询
    """

    def __init__(self, api_key: str | None = None, model: str = "", **kwargs):
        """
        初始化LLM实例。

        参数:
            api_key: API密钥，为None时从环境变量读取
            model: 模型名称
            **kwargs: 其他配置参数
        """
        self._api_key = api_key
        self.model = model
        self._extra_kwargs = kwargs
        self._client: AsyncOpenAI | None = None

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """厂商标识名，如 'qwen', 'kimi' 等"""
        ...

    @property
    @abstractmethod
    def base_url(self) -> str:
        """API基础URL"""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """该厂商/模型的能力声明"""
        ...

    @abstractmethod
    def _get_available_param_names(self) -> set[str]:
        """
        返回该厂商支持的所有参数名称集合。
        用于参数校验和 get_available_params()。
        """
        ...

    @abstractmethod
    async def chat(self, messages: list[Message], **kwargs) -> AsyncIterator[StreamChunk]:
        """
        统一的流式聊天接口。

        参数:
            messages: 消息列表
            **kwargs: 模型参数（temperature, max_tokens等）

        返回:
            异步迭代器，逐个产出 StreamChunk
        """
        ...

    def _get_api_key(self) -> str:
        """
        获取API密钥。
        优先级：构造函数传入 > 环境变量 {PROVIDER_NAME}_API_KEY
        """
        if self._api_key:
            return self._api_key

        env_key = f"{self.provider_name.upper()}_API_KEY"
        key = os.environ.get(env_key)
        if not key:
            raise AuthenticationError(
                f"未找到API Key。请通过构造函数传入或设置环境变量 {env_key}"
            )
        return key

    def _get_client(self) -> AsyncOpenAI:
        """获取或创建OpenAI异步客户端实例（懒加载）"""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._get_api_key(),
                base_url=self.base_url,
            )
        return self._client

    def get_available_params(self) -> dict[str, dict]:
        """
        返回当前模型可用的参数及其元信息。

        返回格式: {参数名: {"type": 类型, "default": 默认值}}
        仅包含该厂商实际支持的参数。
        """
        # 所有可能参数的元信息定义
        all_params = {
            "temperature": {"type": float, "default": None},
            "top_p": {"type": float, "default": None},
            "max_tokens": {"type": int, "default": None},
            "stop": {"type": list, "default": None},
            "frequency_penalty": {"type": float, "default": None},
            "presence_penalty": {"type": float, "default": None},
            "web_search": {"type": bool, "default": False},
            "web_search_strategy": {"type": str, "default": "auto"},
            "enable_thinking": {"type": bool, "default": False},
            "thinking_level": {"type": str, "default": "medium"},
            "tools": {"type": list, "default": None},
            "tool_choice": {"type": str, "default": "auto"},
            "image_size": {"type": str, "default": "1024x1024"},
            "image_quality": {"type": str, "default": "standard"},
            "num_images": {"type": int, "default": 1},
            "voice": {"type": str, "default": None},
            "audio_format": {"type": str, "default": "mp3"},
            "speed": {"type": float, "default": 1.0},
        }

        available_names = self._get_available_param_names()
        return {name: info for name, info in all_params.items() if name in available_names}

    def _validate_params(self, params: dict) -> dict:
        """
        校验并过滤参数：移除不支持的参数并记录警告。

        参数:
            params: 原始参数字典

        返回:
            过滤后的参数字典（仅包含该厂商支持的参数）
        """
        available = self._get_available_param_names()
        validated = {}
        for key, value in params.items():
            if key in available:
                validated[key] = value
            else:
                logger.warning(
                    f"[{self.provider_name}] 参数 '{key}' 不被支持，已忽略"
                )
        return validated

    def _messages_to_dicts(self, messages: list[Message]) -> list[dict]:
        """将Message对象列表转换为API兼容的字典列表"""
        return [msg.to_dict() for msg in messages]
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_base_provider.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/milu/providers/base.py tests/test_base_provider.py
git commit -m "feat: 添加BaseLLM抽象基类和ModelCapabilities能力描述符"
```

---

## Task 5: ModelRegistry 注册表

**Files:**
- Create: `src/milu/providers/__init__.py`
- Create: `tests/test_registry.py`

- [ ] **Step 1: 编写失败测试 tests/test_registry.py**

```python
"""测试 ModelRegistry 注册表"""

import pytest
from milu.providers import ModelRegistry
from milu.providers.base import BaseLLM, ModelCapabilities
from milu.models.message import Message
from milu.models.response import StreamChunk


class FakeLLM(BaseLLM):
    """测试用的假LLM实现"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def base_url(self) -> str:
        return "https://fake.api/v1"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    def _get_available_param_names(self) -> set[str]:
        return {"temperature"}

    async def chat(self, messages, **kwargs):
        yield StreamChunk(content="fake")


class TestModelRegistry:
    """测试模型注册表"""

    def setup_method(self):
        """每个测试前清空注册表"""
        ModelRegistry._providers.clear()

    def test_register_and_create(self):
        """注册并创建实例"""
        ModelRegistry.register("fake", FakeLLM)
        model = ModelRegistry.create("fake", api_key="key", model="test")
        assert isinstance(model, FakeLLM)
        assert model.model == "test"

    def test_list_providers(self):
        """列出已注册厂商"""
        ModelRegistry.register("fake", FakeLLM)
        providers = ModelRegistry.list_providers()
        assert "fake" in providers

    def test_create_unknown_provider(self):
        """创建未注册厂商时抛异常"""
        with pytest.raises(ValueError, match="不支持的厂商"):
            ModelRegistry.create("unknown")

    def test_register_duplicate_overwrites(self):
        """重复注册同一名称会覆盖"""
        ModelRegistry.register("fake", FakeLLM)
        ModelRegistry.register("fake", FakeLLM)
        assert "fake" in ModelRegistry.list_providers()

    def test_create_passes_kwargs(self):
        """create方法正确传递参数"""
        ModelRegistry.register("fake", FakeLLM)
        model = ModelRegistry.create("fake", api_key="my-key", model="my-model", temperature=0.5)
        assert model.model == "my-model"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_registry.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: 创建 src/milu/providers/__init__.py**

```python
"""
模型厂商包 - 包含所有厂商实现和注册表。

使用方式:
    from milu.providers import ModelRegistry

    # 工厂方式创建
    model = ModelRegistry.create("qwen", model="qwen-max")

    # 或直接导入
    from milu.providers.qwen import QwenLLM
    model = QwenLLM(model="qwen-max")
"""

from __future__ import annotations

from milu.providers.base import BaseLLM, ModelCapabilities


class ModelRegistry:
    """
    模型注册表 - 管理所有厂商的LLM实现。

    提供统一的工厂方法创建模型实例，支持动态注册新厂商。
    各厂商模块加载时自动注册到此类。
    """

    _providers: dict[str, type[BaseLLM]] = {}

    @classmethod
    def register(cls, name: str, provider_class: type[BaseLLM]) -> None:
        """
        注册一个厂商实现。

        参数:
            name: 厂商名称（小写），如 "qwen", "kimi"
            provider_class: BaseLLM的子类
        """
        cls._providers[name] = provider_class

    @classmethod
    def create(cls, provider: str, **kwargs) -> BaseLLM:
        """
        根据厂商名创建模型实例。

        参数:
            provider: 厂商名称
            **kwargs: 传递给厂商构造函数的参数

        返回:
            BaseLLM实例

        异常:
            ValueError: 厂商未注册
        """
        if provider not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(f"不支持的厂商: {provider}，可用: {available}")
        return cls._providers[provider](**kwargs)

    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有已注册的厂商名称"""
        return list(cls._providers.keys())


# 导入所有厂商模块以触发自动注册
# 注意：各厂商模块在后续Task中创建，此处先注释
# from milu.providers import qwen, kimi, glm, deepseek, minimax, doubao

__all__ = [
    "ModelRegistry",
    "BaseLLM",
    "ModelCapabilities",
]
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_registry.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/milu/providers/__init__.py tests/test_registry.py
git commit -m "feat: 添加ModelRegistry模型注册表"
```

---

## Task 6: QwenLLM 实现

**Files:**
- Create: `src/milu/providers/qwen.py`
- Create: `tests/conftest.py`
- Create: `tests/test_qwen.py`

- [ ] **Step 1: 创建 tests/conftest.py 公共测试fixtures**

```python
"""pytest 公共 fixtures - 用于所有provider测试的模拟工具"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest


@dataclass
class MockChoice:
    """模拟 OpenAI SDK 的 Choice 对象"""
    delta: "MockDelta"
    finish_reason: str | None = None
    index: int = 0


@dataclass
class MockDelta:
    """模拟流式输出的 delta 对象"""
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list | None = None
    role: str | None = None


@dataclass
class MockUsage:
    """模拟 OpenAI SDK 的 Usage 对象"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class MockChunk:
    """模拟 OpenAI SDK 的流式响应 Chunk"""
    choices: list[MockChoice]
    usage: MockUsage | None = None


@pytest.fixture
def mock_openai_client():
    """
    创建模拟的 AsyncOpenAI 客户端。
    用于测试 provider 的 chat 方法而无需真实 API 调用。
    """
    client = AsyncMock()
    # chat.completions.create 返回一个异步迭代器
    client.chat.completions.create = AsyncMock()
    return client
```

- [ ] **Step 2: 编写失败测试 tests/test_qwen.py**

```python
"""测试 QwenLLM（通义千问）实现"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from milu.models.message import Message, MessageRole
from milu.models.response import StreamChunk
from milu.providers.qwen import QwenLLM
from tests.conftest import MockChunk, MockChoice, MockDelta, MockUsage


class TestQwenCapabilities:
    """测试 Qwen 能力声明"""

    def test_capabilities(self):
        llm = QwenLLM(api_key="test", model="qwen-max")
        caps = llm.capabilities
        assert caps.supports_streaming is True
        assert caps.supports_function_calling is True
        assert caps.supports_json_mode is True
        assert caps.supports_web_search is True
        assert caps.supports_thinking is True
        assert caps.supports_embedding is True
        assert caps.supports_vision is True
        assert caps.supports_audio_understand is True
        assert caps.supports_video is True
        assert caps.supports_document is True
        assert caps.supports_image_generation is True
        assert caps.supports_audio_generation is False
        assert caps.max_context_window == 131072

    def test_provider_name(self):
        llm = QwenLLM(api_key="test", model="qwen-max")
        assert llm.provider_name == "qwen"

    def test_base_url(self):
        llm = QwenLLM(api_key="test", model="qwen-max")
        assert "dashscope" in llm.base_url


class TestQwenAvailableParams:
    """测试 Qwen 可用参数"""

    def test_has_web_search_params(self):
        llm = QwenLLM(api_key="test", model="qwen-max")
        params = llm.get_available_params()
        assert "web_search" in params
        assert "web_search_strategy" in params

    def test_has_thinking_params(self):
        llm = QwenLLM(api_key="test", model="qwen-max")
        params = llm.get_available_params()
        assert "enable_thinking" in params
        assert "thinking_level" in params

    def test_has_function_calling_params(self):
        llm = QwenLLM(api_key="test", model="qwen-max")
        params = llm.get_available_params()
        assert "tools" in params
        assert "tool_choice" in params

    def test_has_basic_params(self):
        llm = QwenLLM(api_key="test", model="qwen-max")
        params = llm.get_available_params()
        assert "temperature" in params
        assert "top_p" in params
        assert "max_tokens" in params


class TestQwenChat:
    """测试 Qwen 流式聊天"""

    @pytest.mark.asyncio
    async def test_basic_chat(self):
        """基础聊天流式响应"""
        llm = QwenLLM(api_key="test-key", model="qwen-max")

        # 模拟流式响应
        mock_chunks = [
            MockChunk(choices=[MockChoice(delta=MockDelta(content="你"))]),
            MockChunk(choices=[MockChoice(delta=MockDelta(content="好"))]),
            MockChunk(
                choices=[MockChoice(delta=MockDelta(content=""), finish_reason="stop")],
                usage=MockUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            ),
        ]

        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk

        with patch.object(llm, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
            mock_get_client.return_value = mock_client

            messages = [Message(role=MessageRole.USER, content="你好")]
            chunks = []
            async for chunk in llm.chat(messages):
                chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].content == "你"
        assert chunks[1].content == "好"
        assert chunks[2].finish_reason == "stop"
        assert chunks[2].usage is not None
        assert chunks[2].usage.total_tokens == 12

    @pytest.mark.asyncio
    async def test_thinking_mode_mapping(self):
        """思考模式参数映射：Qwen仅支持enable_thinking，忽略thinking_level"""
        llm = QwenLLM(api_key="test-key", model="qwen-max")

        mock_chunks = [
            MockChunk(choices=[MockChoice(delta=MockDelta(reasoning_content="让我想想"))]),
            MockChunk(choices=[MockChoice(delta=MockDelta(content="答案是42"))]),
            MockChunk(choices=[MockChoice(delta=MockDelta(), finish_reason="stop")]),
        ]

        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk

        with patch.object(llm, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_create = AsyncMock(return_value=mock_stream())
            mock_client.chat.completions.create = mock_create
            mock_get_client.return_value = mock_client

            messages = [Message(role=MessageRole.USER, content="1+1=?")]
            async for _ in llm.chat(messages, enable_thinking=True, thinking_level="high"):
                pass

            # 验证传递给OpenAI SDK的参数
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs.get("extra_body", {}).get("enable_thinking") is True

    @pytest.mark.asyncio
    async def test_web_search_param(self):
        """联网搜索参数传递"""
        llm = QwenLLM(api_key="test-key", model="qwen-max")

        mock_chunks = [
            MockChunk(choices=[MockChoice(delta=MockDelta(content="搜索结果"))]),
            MockChunk(choices=[MockChoice(delta=MockDelta(), finish_reason="stop")]),
        ]

        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk

        with patch.object(llm, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_create = AsyncMock(return_value=mock_stream())
            mock_client.chat.completions.create = mock_create
            mock_get_client.return_value = mock_client

            messages = [Message(role=MessageRole.USER, content="今日新闻")]
            async for _ in llm.chat(messages, web_search=True):
                pass

            call_kwargs = mock_create.call_args.kwargs
            # Qwen的联网搜索通过extra_body传递enable_search
            assert call_kwargs.get("extra_body", {}).get("enable_search") is True
```

- [ ] **Step 3: 运行测试确认失败**

```bash
uv run pytest tests/test_qwen.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'milu.providers.qwen'`

- [ ] **Step 4: 创建 src/milu/providers/qwen.py**

```python
"""
QwenLLM - 通义千问（阿里云 DashScope）模型实现。

支持能力：
- 流式输出、函数调用、JSON模式、联网搜索
- 思考模式（enable_thinking，忽略thinking_level）
- 文本嵌入、图片理解、音频理解、视频理解、文档理解
- 图片生成（通义万相）

API文档: https://help.aliyun.com/zh/model-studio/
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from milu.exceptions import StreamError
from milu.models.message import Message
from milu.models.response import StreamChunk, TokenUsage
from milu.providers.base import BaseLLM, ModelCapabilities

logger = logging.getLogger(__name__)


class QwenLLM(BaseLLM):
    """通义千问模型实现"""

    # 通义千问能力声明
    _capabilities = ModelCapabilities(
        supports_streaming=True,
        supports_function_calling=True,
        supports_json_mode=True,
        supports_web_search=True,
        supports_thinking=True,
        supports_embedding=True,
        supports_vision=True,
        supports_audio_understand=True,
        supports_video=True,
        supports_document=True,
        supports_image_generation=True,
        supports_audio_generation=False,
        max_context_window=131072,
        supported_output_formats=("text", "json"),
    )

    # 通义千问支持的参数名集合
    _param_names = {
        "temperature", "top_p", "max_tokens", "stop",
        "frequency_penalty", "presence_penalty",
        "web_search", "web_search_strategy",
        "enable_thinking", "thinking_level",
        "tools", "tool_choice",
    }

    @property
    def provider_name(self) -> str:
        return "qwen"

    @property
    def base_url(self) -> str:
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def _get_available_param_names(self) -> set[str]:
        return self._param_names

    async def chat(self, messages: list[Message], **kwargs) -> AsyncIterator[StreamChunk]:
        """
        流式聊天接口。

        Qwen特有参数:
            web_search (bool): 是否开启联网搜索，映射为 extra_body.enable_search
            enable_thinking (bool): 是否开启思考模式，映射为 extra_body.enable_thinking
            thinking_level (str): 思考级别，Qwen不支持此参数，静默忽略
        """
        validated = self._validate_params(kwargs)
        client = self._get_client()

        # 构建请求参数
        request_params = {
            "model": self.model,
            "messages": self._messages_to_dicts(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        # 基础参数
        for key in ("temperature", "top_p", "max_tokens", "stop",
                     "frequency_penalty", "presence_penalty"):
            if key in validated:
                request_params[key] = validated[key]

        # 函数调用参数
        if "tools" in validated:
            request_params["tools"] = validated["tools"]
        if "tool_choice" in validated:
            request_params["tool_choice"] = validated["tool_choice"]

        # extra_body: 联网搜索和思考模式
        extra_body = {}
        if validated.get("web_search"):
            extra_body["enable_search"] = True
        if validated.get("enable_thinking"):
            extra_body["enable_thinking"] = True
        if extra_body:
            request_params["extra_body"] = extra_body

        try:
            response = await client.chat.completions.create(**request_params)
            async for chunk in response:
                yield self._parse_chunk(chunk)
        except Exception as e:
            raise StreamError(f"Qwen 流式调用异常: {e}") from e

    def _parse_chunk(self, chunk) -> StreamChunk:
        """解析OpenAI SDK的流式chunk为统一的StreamChunk"""
        result = StreamChunk()

        if chunk.choices:
            delta = chunk.choices[0].delta
            result.content = delta.content
            result.finish_reason = chunk.choices[0].finish_reason

            # Qwen 思考内容通过 reasoning_content 字段返回
            if hasattr(delta, "reasoning_content"):
                result.reasoning_content = delta.reasoning_content

            # 工具调用
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                result.tool_calls = delta.tool_calls

        # Token用量（仅最后一个chunk）
        if chunk.usage:
            result.usage = TokenUsage(
                prompt_tokens=chunk.usage.prompt_tokens or 0,
                completion_tokens=chunk.usage.completion_tokens or 0,
                total_tokens=chunk.usage.total_tokens or 0,
            )

        return result


# 自动注册到 ModelRegistry
from milu.providers import ModelRegistry
ModelRegistry.register("qwen", QwenLLM)
```

- [ ] **Step 5: 运行测试确认通过**

```bash
uv run pytest tests/test_qwen.py -v
```

Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add src/milu/providers/qwen.py tests/conftest.py tests/test_qwen.py
git commit -m "feat: 添加QwenLLM通义千问模型实现"
```

---

## Task 7: KimiLLM 实现

**Files:**
- Create: `src/milu/providers/kimi.py`
- Create: `tests/test_kimi.py`

- [ ] **Step 1: 编写失败测试 tests/test_kimi.py**

```python
"""测试 KimiLLM（月之暗面 Kimi）实现"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from milu.models.message import Message, MessageRole
from milu.models.response import StreamChunk
from milu.providers.kimi import KimiLLM
from tests.conftest import MockChunk, MockChoice, MockDelta, MockUsage


class TestKimiCapabilities:
    """测试 Kimi 能力声明"""

    def test_capabilities(self):
        llm = KimiLLM(api_key="test", model="moonshot-v1-128k")
        caps = llm.capabilities
        assert caps.supports_streaming is True
        assert caps.supports_function_calling is True
        assert caps.supports_json_mode is True
        assert caps.supports_web_search is True
        assert caps.supports_thinking is True
        assert caps.supports_embedding is False
        assert caps.supports_vision is True
        assert caps.supports_audio_understand is False
        assert caps.supports_video is False
        assert caps.supports_document is True
        assert caps.supports_image_generation is False
        assert caps.max_context_window == 262144

    def test_provider_name(self):
        llm = KimiLLM(api_key="test", model="moonshot-v1-128k")
        assert llm.provider_name == "kimi"

    def test_base_url(self):
        llm = KimiLLM(api_key="test", model="moonshot-v1-128k")
        assert "moonshot" in llm.base_url


class TestKimiAvailableParams:
    """测试 Kimi 可用参数"""

    def test_has_web_search(self):
        params = KimiLLM(api_key="test", model="m").get_available_params()
        assert "web_search" in params

    def test_has_thinking_with_level(self):
        params = KimiLLM(api_key="test", model="m").get_available_params()
        assert "enable_thinking" in params
        assert "thinking_level" in params

    def test_no_image_generation(self):
        params = KimiLLM(api_key="test", model="m").get_available_params()
        assert "image_size" not in params


class TestKimiChat:
    """测试 Kimi 流式聊天"""

    @pytest.mark.asyncio
    async def test_basic_chat(self):
        llm = KimiLLM(api_key="test-key", model="moonshot-v1-128k")
        mock_chunks = [
            MockChunk(choices=[MockChoice(delta=MockDelta(content="你好"))]),
            MockChunk(
                choices=[MockChoice(delta=MockDelta(), finish_reason="stop")],
                usage=MockUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
            ),
        ]

        async def mock_stream():
            for c in mock_chunks:
                yield c

        with patch.object(llm, "_get_client") as mock_get:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=mock_stream())
            mock_get.return_value = client

            messages = [Message(role=MessageRole.USER, content="你好")]
            chunks = [c async for c in llm.chat(messages)]

        assert len(chunks) == 2
        assert chunks[0].content == "你好"
        assert chunks[1].usage.total_tokens == 7

    @pytest.mark.asyncio
    async def test_thinking_level_mapping(self):
        """Kimi思考级别映射为reasoning_effort"""
        llm = KimiLLM(api_key="test-key", model="moonshot-v1-128k")

        mock_chunks = [
            MockChunk(choices=[MockChoice(delta=MockDelta(content="ok"))]),
            MockChunk(choices=[MockChoice(delta=MockDelta(), finish_reason="stop")]),
        ]

        async def mock_stream():
            for c in mock_chunks:
                yield c

        with patch.object(llm, "_get_client") as mock_get:
            client = AsyncMock()
            mock_create = AsyncMock(return_value=mock_stream())
            client.chat.completions.create = mock_create
            mock_get.return_value = client

            messages = [Message(role=MessageRole.USER, content="test")]
            async for _ in llm.chat(messages, enable_thinking=True, thinking_level="high"):
                pass

            call_kwargs = mock_create.call_args.kwargs
            extra = call_kwargs.get("extra_body", {})
            assert extra.get("reasoning_effort") == "high"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_kimi.py -v
```

Expected: FAIL

- [ ] **Step 3: 创建 src/milu/providers/kimi.py**

```python
"""
KimiLLM - 月之暗面 Kimi 模型实现。

支持能力：
- 流式输出、函数调用、JSON模式、联网搜索
- 思考模式（reasoning_effort: low/medium/high）
- 图片理解、文档理解（256K超长上下文）

API文档: https://platform.moonshot.cn/docs/api/chat
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from milu.exceptions import StreamError
from milu.models.message import Message
from milu.models.response import StreamChunk, TokenUsage
from milu.providers.base import BaseLLM, ModelCapabilities

logger = logging.getLogger(__name__)


class KimiLLM(BaseLLM):
    """月之暗面 Kimi 模型实现"""

    _capabilities = ModelCapabilities(
        supports_streaming=True,
        supports_function_calling=True,
        supports_json_mode=True,
        supports_web_search=True,
        supports_thinking=True,
        supports_embedding=False,
        supports_vision=True,
        supports_audio_understand=False,
        supports_video=False,
        supports_document=True,
        supports_image_generation=False,
        supports_audio_generation=False,
        max_context_window=262144,
        supported_output_formats=("text", "json"),
    )

    _param_names = {
        "temperature", "top_p", "max_tokens", "stop",
        "frequency_penalty", "presence_penalty",
        "web_search", "web_search_strategy",
        "enable_thinking", "thinking_level",
        "tools", "tool_choice",
    }

    # 思考级别到 reasoning_effort 的映射
    _THINKING_LEVEL_MAP = {"low": "low", "medium": "medium", "high": "high"}

    @property
    def provider_name(self) -> str:
        return "kimi"

    @property
    def base_url(self) -> str:
        return "https://api.moonshot.cn/v1"

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def _get_available_param_names(self) -> set[str]:
        return self._param_names

    async def chat(self, messages: list[Message], **kwargs) -> AsyncIterator[StreamChunk]:
        """
        流式聊天接口。

        Kimi特有参数:
            web_search (bool): 是否开启联网搜索，通过内置 $web_search 工具实现
            enable_thinking (bool): 是否开启思考模式
            thinking_level (str): 思考级别，映射为 extra_body.reasoning_effort
        """
        validated = self._validate_params(kwargs)
        client = self._get_client()

        request_params = {
            "model": self.model,
            "messages": self._messages_to_dicts(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        # 基础参数
        for key in ("temperature", "top_p", "max_tokens", "stop",
                     "frequency_penalty", "presence_penalty"):
            if key in validated:
                request_params[key] = validated[key]

        # 函数调用参数
        tools = list(validated.get("tools") or [])

        # Kimi 联网搜索：通过内置 $web_search 工具实现
        if validated.get("web_search"):
            tools.append({
                "type": "builtin_function",
                "function": {"name": "$web_search"},
            })

        if tools:
            request_params["tools"] = tools
        if "tool_choice" in validated:
            request_params["tool_choice"] = validated["tool_choice"]

        # 思考模式：映射 thinking_level 为 reasoning_effort
        extra_body = {}
        if validated.get("enable_thinking"):
            level = validated.get("thinking_level", "medium")
            extra_body["reasoning_effort"] = self._THINKING_LEVEL_MAP.get(level, "medium")
        if extra_body:
            request_params["extra_body"] = extra_body

        try:
            response = await client.chat.completions.create(**request_params)
            async for chunk in response:
                yield self._parse_chunk(chunk)
        except Exception as e:
            raise StreamError(f"Kimi 流式调用异常: {e}") from e

    def _parse_chunk(self, chunk) -> StreamChunk:
        """解析流式chunk"""
        result = StreamChunk()

        if chunk.choices:
            delta = chunk.choices[0].delta
            result.content = delta.content
            result.finish_reason = chunk.choices[0].finish_reason

            if hasattr(delta, "reasoning_content"):
                result.reasoning_content = delta.reasoning_content
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                result.tool_calls = delta.tool_calls

        if chunk.usage:
            result.usage = TokenUsage(
                prompt_tokens=chunk.usage.prompt_tokens or 0,
                completion_tokens=chunk.usage.completion_tokens or 0,
                total_tokens=chunk.usage.total_tokens or 0,
            )

        return result


from milu.providers import ModelRegistry
ModelRegistry.register("kimi", KimiLLM)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_kimi.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/milu/providers/kimi.py tests/test_kimi.py
git commit -m "feat: 添加KimiLLM月之暗面模型实现"
```

---

## Task 8: GLMLLM 实现

**Files:**
- Create: `src/milu/providers/glm.py`
- Create: `tests/test_glm.py`

- [ ] **Step 1: 编写失败测试 tests/test_glm.py**

```python
"""测试 GLMLLM（智谱AI GLM）实现"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from milu.models.message import Message, MessageRole
from milu.providers.glm import GLMLLM
from tests.conftest import MockChunk, MockChoice, MockDelta, MockUsage


class TestGLMCapabilities:
    def test_capabilities(self):
        llm = GLMLLM(api_key="test", model="glm-4-plus")
        caps = llm.capabilities
        assert caps.supports_function_calling is True
        assert caps.supports_web_search is True
        assert caps.supports_thinking is True
        assert caps.supports_embedding is True
        assert caps.supports_vision is True
        assert caps.supports_video is True
        assert caps.supports_audio_understand is False
        assert caps.supports_document is False
        assert caps.supports_image_generation is False
        assert caps.max_context_window == 131072

    def test_provider_name(self):
        assert GLMLLM(api_key="test", model="m").provider_name == "glm"

    def test_base_url(self):
        assert "bigmodel" in GLMLLM(api_key="test", model="m").base_url


class TestGLMAvailableParams:
    def test_has_web_search(self):
        params = GLMLLM(api_key="test", model="m").get_available_params()
        assert "web_search" in params

    def test_has_thinking(self):
        params = GLMLLM(api_key="test", model="m").get_available_params()
        assert "enable_thinking" in params


class TestGLMChat:
    @pytest.mark.asyncio
    async def test_basic_chat(self):
        llm = GLMLLM(api_key="test-key", model="glm-4-plus")
        mock_chunks = [
            MockChunk(choices=[MockChoice(delta=MockDelta(content="你好"))]),
            MockChunk(
                choices=[MockChoice(delta=MockDelta(), finish_reason="stop")],
                usage=MockUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
            ),
        ]

        async def mock_stream():
            for c in mock_chunks:
                yield c

        with patch.object(llm, "_get_client") as mock_get:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=mock_stream())
            mock_get.return_value = client

            messages = [Message(role=MessageRole.USER, content="你好")]
            chunks = [c async for c in llm.chat(messages)]

        assert len(chunks) == 2
        assert chunks[0].content == "你好"

    @pytest.mark.asyncio
    async def test_thinking_ignores_level(self):
        """GLM思考模式仅用enable_thinking，忽略thinking_level"""
        llm = GLMLLM(api_key="test-key", model="glm-4-plus")
        mock_chunks = [
            MockChunk(choices=[MockChoice(delta=MockDelta(content="ok"))]),
            MockChunk(choices=[MockChoice(delta=MockDelta(), finish_reason="stop")]),
        ]

        async def mock_stream():
            for c in mock_chunks:
                yield c

        with patch.object(llm, "_get_client") as mock_get:
            client = AsyncMock()
            mock_create = AsyncMock(return_value=mock_stream())
            client.chat.completions.create = mock_create
            mock_get.return_value = client

            messages = [Message(role=MessageRole.USER, content="test")]
            async for _ in llm.chat(messages, enable_thinking=True, thinking_level="high"):
                pass

            call_kwargs = mock_create.call_args.kwargs
            extra = call_kwargs.get("extra_body", {})
            # GLM 只传 enable_thinking，不传 level
            assert extra.get("enable_thinking") is True
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_glm.py -v
```

Expected: FAIL

- [ ] **Step 3: 创建 src/milu/providers/glm.py**

```python
"""
GLMLLM - 智谱AI GLM 模型实现。

支持能力：
- 流式输出、函数调用（并行调用）、JSON模式、联网搜索
- 思考模式（仅enable_thinking开关，忽略thinking_level）
- 文本嵌入、图片理解、视频理解

API文档: https://open.bigmodel.cn/dev/api
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from milu.exceptions import StreamError
from milu.models.message import Message
from milu.models.response import StreamChunk, TokenUsage
from milu.providers.base import BaseLLM, ModelCapabilities

logger = logging.getLogger(__name__)


class GLMLLM(BaseLLM):
    """智谱AI GLM 模型实现"""

    _capabilities = ModelCapabilities(
        supports_streaming=True,
        supports_function_calling=True,
        supports_json_mode=True,
        supports_web_search=True,
        supports_thinking=True,
        supports_embedding=True,
        supports_vision=True,
        supports_audio_understand=False,
        supports_video=True,
        supports_document=False,
        supports_image_generation=False,
        supports_audio_generation=False,
        max_context_window=131072,
        supported_output_formats=("text", "json"),
    )

    _param_names = {
        "temperature", "top_p", "max_tokens", "stop",
        "frequency_penalty", "presence_penalty",
        "web_search", "web_search_strategy",
        "enable_thinking", "thinking_level",
        "tools", "tool_choice",
    }

    @property
    def provider_name(self) -> str:
        return "glm"

    @property
    def base_url(self) -> str:
        return "https://open.bigmodel.cn/api/paas/v4"

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def _get_available_param_names(self) -> set[str]:
        return self._param_names

    async def chat(self, messages: list[Message], **kwargs) -> AsyncIterator[StreamChunk]:
        """
        流式聊天接口。

        GLM特有参数:
            web_search (bool): 联网搜索，通过内置web_search工具实现
            enable_thinking (bool): 思考模式开关
            thinking_level (str): GLM不支持此参数，静默忽略
        """
        validated = self._validate_params(kwargs)
        client = self._get_client()

        request_params = {
            "model": self.model,
            "messages": self._messages_to_dicts(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        # 基础参数
        for key in ("temperature", "top_p", "max_tokens", "stop",
                     "frequency_penalty", "presence_penalty"):
            if key in validated:
                request_params[key] = validated[key]

        # 函数调用 + 联网搜索
        tools = list(validated.get("tools") or [])
        if validated.get("web_search"):
            tools.append({"type": "web_search"})
        if tools:
            request_params["tools"] = tools
        if "tool_choice" in validated:
            request_params["tool_choice"] = validated["tool_choice"]

        # 思考模式：仅用enable_thinking，忽略thinking_level
        extra_body = {}
        if validated.get("enable_thinking"):
            extra_body["enable_thinking"] = True
        if extra_body:
            request_params["extra_body"] = extra_body

        try:
            response = await client.chat.completions.create(**request_params)
            async for chunk in response:
                yield self._parse_chunk(chunk)
        except Exception as e:
            raise StreamError(f"GLM 流式调用异常: {e}") from e

    def _parse_chunk(self, chunk) -> StreamChunk:
        result = StreamChunk()
        if chunk.choices:
            delta = chunk.choices[0].delta
            result.content = delta.content
            result.finish_reason = chunk.choices[0].finish_reason
            if hasattr(delta, "reasoning_content"):
                result.reasoning_content = delta.reasoning_content
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                result.tool_calls = delta.tool_calls
        if chunk.usage:
            result.usage = TokenUsage(
                prompt_tokens=chunk.usage.prompt_tokens or 0,
                completion_tokens=chunk.usage.completion_tokens or 0,
                total_tokens=chunk.usage.total_tokens or 0,
            )
        return result


from milu.providers import ModelRegistry
ModelRegistry.register("glm", GLMLLM)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_glm.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/milu/providers/glm.py tests/test_glm.py
git commit -m "feat: 添加GLMLLM智谱AI模型实现"
```

---

## Task 9: DeepSeekLLM 实现

**Files:**
- Create: `src/milu/providers/deepseek.py`
- Create: `tests/test_deepseek.py`

- [ ] **Step 1: 编写失败测试 tests/test_deepseek.py**

```python
"""测试 DeepSeekLLM 实现"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from milu.models.message import Message, MessageRole
from milu.providers.deepseek import DeepSeekLLM
from tests.conftest import MockChunk, MockChoice, MockDelta, MockUsage


class TestDeepSeekCapabilities:
    def test_capabilities(self):
        llm = DeepSeekLLM(api_key="test", model="deepseek-chat")
        caps = llm.capabilities
        assert caps.supports_function_calling is True
        assert caps.supports_thinking is True
        assert caps.supports_web_search is False
        assert caps.supports_embedding is False
        assert caps.supports_vision is False
        assert caps.max_context_window == 131072

    def test_provider_name(self):
        assert DeepSeekLLM(api_key="test", model="m").provider_name == "deepseek"


class TestDeepSeekAvailableParams:
    def test_no_web_search(self):
        params = DeepSeekLLM(api_key="test", model="m").get_available_params()
        assert "web_search" not in params

    def test_has_thinking(self):
        params = DeepSeekLLM(api_key="test", model="m").get_available_params()
        assert "enable_thinking" in params
        assert "thinking_level" in params


class TestDeepSeekChat:
    @pytest.mark.asyncio
    async def test_basic_chat(self):
        llm = DeepSeekLLM(api_key="test-key", model="deepseek-chat")
        mock_chunks = [
            MockChunk(choices=[MockChoice(delta=MockDelta(content="你好"))]),
            MockChunk(
                choices=[MockChoice(delta=MockDelta(), finish_reason="stop")],
                usage=MockUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
            ),
        ]

        async def mock_stream():
            for c in mock_chunks:
                yield c

        with patch.object(llm, "_get_client") as mock_get:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=mock_stream())
            mock_get.return_value = client

            messages = [Message(role=MessageRole.USER, content="你好")]
            chunks = [c async for c in llm.chat(messages)]

        assert chunks[0].content == "你好"

    @pytest.mark.asyncio
    async def test_thinking_level_to_budget_mapping(self):
        """DeepSeek思考级别映射为thinking_budget"""
        llm = DeepSeekLLM(api_key="test-key", model="deepseek-reasoner")
        mock_chunks = [
            MockChunk(choices=[MockChoice(delta=MockDelta(content="ok"))]),
            MockChunk(choices=[MockChoice(delta=MockDelta(), finish_reason="stop")]),
        ]

        async def mock_stream():
            for c in mock_chunks:
                yield c

        # 测试三个级别的映射
        expected_budgets = {"low": 1024, "medium": 4096, "high": 8192}
        for level, expected_budget in expected_budgets.items():
            with patch.object(llm, "_get_client") as mock_get:
                client = AsyncMock()
                mock_create = AsyncMock(return_value=mock_stream())
                client.chat.completions.create = mock_create
                mock_get.return_value = client

                messages = [Message(role=MessageRole.USER, content="test")]
                async for _ in llm.chat(messages, enable_thinking=True, thinking_level=level):
                    pass

                call_kwargs = mock_create.call_args.kwargs
                extra = call_kwargs.get("extra_body", {})
                assert extra.get("thinking_budget") == expected_budget, (
                    f"level={level} expected budget={expected_budget}, got={extra.get('thinking_budget')}"
                )
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_deepseek.py -v
```

Expected: FAIL

- [ ] **Step 3: 创建 src/milu/providers/deepseek.py**

```python
"""
DeepSeekLLM - 深度求索 DeepSeek 模型实现。

支持能力：
- 流式输出、函数调用
- 思考模式（reasoning_content + thinking_budget 映射）
- 不支持联网搜索、嵌入、多模态

API文档: https://api-docs.deepseek.com/
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from milu.exceptions import StreamError
from milu.models.message import Message
from milu.models.response import StreamChunk, TokenUsage
from milu.providers.base import BaseLLM, ModelCapabilities

logger = logging.getLogger(__name__)


class DeepSeekLLM(BaseLLM):
    """DeepSeek 模型实现"""

    _capabilities = ModelCapabilities(
        supports_streaming=True,
        supports_function_calling=True,
        supports_json_mode=True,
        supports_web_search=False,
        supports_thinking=True,
        supports_embedding=False,
        supports_vision=False,
        supports_audio_understand=False,
        supports_video=False,
        supports_document=False,
        supports_image_generation=False,
        supports_audio_generation=False,
        max_context_window=131072,
        supported_output_formats=("text", "json"),
    )

    _param_names = {
        "temperature", "top_p", "max_tokens", "stop",
        "frequency_penalty", "presence_penalty",
        "enable_thinking", "thinking_level",
        "tools", "tool_choice",
    }

    # 思考级别到 thinking_budget 的映射
    _THINKING_BUDGET_MAP = {"low": 1024, "medium": 4096, "high": 8192}

    @property
    def provider_name(self) -> str:
        return "deepseek"

    @property
    def base_url(self) -> str:
        return "https://api.deepseek.com/v1"

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def _get_available_param_names(self) -> set[str]:
        return self._param_names

    async def chat(self, messages: list[Message], **kwargs) -> AsyncIterator[StreamChunk]:
        """
        流式聊天接口。

        DeepSeek特有参数:
            enable_thinking (bool): 思考模式开关
            thinking_level (str): 思考级别，映射为 extra_body.thinking_budget
        """
        validated = self._validate_params(kwargs)
        client = self._get_client()

        request_params = {
            "model": self.model,
            "messages": self._messages_to_dicts(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        # 基础参数
        for key in ("temperature", "top_p", "max_tokens", "stop",
                     "frequency_penalty", "presence_penalty"):
            if key in validated:
                request_params[key] = validated[key]

        # 函数调用
        if "tools" in validated:
            request_params["tools"] = validated["tools"]
        if "tool_choice" in validated:
            request_params["tool_choice"] = validated["tool_choice"]

        # 思考模式：thinking_level 映射为 thinking_budget
        extra_body = {}
        if validated.get("enable_thinking"):
            level = validated.get("thinking_level", "medium")
            extra_body["thinking_budget"] = self._THINKING_BUDGET_MAP.get(level, 4096)
        if extra_body:
            request_params["extra_body"] = extra_body

        try:
            response = await client.chat.completions.create(**request_params)
            async for chunk in response:
                yield self._parse_chunk(chunk)
        except Exception as e:
            raise StreamError(f"DeepSeek 流式调用异常: {e}") from e

    def _parse_chunk(self, chunk) -> StreamChunk:
        """解析流式chunk，支持reasoning_content"""
        result = StreamChunk()
        if chunk.choices:
            delta = chunk.choices[0].delta
            result.content = delta.content
            result.finish_reason = chunk.choices[0].finish_reason

            # DeepSeek reasoner 模型返回 reasoning_content
            if hasattr(delta, "reasoning_content"):
                result.reasoning_content = delta.reasoning_content

            if hasattr(delta, "tool_calls") and delta.tool_calls:
                result.tool_calls = delta.tool_calls
        if chunk.usage:
            usage = chunk.usage
            result.usage = TokenUsage(
                prompt_tokens=usage.prompt_tokens or 0,
                completion_tokens=usage.completion_tokens or 0,
                total_tokens=usage.total_tokens or 0,
                # DeepSeek 可能返回 reasoning_tokens
                reasoning_tokens=getattr(usage, "completion_tokens_details", None)
                    and getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0,
            )
        return result


from milu.providers import ModelRegistry
ModelRegistry.register("deepseek", DeepSeekLLM)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_deepseek.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/milu/providers/deepseek.py tests/test_deepseek.py
git commit -m "feat: 添加DeepSeekLLM深度求索模型实现"
```

---

## Task 10: MiniMaxLLM 实现

**Files:**
- Create: `src/milu/providers/minimax.py`
- Create: `tests/test_minimax.py`

- [ ] **Step 1: 编写失败测试 tests/test_minimax.py**

```python
"""测试 MiniMaxLLM 实现"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from milu.models.message import Message, MessageRole
from milu.providers.minimax import MiniMaxLLM
from tests.conftest import MockChunk, MockChoice, MockDelta, MockUsage


class TestMiniMaxCapabilities:
    def test_capabilities(self):
        llm = MiniMaxLLM(api_key="test", model="MiniMax-Text-01")
        caps = llm.capabilities
        assert caps.supports_function_calling is True
        assert caps.supports_web_search is False
        assert caps.supports_thinking is False
        assert caps.supports_vision is True
        assert caps.supports_audio_understand is True
        assert caps.supports_image_generation is True
        assert caps.supports_audio_generation is True
        assert caps.max_context_window == 1000000

    def test_provider_name(self):
        assert MiniMaxLLM(api_key="test", model="m").provider_name == "minimax"


class TestMiniMaxAvailableParams:
    def test_no_web_search(self):
        params = MiniMaxLLM(api_key="test", model="m").get_available_params()
        assert "web_search" not in params

    def test_no_thinking(self):
        params = MiniMaxLLM(api_key="test", model="m").get_available_params()
        assert "enable_thinking" not in params

    def test_has_basic_params(self):
        params = MiniMaxLLM(api_key="test", model="m").get_available_params()
        assert "temperature" in params
        assert "tools" in params


class TestMiniMaxChat:
    @pytest.mark.asyncio
    async def test_basic_chat(self):
        llm = MiniMaxLLM(api_key="test-key", model="MiniMax-Text-01")
        mock_chunks = [
            MockChunk(choices=[MockChoice(delta=MockDelta(content="你好"))]),
            MockChunk(
                choices=[MockChoice(delta=MockDelta(), finish_reason="stop")],
                usage=MockUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
            ),
        ]

        async def mock_stream():
            for c in mock_chunks:
                yield c

        with patch.object(llm, "_get_client") as mock_get:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=mock_stream())
            mock_get.return_value = client

            messages = [Message(role=MessageRole.USER, content="你好")]
            chunks = [c async for c in llm.chat(messages)]

        assert chunks[0].content == "你好"
        assert chunks[1].usage.total_tokens == 7
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_minimax.py -v
```

Expected: FAIL

- [ ] **Step 3: 创建 src/milu/providers/minimax.py**

```python
"""
MiniMaxLLM - MiniMax 模型实现。

支持能力：
- 流式输出、函数调用、JSON模式
- 图片理解、音频理解
- 图片生成、音频生成（TTS）
- 1,000,000 超长上下文窗口
- 不支持联网搜索和思考模式

API文档: https://www.minimaxi.com/document/introduction
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from milu.exceptions import StreamError
from milu.models.message import Message
from milu.models.response import StreamChunk, TokenUsage
from milu.providers.base import BaseLLM, ModelCapabilities

logger = logging.getLogger(__name__)


class MiniMaxLLM(BaseLLM):
    """MiniMax 模型实现"""

    _capabilities = ModelCapabilities(
        supports_streaming=True,
        supports_function_calling=True,
        supports_json_mode=True,
        supports_web_search=False,
        supports_thinking=False,
        supports_embedding=True,
        supports_vision=True,
        supports_audio_understand=True,
        supports_video=False,
        supports_document=False,
        supports_image_generation=True,
        supports_audio_generation=True,
        max_context_window=1000000,
        supported_output_formats=("text", "json"),
    )

    # MiniMax 不支持 web_search 和 thinking 相关参数
    _param_names = {
        "temperature", "top_p", "max_tokens", "stop",
        "frequency_penalty", "presence_penalty",
        "tools", "tool_choice",
    }

    @property
    def provider_name(self) -> str:
        return "minimax"

    @property
    def base_url(self) -> str:
        return "https://api.minimax.chat/v1"

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def _get_available_param_names(self) -> set[str]:
        return self._param_names

    async def chat(self, messages: list[Message], **kwargs) -> AsyncIterator[StreamChunk]:
        """流式聊天接口。MiniMax不支持联网搜索和思考模式的特有参数。"""
        validated = self._validate_params(kwargs)
        client = self._get_client()

        request_params = {
            "model": self.model,
            "messages": self._messages_to_dicts(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        # 基础参数
        for key in ("temperature", "top_p", "max_tokens", "stop",
                     "frequency_penalty", "presence_penalty"):
            if key in validated:
                request_params[key] = validated[key]

        # 函数调用
        if "tools" in validated:
            request_params["tools"] = validated["tools"]
        if "tool_choice" in validated:
            request_params["tool_choice"] = validated["tool_choice"]

        try:
            response = await client.chat.completions.create(**request_params)
            async for chunk in response:
                yield self._parse_chunk(chunk)
        except Exception as e:
            raise StreamError(f"MiniMax 流式调用异常: {e}") from e

    def _parse_chunk(self, chunk) -> StreamChunk:
        result = StreamChunk()
        if chunk.choices:
            delta = chunk.choices[0].delta
            result.content = delta.content
            result.finish_reason = chunk.choices[0].finish_reason
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                result.tool_calls = delta.tool_calls
        if chunk.usage:
            result.usage = TokenUsage(
                prompt_tokens=chunk.usage.prompt_tokens or 0,
                completion_tokens=chunk.usage.completion_tokens or 0,
                total_tokens=chunk.usage.total_tokens or 0,
            )
        return result


from milu.providers import ModelRegistry
ModelRegistry.register("minimax", MiniMaxLLM)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_minimax.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/milu/providers/minimax.py tests/test_minimax.py
git commit -m "feat: 添加MiniMaxLLM模型实现"
```

---

## Task 11: DoubaoLLM 实现

**Files:**
- Create: `src/milu/providers/doubao.py`
- Create: `tests/test_doubao.py`

- [ ] **Step 1: 编写失败测试 tests/test_doubao.py**

```python
"""测试 DoubaoLLM（豆包/火山引擎）实现"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from milu.models.message import Message, MessageRole
from milu.providers.doubao import DoubaoLLM
from tests.conftest import MockChunk, MockChoice, MockDelta, MockUsage


class TestDoubaoCapabilities:
    def test_capabilities(self):
        llm = DoubaoLLM(api_key="test", model="doubao-1.5-pro-32k")
        caps = llm.capabilities
        assert caps.supports_function_calling is True
        assert caps.supports_web_search is True
        assert caps.supports_thinking is False
        assert caps.supports_vision is True
        assert caps.supports_embedding is True
        assert caps.supports_image_generation is True
        assert caps.supports_audio_understand is False
        assert caps.max_context_window == 131072

    def test_provider_name(self):
        assert DoubaoLLM(api_key="test", model="m").provider_name == "doubao"

    def test_base_url(self):
        assert "volces" in DoubaoLLM(api_key="test", model="m").base_url


class TestDoubaoAvailableParams:
    def test_has_web_search(self):
        params = DoubaoLLM(api_key="test", model="m").get_available_params()
        assert "web_search" in params

    def test_no_thinking(self):
        params = DoubaoLLM(api_key="test", model="m").get_available_params()
        assert "enable_thinking" not in params


class TestDoubaoChat:
    @pytest.mark.asyncio
    async def test_basic_chat(self):
        llm = DoubaoLLM(api_key="test-key", model="doubao-1.5-pro-32k")
        mock_chunks = [
            MockChunk(choices=[MockChoice(delta=MockDelta(content="你好"))]),
            MockChunk(
                choices=[MockChoice(delta=MockDelta(), finish_reason="stop")],
                usage=MockUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
            ),
        ]

        async def mock_stream():
            for c in mock_chunks:
                yield c

        with patch.object(llm, "_get_client") as mock_get:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=mock_stream())
            mock_get.return_value = client

            messages = [Message(role=MessageRole.USER, content="你好")]
            chunks = [c async for c in llm.chat(messages)]

        assert chunks[0].content == "你好"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_doubao.py -v
```

Expected: FAIL

- [ ] **Step 3: 创建 src/milu/providers/doubao.py**

```python
"""
DoubaoLLM - 豆包（字节跳动/火山引擎）模型实现。

支持能力：
- 流式输出、函数调用（部分支持）、JSON模式、联网搜索
- 图片理解、文本嵌入
- 图片生成
- 不支持思考模式和音频理解

API文档: https://www.volcengine.com/docs/82379/1399008
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from milu.exceptions import StreamError
from milu.models.message import Message
from milu.models.response import StreamChunk, TokenUsage
from milu.providers.base import BaseLLM, ModelCapabilities

logger = logging.getLogger(__name__)


class DoubaoLLM(BaseLLM):
    """豆包（火山引擎）模型实现"""

    _capabilities = ModelCapabilities(
        supports_streaming=True,
        supports_function_calling=True,
        supports_json_mode=True,
        supports_web_search=True,
        supports_thinking=False,
        supports_embedding=True,
        supports_vision=True,
        supports_audio_understand=False,
        supports_video=False,
        supports_document=False,
        supports_image_generation=True,
        supports_audio_generation=False,
        max_context_window=131072,
        supported_output_formats=("text", "json"),
    )

    # Doubao 不支持 thinking 相关参数
    _param_names = {
        "temperature", "top_p", "max_tokens", "stop",
        "frequency_penalty", "presence_penalty",
        "web_search", "web_search_strategy",
        "tools", "tool_choice",
    }

    @property
    def provider_name(self) -> str:
        return "doubao"

    @property
    def base_url(self) -> str:
        return "https://ark.cn-beijing.volces.com/api/v3"

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def _get_available_param_names(self) -> set[str]:
        return self._param_names

    async def chat(self, messages: list[Message], **kwargs) -> AsyncIterator[StreamChunk]:
        """
        流式聊天接口。

        Doubao特有参数:
            web_search (bool): 联网搜索，通过内置web_search工具实现
        """
        validated = self._validate_params(kwargs)
        client = self._get_client()

        request_params = {
            "model": self.model,
            "messages": self._messages_to_dicts(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        # 基础参数
        for key in ("temperature", "top_p", "max_tokens", "stop",
                     "frequency_penalty", "presence_penalty"):
            if key in validated:
                request_params[key] = validated[key]

        # 函数调用 + 联网搜索
        tools = list(validated.get("tools") or [])
        if validated.get("web_search"):
            tools.append({"type": "web_search"})
        if tools:
            request_params["tools"] = tools
        if "tool_choice" in validated:
            request_params["tool_choice"] = validated["tool_choice"]

        try:
            response = await client.chat.completions.create(**request_params)
            async for chunk in response:
                yield self._parse_chunk(chunk)
        except Exception as e:
            raise StreamError(f"Doubao 流式调用异常: {e}") from e

    def _parse_chunk(self, chunk) -> StreamChunk:
        result = StreamChunk()
        if chunk.choices:
            delta = chunk.choices[0].delta
            result.content = delta.content
            result.finish_reason = chunk.choices[0].finish_reason
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                result.tool_calls = delta.tool_calls
        if chunk.usage:
            result.usage = TokenUsage(
                prompt_tokens=chunk.usage.prompt_tokens or 0,
                completion_tokens=chunk.usage.completion_tokens or 0,
                total_tokens=chunk.usage.total_tokens or 0,
            )
        return result


from milu.providers import ModelRegistry
ModelRegistry.register("doubao", DoubaoLLM)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_doubao.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/milu/providers/doubao.py tests/test_doubao.py
git commit -m "feat: 添加DoubaoLLM豆包模型实现"
```

---

## Task 12: 集成导出与使用示例

**Files:**
- Modify: `src/milu/providers/__init__.py`
- Modify: `src/milu/__init__.py`
- Create: `examples/basic_usage.py`

- [ ] **Step 1: 更新 providers/__init__.py 导入所有厂商触发自动注册**

替换文件底部的注释部分：

```python
# 导入所有厂商模块以触发自动注册
from milu.providers import qwen as _qwen
from milu.providers import kimi as _kimi
from milu.providers import glm as _glm
from milu.providers import deepseek as _deepseek
from milu.providers import minimax as _minimax
from milu.providers import doubao as _doubao
```

- [ ] **Step 2: 更新 src/milu/__init__.py 完整导出**

```python
"""milu - 统一AI模型抽象层"""

from milu.exceptions import (
    MiluError,
    AuthenticationError,
    FeatureNotSupportedError,
    ModelConfigError,
    ModelNotAvailableError,
    RateLimitError,
    StreamError,
)
from milu.models import Message, MessageRole, StreamChunk, TokenUsage
from milu.providers import ModelRegistry
from milu.providers.base import BaseLLM, ModelCapabilities

__all__ = [
    # 异常
    "MiluError",
    "ModelConfigError",
    "AuthenticationError",
    "RateLimitError",
    "ModelNotAvailableError",
    "StreamError",
    "FeatureNotSupportedError",
    # 数据模型
    "Message",
    "MessageRole",
    "StreamChunk",
    "TokenUsage",
    # 核心类
    "BaseLLM",
    "ModelCapabilities",
    "ModelRegistry",
]
```

- [ ] **Step 3: 创建 examples/basic_usage.py**

```python
"""
基础使用示例 - 展示如何使用统一抽象层调用不同厂商的模型。

运行前请确保已设置对应厂商的API Key环境变量：
    export QWEN_API_KEY=sk-xxx
    export DEEPSEEK_API_KEY=sk-xxx
"""

import asyncio

from milu import Message, MessageRole, ModelRegistry
from milu.providers.qwen import QwenLLM
from milu.providers.deepseek import DeepSeekLLM


async def demo_direct_usage():
    """示例1：直接导入使用"""
    print("=== 直接导入使用 ===")

    # 直接创建 QwenLLM 实例
    qwen = QwenLLM(model="qwen-max")

    # 查看该模型的能力
    caps = qwen.capabilities
    print(f"厂商: {qwen.provider_name}")
    print(f"模型: {qwen.model}")
    print(f"最大上下文: {caps.max_context_window}")
    print(f"支持联网搜索: {caps.supports_web_search}")
    print(f"支持思考模式: {caps.supports_thinking}")
    print(f"支持图片理解: {caps.supports_vision}")
    print()

    # 查看可用参数
    params = qwen.get_available_params()
    print(f"可用参数: {list(params.keys())}")
    print()

    # 流式聊天
    messages = [
        Message(role=MessageRole.SYSTEM, content="你是一个简洁的助手，回答控制在50字以内。"),
        Message(role=MessageRole.USER, content="什么是Python？"),
    ]

    print("Qwen 回复: ", end="", flush=True)
    async for chunk in qwen.chat(messages, temperature=0.7):
        if chunk.content:
            print(chunk.content, end="", flush=True)
        if chunk.usage:
            print(f"\n[Token用量: {chunk.usage.total_tokens}]")
    print()


async def demo_factory_usage():
    """示例2：通过工厂注册表创建"""
    print("=== 工厂模式创建 ===")

    # 列出所有可用厂商
    providers = ModelRegistry.list_providers()
    print(f"已注册厂商: {providers}")
    print()

    # 通过工厂创建 DeepSeek
    deepseek = ModelRegistry.create(
        "deepseek",
        model="deepseek-chat",
    )

    messages = [
        Message(role=MessageRole.USER, content="用一句话解释量子计算"),
    ]

    print("DeepSeek 回复: ", end="", flush=True)
    async for chunk in deepseek.chat(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


async def demo_thinking_mode():
    """示例3：思考模式"""
    print("=== 思考模式 ===")

    deepseek = DeepSeekLLM(model="deepseek-reasoner")

    messages = [
        Message(role=MessageRole.USER, content="24点游戏：用 3, 3, 8, 8 凑出24"),
    ]

    print("DeepSeek (思考模式) 回复:")
    thinking_text = ""
    content_text = ""

    async for chunk in deepseek.chat(
        messages,
        enable_thinking=True,
        thinking_level="high",  # 映射为 thinking_budget=8192
    ):
        if chunk.reasoning_content:
            thinking_text += chunk.reasoning_content
        if chunk.content:
            content_text += chunk.content

    print(f"  思考过程: {thinking_text[:200]}...")
    print(f"  最终回答: {content_text}")
    print()


async def demo_web_search():
    """示例4：联网搜索"""
    print("=== 联网搜索 ===")

    qwen = QwenLLM(model="qwen-max")

    messages = [
        Message(role=MessageRole.USER, content="今天的科技新闻有什么？"),
    ]

    print("Qwen (联网搜索) 回复: ", end="", flush=True)
    async for chunk in qwen.chat(messages, web_search=True):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


async def demo_capability_check():
    """示例5：运行时能力检查"""
    print("=== 能力检查 ===")

    # 对比不同厂商的能力
    models = {
        "qwen": QwenLLM(model="qwen-max"),
        "deepseek": DeepSeekLLM(model="deepseek-chat"),
    }

    for name, model in models.items():
        caps = model.capabilities
        print(f"\n{name}:")
        print(f"  联网搜索: {'✅' if caps.supports_web_search else '❌'}")
        print(f"  思考模式: {'✅' if caps.supports_thinking else '❌'}")
        print(f"  图片理解: {'✅' if caps.supports_vision else '❌'}")
        print(f"  上下文窗口: {caps.max_context_window:,}")


async def main():
    """运行所有示例"""
    # 取消注释想运行的示例（需要对应API Key）
    # await demo_direct_usage()
    # await demo_factory_usage()
    # await demo_thinking_mode()
    # await demo_web_search()
    await demo_capability_check()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: 运行测试确认全部通过**

```bash
uv run pytest tests/ -v
```

Expected: 全部 PASS

- [ ] **Step 5: 运行示例验证能力检查（无需API Key）**

```bash
uv run python examples/basic_usage.py
```

Expected: 输出各厂商的能力对比信息

- [ ] **Step 6: 提交**

```bash
git add src/milu/ examples/
git commit -m "feat: 完善包导出和基础使用示例"
```

---

## Task 13: 运行全部测试并验证

- [ ] **Step 1: 运行完整测试套件**

```bash
uv run pytest tests/ -v --tb=short
```

Expected: 所有测试通过

- [ ] **Step 2: 验证导入链完整性**

```bash
uv run python -c "
from milu import (
    ModelRegistry, BaseLLM, ModelCapabilities,
    Message, MessageRole, StreamChunk, TokenUsage,
    MiluError, AuthenticationError,
)

# 验证所有厂商已注册
providers = ModelRegistry.list_providers()
print(f'已注册厂商: {sorted(providers)}')
assert len(providers) == 6, f'期望6个厂商，实际{len(providers)}个'

# 验证每个厂商可实例化
for name in providers:
    model = ModelRegistry.create(name, api_key='test', model='test')
    caps = model.capabilities
    params = model.get_available_params()
    print(f'{name}: 能力数={sum(1 for v in vars(caps).values() if v is True)}, 参数数={len(params)}')

print('全部验证通过！')
"
```

Expected: `全部验证通过！`

- [ ] **Step 3: 最终提交（如有修复）**

```bash
git add -A
git commit -m "fix: 最终验证和修复"
```
