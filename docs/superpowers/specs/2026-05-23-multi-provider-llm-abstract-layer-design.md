# AI Agent Framework - 多厂商模型兼容层设计

> 日期：2026-05-23
> 状态：已批准
> 阶段：第一步 - 基础模型兼容层

## 一、项目概述

构建一个统一的AI模型抽象层，兼容6家国内主流大模型厂商的API调用。通过抽象基类和能力描述符模式，实现"一个接口，多厂商适配"，同时确保各厂商特有功能按需暴露，不支持的功能在对象中不存在。

### 目标厂商

| 厂商 | 别名 | Base URL | API Key格式 |
|------|------|----------|------------|
| Qwen | 通义千问 / DashScope | `dashscope.aliyuncs.com/compatible-mode/v1` | `sk-` 前缀 |
| Kimi | 月之暗面 / Moonshot | `api.moonshot.cn/v1` | `sk-` 前缀 |
| GLM | 智谱AI / ZhipuAI | `open.bigmodel.cn/api/paas/v4/` | 无前缀 |
| DeepSeek | 深度求索 | `api.deepseek.com` | `sk-` 前缀 |
| MiniMax | - | `api.minimax.chat/v1` | 无前缀 |
| Doubao | 豆包 / 火山引擎 | `ark.cn-beijing.volces.com/api/v3` | `ark-cn-beijing-` 前缀 |

## 二、技术选型

| 维度 | 选型 | 理由 |
|------|------|------|
| 语言版本 | Python 3.10+ | 类型注解、模式匹配支持 |
| HTTP客户端 | `openai` Python SDK | 6家厂商全部兼容OpenAI格式，统一使用官方SDK最稳定 |
| 异步支持 | 全异步 async/await | Agent循环天然适合异步，性能更好 |
| 包管理 | uv + pyproject.toml | 现代化、速度快 |
| 配置管理 | 环境变量 + 代码配置类 | API Key从环境变量读取，功能参数通过代码设置 |

## 三、架构设计

### 3.1 核心模式：抽象基类 + 能力描述符

```
BaseLLM (抽象基类)
├── ModelCapabilities (数据类：声明该厂商支持的功能清单)
├── ModelConfig (配置类：根据Capabilities动态暴露参数)
├── QwenLLM (继承实现)
├── KimiLLM
├── GLMLLM
├── DeepSeekLLM
├── MiniMaxLLM
└── DoubaoLLM
```

**设计原则**：
- 扩展性：新增厂商只需继承BaseLLM + 声明能力
- 类型安全：通过dataclass提供完整类型提示
- 按需暴露：不支持的功能在对象中不存在，而非抛异常

### 3.2 项目结构

```
ai-agent-framework/
├── pyproject.toml                  # 项目配置，uv管理
├── .env.example                    # 环境变量模板
├── src/
│   └── agent_framework/
│       ├── __init__.py
│       ├── providers/              # 各厂商模型实现
│       │   ├── __init__.py         # ModelRegistry注册表
│       │   ├── base.py             # BaseLLM抽象基类 + ModelCapabilities
│       │   ├── qwen.py             # 通义千问
│       │   ├── kimi.py             # 月之暗面Kimi
│       │   ├── glm.py              # 智谱GLM
│       │   ├── deepseek.py         # DeepSeek
│       │   ├── minimax.py          # MiniMax
│       │   └── doubao.py           # 豆包/火山引擎
│       ├── models/                 # 通用数据模型
│       │   ├── __init__.py
│       │   ├── message.py          # 消息类型定义
│       │   ├── response.py         # 统一响应模型（含流式chunk）
│       │   └── config.py           # 模型配置参数定义
│       └── exceptions.py           # 统一异常体系
├── tests/
│   └── test_providers/
└── examples/
    └── basic_usage.py              # 基础使用示例
```

## 四、核心数据结构

### 4.1 ModelCapabilities - 能力描述符

```python
@dataclass(frozen=True)
class ModelCapabilities:
    """声明某个厂商/模型支持的功能集合"""
    # 基础能力
    supports_streaming: bool = True          # 流式输出
    supports_function_calling: bool = False  # 函数/工具调用
    supports_json_mode: bool = False         # JSON结构化输出
    supports_web_search: bool = False        # 联网搜索
    supports_thinking: bool = False          # 思考/推理模式
    supports_embedding: bool = False         # 文本嵌入

    # 多模态理解能力
    supports_vision: bool = False            # 图片理解
    supports_audio_understand: bool = False   # 音频理解
    supports_video: bool = False             # 视频理解
    supports_document: bool = False          # 文档理解（PDF、Word等）

    # 多模态生成能力
    supports_image_generation: bool = False   # 图片生成
    supports_audio_generation: bool = False   # 音频生成（TTS等）

    # 模型规格
    max_context_window: int = 8192           # 最大上下文窗口
    supported_output_formats: tuple = ("text",)  # 输出格式
```

### 4.2 各厂商能力矩阵

| 能力 | Qwen | Kimi | GLM | DeepSeek | MiniMax | Doubao |
|------|------|------|-----|----------|---------|--------|
| 流式输出 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 函数调用 | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️部分 |
| JSON模式 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 联网搜索 | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| 思考模式 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| 文本嵌入 | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ |
| 图片理解 | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| 音频理解 | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| 视频理解 | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| 文档理解 | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| 图片生成 | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| 音频生成 | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |

### 4.3 Message - 统一消息类型

```python
class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

@dataclass
class Message:
    """统一消息结构，兼容所有厂商的消息格式"""
    role: MessageRole
    content: str | list | None = None      # str=纯文本, list=多模态内容
    tool_calls: list[dict] | None = None    # 工具调用（assistant消息）
    tool_call_id: str | None = None         # 工具调用ID（tool消息）
    name: str | None = None                 # 工具/函数名称
```

多模态消息示例（content为list时）：
```python
Message(role=MessageRole.USER, content=[
    {"type": "text", "text": "这张图是什么？"},
    {"type": "image_url", "image_url": {"url": "https://..."}}
])
```

### 4.4 StreamChunk - 流式输出统一数据块

```python
@dataclass
class StreamChunk:
    """流式输出的统一数据块"""
    content: str | None = None              # 正文内容片段
    reasoning_content: str | None = None     # 思考过程片段
    tool_calls: list | None = None          # 工具调用信息
    finish_reason: str | None = None        # 结束原因
    usage: TokenUsage | None = None         # token统计（仅最后一条）
```

### 4.5 TokenUsage - 统一token统计

```python
@dataclass
class TokenUsage:
    """统一的token用量统计"""
    prompt_tokens: int = 0           # 输入token
    completion_tokens: int = 0       # 输出token
    total_tokens: int = 0            # 总计token
    reasoning_tokens: int = 0        # 思考过程token
```

## 五、配置参数体系

### 5.1 ModelConfig - 基础参数（所有厂商通用）

```python
@dataclass
class ModelConfig:
    """模型调用的基础参数配置"""
    model: str                        # 模型名称
    temperature: float | None = None  # 温度
    top_p: float | None = None        # 核采样概率
    max_tokens: int | None = None     # 最大输出token数
    stop: list[str] | None = None     # 停止词
    stream: bool = True               # 固定True（本框架仅流式）
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
```

### 5.2 扩展参数配置

根据厂商能力按需暴露：

```python
@dataclass
class WebSearchConfig(ModelConfig):
    """联网搜索扩展参数"""
    web_search: bool = False
    web_search_strategy: str = "auto"

@dataclass
class ThinkingConfig(ModelConfig):
    """思考/推理模式扩展参数"""
    enable_thinking: bool = False
    thinking_level: str = "medium"     # 统一接口：low / medium / high

@dataclass
class FunctionCallingConfig(ModelConfig):
    """函数调用扩展参数"""
    tools: list[dict] | None = None
    tool_choice: str | dict = "auto"

@dataclass
class ImageGenerationConfig(ModelConfig):
    """图片生成扩展参数"""
    image_size: str = "1024x1024"
    image_quality: str = "standard"
    num_images: int = 1
```

### 5.3 思考级别内部映射

调用方统一使用 `thinking_level`（low/medium/high，默认medium），各厂商内部映射：

| 级别 | Kimi | DeepSeek | Qwen | GLM |
|------|------|----------|------|-----|
| low | reasoning_effort="low" | thinking_budget=1024 | 仅enable_thinking | 仅enable_thinking |
| medium | reasoning_effort="medium" | thinking_budget=4096 | 仅enable_thinking | 仅enable_thinking |
| high | reasoning_effort="high" | thinking_budget=8192 | 仅enable_thinking | 仅enable_thinking |

不支持level设置的厂商（Qwen、GLM）会静默忽略该参数，不报错。

### 5.4 各厂商配置组合

| 厂商 | 配置组合 |
|------|---------|
| Qwen | ModelConfig + WebSearch + Thinking + FunctionCalling + ImageGeneration |
| Kimi | ModelConfig + WebSearch + Thinking + FunctionCalling |
| GLM | ModelConfig + WebSearch + Thinking + FunctionCalling |
| DeepSeek | ModelConfig + Thinking + FunctionCalling |
| MiniMax | ModelConfig + FunctionCalling + ImageGeneration + AudioGeneration |
| Doubao | ModelConfig + WebSearch + FunctionCalling + ImageGeneration |

## 六、BaseLLM 抽象基类

```python
class BaseLLM(ABC):
    """所有厂商模型的抽象基类"""

    @property
    @abstractmethod
    def provider_name(self) -> str: ...
        # 厂商标识："qwen", "kimi", "glm", "deepseek", "minimax", "doubao"

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities: ...
        # 该厂商的能力声明

    @abstractmethod
    async def chat(self, messages: list[Message], **kwargs) -> AsyncIterator[StreamChunk]: ...
        # 统一流式聊天接口

    @abstractmethod
    def get_available_params(self) -> dict: ...
        # 返回当前模型可用的参数及默认值

    def _validate_params(self, params: dict) -> dict: ...
        # 校验参数：过滤不支持的参数，记录警告日志
```

### 调用流程

```
调用方 → model.chat(messages, temperature=0.7, web_search=True)
              ↓
       _validate_params()  # 检查参数是否在capabilities中
              ↓
       构建OpenAI SDK请求（base_url + api_key + model + 参数）
              ↓
       async for chunk in client.chat.completions.create(stream=True):
           yield StreamChunk(content=..., reasoning_content=..., tool_calls=...)
```

## 七、注册与扩展机制

### ModelRegistry - 模型注册表

```python
class ModelRegistry:
    """模型注册表，管理所有厂商的LLM实现"""
    _providers: dict[str, type[BaseLLM]] = {}

    @classmethod
    def register(cls, name: str, provider_class: type[BaseLLM]): ...

    @classmethod
    def create(cls, provider: str, **kwargs) -> BaseLLM: ...

    @classmethod
    def list_providers(cls) -> list[str]: ...
```

### 新增厂商步骤

1. 在 `providers/` 下新建文件，如 `new_vendor.py`
2. 继承 `BaseLLM`，声明 `capabilities`，实现 `chat()` 方法
3. 文件末尾调用 `ModelRegistry.register("new_vendor", NewVendorLLM)`

## 八、异常体系

```python
class AgentFrameworkError(Exception):
    """框架基础异常"""

class ModelConfigError(AgentFrameworkError):
    """模型配置错误（如不支持的参数）"""

class AuthenticationError(AgentFrameworkError):
    """API Key无效或缺失"""

class RateLimitError(AgentFrameworkError):
    """请求频率超限"""

class ModelNotAvailableError(AgentFrameworkError):
    """模型不可用"""

class StreamError(AgentFrameworkError):
    """流式输出异常"""

class FeatureNotSupportedError(AgentFrameworkError):
    """请求的功能该模型不支持"""
```

## 九、环境变量

```bash
# .env.example
QWEN_API_KEY=sk-xxx
KIMI_API_KEY=sk-xxx
GLM_API_KEY=xxx
DEEPSEEK_API_KEY=sk-xxx
MINIMAX_API_KEY=xxx
DOUBAO_API_KEY=ark-cn-beijing-xxx
```

## 十、后续扩展规划（本阶段不实现）

- 第二步：Agent循环核心引擎
- 第三步：工具调用与MCP集成
- 第四步：RAG检索增强生成
- 第五步：复杂记忆系统
- 第六步：Langfuse调试与监控集成
