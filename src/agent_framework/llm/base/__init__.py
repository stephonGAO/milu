"""LLM 基础类型 - 消息、响应、配置和异常"""

from agent_framework.llm.base.message import Message, MessageRole
from agent_framework.llm.base.response import StreamChunk, TokenUsage
from agent_framework.llm.base.config import (
    ModelConfig,
    WebSearchConfig,
    ThinkingConfig,
    FunctionCallingConfig,
    ImageGenerationConfig,
    AudioGenerationConfig,
)
from agent_framework.llm.base.exceptions import (
    AuthenticationError,
    FeatureNotSupportedError,
    ModelConfigError,
    ModelNotAvailableError,
    RateLimitError,
    StreamError,
)

__all__ = [
    "Message",
    "MessageRole",
    "StreamChunk",
    "TokenUsage",
    "ModelConfig",
    "WebSearchConfig",
    "ThinkingConfig",
    "FunctionCallingConfig",
    "ImageGenerationConfig",
    "AudioGenerationConfig",
    "AuthenticationError",
    "FeatureNotSupportedError",
    "ModelConfigError",
    "ModelNotAvailableError",
    "RateLimitError",
    "StreamError",
]
