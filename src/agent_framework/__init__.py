"""AI Agent Framework - 统一AI模型抽象层，兼容6家国内大模型厂商API"""

from agent_framework.exceptions import (
    AgentFrameworkError,
    AuthenticationError,
    FeatureNotSupportedError,
    ModelConfigError,
    ModelNotAvailableError,
    RateLimitError,
    StreamError,
)
from agent_framework.models.message import Message, MessageRole
from agent_framework.models.response import StreamChunk, TokenUsage
from agent_framework.providers import ModelRegistry

__all__ = [
    # Exceptions
    "AgentFrameworkError",
    "AuthenticationError",
    "FeatureNotSupportedError",
    "ModelConfigError",
    "ModelNotAvailableError",
    "RateLimitError",
    "StreamError",
    # Models
    "Message",
    "MessageRole",
    "StreamChunk",
    "TokenUsage",
    # Providers
    "ModelRegistry",
]
