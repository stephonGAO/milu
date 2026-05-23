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
from agent_framework.models.events import (
    AgentEvent,
    TextDelta,
    ReasoningDelta,
    ToolCallStart,
    ToolResult,
    AgentDone,
    AgentError,
)
from agent_framework.providers import ModelRegistry
from agent_framework.agent import Agent, AgentConfig

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
    # Agent Events
    "AgentEvent",
    "TextDelta",
    "ReasoningDelta",
    "ToolCallStart",
    "ToolResult",
    "AgentDone",
    "AgentError",
    # Agent
    "Agent",
    "AgentConfig",
    # Providers
    "ModelRegistry",
]
