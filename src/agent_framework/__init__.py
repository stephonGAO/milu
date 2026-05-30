"""AI Agent Framework - 统一AI模型抽象层，兼容6家国内大模型厂商API"""

from agent_framework.exceptions import AgentFrameworkError
from agent_framework.llm.base.exceptions import (
    AuthenticationError,
    FeatureNotSupportedError,
    ModelConfigError,
    ModelNotAvailableError,
    RateLimitError,
    StreamError,
)
from agent_framework.agent.exceptions import (
    AgentLoopError,
    AgentTimeout,
    MaxTurnsExceeded,
    TokenLimitExceeded,
    ToolCallLimitExceeded,
    ToolExecutionError,
    MCPError,
    MCPConnectionError,
)
from agent_framework.llm.base.message import Message, MessageRole
from agent_framework.llm.base.response import StreamChunk, TokenUsage
from agent_framework.agent.events import (
    AgentEvent,
    TextDelta,
    ReasoningDelta,
    ToolCallStart,
    ToolConfirmRequired,
    ConfirmResponse,
    ToolResult,
    AgentDone,
    AgentError,
    SubAgentEvent,
    SubAgentDone,
    HistoryCompacted,
)
from agent_framework.llm.providers import ModelRegistry
from agent_framework.agent import Agent, AgentConfig, ConversationHistory
from agent_framework.agent.subagent import SubAgentConfig, create_subagent_tools
from agent_framework.skills.config import SkillConfig
from agent_framework.skills.registry import SkillRegistry
from agent_framework.prompts.builder import PromptBuilder
from agent_framework.tools.mcp.config import MCPServerConfig

__all__ = [
    # Root Exception
    "AgentFrameworkError",
    # LLM Exceptions
    "AuthenticationError",
    "FeatureNotSupportedError",
    "ModelConfigError",
    "ModelNotAvailableError",
    "RateLimitError",
    "StreamError",
    # Agent Exceptions
    "AgentLoopError",
    "AgentTimeout",
    "MaxTurnsExceeded",
    "TokenLimitExceeded",
    "ToolCallLimitExceeded",
    "ToolExecutionError",
    "MCPError",
    "MCPConnectionError",
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
    "ToolConfirmRequired",
    "ConfirmResponse",
    "ToolResult",
    "AgentDone",
    "AgentError",
    "SubAgentEvent",
    "SubAgentDone",
    "HistoryCompacted",
    # Agent
    "Agent",
    "AgentConfig",
    "ConversationHistory",
    # SubAgent
    "SubAgentConfig",
    "create_subagent_tools",
    # Providers
    "ModelRegistry",
    # MCP
    "MCPServerConfig",
    # Skills
    "SkillConfig",
    "SkillRegistry",
    # Prompts
    "PromptBuilder",
]
