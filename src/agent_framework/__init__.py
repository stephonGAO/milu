"""AI Agent Framework - 统一AI模型抽象层，兼容6家国内大模型厂商API"""

from agent_framework._env import load_env
from agent_framework.resources import (
    builtin_prompts_dir,
    builtin_skills_dir,
    templates_dir,
)
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
    SessionLoaded,
)
from agent_framework.llm.providers import ModelRegistry
from agent_framework.agent import Agent, AgentConfig, AgentMode, CompactConfig, ConversationHistory, Session
from agent_framework.agent.subagent import SubAgentConfig, create_subagent_tools
from agent_framework.tools.builtin.todo_write import todo_read, todo_write
from agent_framework.skills.config import SkillConfig
from agent_framework.skills.registry import SkillRegistry
from agent_framework.prompts.builder import PromptBuilder
from agent_framework.tools.mcp.config import MCPServerConfig

__all__ = [
    # Env
    "load_env",
    # 内置资源（提示词模板 / 技能）
    "templates_dir",
    "builtin_prompts_dir",
    "builtin_skills_dir",
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
    "SessionLoaded",
    # Agent
    "Agent",
    "AgentConfig",
    "AgentMode",
    "CompactConfig",
    "ConversationHistory",
    "Session",
    # SubAgent
    "SubAgentConfig",
    "create_subagent_tools",
    # Todo
    "todo_read",
    "todo_write",
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
