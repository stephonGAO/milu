"""milu - 统一AI模型抽象层 + Agent 编排引擎，支持 9 个 LLM 提供商"""

from milu._env import load_env
from milu.resources import (
    builtin_prompts_dir,
    builtin_skills_dir,
    templates_dir,
    user_data_dir,
    default_session_dir,
    default_mcp_config_path,
)
from milu.exceptions import MiluError
from milu.llm.base.exceptions import (
    AuthenticationError,
    FeatureNotSupportedError,
    ModelConfigError,
    ModelNotAvailableError,
    RateLimitError,
    StreamError,
)
from milu.agent.exceptions import (
    AgentLoopError,
    AgentTimeout,
    MaxTurnsExceeded,
    TokenLimitExceeded,
    ToolCallLimitExceeded,
    ToolExecutionError,
    MCPError,
    MCPConnectionError,
)
from milu.llm.base.message import Message, MessageRole
from milu.llm.base.response import StreamChunk, TokenUsage
from milu.agent.events import (
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
from milu.llm.providers import ModelRegistry
from milu.agent import Agent, AgentConfig, AgentMode, CompactConfig, ConversationHistory, Session
from milu.agent.subagent import (
    SubAgentConfig,
    builtin_subagent_configs,
    create_subagent_tools,
)
from milu.tools.builtin.todo_write import todo_read, todo_write
from milu.tools.builtin.memory_tool import memory_read, memory_write
from milu.skills.config import SkillConfig
from milu.skills.registry import SkillRegistry
from milu.prompts.builder import PromptBuilder
from milu.tools.mcp.config import MCPServerConfig
from milu.tools.decorator import tool, ToolWrapper
from milu.serving import (
    AgentPool,
    AgentPoolConfig,
    PooledAgent,
    AgentPoolError,
    PoolBusyError,
    PoolExhaustedError,
    KeyedLLMProvider,
    ApiKeyResolver,
)

__all__ = [
    # Env
    "load_env",
    # 内置资源（提示词模板 / 技能）
    "templates_dir",
    "builtin_prompts_dir",
    "builtin_skills_dir",
    # 用户级数据/配置目录（与 CWD 解耦）
    "user_data_dir",
    "default_session_dir",
    "default_mcp_config_path",
    # Root Exception
    "MiluError",
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
    "builtin_subagent_configs",
    "create_subagent_tools",
    # Todo
    "todo_read",
    "todo_write",
    # Memory
    "memory_read",
    "memory_write",
    # Providers
    "ModelRegistry",
    # MCP
    "MCPServerConfig",
    # Skills
    "SkillConfig",
    "SkillRegistry",
    # Prompts
    "PromptBuilder",
    # Tools
    "tool",
    "ToolWrapper",
    # Serving（多用户并发资源池）
    "AgentPool",
    "AgentPoolConfig",
    "PooledAgent",
    "AgentPoolError",
    "PoolBusyError",
    "PoolExhaustedError",
    # 多租户 API Key 隔离
    "KeyedLLMProvider",
    "ApiKeyResolver",
]
