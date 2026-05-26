"""Agent 层异常"""

from agent_framework.exceptions import AgentFrameworkError


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


class MCPError(AgentLoopError):
    """MCP 相关异常基类"""
    pass


class MCPConnectionError(MCPError):
    """MCP 服务器连接失败"""
    pass
