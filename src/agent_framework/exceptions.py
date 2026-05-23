"""统一异常体系 - 所有自定义异常的基类和派生类"""


class AgentFrameworkError(Exception):
    """框架基础异常，所有其他异常的父类"""
    pass


class ModelConfigError(AgentFrameworkError):
    """模型配置错误，如传入了不支持的参数"""
    pass


class AuthenticationError(AgentFrameworkError):
    """API Key 无效或缺失"""
    pass


class RateLimitError(AgentFrameworkError):
    """请求频率超限"""
    pass


class ModelNotAvailableError(AgentFrameworkError):
    """指定的模型不可用"""
    pass


class StreamError(AgentFrameworkError):
    """流式输出过程中发生异常"""
    pass


class FeatureNotSupportedError(AgentFrameworkError):
    """请求的功能该模型不支持"""
    pass


# ==================== Agent 循环相关异常 ====================

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
