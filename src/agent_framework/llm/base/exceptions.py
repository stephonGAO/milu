"""LLM 层异常"""

from agent_framework.exceptions import AgentFrameworkError


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
