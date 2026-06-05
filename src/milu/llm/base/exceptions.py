"""LLM 层异常"""

from milu.exceptions import MiluError


class ModelConfigError(MiluError):
    """模型配置错误，如传入了不支持的参数"""
    pass


class AuthenticationError(MiluError):
    """API Key 无效或缺失"""
    pass


class RateLimitError(MiluError):
    """请求频率超限"""
    pass


class ModelNotAvailableError(MiluError):
    """指定的模型不可用"""
    pass


class StreamError(MiluError):
    """流式输出过程中发生异常"""
    pass


class FeatureNotSupportedError(MiluError):
    """请求的功能该模型不支持"""
    pass
