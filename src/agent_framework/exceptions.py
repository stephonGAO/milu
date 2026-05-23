"""统一异常体系 - 所有自定义异常的公共基类"""


class AgentFrameworkError(Exception):
    """框架基础异常，所有其他异常的父类"""
    pass
