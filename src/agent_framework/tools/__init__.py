"""工具系统 - @tool 装饰器和 ToolRegistry"""
from agent_framework.tools.decorator import tool, ToolWrapper
from agent_framework.tools.registry import ToolRegistry

__all__ = ["tool", "ToolWrapper", "ToolRegistry"]
