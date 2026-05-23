"""工具系统 - @tool 装饰器、ToolRegistry 和 ToolExecutor"""
from agent_framework.tools.decorator import tool, ToolWrapper
from agent_framework.tools.registry import ToolRegistry
from agent_framework.tools.executor import ToolExecutor, ToolExecutionResult

__all__ = ["tool", "ToolWrapper", "ToolRegistry", "ToolExecutor", "ToolExecutionResult"]
