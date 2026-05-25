"""工具系统 - @tool 装饰器、ToolRegistry、ToolExecutor 和内置工具集"""
from agent_framework.tools.decorator import tool, ToolWrapper
from agent_framework.tools.registry import ToolRegistry
from agent_framework.tools.executor import ToolExecutor, ToolExecutionResult
from agent_framework.tools.builtin import BUILTIN_TOOLS

__all__ = [
    "tool", "ToolWrapper", "ToolRegistry", "ToolExecutor", "ToolExecutionResult",
    "BUILTIN_TOOLS",
]
