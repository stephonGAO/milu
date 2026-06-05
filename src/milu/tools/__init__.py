"""工具系统 - @tool 装饰器、ToolRegistry、ToolExecutor 和内置工具集"""
from milu.tools.decorator import tool, ToolWrapper
from milu.tools.registry import ToolRegistry
from milu.tools.executor import ToolExecutor, ToolExecutionResult
from milu.tools.builtin import BUILTIN_TOOLS
from milu.tools.mcp.config import MCPServerConfig
from milu.tools.catalog import create_catalog_tools

__all__ = [
    "tool", "ToolWrapper", "ToolRegistry", "ToolExecutor", "ToolExecutionResult",
    "BUILTIN_TOOLS", "MCPServerConfig", "create_catalog_tools",
]