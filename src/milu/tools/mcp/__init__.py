"""MCP（Model Context Protocol）客户端工具模块

提供连接外部 MCP 服务器的能力，将 MCP 工具转换为框架 ToolWrapper。
"""
from milu.tools.mcp.config import MCPServerConfig
from milu.tools.mcp.converter import convert_mcp_tool
from milu.tools.mcp.connection import MCPServerConnection
from milu.tools.mcp.manager import MCPManager

__all__ = [
    "MCPServerConfig",
    "convert_mcp_tool",
    "MCPServerConnection",
    "MCPManager",
]
