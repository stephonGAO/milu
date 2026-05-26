"""MCP 测试共享 fixtures"""
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_mcp_tool():
    """模拟 MCP Tool 对象"""
    tool = MagicMock()
    tool.name = "read_file"
    tool.description = "Read file contents"
    tool.inputSchema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
        },
        "required": ["path"],
    }
    return tool


@pytest.fixture
def mock_mcp_tool_2():
    """第二个模拟 MCP Tool"""
    tool = MagicMock()
    tool.name = "write_file"
    tool.description = "Write to a file"
    tool.inputSchema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }
    return tool


@pytest.fixture
def mock_text_content():
    """模拟 MCP TextContent"""
    content = MagicMock()
    content.__class__.__name__ = "TextContent"
    content.text = "file contents here"
    return content


@pytest.fixture
def mock_mcp_result(mock_text_content):
    """模拟 MCP CallToolResult（成功）"""
    result = MagicMock()
    result.content = [mock_text_content]
    result.isError = False
    result.structuredContent = None
    return result


@pytest.fixture
def mock_mcp_error_result():
    """模拟 MCP CallToolResult（错误）"""
    text = MagicMock()
    text.__class__.__name__ = "TextContent"
    text.text = "File not found"

    result = MagicMock()
    result.content = [text]
    result.isError = True
    result.structuredContent = None
    return result


@pytest.fixture
def mock_session(mock_mcp_tool, mock_mcp_result):
    """模拟 MCP ClientSession"""
    session = AsyncMock()

    # list_tools
    list_result = MagicMock()
    list_result.tools = [mock_mcp_tool]
    session.list_tools = AsyncMock(return_value=list_result)

    # call_tool
    session.call_tool = AsyncMock(return_value=mock_mcp_result)

    # initialize
    session.initialize = AsyncMock()

    return session
