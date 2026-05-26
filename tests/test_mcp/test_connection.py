"""测试 MCPServerConnection 连接管理"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_framework.tools.mcp.config import MCPServerConfig
from agent_framework.tools.mcp.connection import MCPServerConnection


class TestMCPServerConnection:
    """MCPServerConnection 连接测试"""

    @pytest.mark.asyncio
    async def test_connect_stdio(self):
        """stdio 连接：应调用 stdio_client 并初始化 session"""
        config = MCPServerConfig.stdio(name="test", command="python", args=["-m", "server"])
        conn = MCPServerConnection(config)

        mock_read = MagicMock()
        mock_write = MagicMock()

        with patch.object(MCPServerConnection, "_create_stdio_transport",
                          new_callable=AsyncMock, return_value=(mock_read, mock_write)):
            with patch("mcp.ClientSession") as mock_session_cls:
                mock_session = AsyncMock()
                mock_session.initialize = AsyncMock()

                # ClientSession 作为 async context manager
                mock_cm = AsyncMock()
                mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
                mock_cm.__aexit__ = AsyncMock(return_value=False)
                mock_session_cls.return_value = mock_cm

                await conn.connect()

                mock_session.initialize.assert_called_once()

        await conn.disconnect()

    @pytest.mark.asyncio
    async def test_discover_tools(self, mock_mcp_tool, mock_session):
        """发现工具：应调用 list_tools 并转换为 ToolWrapper"""
        config = MCPServerConfig.stdio(name="fs", command="echo")
        conn = MCPServerConnection(config)
        conn._session = mock_session

        tools = await conn.discover_tools()

        mock_session.list_tools.assert_called_once()
        assert len(tools) == 1
        assert tools[0].name == "fs__read_file"

    @pytest.mark.asyncio
    async def test_discover_tools_with_filter(self, mock_session, mock_mcp_tool, mock_mcp_tool_2):
        """tool_filter 过滤工具"""
        list_result = MagicMock()
        list_result.tools = [mock_mcp_tool, mock_mcp_tool_2]
        mock_session.list_tools = AsyncMock(return_value=list_result)

        config = MCPServerConfig.stdio(
            name="fs", command="echo",
            tool_filter=["read_file"],
        )
        conn = MCPServerConnection(config)
        conn._session = mock_session

        tools = await conn.discover_tools()
        assert len(tools) == 1
        assert tools[0].name == "fs__read_file"

    @pytest.mark.asyncio
    async def test_discover_tools_no_prefix(self, mock_mcp_tool, mock_session):
        """prefix_tools=False 时不添加前缀"""
        config = MCPServerConfig.stdio(name="fs", command="echo", prefix_tools=False)
        conn = MCPServerConnection(config)
        conn._session = mock_session

        tools = await conn.discover_tools()
        assert tools[0].name == "read_file"

    @pytest.mark.asyncio
    async def test_discover_tools_dangerous(self, mock_mcp_tool, mock_session):
        """dangerous_tools 标记"""
        config = MCPServerConfig.stdio(
            name="fs", command="echo",
            dangerous_tools=["read_file"],
        )
        conn = MCPServerConnection(config)
        conn._session = mock_session

        tools = await conn.discover_tools()
        assert tools[0].dangerous is True

    @pytest.mark.asyncio
    async def test_discover_without_connect_raises(self):
        """未连接时调用 discover_tools 应抛异常"""
        config = MCPServerConfig.stdio(name="test", command="echo")
        conn = MCPServerConnection(config)

        with pytest.raises(RuntimeError, match="未连接"):
            await conn.discover_tools()

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self):
        """disconnect 应清理资源"""
        config = MCPServerConfig.stdio(name="test", command="echo")
        conn = MCPServerConnection(config)
        conn._exit_stack = AsyncMock()
        conn._exit_stack.aclose = AsyncMock()

        await conn.disconnect()
        conn._exit_stack is None

    @pytest.mark.asyncio
    async def test_properties(self):
        config = MCPServerConfig.stdio(name="myserver", command="echo")
        conn = MCPServerConnection(config)
        assert conn.name == "myserver"
        assert conn.config is config

    @pytest.mark.asyncio
    async def test_get_tools_returns_copy(self, mock_mcp_tool, mock_session):
        config = MCPServerConfig.stdio(name="fs", command="echo")
        conn = MCPServerConnection(config)
        conn._session = mock_session

        await conn.discover_tools()
        tools1 = conn.get_tools()
        tools2 = conn.get_tools()
        assert tools1 is not tools2
        assert len(tools1) == len(tools2)
