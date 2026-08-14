"""测试 MCPManager 多服务器编排"""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from milu.tools.mcp.config import MCPServerConfig
from milu.tools.mcp.manager import MCPManager
from milu.tools.decorator import ToolWrapper


def _make_mock_tool(name: str) -> ToolWrapper:
    """创建测试用 ToolWrapper"""
    async def _noop(**kwargs):
        return "ok"
    return ToolWrapper(
        name=name, description=f"Tool {name}",
        parameters_schema={"type": "object", "properties": {}},
        func=_noop, is_async=True,
    )


class TestMCPManager:
    """MCPManager 编排测试"""

    @pytest.mark.asyncio
    async def test_connect_all_success(self):
        """全部连接成功"""
        configs = [
            MCPServerConfig.stdio(name="a", command="echo"),
            MCPServerConfig.stdio(name="b", command="echo"),
        ]

        tool_a = _make_mock_tool("a__tool1")
        tool_b = _make_mock_tool("b__tool2")

        with patch("milu.tools.mcp.manager.MCPServerConnection") as MockConn:
            # 模拟两个连接
            conn_a = AsyncMock()
            conn_a.name = "a"
            conn_a.connect = AsyncMock()
            conn_a.discover_tools = AsyncMock(return_value=[tool_a])
            conn_a.disconnect = AsyncMock()

            conn_b = AsyncMock()
            conn_b.name = "b"
            conn_b.connect = AsyncMock()
            conn_b.discover_tools = AsyncMock(return_value=[tool_b])
            conn_b.disconnect = AsyncMock()

            MockConn.side_effect = [conn_a, conn_b]

            manager = MCPManager(configs)
            tools = await manager.connect_all()

            assert len(tools) == 2
            names = {t.name for t in tools}
            assert names == {"a__tool1", "b__tool2"}

    @pytest.mark.asyncio
    async def test_connect_partial_failure(self):
        """部分服务器失败，其他的仍可用"""
        configs = [
            MCPServerConfig.stdio(name="ok", command="echo"),
            MCPServerConfig.stdio(name="fail", command="bad"),
        ]

        tool = _make_mock_tool("ok__tool1")

        with patch("milu.tools.mcp.manager.MCPServerConnection") as MockConn:
            conn_ok = AsyncMock()
            conn_ok.name = "ok"
            conn_ok.connect = AsyncMock()
            conn_ok.discover_tools = AsyncMock(return_value=[tool])
            conn_ok.disconnect = AsyncMock()

            conn_fail = AsyncMock()
            conn_fail.name = "fail"
            conn_fail.connect = AsyncMock(side_effect=Exception("Connection refused"))
            conn_fail.discover_tools = AsyncMock()
            conn_fail.disconnect = AsyncMock()

            MockConn.side_effect = [conn_ok, conn_fail]

            manager = MCPManager(configs)
            tools = await manager.connect_all()

            # 只有成功连接的工具
            assert len(tools) == 1
            assert tools[0].name == "ok__tool1"

    @pytest.mark.asyncio
    async def test_connect_all_failure_raises(self):
        """所有服务器都失败时抛异常"""
        configs = [
            MCPServerConfig.stdio(name="fail1", command="bad1"),
            MCPServerConfig.stdio(name="fail2", command="bad2"),
        ]

        with patch("milu.tools.mcp.manager.MCPServerConnection") as MockConn:
            conn1 = AsyncMock()
            conn1.name = "fail1"
            conn1.connect = AsyncMock(side_effect=Exception("err1"))

            conn2 = AsyncMock()
            conn2.name = "fail2"
            conn2.connect = AsyncMock(side_effect=Exception("err2"))

            MockConn.side_effect = [conn1, conn2]

            manager = MCPManager(configs)
            with pytest.raises(ConnectionError, match="所有 MCP 服务器连接失败"):
                await manager.connect_all()

    @pytest.mark.asyncio
    async def test_discovery_failure_disconnects_established_connection(self):
        """工具发现失败时也必须关闭已建立的连接。"""
        config = MCPServerConfig.stdio(name="broken", command="echo")

        with patch("milu.tools.mcp.manager.MCPServerConnection") as MockConn:
            conn = AsyncMock()
            conn.name = "broken"
            conn.connect = AsyncMock()
            conn.discover_tools = AsyncMock(side_effect=RuntimeError("bad tool list"))
            conn.disconnect = AsyncMock()
            MockConn.return_value = conn

            manager = MCPManager([config])
            with pytest.raises(ConnectionError, match="所有 MCP 服务器连接失败"):
                await manager.connect_all()

            conn.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cancelled_connection_propagates_and_closes_successes(self):
        """单个连接被取消时应传播取消，并关闭同期已建立的连接。"""
        configs = [
            MCPServerConfig.stdio(name="cancelled", command="echo"),
            MCPServerConfig.stdio(name="ready", command="echo"),
        ]
        tool = _make_mock_tool("ready__tool")

        with patch("milu.tools.mcp.manager.MCPServerConnection") as MockConn:
            cancelled = AsyncMock()
            cancelled.name = "cancelled"
            cancelled.connect = AsyncMock(side_effect=asyncio.CancelledError())
            cancelled.disconnect = AsyncMock()

            ready = AsyncMock()
            ready.name = "ready"
            ready.connect = AsyncMock()
            ready.discover_tools = AsyncMock(return_value=[tool])
            ready.disconnect = AsyncMock()

            MockConn.side_effect = [cancelled, ready]

            manager = MCPManager(configs)
            with pytest.raises(asyncio.CancelledError):
                await manager.connect_all()

            cancelled.disconnect.assert_awaited_once()
            ready.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_all(self):
        """断开所有连接"""
        configs = [MCPServerConfig.stdio(name="a", command="echo")]
        tool = _make_mock_tool("a__t")

        with patch("milu.tools.mcp.manager.MCPServerConnection") as MockConn:
            conn = AsyncMock()
            conn.name = "a"
            conn.connect = AsyncMock()
            conn.discover_tools = AsyncMock(return_value=[tool])
            conn.disconnect = AsyncMock()
            MockConn.return_value = conn

            manager = MCPManager(configs)
            await manager.connect_all()
            await manager.disconnect_all()

            conn.disconnect.assert_called_once()
            assert manager.get_tools() == []

    @pytest.mark.asyncio
    async def test_get_tools_returns_copy(self):
        configs = [MCPServerConfig.stdio(name="a", command="echo")]
        tool = _make_mock_tool("a__t")

        with patch("milu.tools.mcp.manager.MCPServerConnection") as MockConn:
            conn = AsyncMock()
            conn.name = "a"
            conn.connect = AsyncMock()
            conn.discover_tools = AsyncMock(return_value=[tool])
            conn.disconnect = AsyncMock()
            MockConn.return_value = conn

            manager = MCPManager(configs)
            await manager.connect_all()

            t1 = manager.get_tools()
            t2 = manager.get_tools()
            assert t1 is not t2

    @pytest.mark.asyncio
    async def test_empty_configs(self):
        """空配置列表"""
        manager = MCPManager([])
        tools = await manager.connect_all()
        assert tools == []
