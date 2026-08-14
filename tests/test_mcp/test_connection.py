"""测试 MCPServerConnection 连接管理"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from milu.tools.mcp.config import MCPServerConfig
from milu.tools.mcp.connection import MCPServerConnection


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
    async def test_connect_and_disconnect_use_same_owner_task(self):
        """AnyIO 上下文必须在进入它的同一个 task 中退出。"""
        config = MCPServerConfig.stdio(name="test", command="echo")
        conn = MCPServerConnection(config)
        entered_in = None
        exited_in = None

        class FakeSession:
            async def initialize(self):
                pass

        class FakeSessionContext:
            async def __aenter__(self):
                nonlocal entered_in
                entered_in = asyncio.current_task()
                return FakeSession()

            async def __aexit__(self, *exc):
                nonlocal exited_in
                exited_in = asyncio.current_task()
                return False

        with patch.object(
            MCPServerConnection,
            "_create_stdio_transport",
            new_callable=AsyncMock,
            return_value=(MagicMock(), MagicMock()),
        ), patch("mcp.ClientSession", return_value=FakeSessionContext()):
            await conn.connect()
            owner_task = conn._lifecycle_task
            assert owner_task is not asyncio.current_task()
            await conn.disconnect()

        assert entered_in is owner_task
        assert exited_in is owner_task
        assert conn._lifecycle_task is None
        assert conn._session is None

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
    async def test_discover_tools_safe(self, mock_mcp_tool, mock_session):
        """safe_tools 标记"""
        config = MCPServerConfig.stdio(
            name="fs", command="echo",
            safe_tools=["read_file"],
        )
        conn = MCPServerConnection(config)
        conn._session = mock_session

        tools = await conn.discover_tools()
        assert tools[0].is_safe is True

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
        assert conn._exit_stack is None

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


class TestHttpTransport:
    """streamable_http 传输：两套 SDK 签名的分派与 client 生命周期

    背景：mcp SDK 同模块存在两个函数，签名不兼容——
      - streamable_http_client(url, *, http_client=...)  新，不收 headers
      - streamablehttp_client(url, headers=...)          旧，已 @deprecated
    历史 bug：导入了前者却按后者传 headers=，任何 streamable_http 服务器必炸 TypeError。
    """

    @staticmethod
    def _conn_with_stack():
        from contextlib import AsyncExitStack

        config = MCPServerConfig.streamable_http(
            name="remote", url="https://example.com/mcp", headers={"Authorization": "Bearer x"}
        )
        conn = MCPServerConnection(config)
        conn._exit_stack = AsyncExitStack()
        return conn

    @pytest.mark.asyncio
    async def test_new_api_passes_headers_via_client(self):
        """新 API：headers 交给自建 client，且该 client 随 exit_stack 关闭（不能泄漏）"""
        from contextlib import asynccontextmanager

        import mcp.client.streamable_http as sh

        closed = []

        class FakeClient:
            def __init__(self, headers):
                self.headers = headers

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                closed.append(self)
                return False

        calls = {}

        @asynccontextmanager
        async def fake_new_api(url, *, http_client=None, **kw):
            calls["url"] = url
            calls["http_client"] = http_client
            yield ("read", "write", lambda: None)

        conn = self._conn_with_stack()
        await conn._exit_stack.__aenter__()

        with patch.object(sh, "streamable_http_client", fake_new_api), \
             patch.object(sh, "create_mcp_http_client", lambda headers=None: FakeClient(headers)):
            read, write = await conn._create_http_transport()

        assert (read, write) == ("read", "write")
        assert calls["url"] == "https://example.com/mcp"
        # headers 必须经 client 携带，而不是作为关键字传给传输函数
        assert calls["http_client"].headers == {"Authorization": "Bearer x"}

        await conn._exit_stack.aclose()
        assert closed == [calls["http_client"]], "外部传入的 client 需由调用方关闭，否则每次连接泄漏一个"

    @pytest.mark.asyncio
    async def test_falls_back_to_legacy_api(self):
        """老 SDK 没有新 API 时，回退到收 headers 的旧函数"""
        from contextlib import asynccontextmanager

        import mcp.client.streamable_http as sh

        calls = {}

        @asynccontextmanager
        async def fake_legacy(url, headers=None, **kw):
            calls["url"] = url
            calls["headers"] = headers
            yield ("read", "write", lambda: None)

        conn = self._conn_with_stack()
        await conn._exit_stack.__aenter__()

        with patch.object(sh, "streamable_http_client", None), \
             patch.object(sh, "streamablehttp_client", fake_legacy):
            read, write = await conn._create_http_transport()

        assert (read, write) == ("read", "write")
        assert calls["headers"] == {"Authorization": "Bearer x"}
        await conn._exit_stack.aclose()

    @pytest.mark.asyncio
    async def test_local_http_skips_proxy_and_tls_initialization(self):
        """本机明文 MCP 不应为系统代理和证书链付出几十秒初始化成本。"""
        from contextlib import AsyncExitStack, asynccontextmanager

        import mcp.client.streamable_http as sh

        calls = {}

        class FakeClient:
            def __init__(self, **kwargs):
                calls["client_kwargs"] = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

        @asynccontextmanager
        async def fake_new_api(url, *, http_client=None, **kw):
            calls["url"] = url
            yield ("read", "write", lambda: None)

        config = MCPServerConfig.streamable_http(
            name="local", url="http://127.0.0.1:8200/mcp"
        )
        conn = MCPServerConnection(config)
        conn._exit_stack = AsyncExitStack()
        await conn._exit_stack.__aenter__()

        with patch.object(sh, "streamable_http_client", fake_new_api), \
             patch("httpx.AsyncClient", FakeClient):
            await conn._create_http_transport()

        assert calls["client_kwargs"]["trust_env"] is False
        assert calls["client_kwargs"]["verify"] is False
        await conn._exit_stack.aclose()

    @pytest.mark.asyncio
    async def test_real_sdk_signature_accepts_our_kwargs(self):
        """对着真实 SDK 校验签名，防止再次按错签名调用（本条即当年 bug 的守卫）"""
        import inspect

        import mcp.client.streamable_http as sh

        new_api = getattr(sh, "streamable_http_client", None)
        legacy_api = getattr(sh, "streamablehttp_client", None)
        assert new_api is not None or legacy_api is not None

        if new_api is not None:
            new_sig = inspect.signature(new_api)
            assert "http_client" in new_sig.parameters
            assert "headers" not in new_sig.parameters
        if legacy_api is not None:
            assert "headers" in inspect.signature(legacy_api).parameters
