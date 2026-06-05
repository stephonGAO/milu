"""测试 Agent + MCP 集成"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from milu import Agent, AgentConfig
from milu.tools.decorator import ToolWrapper


def _make_mock_llm():
    """创建模拟 LLM"""
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=_empty_stream())
    return llm


async def _empty_stream():
    """空的 LLM 响应流"""
    from milu.llm.base.response import StreamChunk
    yield StreamChunk(content="你好", finish_reason="stop")


def _make_mock_tool(name: str) -> ToolWrapper:
    async def _noop(**kwargs):
        return "ok"
    return ToolWrapper(
        name=name, description=f"Tool {name}",
        parameters_schema={"type": "object", "properties": {}},
        func=_noop, is_async=True,
    )


def _write_mcp_config(tmp_path, servers: dict) -> str:
    """写入临时 MCP 配置文件，返回路径"""
    config_file = tmp_path / "mcp.json"
    config_file.write_text(json.dumps(servers), encoding="utf-8")
    return str(config_file)


class TestAgentMCPIntegration:
    """Agent MCP 集成测试"""

    @pytest.mark.asyncio
    async def test_agent_accepts_mcp_config_path(self):
        """Agent 构造函数接受 mcp_config_path 参数"""
        llm = _make_mock_llm()
        agent = Agent(
            llm=llm,
            system_prompt="test",
            mcp_config_path="config/mcp_servers.json",
        )
        assert agent._mcp_config_path == "config/mcp_servers.json"

    @pytest.mark.asyncio
    async def test_connect_mcp_registers_tools(self, tmp_path):
        """connect_mcp 应连接服务器并注册工具（默认进入休眠池）"""
        config_path = _write_mcp_config(tmp_path, {
            "srv": {"type": "stdio", "command": "echo"},
        })

        tool = _make_mock_tool("srv__hello")

        with patch("milu.tools.mcp.manager.MCPServerConnection") as MockConn:
            conn = AsyncMock()
            conn.name = "srv"
            conn.connect = AsyncMock()
            conn.discover_tools = AsyncMock(return_value=[tool])
            conn.disconnect = AsyncMock()
            MockConn.return_value = conn

            llm = _make_mock_llm()
            agent = Agent(
                llm=llm,
                system_prompt="test",
                mcp_config_path=config_path,
            )
            await agent.connect_mcp()

            # MCP 工具默认进入休眠池
            assert "srv__hello" in agent.tools.list_dormant_names()

            await agent.disconnect_mcp()

    @pytest.mark.asyncio
    async def test_context_manager_lifecycle(self, tmp_path):
        """async with 应自动 connect/disconnect"""
        config_path = _write_mcp_config(tmp_path, {
            "srv": {"type": "stdio", "command": "echo"},
        })
        tool = _make_mock_tool("srv__hello")

        with patch("milu.tools.mcp.manager.MCPServerConnection") as MockConn:
            conn = AsyncMock()
            conn.name = "srv"
            conn.connect = AsyncMock()
            conn.discover_tools = AsyncMock(return_value=[tool])
            conn.disconnect = AsyncMock()
            MockConn.return_value = conn

            llm = _make_mock_llm()
            async with Agent(
                llm=llm,
                system_prompt="test",
                mcp_config_path=config_path,
            ) as agent:
                # MCP 工具默认休眠，可通过 list_dormant_names 查看
                assert "srv__hello" in agent.tools.list_dormant_names()

            # 退出后应已断开
            conn.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_mcp_config_skips_connect(self, tmp_path, monkeypatch):
        """无 MCP 配置时 connect_mcp 应跳过"""
        # 切换到空目录，避免自动搜索找到项目的 config/mcp_servers.json
        monkeypatch.chdir(tmp_path)
        # 确保无环境变量
        monkeypatch.delenv("MCP_CONFIG_PATH", raising=False)

        llm = _make_mock_llm()
        agent = Agent(llm=llm, system_prompt="test")

        # 不应抛异常
        await agent.connect_mcp()
        assert agent._mcp_manager is None

        await agent.disconnect_mcp()

    @pytest.mark.asyncio
    async def test_connect_mcp_from_env(self, tmp_path, monkeypatch):
        """通过 .env 环境变量 MCP_CONFIG_PATH 加载配置"""
        config_path = _write_mcp_config(tmp_path, {
            "env_srv": {"type": "stdio", "command": "echo"},
        })
        monkeypatch.setenv("MCP_CONFIG_PATH", config_path)

        tool = _make_mock_tool("env_srv__hello")

        with patch("milu.tools.mcp.manager.MCPServerConnection") as MockConn:
            conn = AsyncMock()
            conn.name = "env_srv"
            conn.connect = AsyncMock()
            conn.discover_tools = AsyncMock(return_value=[tool])
            conn.disconnect = AsyncMock()
            MockConn.return_value = conn

            llm = _make_mock_llm()
            agent = Agent(llm=llm, system_prompt="test")
            await agent.connect_mcp()
            assert "env_srv__hello" in agent.tools.list_dormant_names()
            await agent.disconnect_mcp()

    @pytest.mark.asyncio
    async def test_param_overrides_env(self, tmp_path, monkeypatch):
        """构造函数 mcp_config_path 优先于环境变量"""
        # 环境变量指向一个配置
        env_config = _write_mcp_config(tmp_path, {
            "env_srv": {"type": "stdio", "command": "echo"},
        })
        monkeypatch.setenv("MCP_CONFIG_PATH", env_config)

        # 参数指向另一个配置
        sub_dir = tmp_path / "sub"
        sub_dir.mkdir()
        param_config = _write_mcp_config(sub_dir, {
            "param_srv": {"type": "stdio", "command": "echo"},
        })

        tool = _make_mock_tool("param_srv__hello")

        with patch("milu.tools.mcp.manager.MCPServerConnection") as MockConn:
            conn = AsyncMock()
            conn.name = "param_srv"
            conn.connect = AsyncMock()
            conn.discover_tools = AsyncMock(return_value=[tool])
            conn.disconnect = AsyncMock()
            MockConn.return_value = conn

            llm = _make_mock_llm()
            agent = Agent(
                llm=llm,
                system_prompt="test",
                mcp_config_path=param_config,
            )
            await agent.connect_mcp()
            # 应使用参数指定的配置
            assert "param_srv__hello" in agent.tools.list_dormant_names()
            assert "env_srv__hello" not in agent.tools.list_dormant_names()
            await agent.disconnect_mcp()

    @pytest.mark.asyncio
    async def test_mixed_native_and_mcp_tools(self, tmp_path):
        """本地 @tool 和 MCP 工具共存"""
        from milu.tools.decorator import tool

        @tool(name="native_tool", description="本地工具")
        async def native_tool() -> str:
            return "native"

        config_path = _write_mcp_config(tmp_path, {
            "srv": {"type": "stdio", "command": "echo"},
        })
        mcp_tool = _make_mock_tool("srv__mcp_tool")

        with patch("milu.tools.mcp.manager.MCPServerConnection") as MockConn:
            conn = AsyncMock()
            conn.name = "srv"
            conn.connect = AsyncMock()
            conn.discover_tools = AsyncMock(return_value=[mcp_tool])
            conn.disconnect = AsyncMock()
            MockConn.return_value = conn

            llm = _make_mock_llm()
            async with Agent(
                llm=llm,
                system_prompt="test",
                tools=[native_tool],
                mcp_config_path=config_path,
            ) as agent:
                # 本地工具在活跃池
                all_active = agent.tools.list_tools()
                assert "native_tool" in all_active
                # MCP 工具在休眠池
                all_dormant = agent.tools.list_dormant_names()
                assert "srv__mcp_tool" in all_dormant

    @pytest.mark.asyncio
    async def test_mcp_tools_active_by_default(self, tmp_path):
        """mcp_tools_active_by_default=True 时 MCP 工具直接进入活跃池"""
        config_path = _write_mcp_config(tmp_path, {
            "srv": {"type": "stdio", "command": "echo"},
        })
        tool = _make_mock_tool("srv__hello")

        with patch("milu.tools.mcp.manager.MCPServerConnection") as MockConn:
            conn = AsyncMock()
            conn.name = "srv"
            conn.connect = AsyncMock()
            conn.discover_tools = AsyncMock(return_value=[tool])
            conn.disconnect = AsyncMock()
            MockConn.return_value = conn

            llm = _make_mock_llm()
            agent = Agent(
                llm=llm,
                system_prompt="test",
                mcp_config_path=config_path,
                mcp_tools_active_by_default=True,
            )
            await agent.connect_mcp()

            # 直接进入活跃池
            assert "srv__hello" in agent.tools.list_tools()
            assert "srv__hello" not in agent.tools.list_dormant_names()

            await agent.disconnect_mcp()

    @pytest.mark.asyncio
    async def test_dormant_activate_flow(self, tmp_path):
        """MCP 工具休眠 → 元工具发现 → 激活 → 可调用"""
        config_path = _write_mcp_config(tmp_path, {
            "srv": {"type": "stdio", "command": "echo"},
        })
        mcp_tool = _make_mock_tool("srv__query")

        with patch("milu.tools.mcp.manager.MCPServerConnection") as MockConn:
            conn = AsyncMock()
            conn.name = "srv"
            conn.connect = AsyncMock()
            conn.discover_tools = AsyncMock(return_value=[mcp_tool])
            conn.disconnect = AsyncMock()
            MockConn.return_value = conn

            llm = _make_mock_llm()
            async with Agent(
                llm=llm,
                system_prompt="test",
                mcp_config_path=config_path,
            ) as agent:
                # 1. MCP 工具在休眠池
                assert "srv__query" in agent.tools.list_dormant_names()
                assert "srv__query" not in agent.tools.list_tools()

                # 2. 元工具在活跃池
                assert "list_catalog" in agent.tools.list_tools()
                assert "search_tools" in agent.tools.list_tools()
                assert "activate_tools" in agent.tools.list_tools()

                # 3. 通过元工具搜索
                search = agent.tools.get_tool("search_tools")
                result = await search.func(query="query")
                assert "srv__query" in result

                # 4. 通过元工具激活
                activate = agent.tools.get_tool("activate_tools")
                result = await activate.func(tool_names=["srv__query"])
                assert "已激活" in result

                # 5. 工具现在在活跃池
                assert "srv__query" in agent.tools.list_tools()
                assert "srv__query" not in agent.tools.list_dormant_names()

                # 6. schemas 包含已激活工具
                schema_names = [s["function"]["name"] for s in agent.tools.get_schemas()]
                assert "srv__query" in schema_names
