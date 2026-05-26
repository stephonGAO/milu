"""MCP 多服务器编排器

并行连接多个 MCP 服务器，汇总工具列表，处理名称冲突。
"""
from __future__ import annotations

import asyncio
import logging

from agent_framework.tools.decorator import ToolWrapper
from agent_framework.tools.mcp.config import MCPServerConfig
from agent_framework.tools.mcp.connection import MCPServerConnection

logger = logging.getLogger(__name__)


class MCPManager:
    """编排多个 MCP 服务器连接。

    使用 asyncio.gather 并行连接，单个服务器失败不阻塞其他。
    """

    def __init__(self, configs: list[MCPServerConfig]):
        self._configs = configs
        self._connections: list[MCPServerConnection] = []
        self._all_tools: list[ToolWrapper] = []

    async def connect_all(self) -> list[ToolWrapper]:
        """并行连接所有 MCP 服务器并发现工具。

        Returns:
            所有服务器提供的 ToolWrapper 列表。

        Raises:
            ConnectionError: 所有服务器都连接失败时。
        """
        connections = [MCPServerConnection(cfg) for cfg in self._configs]

        # 并行连接
        async def _connect_and_discover(conn: MCPServerConnection):
            await conn.connect()
            return await conn.discover_tools()

        results = await asyncio.gather(
            *[_connect_and_discover(c) for c in connections],
            return_exceptions=True,
        )

        # 收集成功的连接和工具
        self._connections = []
        self._all_tools = []
        failures = []

        for conn, result in zip(connections, results):
            if isinstance(result, Exception):
                logger.error(
                    "MCP 服务器 [%s] 连接失败: %s",
                    conn.name,
                    result,
                )
                failures.append((conn.name, result))
            else:
                self._connections.append(conn)
                self._all_tools.extend(result)

        # 检测工具名冲突
        self._check_name_conflicts()

        # 全部失败时抛异常
        if not self._connections and failures:
            names = ", ".join(n for n, _ in failures)
            raise ConnectionError(
                f"所有 MCP 服务器连接失败 ({names})"
            )

        logger.info(
            "MCP 管理器: %d/%d 服务器已连接, %d 个工具可用",
            len(self._connections),
            len(self._configs),
            len(self._all_tools),
        )
        return self._all_tools

    def get_tools(self) -> list[ToolWrapper]:
        """获取所有已发现的工具。"""
        return list(self._all_tools)

    async def disconnect_all(self) -> None:
        """并行断开所有 MCP 服务器连接。"""
        if not self._connections:
            return

        await asyncio.gather(
            *[c.disconnect() for c in self._connections],
            return_exceptions=True,
        )
        self._connections.clear()
        self._all_tools.clear()
        logger.info("MCP 管理器: 所有服务器已断开")

    def _check_name_conflicts(self) -> None:
        """检测工具名冲突并记录警告。"""
        seen: dict[str, str] = {}  # tool_name → server_name
        for tool in self._all_tools:
            # 从工具名前缀推断服务器名
            parts = tool.name.split("__", 1)
            server = parts[0] if len(parts) > 1 else "unknown"

            if tool.name in seen:
                logger.warning(
                    "MCP 工具名冲突: '%s' 同时存在于服务器 '%s' 和 '%s'",
                    tool.name,
                    seen[tool.name],
                    server,
                )
            else:
                seen[tool.name] = server
