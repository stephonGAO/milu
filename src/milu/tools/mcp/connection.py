"""MCP 单服务器连接管理

使用 AsyncExitStack 管理 MCP SDK 的异步上下文（传输层 + 会话层）。
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack

from milu.tools.decorator import ToolWrapper
from milu.tools.mcp.config import MCPServerConfig
from milu.tools.mcp.converter import convert_mcp_tool

logger = logging.getLogger(__name__)


def suppress_mcp_asyncgen_errors() -> None:
    """屏蔽 MCP stdio_client 异步生成器在事件循环关闭阶段产生的噪声错误。

    asyncio.run() 结束时调用 shutdown_asyncgens() 关闭残留的异步生成器。
    MCP 的 stdio_client 内部使用 anyio TaskGroup，在此阶段关闭时 cancel scope
    跨任务退出会抛出 RuntimeError，asyncio 默认将其打印到 stderr。
    本函数在当前运行的事件循环上安装过滤器，对来自 mcp 包的 asyncgen 错误静默处理。
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    original = loop.get_exception_handler()

    def _handler(loop, context):
        if "asyncgen" in context:
            agen = context.get("asyncgen")
            if agen is not None:
                try:
                    code = getattr(agen, "ag_code", None)
                    filename = getattr(code, "co_filename", "") or ""
                    name = getattr(code, "co_name", "") or ""
                    if "mcp" in filename.lower() or name == "stdio_client":
                        return
                except Exception:
                    pass
        if original:
            original(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(_handler)


class MCPServerConnection:
    """管理单个 MCP 服务器的连接生命周期。

    使用方式::

        conn = MCPServerConnection(config)
        await conn.connect()
        tools = conn.get_tools()
        # ... 使用工具 ...
        await conn.disconnect()
    """

    def __init__(self, config: MCPServerConfig):
        self._config = config
        self._session = None
        self._tools: list[ToolWrapper] = []
        self._exit_stack: AsyncExitStack | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def config(self) -> MCPServerConfig:
        return self._config

    async def connect(self) -> None:
        """建立传输连接 + 创建会话 + 初始化。"""
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        try:
            read, write = await self._create_transport()
            # 延迟导入 mcp，使 mcp 成为真正的可选依赖
            from mcp import ClientSession

            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await self._session.initialize()
            logger.info("MCP 服务器 [%s] 已连接", self._config.name)
        except ImportError:
            await self._cleanup()
            raise ImportError(
                "未安装 mcp 包（核心依赖，通常已随 milu 安装）。"
                "请修复: pip install 'mcp>=1.0.0'"
            )
        except Exception:
            await self._cleanup()
            raise

    async def discover_tools(self) -> list[ToolWrapper]:
        """发现并转换 MCP 服务器上的工具。"""
        if self._session is None:
            raise RuntimeError(f"MCP 服务器 [{self._config.name}] 未连接")

        list_result = await self._session.list_tools()
        all_tools = list_result.tools or []

        # 应用 tool_filter
        if self._config.tool_filter:
            allowed = set(self._config.tool_filter)
            all_tools = [t for t in all_tools if t.name in allowed]

        safe_set = set(self._config.safe_tools)
        self._tools = []
        for mcp_tool in all_tools:
            wrapper = convert_mcp_tool(
                mcp_tool=mcp_tool,
                session=self._session,
                server_name=self._config.name,
                prefix=self._config.prefix_tools,
                is_safe=mcp_tool.name in safe_set,
            )
            self._tools.append(wrapper)

        logger.info(
            "MCP 服务器 [%s] 发现 %d 个工具",
            self._config.name,
            len(self._tools),
        )
        return self._tools

    def get_tools(self) -> list[ToolWrapper]:
        """获取已发现的工具列表。"""
        return list(self._tools)

    async def disconnect(self) -> None:
        """断开连接并清理资源。"""
        await self._cleanup()
        logger.info("MCP 服务器 [%s] 已断开", self._config.name)

    async def _create_transport(self):
        """根据配置创建对应的传输层（read, write）流。"""
        transport = self._config.transport

        if transport == "stdio":
            return await self._create_stdio_transport()
        elif transport == "streamable_http":
            return await self._create_http_transport()
        elif transport == "sse":
            return await self._create_sse_transport()
        else:
            raise ValueError(f"不支持的传输类型: {transport}")

    async def _create_stdio_transport(self):
        """创建 stdio 传输（子进程通信）。"""
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self._config.command,
            args=self._config.args or [],
            env=self._config.env,
        )
        transport_cm = stdio_client(params)
        read, write = await self._exit_stack.enter_async_context(transport_cm)
        return read, write

    async def _create_http_transport(self):
        """创建 streamable_http 传输。"""
        from mcp.client.streamable_http import streamable_http_client

        transport_cm = streamable_http_client(
            self._config.url,
            headers=self._config.headers,
        )
        read, write, _ = await self._exit_stack.enter_async_context(transport_cm)
        return read, write

    async def _create_sse_transport(self):
        """创建 SSE 传输。"""
        from mcp.client.sse import sse_client

        transport_cm = sse_client(
            self._config.url,
            headers=self._config.headers,
        )
        read, write = await self._exit_stack.enter_async_context(transport_cm)
        return read, write

    async def _cleanup(self):
        """安全清理 AsyncExitStack。"""
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logger.warning("MCP [%s] 清理资源时出错: %s", self._config.name, e)
            finally:
                self._exit_stack = None
                self._session = None
