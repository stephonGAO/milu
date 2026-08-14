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
        self._lifecycle_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def config(self) -> MCPServerConfig:
        return self._config

    async def connect(self) -> None:
        """建立传输连接 + 创建会话 + 初始化。"""
        existing_task = self._lifecycle_task
        if existing_task is not None:
            if not existing_task.done():
                raise RuntimeError(f"MCP 服务器 [{self._config.name}] 已连接或正在连接")
            self._clear_lifecycle(existing_task)

        loop = asyncio.get_running_loop()
        ready: asyncio.Future[None] = loop.create_future()
        stop_event = asyncio.Event()
        task = asyncio.create_task(
            self._run_lifecycle(ready, stop_event),
            name=f"mcp-{self._config.name}-lifecycle",
        )
        self._lifecycle_task = task
        self._stop_event = stop_event

        try:
            # shield 防止调用方取消时连带取消 ready；取消分支会显式停止 owner task，
            # 让已经进入的 AnyIO 上下文仍在原 task 中退出。
            await asyncio.shield(ready)
        except asyncio.CancelledError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            self._clear_lifecycle(task)
            raise
        except Exception:
            await task
            self._clear_lifecycle(task)
            raise

    async def _run_lifecycle(
        self,
        ready: asyncio.Future[None],
        stop_event: asyncio.Event,
    ) -> None:
        """在单一 owner task 中进入并退出 MCP/AnyIO 异步上下文。"""
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
            ready.set_result(None)
            await stop_event.wait()
        except ImportError:
            if not ready.done():
                ready.set_exception(
                    ImportError(
                        "未安装 mcp 包（核心依赖，通常已随 milu 安装）。"
                        "请修复: pip install 'mcp>=1.8.0,<2'"
                    )
                )
        except asyncio.CancelledError:
            if not ready.done():
                ready.cancel()
            raise
        except Exception as exc:
            if not ready.done():
                ready.set_exception(exc)
            else:
                logger.warning("MCP [%s] 连接生命周期异常: %s", self._config.name, exc)
        finally:
            await self._cleanup()

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
        task = self._lifecycle_task
        stop_event = self._stop_event

        if task is not None:
            if stop_event is not None:
                stop_event.set()
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # owner task 仍会继续清理；保留引用以阻止清理完成前重新连接。
                raise
            else:
                self._clear_lifecycle(task)
        else:
            # 兼容尚未通过 owner task 创建的测试/旧状态；正常连接不会走此分支。
            await self._cleanup()
        logger.info("MCP 服务器 [%s] 已断开", self._config.name)

    def _clear_lifecycle(self, task: asyncio.Task[None]) -> None:
        """仅清除仍指向指定 owner task 的生命周期状态。"""
        if self._lifecycle_task is task:
            self._lifecycle_task = None
            self._stop_event = None

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
        """创建 streamable_http 传输。

        mcp SDK 在此处有一次签名变更，同模块两个函数不可混用：
          - 新 API `streamable_http_client(url, *, http_client=...)`：不收 headers，
            headers 由调用方构造的 AsyncClient 携带；
          - 旧 API `streamablehttp_client(url, headers=...)`：收 headers，1.2x 起已标
            @deprecated（实现即转调新 API），将来会被移除。
        优先走新 API（未废弃、且是唯一长期存在的路径），仅在老 SDK 上回退旧 API；
        用能力探测兼容 mcp 1.x 内的签名变化（pyproject 要求 mcp>=1.8.0,<2）。
        """
        from mcp.client import streamable_http as _sh

        new_api = getattr(_sh, "streamable_http_client", None)
        if new_api is not None:
            # 远程地址使用 SDK 自己的工厂，保持与当前 SDK 的 HTTP 客户端实现一致。
            # 该工厂的默认值与旧 API 一致（follow_redirects + Timeout(30, read=300)），
            # 故切换新 API 不改变超时行为。
            factory = getattr(_sh, "create_mcp_http_client", None)
            if factory is None:  # 极老/极新 SDK 未再导出时的兜底
                from mcp.shared._httpx_utils import create_mcp_http_client as factory

            # 外部传入的 client 由调用方负责关闭（SDK 只管理自己创建的那个），
            # 因此挂进 exit_stack，随连接一起释放，避免每次连接泄漏一个 AsyncClient。
            # 本机 HTTP MCP 不需要代理和 TLS。Windows 上 httpx 构造默认客户端时
            # 读取系统代理与证书链可能耗时几十秒，即使目标只是 127.0.0.1；本机明文
            # 地址显式关闭二者，将每个 Agent 的 MCP 建连从约 30 秒降到毫秒级。
            from urllib.parse import urlparse

            parsed = urlparse(self._config.url or "")
            is_local_http = (
                parsed.scheme == "http"
                and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
            )
            if is_local_http:
                import httpx

                http_client = httpx.AsyncClient(
                    headers=self._config.headers,
                    timeout=httpx.Timeout(30.0, read=300.0),
                    follow_redirects=True,
                    trust_env=False,
                    verify=False,
                )
            else:
                http_client = factory(headers=self._config.headers)
            await self._exit_stack.enter_async_context(http_client)
            transport_cm = new_api(self._config.url, http_client=http_client)
        else:
            transport_cm = _sh.streamablehttp_client(
                self._config.url,
                headers=self._config.headers,
            )

        # 返回值元数可能变化（当前为 (read, write, get_session_id)），按下标取前两项。
        streams = await self._exit_stack.enter_async_context(transport_cm)
        return streams[0], streams[1]

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
