"""MCP Tool → ToolWrapper 转换器

将 MCP 服务器提供的 Tool 对象转换为框架的 ToolWrapper，
使 MCP 工具能够像本地 @tool 工具一样被 Agent 使用。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from milu.tools.decorator import ToolWrapper

logger = logging.getLogger(__name__)


def convert_mcp_tool(
    mcp_tool: Any,
    session: Any,
    server_name: str,
    prefix: bool = True,
    is_safe: bool = False,
) -> ToolWrapper:
    """将一个 MCP Tool 转换为框架 ToolWrapper。

    Args:
        mcp_tool: MCP SDK 返回的 Tool 对象（含 name, description, inputSchema）
        session: MCP ClientSession（用于调用 call_tool）
        server_name: 服务器名称（用于工具名前缀）
        prefix: 是否添加 "{server_name}__" 前缀
        is_safe: 是否标记为安全工具

    Returns:
        可在 ToolRegistry 中注册的 ToolWrapper 实例
    """
    tool_name = f"{server_name}__{mcp_tool.name}" if prefix else mcp_tool.name
    description = mcp_tool.description or f"MCP tool: {mcp_tool.name}"
    input_schema = mcp_tool.inputSchema or {"type": "object", "properties": {}}

    # 用闭包捕获 session 和原始工具名
    original_name = mcp_tool.name

    async def _call(**kwargs) -> str:
        """通过 MCP session 调用远程工具。"""
        result = await session.call_tool(original_name, kwargs)
        return _convert_result(result)

    return ToolWrapper(
        name=tool_name,
        description=description,
        parameters_schema=input_schema,
        func=_call,
        is_async=True,
        is_safe=is_safe,
    )


def _convert_result(result: Any) -> str:
    """将 MCP CallToolResult 转换为字符串。

    处理规则：
    - TextContent → 提取 .text
    - ImageContent → 描述性文本
    - structuredContent → JSON 序列化
    - 错误结果 → 添加 [MCP Error] 前缀
    """
    parts: list[str] = []

    # 处理 content 列表
    content_list = getattr(result, "content", None) or []
    for block in content_list:
        block_type = type(block).__name__
        if block_type == "TextContent" or hasattr(block, "text"):
            text = getattr(block, "text", "")
            if text:
                parts.append(text)
        elif block_type == "ImageContent" or hasattr(block, "data"):
            mime = getattr(block, "mimeType", "image/unknown")
            data = getattr(block, "data", "")
            size = len(data) if data else 0
            parts.append(f"[Image: {mime}, {size} bytes base64]")
        elif block_type == "EmbeddedResource" or hasattr(block, "resource"):
            resource = getattr(block, "resource", None)
            if resource:
                text = getattr(resource, "text", None)
                if text:
                    parts.append(text)
                else:
                    uri = getattr(resource, "uri", "unknown")
                    parts.append(f"[EmbeddedResource: {uri}]")
            else:
                parts.append("[EmbeddedResource: empty]")
        else:
            # 未知类型，尝试转字符串
            parts.append(str(block))

    # 处理 structuredContent
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        parts.append(json.dumps(structured, ensure_ascii=False, indent=2))

    # 组合结果
    output = "\n".join(parts) if parts else ""

    # 处理错误
    is_error = getattr(result, "isError", False)
    if is_error:
        output = f"[MCP Error] {output}" if output else "[MCP Error] 工具执行失败"

    return output
