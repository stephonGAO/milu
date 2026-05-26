"""工具注册表 - 管理 @tool 装饰过的函数"""
from __future__ import annotations

import logging

from agent_framework.tools.decorator import ToolWrapper

logger = logging.getLogger(__name__)

# 注释：
# 这里相当于把所有装饰后的方法集中保存在这里。
# 并且增加get_schemas和其他统一管理功能。
class ToolRegistry:
    """工具注册表 - 管理 @tool 装饰过的函数"""

    def __init__(self):
        self._tools: dict[str, ToolWrapper] = {}

    def register(self, func) -> None:
        """注册一个 @tool 装饰的函数"""
        if not hasattr(func, "_tool_wrapper"):
            raise ValueError(f"函数 {func.__name__} 未被 @tool 装饰")

        wrapper: ToolWrapper = func._tool_wrapper
        self._tools[wrapper.name] = wrapper

    def register_many(self, funcs: list) -> None:
        """批量注册"""
        for func in funcs:
            self.register(func)

    def register_wrapper(self, wrapper: ToolWrapper) -> None:
        """直接注册 ToolWrapper 实例（无需 @tool 装饰器，用于 MCP 等外部工具）"""
        if wrapper.name in self._tools:
            logger.warning("工具 %s 已存在，将被覆盖", wrapper.name)
        self._tools[wrapper.name] = wrapper

    def get_schemas(self) -> list[dict]:
        """
        返回所有工具的 OpenAI function calling schema 列表。
        可直接传给 llm.chat(tools=...) 使用。
        """
        schemas = []
        for wrapper in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": wrapper.name,
                    "description": wrapper.description,
                    "parameters": wrapper.parameters_schema,
                },
            })
        return schemas

    def get_tool(self, name: str) -> ToolWrapper | None:
        """根据名称获取工具"""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """列出所有已注册的工具名"""
        return list(self._tools.keys())
