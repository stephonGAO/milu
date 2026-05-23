"""工具注册表 - 管理 @tool 装饰过的函数"""
from __future__ import annotations

from agent_framework.tools.decorator import ToolWrapper


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
