"""@tool 装饰器实现"""
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Callable

from agent_framework.tools.schema import generate_schema_from_function


@dataclass
class ToolWrapper:
    """包装一个 @tool 函数，持有其元数据和 schema"""
    name: str
    description: str
    parameters_schema: dict
    func: Callable
    is_async: bool
    dangerous: bool


def tool(name: str, description: str, dangerous: bool = False):
    """装饰器：将函数标记为 Agent 可调用的工具。"""
    def decorator(func: Callable) -> Callable:
        parameters_schema = generate_schema_from_function(func)
        is_async = asyncio.iscoroutinefunction(func)
        wrapper = ToolWrapper(
            name=name,
            description=description,
            parameters_schema=parameters_schema,
            func=func,
            is_async=is_async,
            dangerous=dangerous,
        )
        func._tool_wrapper = wrapper
        return func
    return decorator
