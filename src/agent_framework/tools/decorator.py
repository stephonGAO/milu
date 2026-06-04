"""@tool 装饰器实现"""
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Callable

from agent_framework.tools.schema import generate_schema_from_function

# 注释：
# 这个类是定义装饰。就是把一个方法加上额外的信息（装饰的信息）。
# 这里相当于定义要装饰的内容与结构，和装饰方法。装饰的时候把额外的结构加上。
# 相当于原方法里增加了一个_tool_wrapper变量，内部是包装后的整体。包装包里也有原方法，有所有信息。
@dataclass
class ToolWrapper:
    """包装一个 @tool 函数，持有其元数据和 schema"""
    name: str
    description: str
    parameters_schema: dict
    func: Callable
    is_async: bool
    is_safe: bool = True                            # 工具是否安全（talk 模式可用，manual 模式免审批）
    safe_check: Callable[[dict], bool] | None = None  # 动态安全检查（每次调用时判定是否安全）
    category: str = ""    # 工具分类（MCP 服务器名等，用于 Tool Catalog 分组显示）
    meta: bool = False    # 是否为元工具（元工具不可被停用）


def tool(name: str, description: str, is_safe: bool = True,
         safe_check: Callable[[dict], bool] | None = None):
    """装饰器：将函数标记为 Agent 可调用的工具。

    :param is_safe: 是否安全工具（talk 模式可用，manual 模式免审批）
    :param safe_check: 动态安全检查函数（按调用参数判定是否安全）
    """
    def decorator(func: Callable) -> Callable:
        parameters_schema = generate_schema_from_function(func)
        is_async = asyncio.iscoroutinefunction(func)
        wrapper = ToolWrapper(
            name=name,
            description=description,
            parameters_schema=parameters_schema,
            func=func,
            is_async=is_async,
            is_safe=is_safe,
            safe_check=safe_check,
        )
        # 相当于原方法里增加了一个_tool_wrapper变量，内部是包装后的整体。包装包里也有原方法，有所有信息。
        func._tool_wrapper = wrapper
        return func
    return decorator
