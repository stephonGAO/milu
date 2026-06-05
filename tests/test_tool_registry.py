"""测试 ToolRegistry"""
import pytest
from milu.tools import tool, ToolRegistry


def test_register_single_tool():
    """应能注册单个工具"""
    @tool(name="get_weather", description="获取天气")
    async def get_weather(city: str) -> str:
        return "晴"

    registry = ToolRegistry()
    registry.register(get_weather)

    assert "get_weather" in registry.list_tools()


def test_register_many_tools():
    """应能批量注册工具"""
    @tool(name="tool1", description="工具1")
    async def tool1() -> str:
        return "1"

    @tool(name="tool2", description="工具2")
    async def tool2() -> str:
        return "2"

    registry = ToolRegistry()
    registry.register_many([tool1, tool2])

    assert "tool1" in registry.list_tools()
    assert "tool2" in registry.list_tools()


def test_get_tool():
    """应能根据名称获取工具"""
    @tool(name="my_tool", description="我的工具")
    async def my_tool() -> str:
        return "ok"

    registry = ToolRegistry()
    registry.register(my_tool)

    wrapper = registry.get_tool("my_tool")
    assert wrapper is not None
    assert wrapper.name == "my_tool"

    # 不存在的工具
    assert registry.get_tool("nonexistent") is None


def test_get_schemas():
    """应返回所有工具的 OpenAI schema"""
    @tool(name="get_weather", description="获取天气")
    async def get_weather(city: str) -> str:
        return "晴"

    @tool(name="calculator", description="计算器")
    async def calculator(expression: str) -> str:
        return "result"

    registry = ToolRegistry()
    registry.register_many([get_weather, calculator])

    schemas = registry.get_schemas()

    assert len(schemas) == 2
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "get_weather"
    assert schemas[1]["function"]["name"] == "calculator"


def test_get_schemas_empty():
    """空注册表应返回空列表"""
    registry = ToolRegistry()
    assert registry.get_schemas() == []


def test_duplicate_registration():
    """重复注册应覆盖"""
    @tool(name="my_tool", description="版本1")
    async def my_tool_v1() -> str:
        return "v1"

    @tool(name="my_tool", description="版本2")
    async def my_tool_v2() -> str:
        return "v2"

    registry = ToolRegistry()
    registry.register(my_tool_v1)
    registry.register(my_tool_v2)

    wrapper = registry.get_tool("my_tool")
    assert wrapper.description == "版本2"


def test_register_non_decorated_function():
    """注册未用 @tool 装饰的函数应报错"""
    async def plain_func() -> str:
        return "ok"

    registry = ToolRegistry()
    with pytest.raises(ValueError, match="未被 @tool 装饰"):
        registry.register(plain_func)
