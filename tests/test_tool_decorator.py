"""测试 @tool 装饰器和 schema 生成"""
import pytest
from agent_framework.tools import tool
from agent_framework.tools.decorator import ToolWrapper


def test_basic_tool_decorator():
    @tool(name="get_weather", description="获取天气")
    async def get_weather(city: str) -> str:
        return f"{city}：晴"

    assert hasattr(get_weather, "_tool_wrapper")
    wrapper: ToolWrapper = get_weather._tool_wrapper
    assert wrapper.name == "get_weather"
    assert wrapper.description == "获取天气"
    assert wrapper.is_async is True


def test_schema_generation_basic_types():
    @tool(name="test_func", description="测试函数")
    async def test_func(text: str, number: int, flag: bool, value: float = 3.14) -> str:
        return "ok"

    wrapper: ToolWrapper = test_func._tool_wrapper
    schema = wrapper.parameters_schema

    assert schema["type"] == "object"
    assert schema["properties"]["text"]["type"] == "string"
    assert schema["properties"]["number"]["type"] == "integer"
    assert schema["properties"]["flag"]["type"] == "boolean"
    assert schema["properties"]["value"]["type"] == "number"
    assert "text" in schema["required"]
    assert "number" in schema["required"]
    assert "flag" in schema["required"]
    assert "value" not in schema["required"]


def test_schema_optional_types():
    from typing import Optional

    @tool(name="test_optional", description="测试")
    async def test_optional(required: str, optional: Optional[str] = None) -> str:
        return "ok"

    wrapper: ToolWrapper = test_optional._tool_wrapper
    schema = wrapper.parameters_schema
    assert "required" in schema["required"]
    assert "optional" not in schema["required"]


def test_schema_literal_type():
    from typing import Literal

    @tool(name="test_literal", description="测试")
    async def test_literal(mode: Literal["fast", "slow"]) -> str:
        return "ok"

    wrapper: ToolWrapper = test_literal._tool_wrapper
    schema = wrapper.parameters_schema
    assert schema["properties"]["mode"]["type"] == "string"
    assert schema["properties"]["mode"]["enum"] == ["fast", "slow"]


def test_schema_list_type():
    @tool(name="test_list", description="测试")
    async def test_list(items: list[str]) -> str:
        return "ok"

    wrapper: ToolWrapper = test_list._tool_wrapper
    schema = wrapper.parameters_schema
    assert schema["properties"]["items"]["type"] == "array"


def test_docstring_param_extraction():
    @tool(name="test_doc", description="测试函数")
    async def test_doc(query: str, limit: int = 10) -> str:
        """
        搜索信息。
        :param query: 搜索关键词
        :param limit: 最大返回数
        """
        return "ok"

    wrapper: ToolWrapper = test_doc._tool_wrapper
    schema = wrapper.parameters_schema
    assert schema["properties"]["query"]["description"] == "搜索关键词"
    assert schema["properties"]["limit"]["description"] == "最大返回数"


def test_sync_function():
    @tool(name="sync_tool", description="同步工具")
    def sync_tool(x: int) -> str:
        return str(x)

    wrapper: ToolWrapper = sync_tool._tool_wrapper
    assert wrapper.is_async is False


def test_dangerous_flag():
    @tool(name="dangerous_tool", description="危险操作", dangerous=True)
    async def dangerous_tool(path: str) -> str:
        return "deleted"

    wrapper: ToolWrapper = dangerous_tool._tool_wrapper
    assert wrapper.dangerous is True
