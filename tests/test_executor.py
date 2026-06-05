"""测试 ToolExecutor"""
import pytest
from milu.tools import tool, ToolRegistry
from milu.agent.config import AgentConfig
from milu.tools.executor import ToolExecutor, ToolExecutionResult


@pytest.fixture
def registry():
    """创建带测试工具的注册表"""
    @tool(name="add", description="加法")
    async def add(a: int, b: int) -> str:
        return str(a + b)

    @tool(name="fail", description="会失败的工具")
    async def fail() -> str:
        raise ValueError("故意失败")

    @tool(name="sync_tool", description="同步工具")
    def sync_tool(x: int) -> str:
        return str(x * 2)

    reg = ToolRegistry()
    reg.register_many([add, fail, sync_tool])
    return reg


@pytest.fixture
def executor(registry):
    return ToolExecutor(registry, AgentConfig())


@pytest.mark.asyncio
async def test_execute_success(executor):
    """应成功执行工具"""
    tool_call = {
        "id": "call_1",
        "function": {"name": "add", "arguments": '{"a": 5, "b": 3}'}
    }

    result = await executor.execute(tool_call)
    assert result.output == "8"
    assert result.is_error is False


@pytest.mark.asyncio
async def test_execute_sync_function(executor):
    """应支持同步函数"""
    tool_call = {
        "id": "call_2",
        "function": {"name": "sync_tool", "arguments": '{"x": 10}'}
    }

    result = await executor.execute(tool_call)
    assert result.output == "20"
    assert result.is_error is False


@pytest.mark.asyncio
async def test_execute_tool_not_found(executor):
    """工具不存在应返回错误"""
    tool_call = {
        "id": "call_3",
        "function": {"name": "nonexistent", "arguments": "{}"}
    }

    result = await executor.execute(tool_call)
    assert result.is_error is True
    assert "工具不存在" in result.output


@pytest.mark.asyncio
async def test_execute_invalid_json(executor):
    """参数 JSON 解析失败应返回错误"""
    tool_call = {
        "id": "call_4",
        "function": {"name": "add", "arguments": "invalid json"}
    }

    result = await executor.execute(tool_call)
    assert result.is_error is True
    assert "参数解析失败" in result.output


@pytest.mark.asyncio
async def test_execute_exception_handling(executor):
    """工具执行异常应返回错误而非抛出"""
    tool_call = {
        "id": "call_5",
        "function": {"name": "fail", "arguments": "{}"}
    }

    result = await executor.execute(tool_call)
    assert result.is_error is True
    assert "执行异常" in result.output
