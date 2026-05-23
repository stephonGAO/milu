"""Agent 端到端集成测试（需要真实 API Key）"""
import asyncio
import os
import pytest

from agent_framework import Agent, AgentConfig, TextDelta, ToolCallStart, ToolResult, AgentDone
from agent_framework.providers import ModelRegistry
from agent_framework.tools import tool


# 跳过条件：没有 API Key
pytestmark = pytest.mark.skipif(
    not os.environ.get("QWEN_API_KEY"),
    reason="需要 QWEN_API_KEY 环境变量"
)


@tool(name="get_weather", description="获取指定城市的天气")
async def get_weather(city: str) -> str:
    """模拟天气工具"""
    weather_data = {
        "北京": "晴，25°C",
        "上海": "多云，23°C",
        "广州": "阵雨，28°C",
    }
    return weather_data.get(city, f"{city}：未知")


@tool(name="calculator", description="数学计算")
async def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


@pytest.mark.asyncio
async def test_agent_with_tools():
    """测试 Agent + 工具调用端到端流程"""
    llm = ModelRegistry.create("qwen", model="qwen-max")

    agent = Agent(
        llm=llm,
        system_prompt="你是一个智能助手，可以查天气和做计算。",
        tools=[get_weather, calculator],
        config=AgentConfig(max_turns=5, timeout=60),
    )

    events = []
    async for event in agent.run("北京天气怎么样？顺便算一下 17*23+45"):
        events.append(event)

    # 应有工具调用
    tool_starts = [e for e in events if isinstance(e, ToolCallStart)]
    assert len(tool_starts) >= 1

    # 应有天气工具调用
    weather_call = next((e for e in tool_starts if e.tool_name == "get_weather"), None)
    assert weather_call is not None
    assert "北京" in weather_call.arguments

    # 应有计算工具调用
    calc_call = next((e for e in tool_starts if e.tool_name == "calculator"), None)
    assert calc_call is not None

    # 应有最终回复
    done = next((e for e in events if isinstance(e, AgentDone)), None)
    assert done is not None
    assert len(done.final_text) > 0


@pytest.mark.asyncio
async def test_multi_turn_conversation():
    """测试多轮对话历史保留"""
    llm = ModelRegistry.create("qwen", model="qwen-max")

    agent = Agent(
        llm=llm,
        system_prompt="你是一个简洁的助手。",
        config=AgentConfig(max_turns=5, timeout=60),
    )

    # 第一轮
    events1 = []
    async for event in agent.run("我叫张三"):
        events1.append(event)

    # 第二轮
    events2 = []
    async for event in agent.run("我叫什么名字？"):
        events2.append(event)

    # 第二轮应能记住第一轮的信息
    done = next((e for e in events2 if isinstance(e, AgentDone)), None)
    assert done is not None
    # 模型应该能回答出"张三"
    assert "张三" in done.final_text or "您" in done.final_text


@pytest.mark.asyncio
async def test_tool_error_handling():
    """测试工具执行异常不中断循环"""
    @tool(name="failing_tool", description="会失败的工具")
    async def failing_tool() -> str:
        raise ValueError("故意失败")

    llm = ModelRegistry.create("qwen", model="qwen-max")

    agent = Agent(
        llm=llm,
        system_prompt="你是助手。",
        tools=[failing_tool],
        config=AgentConfig(max_turns=3, timeout=60),
    )

    events = []
    async for event in agent.run("请调用 failing_tool"):
        events.append(event)

    # 应有 ToolResult 且 is_error=True
    tool_results = [e for e in events if isinstance(e, ToolResult)]
    assert len(tool_results) >= 1
    assert any(r.is_error for r in tool_results)

    # 应有最终回复（模型处理错误后继续）
    done = next((e for e in events if isinstance(e, AgentDone)), None)
    assert done is not None
