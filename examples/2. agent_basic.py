"""基础 Agent 使用示例 —— 演示完整事件流（思考、文本、工具调用、错误）"""
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from agent_framework import (
    Agent, AgentConfig,
    TextDelta, ReasoningDelta,
    ToolCallStart, ToolResult,
    AgentDone, AgentError,
)
from agent_framework.llm.providers import ModelRegistry
from agent_framework.tools import tool

# 加载 .env 中的 API Key
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ── 定义工具 ──────────────────────────────────────────────
@tool(name="get_weather", description="获取指定城市的天气信息")
async def get_weather(city: str) -> str:
    return f"{city}：晴，25°C，湿度 42%"


@tool(name="calculator", description="执行数学表达式计算")
async def calculator(expression: str) -> str:
    return str(eval(expression))  # noqa: S307


# ── 主流程 ────────────────────────────────────────────────
async def main():
    llm = ModelRegistry.create("qwen", model="qwen3.7-max")

    agent = Agent(
        llm=llm,
        system_prompt="你是一个智能助手，可以查天气和做计算。回答要简洁。",
        tools=[get_weather, calculator],
        config=AgentConfig(max_turns=5, timeout=60),
    )

    user_input = "北京天气怎么样？顺便算一下 17*23+45"
    print(f"👤 用户: {user_input}")
    print("─" * 50)

    thinking_visible = False  # 控制思考区域的折叠

    async for event in agent.run(user_input):

        # 💭 思考过程（reasoning / chain-of-thought）
        if isinstance(event, ReasoningDelta):
            if not thinking_visible:
                print("💭 思考中...", flush=True)
                thinking_visible = True
            print(event.text, end="", flush=True)

        # 📝 正文输出
        elif isinstance(event, TextDelta):
            if thinking_visible:
                print("\n", flush=True)  # 思考区结束换行
                thinking_visible = False
            print(event.text, end="", flush=True)

        # 🔧 工具调用开始
        elif isinstance(event, ToolCallStart):
            print(f"\n🔧 调用工具: {event.tool_name}({event.arguments})", flush=True)

        # 📋 工具调用结果
        elif isinstance(event, ToolResult):
            if event.is_error:
                print(f"❌ 工具出错: {event.tool_name} -> {event.output}", flush=True)
            else:
                print(f"📋 工具结果: {event.output}", flush=True)

        # ✅ 正常完成
        elif isinstance(event, AgentDone):
            u = event.total_usage
            print(f"\n{'─' * 50}")
            print(f"✅ 完成 | 轮次: {event.turn_count} | "
                  f"tokens: {u.total_tokens} "
                  f"(prompt={u.prompt_tokens}, completion={u.completion_tokens}"
                  f"{f', reasoning={u.reasoning_tokens}' if u.reasoning_tokens else ''})")

        # 🚨 Agent 级别错误
        elif isinstance(event, AgentError):
            print(f"\n{'─' * 50}")
            print(f"🚨 错误 [{event.error_type}]: {event.message}")


if __name__ == "__main__":
    asyncio.run(main())
