"""真实 API 测试 - ChatGPT + Gemini

使用用户提供的 API Key 进行真实网络请求测试。
测试完成后删除此文件。
"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from milu import Agent, AgentConfig
from milu.llm.providers import ModelRegistry
from milu.llm.base.message import Message, MessageRole

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")


async def test_chatgpt_basic():
    """ChatGPT 基础对话测试"""
    print("\n" + "=" * 60)
    print("  ChatGPT (Responses API) - 基础对话")
    print("=" * 60)

    llm = ModelRegistry.create("openai", api_key=OPENAI_KEY, model="gpt-4o-mini")
    messages = [
        Message(role=MessageRole.SYSTEM, content="你是一个简洁的助手，回答控制在50字以内。"),
        Message(role=MessageRole.USER, content="Python 的发明者是谁？"),
    ]

    full_text = ""
    async for chunk in llm.chat(messages, temperature=0):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_text += chunk.content
        if chunk.reasoning_content:
            print(f"[reasoning: {chunk.reasoning_content}]", end="", flush=True)
        if chunk.usage:
            print(f"\n  [usage: prompt={chunk.usage.prompt_tokens}, completion={chunk.usage.completion_tokens}, total={chunk.usage.total_tokens}]")

    print()
    assert full_text, "应有文本输出"
    assert "Guido" in full_text or "荷兰" in full_text or "吉多" in full_text, f"回答应包含发明者信息: {full_text}"
    print("  PASS")


async def test_chatgpt_tool_call():
    """ChatGPT 工具调用测试"""
    print("\n" + "=" * 60)
    print("  ChatGPT (Responses API) - 工具调用")
    print("=" * 60)

    from milu.tools import tool

    @tool(name="get_weather", description="获取城市天气")
    async def get_weather(city: str) -> str:
        return f"{city}: 晴天, 25°C"

    llm = ModelRegistry.create("openai", api_key=OPENAI_KEY, model="gpt-4o-mini")

    agent = Agent(
        llm=llm,
        system_prompt="你是天气助手，使用 get_weather 工具查天气。回答简洁。",
        tools=[get_weather],
        config=AgentConfig(max_turns=3),
    )

    async for event in agent.run("北京今天天气怎么样？"):
        if hasattr(event, "tool_name"):
            print(f"  tool_call: {event.tool_name}")
        if hasattr(event, "text"):
            print(event.text, end="", flush=True)
        if hasattr(event, "total_usage"):
            print(f"\n  [done] turns={event.turn_count}, tokens={event.total_usage.total_tokens}")
        if hasattr(event, "message") and hasattr(event, "error_type"):
            print(f"\n  [error] {event.error_type}: {event.message}")

    print("  PASS")


async def test_gemini_basic():
    """Gemini 基础对话测试"""
    print("\n" + "=" * 60)
    print("  Gemini (OpenAI 兼容) - 基础对话")
    print("=" * 60)

    llm = ModelRegistry.create("gemini", api_key=GEMINI_KEY, model="gemini-2.5-flash-lite")
    messages = [
        Message(role=MessageRole.SYSTEM, content="你是一个简洁的助手，回答控制在50字以内。"),
        Message(role=MessageRole.USER, content="光速是多少？"),
    ]

    full_text = ""
    async for chunk in llm.chat(messages, temperature=0):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_text += chunk.content
        if chunk.usage:
            print(f"\n  [usage: prompt={chunk.usage.prompt_tokens}, completion={chunk.usage.completion_tokens}, total={chunk.usage.total_tokens}]")

    print()
    assert full_text, "应有文本输出"
    assert "299792458" in full_text or "3" in full_text, f"回答应包含光速: {full_text}"
    print("  PASS")


async def test_gemini_tool_call():
    """Gemini 工具调用测试"""
    print("\n" + "=" * 60)
    print("  Gemini (OpenAI 兼容) - 工具调用")
    print("=" * 60)

    from milu.tools import tool

    @tool(name="calculate", description="计算数学表达式")
    async def calculate(expression: str) -> str:
        return str(eval(expression))

    llm = ModelRegistry.create("gemini", api_key=GEMINI_KEY, model="gemini-2.5-flash-lite")

    agent = Agent(
        llm=llm,
        system_prompt="你是计算助手，使用 calculate 工具计算数学表达式。回答简洁。",
        tools=[calculate],
        config=AgentConfig(max_turns=3),
    )

    async for event in agent.run("计算 123 * 456 + 789"):
        if hasattr(event, "tool_name"):
            print(f"  tool_call: {event.tool_name}")
        if hasattr(event, "text"):
            print(event.text, end="", flush=True)
        if hasattr(event, "total_usage"):
            print(f"\n  [done] turns={event.turn_count}, tokens={event.total_usage.total_tokens}")
        if hasattr(event, "message") and hasattr(event, "error_type"):
            print(f"\n  [error] {event.error_type}: {event.message}")

    print("  PASS")


async def test_chatgpt_multi_turn():
    """ChatGPT 多轮对话测试"""
    print("\n" + "=" * 60)
    print("  ChatGPT (Responses API) - 多轮对话")
    print("=" * 60)

    llm = ModelRegistry.create("openai", api_key=OPENAI_KEY, model="gpt-4o-mini")

    agent = Agent(
        llm=llm,
        system_prompt="你是一个简洁的助手，回答控制在30字以内。",
        config=AgentConfig(max_turns=2),
    )

    questions = [
        "中国的首都是哪里？",
        "它的古称是什么？",  # 需要记住上文
    ]

    for q in questions:
        print(f"\n  Q: {q}")
        print("  A: ", end="")
        async for event in agent.run(q):
            if hasattr(event, "text"):
                print(event.text, end="", flush=True)
        print()

    print("  PASS")


async def main():
    if not OPENAI_KEY:
        print("ERROR: 未设置 OPENAI_API_KEY 环境变量")
        return
    if not GEMINI_KEY:
        print("ERROR: 未设置 GEMINI_API_KEY 环境变量")
        return

    results = {}
    tests = [
        # ("ChatGPT 基础对话", test_chatgpt_basic),
        # ("ChatGPT 工具调用", test_chatgpt_tool_call),
        # ("ChatGPT 多轮对话", test_chatgpt_multi_turn),
        ("Gemini 基础对话", test_gemini_basic),
        ("Gemini 工具调用", test_gemini_tool_call),
    ]

    for name, test_fn in tests:
        try:
            await test_fn()
            results[name] = "PASS"
        except Exception as e:
            results[name] = f"FAIL: {e}"
            print(f"  FAIL: {e}")

    # 汇总
    print("\n" + "=" * 60)
    print("  测试汇总")
    print("=" * 60)
    for name, status in results.items():
        icon = "[OK]" if status == "PASS" else "[!!]"
        print(f"  {icon} {name}: {status}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
