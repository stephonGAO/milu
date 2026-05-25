"""基础 LLM 调用示例 —— 演示多厂商流式对话 & 能力查询"""
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from agent_framework import Message, MessageRole
from agent_framework.llm.providers import ModelRegistry

# 加载 .env 中的 API Key
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


async def demo_capabilities():
    """展示所有已注册厂商及其能力"""
    print("=" * 60)
    print("  厂商能力一览")
    print("=" * 60)
    for name in sorted(ModelRegistry.list_providers()):
        llm = ModelRegistry.create(name)
        cap = llm.capabilities
        features = [
            f"streaming={cap.supports_streaming}",
            f"tools={cap.supports_function_calling}",
            f"json={cap.supports_json_mode}",
            f"search={cap.supports_web_search}",
            f"thinking={cap.supports_thinking}",
            f"vision={cap.supports_vision}",
        ]
        print(f"  {name:<10} | {', '.join(features)}")
    print()


async def demo_chat(provider: str, model: str, question: str):
    """与指定厂商进行一次流式对话"""
    print(f"── {provider} ({model}) ──")
    print(f"👤 {question}")
    print("🤖 ", end="", flush=True)

    llm = ModelRegistry.create(provider, model=model)
    messages = [
        Message(role=MessageRole.SYSTEM, content="你是一个简洁的助手，回答控制在两句话以内。"),
        Message(role=MessageRole.USER, content=question),
    ]

    try:
        async for chunk in llm.chat(messages):
            # 思考内容（部分模型支持）
            if chunk.reasoning_content:
                print(f"[思考: {chunk.reasoning_content}]", end="", flush=True)
            # 正文
            if chunk.content:
                print(chunk.content, end="", flush=True)
        print()  # 换行
    except Exception as e:
        print(f"\n⚠️  调用失败: {e}")
    print()


async def main():
    # 1. 能力查询
    await demo_capabilities()

    # 2. 多厂商流式对话（使用 .env 中已配置好 key 的厂商）
    print("=" * 60)
    print("  流式对话演示")
    print("=" * 60)

    await demo_chat("qwen", "qwen-max", "用一句话介绍Python")
    await demo_chat("deepseek", "deepseek-chat", "1+2+3+...+100 等于多少？")


if __name__ == "__main__":
    asyncio.run(main())
