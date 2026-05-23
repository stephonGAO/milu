"""基础使用示例"""

import asyncio
from agent_framework import ModelRegistry, Message, MessageRole


async def main():
    # 列出所有可用厂商
    print("可用厂商:", ModelRegistry.list_providers())

    # 创建 Qwen 实例
    qwen = ModelRegistry.create("qwen", model="qwen-max")
    print(f"Qwen 能力: vision={qwen.capabilities.supports_vision}, web_search={qwen.capabilities.supports_web_search}")

    # 创建 DeepSeek 实例
    deepseek = ModelRegistry.create("deepseek", model="deepseek-chat")
    print(f"DeepSeek 能力: thinking={deepseek.capabilities.supports_thinking}")

    # 创建消息
    messages = [
        Message(role=MessageRole.SYSTEM, content="你是一个有帮助的助手。"),
        Message(role=MessageRole.USER, content="你好！"),
    ]

    # 注意：实际调用需要设置环境变量 API Key
    # QWEN_API_KEY=sk-xxx
    # print("调用 Qwen...")
    # async for chunk in qwen.chat(messages):
    #     if chunk.content:
    #         print(chunk.content, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
