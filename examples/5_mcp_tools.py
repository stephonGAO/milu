"""示例 5：MCP 工具集成

演示如何通过配置文件连接 MCP 服务器，
让 Agent 自动发现并调用 MCP 提供的工具。

运行前提：
1. pip install agent-framework[mcp]
2. 在 config/mcp_servers.json 中配置 MCP 服务器
   或在 .env 中设置 MCP_CONFIG_PATH=路径
"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agent_framework import Agent, AgentConfig
from agent_framework.llm.providers import ModelRegistry


async def example_auto():
    """方式1：自动搜索 config/mcp_servers.json"""
    print("=" * 60)
    print("  MCP 示例：自动搜索配置文件")
    print("=" * 60)

    llm = ModelRegistry.create("deepseek", model="deepseek-chat")

    # Agent 会自动搜索 config/mcp_servers.json
    async with Agent(
        llm=llm,
        system_prompt="你是一个使用外部工具的助手。",
        config=AgentConfig(max_turns=5),
    ) as agent:
        # 查看已注册的工具
        print(f"已注册工具: {agent.tools.list_tools()}")

        async for event in agent.run("使用可用工具完成你的任务"):
            if hasattr(event, "tool_name"):
                print(f"  工具调用: {event.tool_name}")
            if hasattr(event, "text"):
                print(event.text, end="", flush=True)
            if hasattr(event, "turn_count"):
                print(f"\n  [完成] 轮次={event.turn_count}")


async def example_explicit_path():
    """方式2：指定配置文件路径"""
    print("=" * 60)
    print("  MCP 示例：指定配置文件路径")
    print("=" * 60)

    llm = ModelRegistry.create("deepseek", model="deepseek-chat")

    async with Agent(
        llm=llm,
        system_prompt="你是一个使用外部工具的助手。",
        mcp_config_path="config/mcp_servers.json",
        config=AgentConfig(max_turns=5),
    ) as agent:
        print(f"已注册工具: {agent.tools.list_tools()}")


async def example_env_path():
    """方式3：通过 .env 环境变量 MCP_CONFIG_PATH 指定"""
    print("=" * 60)
    print("  MCP 示例：.env 中配置 MCP_CONFIG_PATH")
    print("=" * 60)

    # 在 .env 中设置：MCP_CONFIG_PATH=/path/to/your/mcp.json
    llm = ModelRegistry.create("deepseek", model="deepseek-chat")

    async with Agent(
        llm=llm,
        system_prompt="你是一个使用外部工具的助手。",
        config=AgentConfig(max_turns=5),
    ) as agent:
        print(f"已注册工具: {agent.tools.list_tools()}")


async def main():
    from agent_framework.tools.mcp import MCPServerConfig

    # 简单验证配置加载
    configs = MCPServerConfig.load_file("config/mcp_servers.json")
    print("从配置文件加载的 MCP 服务器:")
    for c in configs:
        print(f"  {c.name}: transport={c.transport}, command={c.command}")

    if not configs:
        print("  (未找到配置文件)")

    # 选择一种方式运行
    # await example_auto()
    # await example_explicit_path()
    # await example_env_path()


if __name__ == "__main__":
    asyncio.run(main())
