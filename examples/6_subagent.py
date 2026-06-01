"""子代理（SubAgent）委派示例 —— 父 Agent 将专项任务委派给独立子 Agent

演示：
  - 定义多个专业子代理（调研助手、编程助手）
  - 父 Agent 自主选择并委派任务
  - 子代理独立执行，结果精炼后返回父 Agent
  - 子代理内部事件（SubAgentEvent）的可视化
  - 完整的事件流处理

用法：
    python examples/6_subagent.py
"""
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from agent_framework import (
    Agent, AgentConfig,
    SubAgentConfig, create_subagent_tools,
    SubAgentEvent, SubAgentDone,
    TextDelta, ReasoningDelta,
    ToolCallStart, ToolResult,
    AgentDone, AgentError,
)
from agent_framework.llm.providers import ModelRegistry
from agent_framework.tools.builtin import (
    BUILTIN_TOOLS,
    web_search,
    http_request,
    file_read,
    file_write,
    python_repl,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ── 样式常量 ──────────────────────────────────────────────

DIVIDER = "-" * 50
BANNER = "=" * 60

COLORS = {
    "reset":   "\033[0m",
    "bold":    "\033[1m",
    "dim":     "\033[2m",
    "green":   "\033[32m",
    "yellow":  "\033[33m",
    "blue":    "\033[34m",
    "magenta": "\033[35m",
    "cyan":    "\033[36m",
    "red":     "\033[31m",
}


def c(color: str, text: str) -> str:
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


# ── 主流程 ────────────────────────────────────────────────

async def main():
    print(f"\n{BANNER}")
    print(c("bold", c("cyan", "  SubAgent 子代理委派示例")))
    print(BANNER)

    llm = ModelRegistry.create("qwen", model="qwen3.7-max", enable_thinking=True, web_search=True)
    llm2 = ModelRegistry.create("qwen", model="qwen3.7-max", enable_thinking=True)

    # 定义子代理
    subagent_tools = create_subagent_tools(
        llm=llm,
        subagents=[
            SubAgentConfig(
                name="researcher",
                description=(
                    "调研助手：擅长搜索和整理信息。"
                    "当需要查找资料、对比分析、总结报告时委派此代理。"
                ),
                system_prompt=(
                    "你是一个专业的调研助手。你的任务是搜索和整理信息，"
                    "提供准确、全面的调研结果。请引用信息来源。"
                ),
                tools=[file_read],
                config=AgentConfig(max_turns=8, timeout=60, total_timeout=120),
            ),
            SubAgentConfig(
                name="coder",
                description=(
                    "编程助手：擅长写代码和调试。"
                    "当需要编写、修改、测试代码或解决编程问题时委派此代理。"
                ),
                system_prompt=(
                    "你是一个专业的编程助手。你擅长 Python 编程，"
                    "可以编写清晰的代码、调试问题、创建示例文件。"
                    "操作文件时先用 index 了解结构，再精确操作。"
                ),
                tools=[file_read, file_write, python_repl],
                config=AgentConfig(max_turns=8, timeout=60, total_timeout=120),
            ),
        ],
    )

    # 构建父 Agent
    agent = Agent(
        llm=llm2,
        system_prompt=(
            "你是一个项目经理，拥有两个专业子代理：\n"
            "  - researcher：调研助手，擅长搜索和整理信息\n"
            "  - coder：编程助手，擅长写代码和调试\n\n"
            "工作原则：\n"
            "  1. 分析用户需求，决定委派给哪个子代理\n"
            "  2. 给子代理清晰的任务描述\n"
            "  3. 综合子代理的结果，给出最终回复\n"
            "  4. 如果任务简单，也可以直接回答（不需要委派）"
        ),
        tools=[*BUILTIN_TOOLS, *subagent_tools],
        config=AgentConfig(max_turns=10, timeout=120, total_timeout=300),
    )

    # 测试任务
    tasks = [
        "帮我调研一下 2026 年最流行的 Python Web 框架，简要对比优缺点",
        "用 Python 写一个快速排序的示例文件，保存到 quicksort_demo.py",
    ]

    for task in tasks:
        print(f"\n{DIVIDER}")
        print(f"{c('green', 'User>')} {task}")
        print(DIVIDER)

        thinking_visible = False
        in_subagent = False

        async for event in agent.run(task):

            # 思考过程
            if isinstance(event, ReasoningDelta):
                if not thinking_visible:
                    print(c("dim", "  [thinking] "), end="", flush=True)
                    thinking_visible = True
                print(c("dim", event.text), end="", flush=True)

            # 正文输出
            elif isinstance(event, TextDelta):
                if thinking_visible:
                    print()
                    thinking_visible = False
                if in_subagent:
                    print()  # 子代理结束后换行
                    in_subagent = False
                print(event.text, end="", flush=True)

            # 工具调用开始
            elif isinstance(event, ToolCallStart):
                args_str = str(event.arguments)
                short_args = args_str[:80] + "..." if len(args_str) > 80 else args_str
                print(
                    f"\n  {c('yellow', '->')} {c('bold', event.tool_name)}"
                    f"{c('dim', '(' + short_args + ')')}",
                    flush=True,
                )

            # 子代理内部事件
            elif isinstance(event, SubAgentEvent):
                if not in_subagent:
                    print(f"\n  {c('magenta', '[SubAgent]')} {c('dim', '开始执行...')}", flush=True)
                    in_subagent = True
                # 显示子代理的工具调用
                if isinstance(event.event, ToolCallStart):
                    print(
                        f"    {c('cyan', '->')} {event.event.tool_name}"
                        f"{c('dim', '(' + str(event.event.arguments)[:60] + ')')}",
                        flush=True,
                    )
                elif isinstance(event.event, ToolResult):
                    tag = c("green", "OK") if not event.event.is_error else c("red", "ERR")
                    output = event.event.output[:120] + "..." if len(event.event.output) > 120 else event.event.output
                    print(f"    {tag} {c('dim', output)}", flush=True)
                elif isinstance(event.event, TextDelta):
                    # 可选：显示子代理的文本输出（缩进显示）
                    pass  # 省略子代理文本，避免信息过多

            # 子代理完成
            elif isinstance(event, SubAgentDone):
                print(
                    f"  {c('magenta', '[SubAgent Done]')} "
                    f"{c('dim', f'turns={event.turn_count}, ')}"
                    f"{c('dim', f'tokens={event.total_usage.total_tokens}')}"
                    f"{c('red', ' [ERROR]') if event.is_error else ''}",
                    flush=True,
                )
                in_subagent = False

            # 工具结果（非子代理工具）
            elif isinstance(event, ToolResult):
                if not in_subagent:
                    tag = c("green", "OK ") if not event.is_error else c("red", "ERR")
                    output = event.output[:200] + "..." if len(event.output) > 200 else event.output
                    print(f"  {tag} {c('dim', output)}", flush=True)

            # 完成
            elif isinstance(event, AgentDone):
                if thinking_visible or in_subagent:
                    print()
                u = event.total_usage
                meta = (
                    f"turns={event.turn_count}, "
                    f"tokens={u.total_tokens}"
                    f" (prompt={u.prompt_tokens}, completion={u.completion_tokens})"
                )
                print(f"\n  {c('dim', '[' + meta + ']')}")

            # 错误
            elif isinstance(event, AgentError):
                print(f"\n  {c('red', '[ERROR]')} {c('red', event.message)}")

        print()  # 任务间空行

    print(f"\n{c('dim', '示例结束。')}")


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        import os
        os.system("")

    asyncio.run(main())
