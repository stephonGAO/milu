"""多轮对话 Agent —— 交互式命令行聊天

演示：
  - 多轮对话记忆（上下文跨轮保持）
  - 工具自动调用（file、python_repl、datetime 等）
  - 对话重置 / 历史查看 / 退出
  - 流式输出 + 思考过程 + 工具调用可视化
  - 优雅的错误处理和超时控制

用法：
    python examples/multi_turn_chat.py
"""
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent_framework import (
    Agent, AgentConfig, ConversationHistory, ConfirmResponse,
    TextDelta, ReasoningDelta,
    ToolCallStart, ToolConfirmRequired, ToolResult,
    AgentDone, AgentError,
)
from agent_framework.llm.providers import ModelRegistry
from agent_framework.tools.builtin import BUILTIN_TOOLS, create_structured_output_tool, create_todo_write_tool

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── 样式常量 ───────────────────────────────────────────────

DIVIDER  = "-" * 50
BANNER   = "=" * 60

COLORS = {
    "reset":    "\033[0m",
    "bold":     "\033[1m",
    "dim":      "\033[2m",
    "green":    "\033[32m",
    "yellow":   "\033[33m",
    "blue":     "\033[34m",
    "magenta":  "\033[35m",
    "cyan":     "\033[36m",
    "red":      "\033[31m",
}


def c(color: str, text: str) -> str:
    """带颜色的文本"""
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def print_header():
    print(f"\n{BANNER}")
    print(c("bold", c("cyan", "  多轮对话 Agent — 交互式聊天")))
    print(BANNER)
    print(f"""
  内置工具: {', '.join(t._tool_wrapper.name for t in [*BUILTIN_TOOLS, create_structured_output_tool(), create_todo_write_tool()])}
  元工具:   list_catalog, search_tools, activate_tools（用于发现和激活 MCP 工具）

  命令:
    /history   — 查看对话历史
    /reset     — 重置对话（清空上下文）
    /tools     — 查看可用工具（含休眠工具）
    /quit      — 退出
""")


# ── 危险工具确认回调 ────────────────────────────────────

async def confirm_dangerous(tool_name: str, args_str: str) -> ConfirmResponse:
    """危险工具执行前请求用户确认，支持自定义消息"""
    short_args = args_str[:100] + "..." if len(args_str) > 100 else args_str
    print(f"\n  {c('red', '[DANGEROUS]')} {c('bold', tool_name)}({c('dim', short_args)})")
    print(f"  {c('dim', '输入 y 同意 / n 拒绝 / 或直接输入指示发给 Agent')}")
    try:
        resp = input(f"  {c('yellow', '> ')}").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ConfirmResponse(approved=False)

    if not resp:
        return ConfirmResponse(approved=False)
    if resp.lower() in ("y", "yes"):
        return ConfirmResponse(approved=True)
    if resp.lower() in ("n", "no"):
        return ConfirmResponse(approved=False, message="用户选择拒绝执行")
    # 自定义消息
    return ConfirmResponse(approved=False, message=resp)


# ── Agent 构建 ──────────────────────────────────────────

def build_agent() -> Agent:
    """构建带全部内置工具的 Agent"""
    llm = ModelRegistry.create("qwen", model="qwen3.7-max", web_search=True, enable_thinking=True)

    so_tool = create_structured_output_tool()
    todo_tool = create_todo_write_tool()
    all_tools = [*BUILTIN_TOOLS, so_tool, todo_tool]

    history = ConversationHistory(
        strategy="sliding_window",
        max_turns=50,
    )

    agent = Agent(
        llm=llm,
        system_prompt=(
            "你是一个功能强大的 AI 助手，拥有丰富的工具集。\n"
            "核心原则：\n"
            "  1. 主动使用工具完成任务，不要只描述怎么做——直接做\n"
            "  2. 保持上下文连贯，记住之前对话中的信息\n"
            "  3. 回答简洁准确，必要时给出代码或数据佐证\n"
            "  4. 操作文件时先用 index 了解结构，再精确操作\n"
            "  5. 你拥有元工具 list_catalog / search_tools / activate_tools，"
            "可以动态发现和激活 MCP 外部工具（如 fetch、数据库、Figma 等）。"
            "当需要外部能力时，先用 search_tools 搜索，再用 activate_tools 激活后调用"
        ),
        tools=all_tools,
        history=history,
        # config=AgentConfig(max_turns=8, timeout=60, total_timeout=300, confirm_dangerous=True),
        config=AgentConfig(),
        on_confirm=confirm_dangerous,
    )

    return agent


# ── 事件处理 ──────────────────────────────────────────────

async def handle_turn(agent: Agent, user_input: str):
    """处理一轮用户输入，流式渲染 Agent 事件"""
    thinking_visible = False
    assistant_text = ""

    async for event in agent.run(user_input):

        # ── 思考过程 ──
        if isinstance(event, ReasoningDelta):
            if not thinking_visible:
                print(c("dim", "  [thinking] "), end="", flush=True)
                thinking_visible = True
            print(c("dim", event.text), end="", flush=True)

        # ── 助手文本 ──
        elif isinstance(event, TextDelta):
            if thinking_visible:
                print()  # 思考结束后换行
                thinking_visible = False
            print(event.text, end="", flush=True)
            assistant_text += event.text

        # ── 工具调用开始 ──
        elif isinstance(event, ToolCallStart):
            args_str = str(event.arguments)
            short_args = args_str[:100] + "..." if len(args_str) > 100 else args_str
            print(
                f"\n  {c('yellow', '->')} {c('bold', event.tool_name)}"
                f"{c('dim', '(' + short_args + ')')}",
                flush=True,
            )

        # ── 危险工具确认结果 ──
        elif isinstance(event, ToolConfirmRequired):
            if event.approved:
                print(f"  {c('green', '[APPROVED]')} 用户同意执行", flush=True)
            else:
                if event.message:
                    print(f"  {c('red', '[REJECTED]')} 用户指示: {c('cyan', event.message)}", flush=True)
                else:
                    print(f"  {c('red', '[REJECTED]')} 用户拒绝执行", flush=True)

        # ── 工具结果 ──
        elif isinstance(event, ToolResult):
            if event.is_error:
                tag = c("red", "ERR")
            else:
                tag = c("green", "OK ")
            output = event.output[:250] + "..." if len(event.output) > 250 else event.output
            print(f"  {tag} {c('dim', output)}", flush=True)

        # ── 完成 ──
        elif isinstance(event, AgentDone):
            if thinking_visible:
                print()
            u = event.total_usage
            meta = (
                f"turns={event.turn_count}, "
                f"tokens={u.total_tokens}"
                f" (prompt={u.prompt_tokens}, completion={u.completion_tokens})"
            )
            print(f"\n  {c('dim', '[' + meta + ']')}")

        # ── 错误 ──
        elif isinstance(event, AgentError):
            print(f"\n  {c('red', '[ERROR]')} {c('red', event.message)}")


# ── 内置命令处理 ──────────────────────────────────────────

def handle_command(agent: Agent, cmd: str) -> bool:
    """
    处理 / 命令。返回 True 表示已处理（不发送给 Agent），
    返回 False 表示应退出循环。
    """
    if cmd == "/quit":
        print(c("dim", "\n再见!"))
        return False  # 信号：退出

    elif cmd == "/reset":
        agent.history.clear()
        print(c("yellow", "\n  对话已重置，上下文已清空。\n"))

    elif cmd == "/history":
        messages = agent.history.all_messages
        print(f"\n{DIVIDER}")
        print(c("bold", "  对话历史") + c("dim", f" ({len(messages)} 条消息)"))
        print(DIVIDER)
        for msg in messages:
            role = msg.role.value.upper()
            raw = msg.content or ""
            content = raw[:120] + "..." if len(raw) > 120 else raw
            content = content.replace("\n", " ")
            print(f"  {c('cyan', f'[{role:<9}]')} {c('dim', content)}")
        print(DIVIDER + "\n")

    elif cmd == "/tools":
        active_tools = agent.tools.list_tools()
        dormant_tools = agent.tools.list_dormant_tools()

        print(f"\n{DIVIDER}")
        print(c("bold", "  活跃工具") + c("dim", f" ({len(active_tools)} 个)"))
        print(DIVIDER)

        # 内置工具
        builtin_names = {w._tool_wrapper.name for w in [*BUILTIN_TOOLS, create_structured_output_tool(), create_todo_write_tool()]}
        for tool_func in [*BUILTIN_TOOLS, create_structured_output_tool(), create_todo_write_tool()]:
            w = tool_func._tool_wrapper
            if w.name in active_tools:
                danger = c("red", " [D]") if w.dangerous else ""
                print(f"  {c('yellow', w.name):<30} {w.description[:40]}{danger}")

        # 元工具
        meta_names = {"list_catalog", "search_tools", "activate_tools"}
        for name in meta_names:
            if name in active_tools:
                wrapper = agent.tools.get_tool(name)
                desc = wrapper.description[:40] if wrapper else ""
                print(f"  {c('cyan', name):<30} {desc}")

        # 已激活的 MCP 工具
        activated_mcp = [n for n in active_tools if n not in builtin_names and n not in meta_names]
        if activated_mcp:
            print(DIVIDER)
            print(c("bold", "  已激活 MCP 工具"))
            print(DIVIDER)
            for name in activated_mcp:
                wrapper = agent.tools.get_tool(name)
                desc = wrapper.description[:40] if wrapper else ""
                print(f"  {c('green', name):<30} {desc}")

        # 休眠工具（按 category 分组）
        if dormant_tools:
            grouped = {}
            for t in dormant_tools:
                cat = t["category"] or "未分类"
                grouped.setdefault(cat, []).append(t)

            print(DIVIDER)
            print(c("bold", "  休眠工具（可通过 search_tools / activate_tools 激活）")
                  + c("dim", f" ({len(dormant_tools)} 个)"))
            print(DIVIDER)
            for cat, items in grouped.items():
                print(f"  {c('magenta', f'【{cat}】')}")
                for item in items:
                    print(f"    {c('dim', item['name']):<32} {item['description'][:38]}")

        print(DIVIDER + "\n")

    elif cmd == "/help":
        print("""
  /history   — 查看对话历史
  /reset     — 重置对话（清空上下文）
  /tools     — 查看可用工具
  /help      — 显示帮助
  /quit      — 退出
""")
    else:
        print(c("red", f"\n  未知命令: {cmd}  (输入 /help 查看帮助)\n"))

    return True  # 已处理，继续循环


# ── 主循环 ────────────────────────────────────────────────

async def main():
    print_header()
    agent = build_agent()

    # 连接 MCP 服务器（如果存在 config/mcp_servers.json）
    await agent.connect_mcp()
    dormant = agent.tools.list_dormant_tools()
    if dormant:
        categories = {}
        for t in dormant:
            cat = t["category"] or "MCP"
            categories.setdefault(cat, []).append(t["name"])
        summary = ", ".join(f"{cat}({len(names)}个)" for cat, names in categories.items())
        print(c("cyan", f"  MCP 工具已加载（休眠态）: {summary}"))
        print(c("dim", "  输入 /tools 查看详情，或通过 search_tools / activate_tools 按需激活\n"))

    turn_count = 0

    try:
        while True:
            # ── 读取用户输入 ──
            try:
                user_input = input(c("green", "\nYou> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print(c("dim", "\n再见!"))
                break

            if not user_input:
                continue

            # ── / 命令 ──
            if user_input.startswith("/"):
                if not handle_command(agent, user_input):
                    break
                continue

            # ── 发送给 Agent ──
            turn_count += 1
            print(c("blue", f"\n  [Turn {turn_count}]"))
            print(DIVIDER)

            try:
                await handle_turn(agent, user_input)
            except Exception as e:
                print(f"\n  {c('red', '[EXCEPTION]')} {c('red', str(e))}")
    finally:
        # 断开 MCP 连接
        await agent.disconnect_mcp()


if __name__ == "__main__":
    # Windows 下启用 ANSI 颜色
    if sys.platform == "win32":
        import os
        os.system("")

    asyncio.run(main())
