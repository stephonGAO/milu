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
    SubAgentConfig, create_subagent_tools,
    SubAgentEvent, SubAgentDone,
)
from agent_framework.llm.providers import ModelRegistry
from agent_framework.tools.builtin import (
    BUILTIN_TOOLS,
    create_structured_output_tool,
    create_todo_write_tool,
    web_search,
    http_request,
    file,
    python_repl,
)
from agent_framework.tools.builtin.todo_write import TodoManager

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── 样式常量 ───────────────────────────────────────────────

DIVIDER  = "-" * 50
BANNER   = "=" * 60
PLAN_FILE = Path.cwd() / ".plan.json"

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
    # 构造完整的内置工具名列表（create_todo_write_tool 返回元组）
    _todo_tools = create_todo_write_tool()
    _all_names = [t._tool_wrapper.name for t in [*BUILTIN_TOOLS, create_structured_output_tool(), *_todo_tools]]
    print(f"""
  内置工具: {', '.join(_all_names)}
  子代理:   researcher（调研助手）, coder（编程助手）
  元工具:   list_catalog, search_tools, activate_tools（用于发现和激活 MCP 工具）
  计划文件: {PLAN_FILE}

  命令:
    /history   — 查看对话历史
    /reset     — 重置对话（清空上下文）
    /tools     — 查看可用工具（含休眠工具）
    /plan      — 查看当前会话计划
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
    """构建带全部内置工具和子代理的 Agent"""
    llm = ModelRegistry.create("qwen", model="qwen3.7-max", web_search=True, enable_thinking=True)

    so_tool = create_structured_output_tool()
    todo_write, todo_read = create_todo_write_tool(plan_file=PLAN_FILE)

    # 创建子代理工具
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
                tools=[web_search, http_request],
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
                tools=[file, python_repl],
                config=AgentConfig(max_turns=8, timeout=60, total_timeout=120),
            ),
        ],
    )

    all_tools = [*BUILTIN_TOOLS, so_tool, todo_write, todo_read, *subagent_tools]

    history = ConversationHistory(
        strategy="sliding_window",
        max_turns=50,
    )

    agent = Agent(
        llm=llm,
        system_prompt=(
            "你是一个功能强大的 AI 助手，拥有丰富的工具集和专业子代理团队。\n"
            "核心原则：\n"
            "  1. 主动使用工具完成任务，不要只描述怎么做——直接做\n"
            "  2. 保持上下文连贯，记住之前对话中的信息\n"
            "  3. 回答简洁准确，必要时给出代码或数据佐证\n"
            "  4. 操作文件时先用 index 了解结构，再精确操作\n"
            "  5. 你拥有元工具 list_catalog / search_tools / activate_tools，"
            "可以动态发现和激活 MCP 外部工具（如 fetch、数据库、Figma 等）。"
            "当需要外部能力时，先用 search_tools 搜索，再用 activate_tools 激活后调用\n"
            "  6. 对于多步骤任务，主动使用 todo_write 制定并跟踪计划。"
            "计划会自动保存到本地文件，如果看到提醒消息中的当前计划，"
            "请根据进度继续工作或更新计划状态。"
            "不确定当前计划时，调用 todo_read 查看\n"
            "  7. 你拥有两个专业子代理：researcher（调研助手）和 coder（编程助手）。"
            "当任务需要专项能力时，委派给对应子代理。给子代理清晰的任务描述，"
            "综合其结果给出最终回复。简单任务可直接处理，不必委派"
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

        # ── 子代理内部事件 ──
        elif isinstance(event, SubAgentEvent):
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

        # ── 子代理完成 ──
        elif isinstance(event, SubAgentDone):
            status = c("red", " [ERROR]") if event.is_error else ""
            print(
                f"  {c('magenta', '[SubAgent Done]')} "
                f"{c('dim', f'turns={event.turn_count}, tokens={event.total_usage.total_tokens}')}"
                f"{status}",
                flush=True,
            )

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
        # 同时清空计划文件
        if PLAN_FILE.exists():
            PLAN_FILE.unlink()
        print(c("yellow", "\n  对话已重置，上下文和计划已清空。\n"))

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
        _builtin_factory_tools = [*BUILTIN_TOOLS, create_structured_output_tool(), *create_todo_write_tool()]
        builtin_names = {w._tool_wrapper.name for w in _builtin_factory_tools}
        for tool_func in _builtin_factory_tools:
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

        # 子代理工具
        subagent_names = {"researcher", "coder"}
        for name in subagent_names:
            if name in active_tools:
                wrapper = agent.tools.get_tool(name)
                desc = wrapper.description[:40] if wrapper else ""
                print(f"  {c('magenta', name + ' [SubAgent]'):<30} {desc}")

        # 已激活的 MCP 工具
        all_special_names = builtin_names | meta_names | subagent_names
        activated_mcp = [n for n in active_tools if n not in all_special_names]
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
  /plan      — 查看当前会话计划
  /help      — 显示帮助
  /quit      — 退出
""")

    elif cmd == "/plan":
        if PLAN_FILE.exists():
            import json
            try:
                data = json.loads(PLAN_FILE.read_text(encoding="utf-8"))
                items = data.get("items", [])
            except Exception as e:
                print(f"\n  {c('red', '计划文件读取失败')}: {e}\n")
                return True

            if not items:
                print(f"\n  {c('dim', '暂无会话计划。')}\n")
                return True

            print(f"\n{DIVIDER}")
            print(c("bold", "  当前会话计划") + c("dim", f" ({len(items)} 个条目)"))
            print(DIVIDER)
            markers = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
            colors = {"pending": "yellow", "in_progress": "green", "completed": "dim"}
            completed = 0
            for item in items:
                status = item.get("status", "pending")
                marker = markers.get(status, "[ ]")
                color = colors.get(status, "yellow")
                line = f"  {marker} {item['content']}"
                if status == "in_progress" and item.get("activeForm"):
                    line += f"  ({c('cyan', item['activeForm'])})"
                if status == "completed":
                    completed += 1
                print(f"  {c(color, line)}")
            print(f"\n  {c('dim', f'({completed}/{len(items)} 已完成)')}")
            print(f"  {c('dim', f'文件: {PLAN_FILE}')}")
            print(DIVIDER + "\n")
        else:
            print(f"\n  {c('dim', '暂无会话计划（计划文件不存在）。')}\n")

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

    # 显示恢复的计划状态
    if PLAN_FILE.exists():
        import json
        try:
            data = json.loads(PLAN_FILE.read_text(encoding="utf-8"))
            items = data.get("items", [])
            if items:
                completed = sum(1 for i in items if i.get("status") == "completed")
                in_progress = next(
                    (i["content"] for i in items if i.get("status") == "in_progress"), None
                )
                status_line = f"  计划已恢复: {completed}/{len(items)} 已完成"
                if in_progress:
                    status_line += f" | 当前: {c('green', in_progress)}"
                print(c("yellow", status_line))
                print(c("dim", f"  输入 /plan 查看详情\n"))
        except Exception:
            pass

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
