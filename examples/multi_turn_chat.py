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
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from milu import (
    Agent, AgentMode, ConfirmResponse,
    TextDelta, ReasoningDelta,
    ToolCallStart, ToolConfirmRequired, ToolResult,
    AgentDone, AgentError,
    SubAgentEvent, SubAgentDone,
    HistoryCompacted, SessionLoaded,
)
from milu.llm.providers import ModelRegistry
from milu.tools.builtin import (
    BUILTIN_TOOLS,
    create_structured_output_tool,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _read_plan_items(agent) -> list[dict]:
    """读取当前会话的计划条目（todo 工具持久化在 session 目录的 plan.json）。

    v2 起 todo 工具无状态化，计划不再挂在 agent 上，而是落盘到
    {session_dir}/plan.json，格式为 {"items": [{content, status, activeForm}]}。
    """
    if not agent.session:
        return []
    plan_file = agent.session.dir_path / "plan.json"
    if not plan_file.exists():
        return []
    try:
        return json.loads(plan_file.read_text(encoding="utf-8")).get("items", [])
    except (OSError, ValueError):
        return []


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
    # 构造完整的内置工具名列表（todo 工具已包含在 BUILTIN_TOOLS 中）
    _all_names = [t._tool_wrapper.name for t in [*BUILTIN_TOOLS, create_structured_output_tool()]]
    print(f"""
  内置工具: {', '.join(_all_names)}
  子代理:   researcher（调研员）, reader（长内容阅读员）, coder（编码执行员）
  元工具:   list_catalog, search_tools, activate_tools（用于发现和激活 MCP 工具）
  技能:     内置 skill-creator / deep-research / mcp-builder 等，LLM 按需 load_skill 加载
  计划文件: 每个会话独立保存（~/.milu/sessions/{{session_id}}/plan.json）

  命令:
    /history   — 查看对话历史
    /reset     — 重置对话（清空上下文）
    /tools     — 查看可用工具（含休眠工具）
    /skills    — 查看可用技能
    /plan      — 查看当前会话计划
    /mode      — 查看/切换操作模式（talk/manual/auto/superwork）
    /save      — 保存当前会话
    /sessions  — 查看所有会话
    /new       — 新建会话
    /load <id> — 加载历史会话
    /quit      — 退出
""")


# ── 不安全工具确认回调 ────────────────────────────────────

async def confirm_unsafe(tool_name: str, args_str: str) -> ConfirmResponse:
    """不安全工具执行前请求用户确认，支持自定义消息"""
    short_args = args_str[:100] + "..." if len(args_str) > 100 else args_str
    print(f"\n  {c('red', '[UNSAFE]')} {c('bold', tool_name)}({c('dim', short_args)})")
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
    """构建 Agent —— 展示「全配默认」：几乎零配置。

    顶层 Agent 不传 tools / subagents / prompt_dir / skills_dir 时自动获得：
      - 全套内置工具 BUILTIN_TOOLS（file / python_repl / http / web_search /
        shell / datetime / todo）
      - 内置子代理三件套 researcher / reader / coder（操作模式与人工确认回调
        经 ContextVar 自动继承父 Agent）
      - 内置 main 角色提示词 + 内置技能（skill-creator / deep-research 等）
      - 会话持久化到 ~/.milu/sessions（MILU_HOME 可覆盖）

    需要定制时显式传参覆盖，例如：
      tools=[...]                                # 覆盖默认全套工具（[] 即无工具）
      subagents=builtin_subagent_configs(include=("reader",))   # 只要部分子代理
      prompt_dir=... / system_prompt="..."       # 自定义提示词
    """
    llm = ModelRegistry.create("qwen", model="qwen-plus", web_search=True, enable_thinking=False)

    # structured_output 需注入 LLM 做自修复，须经工厂创建后追加；
    # 显式传 tools 会覆盖默认全套，因此这里自带 BUILTIN_TOOLS
    so_tool = create_structured_output_tool()

    return Agent(
        llm=llm,
        tools=[*BUILTIN_TOOLS, so_tool],
        on_confirm=confirm_unsafe,
    )


# ── 事件处理 ──────────────────────────────────────────────

async def handle_turn(agent: Agent, user_input: str):
    """处理一轮用户输入，流式渲染 Agent 事件"""
    thinking_visible = False
    assistant_text = ""
    announced_subagents: set[str] = set()  # 已显示开始标签的子代理
    # 子代理输出状态：跟踪是否需要在下一行打印标签前缀
    sa_needs_tag: dict[str, bool] = {}   # True = 下次文本输出前需打印标签

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
            name = event.subagent_name
            tag = c("magenta", f"[{name}]")

            # 首次出现该子代理事件时，打印开始标签
            if name not in announced_subagents:
                announced_subagents.add(name)
                sa_needs_tag[name] = True  # 首个文本前需要标签
                print(
                    f"\n  {c('magenta', '╭─')} {tag} {c('magenta', '开始执行')} {c('magenta', '─' * 30)}",
                    flush=True,
                )

            if isinstance(event.event, TextDelta):
                if sa_needs_tag.get(name, True):
                    print(f"    {tag} ", end="", flush=True)
                    sa_needs_tag[name] = False
                print(event.event.text, end="", flush=True)
                # 文本以换行结尾时，下次输出需要重新打印标签
                if event.event.text.endswith("\n"):
                    sa_needs_tag[name] = True
            elif isinstance(event.event, ReasoningDelta):
                if sa_needs_tag.get(name, True):
                    print(f"    {tag} ", end="", flush=True)
                    sa_needs_tag[name] = False
                print(c("dim", event.event.text), end="", flush=True)
                if event.event.text.endswith("\n"):
                    sa_needs_tag[name] = True
            elif isinstance(event.event, ToolCallStart):
                # 工具调用始终另起一行
                print(
                    f"\n    {tag} {c('cyan', '->')} {event.event.tool_name}"
                    f"{c('dim', '(' + str(event.event.arguments)[:60] + ')')}",
                    flush=True,
                )
                sa_needs_tag[name] = True  # 工具调用后需要新行+标签
            elif isinstance(event.event, ToolResult):
                result_tag = c("green", "OK") if not event.event.is_error else c("red", "ERR")
                output = event.event.output[:120] + "..." if len(event.event.output) > 120 else event.event.output
                print(f"\n    {tag} {result_tag} {c('dim', output)}", flush=True)
                sa_needs_tag[name] = True  # 工具结果后需要新行+标签

        # ── 子代理完成 ──
        elif isinstance(event, SubAgentDone):
            name = event.subagent_name
            tag = c("magenta", f"[{name}]")
            status = c("red", " [ERROR]") if event.is_error else ""
            # 如果子代理文本未以换行结束，先补一个换行
            if not sa_needs_tag.get(name, True):
                print()
            print(
                f"  {c('magenta', '╰─')} {tag} {c('magenta', '完成')} "
                f"{c('dim', f'turns={event.turn_count}, tokens={event.total_usage.total_tokens}')}"
                f"{status} {c('magenta', '─' * 28)}",
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

        # ── 上下文压缩 ──
        elif isinstance(event, HistoryCompacted):
            print(
                f"\n  {c('cyan', '[COMPACT]')} "
                f"对话历史已压缩: {event.original_count} → {event.compacted_count} 条消息 "
                f"({c('dim', event.strategy)})",
                flush=True,
            )


# ── 内置命令处理 ──────────────────────────────────────────

async def handle_command(agent: Agent, cmd: str) -> bool:
    """
    处理 / 命令。返回 True 表示已处理（不发送给 Agent），
    返回 False 表示应退出循环。
    """
    if cmd == "/quit":
        print(c("dim", "\n再见!"))
        return False  # 信号：退出

    elif cmd == "/reset":
        await agent.reset()
        print(c("yellow", "\n  对话已重置，上下文和计划已清空。\n"))

    elif cmd == "/history":
        messages = agent.history.all_messages
        session = agent.history.session

        print(f"\n{DIVIDER}")
        print(c("bold", "  对话历史") + c("dim", f" ({len(messages)} 条消息)"))

        # 显示 session 文件位置
        if session:
            print(c("dim", f"  会话 ID: {session.session_id}"))
            print(c("dim", f"  日志文件: {session.conversation_path}"))
            print(c("dim", f"  已记录消息: {session.message_count} 条 | "
                            f"当前内存: {len(messages)} 条"))

        print(DIVIDER)
        for msg in messages:
            role = msg.role.value.upper()
            raw = msg.content or ""
            content = raw[:500] + "..." if len(raw) > 500 else raw
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
        _builtin_factory_tools = [*BUILTIN_TOOLS, create_structured_output_tool()]
        builtin_names = {w._tool_wrapper.name for w in _builtin_factory_tools}
        for tool_func in _builtin_factory_tools:
            w = tool_func._tool_wrapper
            if w.name in active_tools:
                safe = c("green", " [S]") if w.is_safe else ""
                print(f"  {c('yellow', w.name):<30} {w.description[:40]}{safe}")

        # 元工具
        meta_names = {"list_catalog", "search_tools", "activate_tools", "load_skill"}
        for name in meta_names:
            if name in active_tools:
                wrapper = agent.tools.get_tool(name)
                desc = wrapper.description[:40] if wrapper else ""
                print(f"  {c('cyan', name):<30} {desc}")

        # 子代理工具
        subagent_names = {"researcher", "reader", "coder"}
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
  /skills    — 查看可用技能
  /plan      — 查看当前会话计划
  /mode      — 查看/切换操作模式（talk/manual/auto/superwork）
  /prompt    — 查看当前系统提示词
  /compact   — 手动压缩对话历史
  /save      — 保存当前会话
  /sessions  — 查看所有会话
  /new       — 新建会话（自动保存当前）
  /load <id> — 加载历史会话
  /help      — 显示帮助
  /quit      — 退出
""")

    elif cmd == "/mode":
        mode = agent.mode
        mode_colors = {
            AgentMode.TALK: "cyan",
            AgentMode.MANUAL: "yellow",
            AgentMode.AUTO: "green",
            AgentMode.SUPERWORK: "red",
        }
        mode_desc = {
            AgentMode.TALK: "只读模式（仅允许安全操作）",
            AgentMode.MANUAL: "人工审批模式（不安全操作需确认）",
            AgentMode.AUTO: "自主模式（自主决策，可选 AI 安全判定兜底）",
            AgentMode.SUPERWORK: "全权限模式（无安全检查）",
        }
        print(f"\n{DIVIDER}")
        print(c("bold", "  操作模式"))
        print(DIVIDER)
        for m in AgentMode:
            marker = c("bold", " → 当前") if m == mode else ""
            color = mode_colors.get(m, "reset")
            print(f"  {c(color, m.value):<20} {c('dim', mode_desc.get(m, ''))}{marker}")
        print(DIVIDER)
        print(c("dim", "  用法: /mode <talk|manual|auto|superwork>\n"))

    elif cmd.startswith("/mode "):
        new_mode = cmd.split(maxsplit=1)[1].strip()
        try:
            agent.set_mode(new_mode)
            mode_colors = {
                "talk": "cyan",
                "manual": "yellow",
                "auto": "green",
                "superwork": "red",
            }
            color = mode_colors.get(new_mode, "reset")
            print(f"\n  {c('green', '模式已切换为:')} {c(color, c('bold', new_mode))}\n")
        except ValueError:
            print(f"\n  {c('red', f'无效模式: {new_mode}')}（可选: talk, manual, auto, superwork）\n")

    elif cmd == "/plan":
        items = _read_plan_items(agent)
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
            line = f"  {marker} {item.get('content', '')}"
            if status == "in_progress" and item.get("activeForm"):
                line += f"  ({c('cyan', item['activeForm'])})"
            if status == "completed":
                completed += 1
            print(f"  {c(color, line)}")
        print(f"\n  {c('dim', f'({completed}/{len(items)} 已完成)')}")
        if agent.session:
            plan_path = agent.session.dir_path / "plan.json"
            print(f"  {c('dim', f'文件: {plan_path}')}")
        print(DIVIDER + "\n")

    elif cmd == "/skills":
        skill_names = agent.skill_registry.list_names()
        if not skill_names:
            print(f"\n  {c('dim', '暂无可用技能（skills/ 目录为空）。')}\n")
            return True

        print(f"\n{DIVIDER}")
        print(c("bold", "  可用技能") + c("dim", f" ({len(skill_names)} 个)"))
        print(DIVIDER)
        for name in skill_names:
            cfg = agent.skill_registry.get(name)
            trigger_part = f"  {c('dim', '[' + ', '.join(cfg.triggers) + ']')}" if cfg.triggers else ""
            print(f"  {c('blue', name):<30} {cfg.description}{trigger_part}")
        print(f"\n  {c('dim', 'LLM 会自动调用 load_skill 按需加载技能正文')}")
        print(DIVIDER + "\n")

    elif cmd == "/prompt":
        # 触发一次最新拼装（支持热重载：用户可能在运行中改了 .md 文件）
        agent._build_system_prompt()
        system_msg = agent.history.all_messages[0] if agent.history.all_messages else None
        content = system_msg.content if system_msg else ""

        if not content:
            print(f"\n  {c('dim', '当前系统提示词为空。')}\n")
            return True

        # 统计：按 section 拆分（如果来自 PromptBuilder）
        line_count = content.count("\n") + 1
        char_count = len(content)

        # 提取 PromptBuilder 文件列表（如果存在）
        file_list = []
        if agent.prompt_builder:
            file_list = agent.prompt_builder.list_files()

        print(f"\n{DIVIDER}")
        print(c("bold", "  当前系统提示词")
              + c("dim", f"  ({line_count} 行, {char_count} 字符)"))
        if file_list:
            names = ", ".join(f["file"] for f in file_list)
            print(c("dim", f"  来源: {names}"))
        print(DIVIDER)
        print(content)
        print(DIVIDER + "\n")

    elif cmd == "/compact":
        if not agent._history.compact_enabled:
            print(f"\n  {c('dim', '上下文压缩未启用。')}\n")
            return True
        original_count = len(agent.history._messages)
        compacted, summary = await agent._history.manual_compact()
        agent.history.replace_all(compacted)
        agent._history._log_compaction(compacted)
        print(f"\n{DIVIDER}")
        print(c("bold", "  手动压缩完成")
              + c("dim", f"  ({original_count} → {len(compacted)} 条消息)"))
        print(DIVIDER)
        # 显示摘要前 500 字符
        preview = summary[:500] + "..." if len(summary) > 500 else summary
        print(f"  {c('dim', preview)}")
        print(DIVIDER + "\n")

    elif cmd == "/save":
        agent.save_session()
        if agent.session:
            print(f"\n  {c('green', '会话已保存')}")
            print(f"  ID: {c('cyan', agent.session.session_id)}")
            print(f"  路径: {c('dim', str(agent.session.dir_path))}")
            print(f"  消息数: {agent.session.message_count}\n")
        else:
            print(f"\n  {c('dim', '会话功能未启用。')}\n")

    elif cmd == "/sessions":
        from milu.agent.session import Session as SessionClass
        base_dir = Path(agent.session.base_dir) if agent.session else Path(".sessions")
        sessions = SessionClass.list_sessions(base_dir)
        if not sessions:
            print(f"\n  {c('dim', '暂无历史会话。')}\n")
            return True

        print(f"\n{DIVIDER}")
        print(c("bold", "  历史会话") + c("dim", f" ({len(sessions)} 个)"))
        print(DIVIDER)
        current_id = agent.session.session_id if agent.session else None
        for s in sessions:
            sid = s.get("session_id", "?")
            model = s.get("model", "")
            msg_count = s.get("message_count", 0)
            updated = s.get("updated_at", 0)
            from datetime import datetime
            time_str = datetime.fromtimestamp(updated).strftime("%m-%d %H:%M") if updated else "?"
            marker = c("green", " ← 当前") if sid == current_id else ""
            model_str = c("dim", f" ({model})") if model else ""
            print(f"  {c('cyan', sid)}  {c('dim', time_str)}  {msg_count} 条消息{model_str}{marker}")
        print(DIVIDER + "\n")

    elif cmd == "/new":
        agent.new_session()
        print(f"\n  {c('green', '新会话已创建')}")
        if agent.session:
            print(f"  ID: {c('cyan', agent.session.session_id)}\n")

    elif cmd.startswith("/load"):
        parts = cmd.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            print(f"\n  {c('yellow', '用法: /load <session_id>')}\n")
            return True
        session_id = parts[1].strip()
        try:
            msg_count = agent.load_session(session_id)
            print(f"\n  {c('green', '会话已加载')}")
            print(f"  ID: {c('cyan', session_id)}")
            print(f"  消息数: {msg_count}\n")
        except FileNotFoundError:
            print(f"\n  {c('red', f'会话不存在: {session_id}')}\n")
        except Exception as e:
            print(f"\n  {c('red', f'加载失败: {e}')}\n")

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

    # 显示会话信息
    if agent.session:
        print(c("cyan", f"  会话 ID: {agent.session.session_id}"))
        print(c("dim", f"  日志路径: {agent.session.dir_path}\n"))

    # 显示压缩器参数
    if agent._history.compact_enabled:
        cp = agent._history._compactor
        print(c("dim", "  压缩器参数:"))
        print(c("dim", f"    最大上下文窗口: {cp._max_context_window:,} tokens"))
        print(c("dim", f"    占位符轮次阈值: {cp._old_round_threshold} 轮（age > 此值 → 全部占位符）"))
        print(c("dim", f"    截断字符数阈值: {cp._truncate_threshold} 字符（age 中间轮次 → 超过截断）"))
        print(c("dim", f"    L4 触发比例:    {cp._config.trigger_ratio:.0%}（token 占比超此值 → LLM 摘要）"))
        print(c("dim", f"    最近保留轮数:   {cp._config.recent_rounds} 轮（最后这部分上下文超 30% 时降为 0）"))
        print()

    # 显示恢复的计划状态
    items = _read_plan_items(agent)
    if items:
        completed = sum(1 for i in items if i.get("status") == "completed")
        in_progress = next(
            (i.get("content") for i in items if i.get("status") == "in_progress"), None
        )
        status_line = f"  计划已恢复: {completed}/{len(items)} 已完成"
        if in_progress:
            status_line += f" | 当前: {c('green', in_progress)}"
        print(c("yellow", status_line))
        print(c("dim", f"  输入 /plan 查看详情\n"))

    turn_count = 0

# 案例输入
# 使用子代理帮我调研一下 2026 年最流行的 Python Web 框架，简要对比优缺点。
# 并用 Python 写一个快速排序的示例文件，保存到 quicksort_demo.py。并对写出的代码进行审查。根据审查意见再进行修改。
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
                if not await handle_command(agent, user_input):
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
        # 保存会话并断开 MCP 连接
        agent.save_session()
        await agent.disconnect_mcp()


if __name__ == "__main__":
    # Windows 下启用 ANSI 颜色
    if sys.platform == "win32":
        import os
        os.system("")

    asyncio.run(main())
