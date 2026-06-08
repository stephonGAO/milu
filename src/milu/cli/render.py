"""终端渲染：ANSI 颜色、不安全工具确认回调、Agent 事件流渲染。

渲染逻辑迁移并整理自 examples/multi_turn_chat.py，供 chat REPL 与一次性 run 共用。
"""
from __future__ import annotations

from milu.i18n import t
from milu.agent.events import (
    AgentDone,
    AgentError,
    ConfirmResponse,
    HistoryCompacted,
    ReasoningDelta,
    SubAgentDone,
    SubAgentEvent,
    TextDelta,
    ToolCallStart,
    ToolConfirmRequired,
    ToolResult,
)

# ── 样式常量 ───────────────────────────────────────────────

DIVIDER = "-" * 50
BANNER = "=" * 60

COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "red": "\033[31m",
}


def c(color: str, text: str) -> str:
    """返回带 ANSI 颜色的文本。"""
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


# ── 不安全工具确认回调（auto 模式下，不安全工具执行前征求用户同意）──────────

async def confirm_unsafe(tool_name: str, args_str: str) -> ConfirmResponse:
    """不安全工具执行前请求用户确认，支持自定义指示消息。

    输入 y/yes 同意；n/no 拒绝；直接输入其他文本则作为指示发回给 Agent。
    """
    short_args = args_str[:100] + "..." if len(args_str) > 100 else args_str
    print(f"\n  {c('red', '[UNSAFE]')} {c('bold', tool_name)}({c('dim', short_args)})")
    print(f"  {c('dim', t('输入 y 同意 / n 拒绝 / 或直接输入指示发给 Agent'))}")
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
        return ConfirmResponse(approved=False, message=t("用户选择拒绝执行"))
    return ConfirmResponse(approved=False, message=resp)


# ── 事件流渲染 ─────────────────────────────────────────────

async def render_turn(
    agent, user_input: str, *, show_meta: bool = True, show_subagent: bool = True
) -> str:
    """运行 Agent 一轮，流式渲染事件，返回助手最终文本。

    :param agent: Agent 实例
    :param user_input: 用户输入
    :param show_meta: 是否在结束时打印 token/轮次等元信息
    :param show_subagent: 是否渲染子代理内部事件（SubAgentEvent/SubAgentDone）。
        为 False 时子代理照常运行，仅不展示其内部过程（由 display.show_subagent_events 控制）。
    :return: 助手正文文本（拼接所有 TextDelta）
    """
    thinking_visible = False
    assistant_text = ""
    announced_subagents: set[str] = set()
    sa_needs_tag: dict[str, bool] = {}

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
                print()
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

        # ── 不安全工具确认结果 ──
        elif isinstance(event, ToolConfirmRequired):
            if event.approved:
                print(f"  {c('green', '[APPROVED]')} {t('用户同意执行')}", flush=True)
            elif event.message:
                print(f"  {c('red', '[REJECTED]')} {t('用户指示: ')}{c('cyan', event.message)}", flush=True)
            else:
                print(f"  {c('red', '[REJECTED]')} {t('用户拒绝执行')}", flush=True)

        # ── 子代理内部事件 ──
        elif isinstance(event, SubAgentEvent):
            if not show_subagent:
                continue
            name = event.subagent_name
            tag = c("magenta", f"[{name}]")

            if name not in announced_subagents:
                announced_subagents.add(name)
                sa_needs_tag[name] = True
                print(
                    f"\n  {c('magenta', '╭─')} {tag} {c('magenta', t('开始执行'))} {c('magenta', '─' * 30)}",
                    flush=True,
                )

            if isinstance(event.event, TextDelta):
                if sa_needs_tag.get(name, True):
                    print(f"    {tag} ", end="", flush=True)
                    sa_needs_tag[name] = False
                print(event.event.text, end="", flush=True)
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
                print(
                    f"\n    {tag} {c('cyan', '->')} {event.event.tool_name}"
                    f"{c('dim', '(' + str(event.event.arguments)[:60] + ')')}",
                    flush=True,
                )
                sa_needs_tag[name] = True
            elif isinstance(event.event, ToolResult):
                result_tag = c("green", "OK") if not event.event.is_error else c("red", "ERR")
                output = event.event.output[:120] + "..." if len(event.event.output) > 120 else event.event.output
                print(f"\n    {tag} {result_tag} {c('dim', output)}", flush=True)
                sa_needs_tag[name] = True

        # ── 子代理完成 ──
        elif isinstance(event, SubAgentDone):
            if not show_subagent:
                continue
            name = event.subagent_name
            tag = c("magenta", f"[{name}]")
            status = c("red", " [ERROR]") if event.is_error else ""
            if not sa_needs_tag.get(name, True):
                print()
            print(
                f"  {c('magenta', '╰─')} {tag} {c('magenta', t('完成'))} "
                f"{c('dim', f'turns={event.turn_count}, tokens={event.total_usage.total_tokens}')}"
                f"{status} {c('magenta', '─' * 28)}",
                flush=True,
            )

        # ── 工具结果 ──
        elif isinstance(event, ToolResult):
            tag = c("red", "ERR") if event.is_error else c("green", "OK ")
            output = event.output[:250] + "..." if len(event.output) > 250 else event.output
            print(f"  {tag} {c('dim', output)}", flush=True)

        # ── 完成 ──
        elif isinstance(event, AgentDone):
            if thinking_visible:
                print()
            if show_meta:
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
                + t("对话历史已压缩: {a} → {b} 条消息", a=event.original_count, b=event.compacted_count)
                + f" ({c('dim', event.strategy)})",
                flush=True,
            )

    return assistant_text
