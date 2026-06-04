"""CLI 入口：命令行参数解析、子命令分发、main()。

子命令：
    chat                 交互式多轮对话（无子命令时的默认行为）
    run [PROMPT]         一次性执行（PROMPT 省略时从 stdin 读，支持管道）
    config ...           查看/修改配置文件
    sessions [list|show] 查看历史会话
    providers            列出支持的厂商及 Key 配置状态
    version              显示版本

全局选项（厂商/模型/模式等）写在子命令之后，例如：
    agent-framework chat -p deepseek -m deepseek-chat --mode superwork
    agent-framework run "你好" -q
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from agent_framework._env import ensure_dotenv_loaded
from agent_framework.agent.events import AgentDone
from agent_framework.exceptions import AgentFrameworkError
from agent_framework.llm.base.exceptions import AuthenticationError
from agent_framework.llm.providers import ModelRegistry

from agent_framework.cli.config import (
    DEFAULT_MODELS,
    DEFAULT_PROVIDER,
    SCALAR_FIELDS,
    CLIConfig,
    coerce_scalar,
    env_key_name,
    resolve_settings,
)
from agent_framework.cli.render import DIVIDER, c, render_turn


# ── 参数解析 ──────────────────────────────────────────────

def _add_common_options(p: argparse.ArgumentParser) -> None:
    """给 chat / run 子命令添加共享的厂商/模型/模式选项。"""
    p.add_argument("-p", "--provider", help=f"厂商名（默认 {DEFAULT_PROVIDER}，或配置文件）")
    p.add_argument("-m", "--model", help="模型名（默认按厂商内置，或配置文件）")
    p.add_argument("--api-key", help="临时指定 API Key（优先级最高）")
    p.add_argument("--mode", choices=["talk", "manual", "auto", "superwork"], help="操作模式")
    p.add_argument("--no-session", action="store_true", help="禁用会话持久化")
    p.add_argument("--no-mcp", action="store_true", help="启动时不连接 MCP 服务器")
    p.add_argument("--no-subagents", action="store_true", help="不挂载内置子代理")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-framework",
        description="agent-framework 命令行：启动 Agent、一次性执行或多轮对话。",
    )
    # 无子命令时（裸 agent-framework → chat）的默认值兜底
    parser.set_defaults(
        provider=None, model=None, api_key=None, mode=None,
        no_session=False, no_mcp=False, no_subagents=False,
    )
    sub = parser.add_subparsers(dest="command")

    # chat
    p_chat = sub.add_parser("chat", help="交互式多轮对话")
    _add_common_options(p_chat)
    p_chat.add_argument("--session", help="续接指定会话 ID")

    # run
    p_run = sub.add_parser("run", help="一次性执行单条指令")
    _add_common_options(p_run)
    p_run.add_argument("prompt", nargs="?", help="指令文本（省略则从 stdin 读取）")
    p_run.add_argument("-q", "--quiet", action="store_true", help="只输出最终回答（适合管道）")
    p_run.add_argument("--session", help="续接指定会话 ID")

    # config
    p_cfg = sub.add_parser("config", help="查看/修改配置")
    cfg_sub = p_cfg.add_subparsers(dest="config_action")
    cfg_sub.add_parser("show", help="打印当前配置")
    cfg_sub.add_parser("path", help="打印配置文件路径")
    g = cfg_sub.add_parser("get", help="读取某个配置项")
    g.add_argument("key", choices=sorted(SCALAR_FIELDS))
    st = cfg_sub.add_parser("set", help="设置某个配置项")
    st.add_argument("key", choices=sorted(SCALAR_FIELDS))
    st.add_argument("value")
    sk = cfg_sub.add_parser("set-key", help="保存某厂商的 API Key 到配置文件")
    sk.add_argument("provider")
    sk.add_argument("api_key")

    # sessions
    p_sess = sub.add_parser("sessions", help="查看历史会话")
    sess_sub = p_sess.add_subparsers(dest="sessions_action")
    sess_sub.add_parser("list", help="列出全部会话（默认）")
    ss = sess_sub.add_parser("show", help="打印某会话的消息")
    ss.add_argument("session_id")

    # providers / version
    sub.add_parser("providers", help="列出支持的厂商及 Key 配置状态")
    sub.add_parser("version", help="显示版本")

    return parser


# ── 一次性执行 ────────────────────────────────────────────

async def _run_once(settings, prompt: str, quiet: bool) -> int:
    from agent_framework.cli.builder import build_agent

    agent = build_agent(settings)
    if settings.use_mcp:
        await agent.connect_mcp()
    if settings.session_id and agent.session:
        try:
            agent.load_session(settings.session_id)
        except FileNotFoundError:
            pass
    try:
        if quiet:
            final = ""
            async for ev in agent.run(prompt):
                if isinstance(ev, AgentDone):
                    final = ev.final_text
            print(final)
        else:
            await render_turn(agent, prompt)
    finally:
        agent.save_session()
        await agent.disconnect_mcp()
    return 0


# ── 子命令处理 ────────────────────────────────────────────

def _cmd_chat(args) -> int:
    from agent_framework.cli.builder import build_agent
    from agent_framework.cli.repl import run_chat

    config = CLIConfig.load()
    settings = resolve_settings(config, args)
    agent = build_agent(settings)
    asyncio.run(run_chat(agent, settings))
    return 0


def _cmd_run(args) -> int:
    config = CLIConfig.load()
    settings = resolve_settings(config, args)
    prompt = args.prompt
    if not prompt:
        # 从 stdin 读取（支持管道：echo "..." | agent-framework run）。
        # 按 UTF-8 解码二进制流，避免 Windows 控制台代码页 + surrogateescape 把
        # 中文等多字节字符解成无法再编码的孤立代理字符。
        if not sys.stdin.isatty():
            prompt = sys.stdin.buffer.read().decode("utf-8", errors="replace").strip()
        if not prompt:
            print(c("red", "错误：未提供指令。用法：agent-framework run \"你的问题\""), file=sys.stderr)
            return 2
    return asyncio.run(_run_once(settings, prompt, args.quiet))


def _cmd_config(args) -> int:
    action = getattr(args, "config_action", None)
    config = CLIConfig.load()

    if action == "path":
        print(CLIConfig.path())
        return 0
    if action == "get":
        print(getattr(config, args.key))
        return 0
    if action == "set":
        try:
            value = coerce_scalar(args.key, args.value)
        except ValueError as e:
            print(c("red", f"错误：{e}"), file=sys.stderr)
            return 2
        setattr(config, args.key, value)
        p = config.save()
        print(c("green", f"已设置 {args.key} = {value!r}") + c("dim", f"  → {p}"))
        return 0
    if action == "set-key":
        config.api_keys[args.provider] = args.api_key
        p = config.save()
        print(c("green", f"已保存 {args.provider} 的 API Key") + c("dim", f"  → {p}"))
        return 0

    # 默认 / show
    import json as _json
    from dataclasses import asdict
    data = asdict(config)
    # 脱敏 api_keys
    data["api_keys"] = {k: (v[:6] + "***") if v else v for k, v in data["api_keys"].items()}
    print(f"配置文件: {CLIConfig.path()}")
    print(_json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def _cmd_sessions(args) -> int:
    from datetime import datetime

    from agent_framework.agent.session import Session
    from agent_framework.resources import default_session_dir

    action = getattr(args, "sessions_action", None) or "list"
    base = default_session_dir()

    if action == "show":
        try:
            sess = Session.load_session(args.session_id, base)
        except FileNotFoundError:
            print(c("red", f"会话不存在: {args.session_id}"), file=sys.stderr)
            return 1
        msgs = sess.load_messages()
        print(f"{DIVIDER}\n  会话 {c('cyan', args.session_id)}（{len(msgs)} 条消息）\n{DIVIDER}")
        for m in msgs:
            role = m.role.value.upper()
            content = (m.content or "").replace("\n", " ")
            content = content[:500] + "..." if len(content) > 500 else content
            print(f"  {c('cyan', f'[{role:<9}]')} {c('dim', content)}")
        return 0

    sessions = Session.list_sessions(base)
    if not sessions:
        print(c("dim", f"暂无历史会话（{base}）"))
        return 0
    print(f"{DIVIDER}\n  历史会话 ({len(sessions)} 个)  {c('dim', str(base))}\n{DIVIDER}")
    for s in sessions:
        updated = s.get("updated_at", 0)
        time_str = datetime.fromtimestamp(updated).strftime("%Y-%m-%d %H:%M") if updated else "?"
        model_str = c("dim", f" ({s.get('model')})") if s.get("model") else ""
        print(f"  {c('cyan', s.get('session_id', '?'))}  {c('dim', time_str)}  "
              f"{s.get('message_count', 0)} 条消息{model_str}")
    return 0


def _cmd_providers(_args) -> int:
    ensure_dotenv_loaded()
    config = CLIConfig.load()
    providers = ModelRegistry.list_providers()
    print(f"{DIVIDER}\n  支持的厂商 ({len(providers)} 个)\n{DIVIDER}")
    for name in providers:
        env_name = env_key_name(name)
        has_env = bool(os.environ.get(env_name))
        has_cfg = bool(config.api_keys.get(name))
        if has_env:
            status = c("green", f"已配置 (env {env_name})")
        elif has_cfg:
            status = c("green", "已配置 (config)")
        else:
            status = c("dim", f"未配置（设 {env_name} 或 config set-key {name} <key>）")
        default_model = DEFAULT_MODELS.get(name, "—")
        mark = c("bold", " *默认") if name == DEFAULT_PROVIDER else ""
        print(f"  {c('cyan', name):<22} 默认模型 {c('dim', default_model):<32} {status}{mark}")
    return 0


def _cmd_version(_args) -> int:
    try:
        from importlib.metadata import version
        print(f"agent-framework {version('agent-framework')}")
    except Exception:
        print("agent-framework (版本未知)")
    return 0


# ── main ─────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    # Windows 下启用 ANSI 颜色
    if sys.platform == "win32":
        os.system("")
    # 进程内加载一次 .env（可被 AGENT_FRAMEWORK_NO_DOTENV 关闭）
    ensure_dotenv_loaded()

    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "chat"  # 无子命令 → 进入交互对话

    handlers = {
        "chat": _cmd_chat,
        "run": _cmd_run,
        "config": _cmd_config,
        "sessions": _cmd_sessions,
        "providers": _cmd_providers,
        "version": _cmd_version,
    }
    handler = handlers[command]

    try:
        return handler(args) or 0
    except KeyboardInterrupt:
        print(c("dim", "\n已中断。"))
        return 130
    except AuthenticationError as e:
        provider = getattr(args, "provider", None) or CLIConfig.load().provider or DEFAULT_PROVIDER
        print(c("red", f"\n鉴权失败：{e}"), file=sys.stderr)
        print(c("dim", f"请设置环境变量 {env_key_name(provider)}，"
                       f"或运行 `agent-framework config set-key {provider} <你的key>`。"),
              file=sys.stderr)
        return 1
    except ValueError as e:
        print(c("red", f"\n错误：{e}"), file=sys.stderr)
        return 2
    except AgentFrameworkError as e:
        print(c("red", f"\n运行失败：{e}"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
