"""CLI 入口：命令行参数解析、子命令分发、main()。

子命令：
    chat                 交互式多轮对话（无子命令时的默认行为）
    run [PROMPT]         一次性执行（PROMPT 省略时从 stdin 读，支持管道）
    config ...           查看/修改配置文件
    sessions [list|show] 查看历史会话
    providers            列出支持的厂商及 Key 配置状态
    version              显示版本
    schedule             定时任务管理（list/run/delete/enable/disable）
    scheduler            调度守护进程（start）

全局选项（厂商/模型/模式等）写在子命令之后，例如：
    milu chat -p deepseek -m deepseek-chat --mode superwork
    milu run "你好" -q
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from milu._env import ensure_dotenv_loaded
from milu.agent.events import AgentDone
from milu.exceptions import MiluError
from milu.llm.base.exceptions import AuthenticationError
from milu.llm.providers import ModelRegistry

from milu.config import load_config, set_user_value, write_project_template
from milu.cli.config import (
    DEFAULT_MODELS,
    DEFAULT_PROVIDER,
    env_key_name,
    resolve_settings,
)
from milu.cli.render import DIVIDER, c, render_turn


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
        prog="milu",
        description="milu 命令行：启动 Agent、一次性执行或多轮对话。",
    )
    # 无子命令时（裸 milu → chat）的默认值兜底
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
    cfg_sub.add_parser("show", help="打印合并后的生效配置及各文件路径")
    cfg_sub.add_parser("path", help="打印项目级 / 用户级配置文件路径")
    cfg_sub.add_parser("init", help="在项目 config/milu.json 生成全量默认配置模板")
    g = cfg_sub.add_parser("get", help="读取配置项（点号路径，如 agent.max_turns）")
    g.add_argument("key")
    st = cfg_sub.add_parser("set", help="设置配置项到用户级配置（点号路径）")
    st.add_argument("key")
    st.add_argument("value")

    # sessions
    p_sess = sub.add_parser("sessions", help="查看历史会话")
    sess_sub = p_sess.add_subparsers(dest="sessions_action")
    sess_sub.add_parser("list", help="列出全部会话（默认）")
    ss = sess_sub.add_parser("show", help="打印某会话的消息")
    ss.add_argument("session_id")

    # providers / version
    sub.add_parser("providers", help="列出支持的厂商及 Key 配置状态")
    sub.add_parser("version", help="显示版本")

    # schedule — 定时任务管理
    p_sch = sub.add_parser("schedule", help="定时任务管理（list/run/delete/enable/disable）")
    sch_sub = p_sch.add_subparsers(dest="schedule_action")
    sch_sub.add_parser("list", help="列出所有定时任务（默认）")
    sr = sch_sub.add_parser("run", help="立即同步执行指定任务")
    sr.add_argument("name", help="任务名称")
    sd = sch_sub.add_parser("delete", help="删除指定任务")
    sd.add_argument("name", help="任务名称")
    se = sch_sub.add_parser("enable", help="启用指定任务")
    se.add_argument("name", help="任务名称")
    sdi = sch_sub.add_parser("disable", help="禁用指定任务")
    sdi.add_argument("name", help="任务名称")

    # scheduler — 调度守护进程
    p_sched = sub.add_parser("scheduler", help="调度守护进程管理（start）")
    sched_sub = p_sched.add_subparsers(dest="scheduler_action")
    sched_sub.add_parser("start", help="启动调度守护进程（前台运行，Ctrl+C 停止）")

    return parser


# ── 一次性执行 ────────────────────────────────────────────

async def _run_once(settings, prompt: str, quiet: bool) -> int:
    from milu.cli.builder import build_agent

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
    from milu.cli.builder import build_agent
    from milu.cli.repl import run_chat

    config = load_config()
    settings = resolve_settings(config, args)
    agent = build_agent(settings)
    asyncio.run(run_chat(agent, settings))
    return 0


def _cmd_run(args) -> int:
    config = load_config()
    settings = resolve_settings(config, args)
    prompt = args.prompt
    if not prompt:
        # 从 stdin 读取（支持管道：echo "..." | milu run）。
        # 按 UTF-8 解码二进制流，避免 Windows 控制台代码页 + surrogateescape 把
        # 中文等多字节字符解成无法再编码的孤立代理字符。
        if not sys.stdin.isatty():
            prompt = sys.stdin.buffer.read().decode("utf-8", errors="replace").strip()
        if not prompt:
            print(c("red", "错误：未提供指令。用法：milu run \"你的问题\""), file=sys.stderr)
            return 2
    return asyncio.run(_run_once(settings, prompt, args.quiet))


def _cmd_config(args) -> int:
    from milu.resources import project_config_path, user_config_path

    action = getattr(args, "config_action", None)

    if action == "path":
        print(f"项目级: {project_config_path()}")
        print(f"用户级: {user_config_path()}")
        return 0
    if action == "init":
        p = write_project_template()
        print(c("green", f"已生成项目配置模板 → {p}"))
        return 0
    if action == "get":
        try:
            print(load_config().get(args.key))
        except KeyError:
            print(c("red", f"未知配置项：{args.key}"), file=sys.stderr)
            return 2
        return 0
    if action == "set":
        try:
            p, value = set_user_value(args.key, args.value)
        except ValueError as e:
            print(c("red", f"错误：{e}"), file=sys.stderr)
            return 2
        print(c("green", f"已设置 {args.key} = {value!r}") + c("dim", f"  → {p}"))
        return 0

    # 默认 / show：打印合并后的生效配置
    import json as _json
    cfg = load_config()
    print(f"项目级: {project_config_path()}")
    print(f"用户级: {user_config_path()}")
    print(c("dim", "（生效 = 内置默认 ← 项目级 ← 用户级；CLI 参数运行时再叠加）"))
    print(_json.dumps(cfg.data, ensure_ascii=False, indent=2))
    return 0


def _cmd_sessions(args) -> int:
    from datetime import datetime

    from milu.agent.session import Session
    from milu.resources import default_session_dir

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
    cfg = load_config()
    default_models = cfg.default_models
    default_provider = cfg.llm.get("provider") or DEFAULT_PROVIDER
    providers = ModelRegistry.list_providers()
    print(f"{DIVIDER}\n  支持的厂商 ({len(providers)} 个)\n{DIVIDER}")
    for name in providers:
        env_name = env_key_name(name)
        has_env = bool(os.environ.get(env_name))
        if has_env:
            status = c("green", f"已配置 (env {env_name})")
        else:
            status = c("dim", f"未配置（在 .env 设置 {env_name}）")
        default_model = default_models.get(name, "—")
        mark = c("bold", " *默认") if name == default_provider else ""
        print(f"  {c('cyan', name):<22} 默认模型 {c('dim', default_model):<32} {status}{mark}")
    return 0


def _cmd_version(_args) -> int:
    try:
        from importlib.metadata import version
        print(f"milu {version('milu')}")
    except Exception:
        print("milu (版本未知)")
    return 0


def _cmd_schedule(args) -> int:
    """定时任务 CRUD 管理（list/run/delete/enable/disable）。"""
    from milu.resources import user_data_dir
    from milu.scheduler.store import ScheduleStore

    action = getattr(args, "schedule_action", None) or "list"
    store = ScheduleStore(user_data_dir() / "schedules.json")

    if action == "list":
        tasks = store.list_all()
        if not tasks:
            print(c("dim", "暂无定时任务。可在对话中让 Agent 调用 schedule_create 工具创建。"))
            return 0
        print(f"{DIVIDER}\n  定时任务 ({len(tasks)} 个)\n{DIVIDER}")
        for t in tasks:
            status_color = "green" if t.enabled else "dim"
            status_str = "启用" if t.enabled else "禁用"
            last = t.last_run[:16].replace("T", " ") if t.last_run else "从未"
            nxt = t.next_run[:16].replace("T", " ") if t.next_run else "待计算"
            print(
                f"  {c('cyan', t.name)}  {c(status_color, f'[{status_str}]')}  "
                f"运行 {t.run_count} 次\n"
                f"    触发: {c('yellow', t.trigger_desc())}\n"
                f"    说明: {t.description or t.prompt[:60]}\n"
                f"    上次: {c('dim', last)}  下次: {c('dim', nxt)}"
            )
        print(DIVIDER)
        return 0

    if action == "run":
        from milu._env import ensure_dotenv_loaded
        from milu.scheduler.engine import ScheduleEngine

        ensure_dotenv_loaded()
        task = store.get(args.name)
        if not task:
            print(c("red", f"错误：任务 '{args.name}' 不存在"), file=sys.stderr)
            return 1
        print(c("cyan", f"  正在执行任务 '{args.name}'..."))
        engine = ScheduleEngine(store)
        try:
            result = asyncio.run(engine.run_task_now(args.name))
            print(f"\n{DIVIDER}")
            print(c("green", f"  任务 '{args.name}' 执行完成"))
            print(DIVIDER)
            print(result)
            print(DIVIDER)
        except Exception as e:
            print(c("red", f"  执行失败: {e}"), file=sys.stderr)
            return 1
        return 0

    if action == "delete":
        if not store.remove(args.name):
            print(c("red", f"错误：任务 '{args.name}' 不存在"), file=sys.stderr)
            return 1
        print(c("green", f"  任务 '{args.name}' 已删除"))
        return 0

    if action in ("enable", "disable"):
        task = store.get(args.name)
        if not task:
            print(c("red", f"错误：任务 '{args.name}' 不存在"), file=sys.stderr)
            return 1
        task.enabled = action == "enable"
        store.update(task)
        verb = "已启用" if task.enabled else "已禁用"
        print(c("green", f"  任务 '{args.name}' {verb}"))
        return 0

    # 默认 list
    return _cmd_schedule_list(store)


def _cmd_schedule_list(store) -> int:
    tasks = store.list_all()
    if not tasks:
        print(c("dim", "暂无定时任务。"))
    return 0


def _cmd_scheduler(args) -> int:
    """调度守护进程（start）。"""
    from milu._env import ensure_dotenv_loaded
    from milu.resources import user_data_dir
    from milu.scheduler.engine import ScheduleEngine
    from milu.scheduler.store import ScheduleStore

    action = getattr(args, "scheduler_action", None) or "start"

    if action == "start":
        ensure_dotenv_loaded()
        data_dir = user_data_dir()
        store = ScheduleStore(data_dir / "schedules.json")
        log_dir = data_dir / "scheduler_logs"
        engine = ScheduleEngine(store, log_dir=log_dir)

        tasks = store.list_all()
        enabled = [t for t in tasks if t.enabled]
        print(f"{DIVIDER}")
        print(c("bold", c("cyan", "  milu 调度守护进程")))
        print(DIVIDER)
        print(f"  任务文件: {c('dim', str(data_dir / 'schedules.json'))}")
        print(f"  日志目录: {c('dim', str(log_dir))}")
        print(f"  已启用任务: {c('yellow', str(len(enabled)))} / {len(tasks)} 个")
        if enabled:
            for t in enabled:
                print(f"    {c('cyan', t.name):<20} {t.trigger_desc()}")
        print(DIVIDER + "\n")

        try:
            asyncio.run(engine.start())
        except KeyboardInterrupt:
            print(c("dim", "\n  调度器已停止。"))
        return 0

    print(c("red", f"未知操作: {action}（可用: start）"), file=sys.stderr)
    return 2


# ── main ─────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    # Windows 下启用 ANSI 颜色
    if sys.platform == "win32":
        os.system("")
    # 进程内加载一次 .env（可被 MILU_NO_DOTENV 关闭）
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
        "schedule": _cmd_schedule,
        "scheduler": _cmd_scheduler,
    }
    handler = handlers.get(command)
    if handler is None:
        print(c("red", f"未知子命令: {command}"), file=sys.stderr)
        return 2

    try:
        return handler(args) or 0  # type: ignore[misc]
    except KeyboardInterrupt:
        print(c("dim", "\n已中断。"))
        return 130
    except AuthenticationError as e:
        provider = (getattr(args, "provider", None)
                    or load_config().llm.get("provider") or DEFAULT_PROVIDER)
        print(c("red", f"\n鉴权失败：{e}"), file=sys.stderr)
        print(c("dim", f"请在 .env 或环境变量中设置 {env_key_name(provider)}。"),
              file=sys.stderr)
        return 1
    except ValueError as e:
        print(c("red", f"\n错误：{e}"), file=sys.stderr)
        return 2
    except MiluError as e:
        print(c("red", f"\n运行失败：{e}"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
