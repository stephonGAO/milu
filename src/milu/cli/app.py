"""CLI 入口：命令行参数解析、子命令分发、main()。

子命令：
    chat                 交互式多轮对话（无子命令时的默认行为）
    run [PROMPT]         一次性执行（PROMPT 省略时从 stdin 读，支持管道）
    setup                初始化引导（选厂商/模型、配置 API Key 与搜索工具）
    config ...           查看/修改配置文件
    sessions [list|show] 查看历史会话
    providers            列出支持的厂商及 Key 配置状态
    version              显示版本
    schedule             定时任务管理（list/run/delete/enable/disable）
    scheduler            调度守护进程（start）
    serve                启动内置 Web 服务（多用户对话 + 全功能演示前端）
    gateway              多渠道接入网关（微信客服/飞书/Telegram → milu Agent）
    trace                运行追踪查看（list/show/compare/stats，可观测性）

全局选项（厂商/模型/模式/语言等）写在子命令之后，例如：
    milu chat -p deepseek -m deepseek-chat --mode superwork
    milu run "你好" -q
    milu --lang en providers        # 英文输出（亦可 MILU_LANG=en 或 config set lang en）
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from milu._env import ensure_dotenv_loaded
from milu.agent.events import AgentDone
from milu.exceptions import MiluError
from milu.i18n import set_lang, t
from milu.llm.base.exceptions import AuthenticationError
from milu.llm.providers import ModelRegistry

from milu.config import load_config, set_user_value, write_project_template
from milu.cli.config import (
    DEFAULT_PROVIDER,
    env_key_name,
    resolve_settings,
)
from milu.cli.render import DIVIDER, c, render_turn


# ── 参数解析 ──────────────────────────────────────────────

def _add_common_options(p: argparse.ArgumentParser) -> None:
    """给 chat / run 子命令添加共享的厂商/模型/模式选项。"""
    p.add_argument("-p", "--provider", help=t("厂商名（默认 {p}，或配置文件）", p=DEFAULT_PROVIDER))
    p.add_argument("-m", "--model", help=t("模型名（默认按厂商内置，或配置文件）"))
    p.add_argument("--api-key", help=t("临时指定 API Key（优先级最高）"))
    p.add_argument("--mode", choices=["talk", "manual", "auto", "superwork"], help=t("操作模式"))
    p.add_argument("--lang", choices=["zh", "en"], help=t("选择语言（zh/en，默认 zh）"))
    p.add_argument("--no-session", action="store_true", help=t("禁用会话持久化"))
    p.add_argument("--no-mcp", action="store_true", help=t("启动时不连接 MCP 服务器"))
    p.add_argument("--no-subagents", action="store_true", help=t("不挂载内置子代理"))
    p.add_argument("--no-scheduler", action="store_true",
                   help=t("对话期间不嵌入定时任务调度引擎（仅 chat 生效）"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="milu",
        description=t("milu 命令行：启动 Agent、一次性执行或多轮对话。"),
    )
    # 无子命令时（裸 milu → chat）的默认值兜底
    parser.set_defaults(
        provider=None, model=None, api_key=None, mode=None, lang=None,
        no_session=False, no_mcp=False, no_subagents=False, no_scheduler=False,
    )
    parser.add_argument("--lang", choices=["zh", "en"], help=t("选择语言（zh/en，默认 zh）"))
    sub = parser.add_subparsers(dest="command")

    # chat
    p_chat = sub.add_parser("chat", help=t("交互式多轮对话"))
    _add_common_options(p_chat)
    p_chat.add_argument("--session", help=t("续接指定会话 ID"))

    # run
    p_run = sub.add_parser("run", help=t("一次性执行单条指令"))
    _add_common_options(p_run)
    p_run.add_argument("prompt", nargs="?", help=t("指令文本（省略则从 stdin 读取）"))
    p_run.add_argument("-q", "--quiet", action="store_true", help=t("只输出最终回答（适合管道）"))
    p_run.add_argument("--session", help=t("续接指定会话 ID"))

    # setup — 初始化引导
    sub.add_parser("setup", help=t("初始化引导：选厂商/模型、配置 API Key 与搜索工具"))

    # config
    p_cfg = sub.add_parser("config", help=t("查看/修改配置"))
    cfg_sub = p_cfg.add_subparsers(dest="config_action")
    cfg_sub.add_parser("show", help=t("打印合并后的生效配置及各文件路径"))
    cfg_sub.add_parser("path", help=t("打印项目级 / 用户级配置文件路径"))
    cfg_sub.add_parser("init", help=t("在项目 config/milu.json 生成全量默认配置模板"))
    g = cfg_sub.add_parser("get", help=t("读取配置项（点号路径，如 agent.max_turns）"))
    g.add_argument("key")
    st = cfg_sub.add_parser("set", help=t("设置配置项到用户级配置（点号路径）"))
    st.add_argument("key")
    st.add_argument("value")

    # sessions
    p_sess = sub.add_parser("sessions", help=t("查看历史会话"))
    sess_sub = p_sess.add_subparsers(dest="sessions_action")
    sess_sub.add_parser("list", help=t("列出全部会话（默认）"))
    ss = sess_sub.add_parser("show", help=t("打印某会话的消息"))
    ss.add_argument("session_id")

    # providers / version
    sub.add_parser("providers", help=t("列出支持的厂商及 Key 配置状态"))
    sub.add_parser("version", help=t("显示版本"))

    # schedule — 定时任务管理（多用户：--user 指定用户，默认 default）
    p_sch = sub.add_parser("schedule", help=t("定时任务管理（list/run/delete/enable/disable）"))
    sch_sub = p_sch.add_subparsers(dest="schedule_action")
    sl = sch_sub.add_parser("list", help=t("列出定时任务（默认）"))
    sl.add_argument("--user", default="default", help=t("用户标识（默认 default）"))
    sl.add_argument("--all", action="store_true", help=t("列出全部用户的任务"))
    sr = sch_sub.add_parser("run", help=t("立即同步执行指定任务"))
    sr.add_argument("name", help=t("任务名称"))
    sr.add_argument("--user", default="default", help=t("用户标识（默认 default）"))
    sd = sch_sub.add_parser("delete", help=t("删除指定任务"))
    sd.add_argument("name", help=t("任务名称"))
    sd.add_argument("--user", default="default", help=t("用户标识（默认 default）"))
    se = sch_sub.add_parser("enable", help=t("启用指定任务"))
    se.add_argument("name", help=t("任务名称"))
    se.add_argument("--user", default="default", help=t("用户标识（默认 default）"))
    sdi = sch_sub.add_parser("disable", help=t("禁用指定任务"))
    sdi.add_argument("name", help=t("任务名称"))
    sdi.add_argument("--user", default="default", help=t("用户标识（默认 default）"))

    # trace — 运行追踪查看（可观测性）
    p_trace = sub.add_parser("trace", help=t("运行追踪查看（list/show/compare/stats）"))
    trace_sub = p_trace.add_subparsers(dest="trace_action")
    tl = trace_sub.add_parser("list", help=t("列出近期运行（默认）"))
    tl.add_argument("-n", type=int, default=20, help=t("条数（默认 20）"))
    tl.add_argument("--model", help=t("按模型名过滤"))
    ts = trace_sub.add_parser("show", help=t("树状展示一次运行的完整 span 树"))
    ts.add_argument("trace_id", help=t("trace_id（支持前缀匹配）"))
    tc = trace_sub.add_parser("compare", help=t("多次运行对比（耗时/token/成本）"))
    tc.add_argument("trace_ids", nargs="+", help=t("两个及以上 trace_id"))
    tst = trace_sub.add_parser("stats", help=t("聚合统计（p50/p95 耗时、token、成本）"))
    tst.add_argument("--by", choices=["model", "provider", "day"], default="model",
                     help=t("分组维度（默认 model）"))

    # scheduler — 调度守护进程
    p_sched = sub.add_parser("scheduler", help=t("调度守护进程管理（start）"))
    sched_sub = p_sched.add_subparsers(dest="scheduler_action")
    sched_sub.add_parser("start", help=t("启动调度守护进程（前台运行，Ctrl+C 停止）"))

    # serve — 内置 Web 服务（FastAPI + 全功能演示前端）
    p_serve = sub.add_parser("serve", help=t("启动内置 Web 服务（多用户对话 + 演示前端）"))
    p_serve.add_argument("--host", default="127.0.0.1", help=t("监听地址（默认 127.0.0.1）"))
    p_serve.add_argument("--port", type=int, default=8000, help=t("监听端口（默认 8000）"))
    p_serve.add_argument("-p", "--provider", help=t("默认厂商（默认按配置）"))
    p_serve.add_argument("-m", "--model", help=t("默认模型（默认按厂商内置）"))
    p_serve.add_argument("--api-key", help=t("临时 API Key（一般用 .env 配置）"))
    p_serve.add_argument("--mode", choices=["talk", "manual", "auto", "superwork"],
                         help=t("默认操作模式"))
    p_serve.add_argument("--lang", choices=["zh", "en"], help=t("选择语言（zh/en，默认 zh）"))
    p_serve.add_argument("--no-mcp", action="store_true", help=t("不连接 MCP 服务器"))
    p_serve.add_argument("--no-subagents", action="store_true", help=t("不挂载内置子代理"))
    p_serve.add_argument("--no-session", action="store_true", help=t("禁用会话持久化"))
    p_serve.add_argument("--no-scheduler", action="store_true", help=t("不嵌入定时任务调度引擎"))
    p_serve.add_argument("--reload", action="store_true", help=t("开发模式：代码变更自动重载"))

    # ── gateway：多渠道接入网关（微信客服 / 飞书 / Telegram）──
    p_gw = sub.add_parser(
        "gateway", help=t("启动多渠道接入网关（微信客服/飞书/Telegram → milu Agent）"))
    p_gw.add_argument("--host", default="0.0.0.0", help=t("监听地址（默认 0.0.0.0）"))
    p_gw.add_argument("--port", type=int, default=8800, help=t("监听端口（默认 8800）"))
    p_gw.add_argument("-p", "--provider", help=t("默认厂商（默认按配置）"))
    p_gw.add_argument("-m", "--model", help=t("默认模型（默认按厂商内置）"))
    p_gw.add_argument("--api-key", help=t("临时 API Key（一般用 .env 配置）"))
    p_gw.add_argument("--mode", choices=["talk", "manual", "auto", "superwork"],
                      help=t("默认操作模式（默认 auto）"))
    p_gw.add_argument("--lang", choices=["zh", "en"], help=t("选择语言（zh/en，默认 zh）"))
    p_gw.add_argument("--channel", help=t(
        "启用的渠道（逗号分隔：wechat_kf,feishu,telegram；缺省按已配置的凭证自动探测）"))
    p_gw.add_argument("--no-subagents", action="store_true", help=t("不挂载内置子代理"))
    p_gw.add_argument("--no-session", action="store_true", help=t("禁用会话持久化"))
    p_gw.add_argument("--no-persist", action="store_true",
                      help=t("去重/游标不落盘（用内存版，重启即丢）"))

    return parser


# ── 一次性执行 ────────────────────────────────────────────

async def _run_once(settings, prompt: str, quiet: bool) -> int:
    from milu.cli.builder import build_agent
    from milu.tools.mcp.connection import suppress_mcp_asyncgen_errors

    suppress_mcp_asyncgen_errors()
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
            await render_turn(agent, prompt, show_subagent=settings.show_subagent_events)
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

    # 首次使用引导：交互终端下检测不到该厂商的 API Key → 询问进入初始化引导
    if not settings.api_key and sys.stdin.isatty():
        from milu.cli.setup_wizard import offer_first_run_setup

        if offer_first_run_setup(settings.provider):
            # 引导可能更换了厂商/模型并写入了 Key → 重新加载配置解析
            config = load_config()
            settings = resolve_settings(config, args)

    agent = build_agent(settings)
    asyncio.run(run_chat(agent, settings))
    return 0


def _cmd_setup(_args) -> int:
    from milu.cli.setup_wizard import run_setup_wizard

    return run_setup_wizard()


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
            print(c("red", t("错误：未提供指令。用法：milu run \"你的问题\"")), file=sys.stderr)
            return 2
    return asyncio.run(_run_once(settings, prompt, args.quiet))


def _cmd_config(args) -> int:
    from milu.resources import project_config_path, user_config_path

    action = getattr(args, "config_action", None)

    if action == "path":
        print(t("项目级: {p}", p=project_config_path()))
        print(t("用户级: {p}", p=user_config_path()))
        return 0
    if action == "init":
        p = write_project_template()
        print(c("green", t("已生成项目配置模板 → {p}", p=p)))
        return 0
    if action == "get":
        try:
            print(load_config().get(args.key))
        except KeyError:
            print(c("red", t("未知配置项：{k}", k=args.key)), file=sys.stderr)
            return 2
        return 0
    if action == "set":
        try:
            p, value = set_user_value(args.key, args.value)
        except ValueError as e:
            print(c("red", t("错误：{e}", e=e)), file=sys.stderr)
            return 2
        print(c("green", t("已设置 {k} = {v}", k=args.key, v=repr(value))) + c("dim", f"  → {p}"))
        return 0

    # 默认 / show：打印合并后的生效配置
    import json as _json
    cfg = load_config()
    print(t("项目级: {p}", p=project_config_path()))
    print(t("用户级: {p}", p=user_config_path()))
    print(c("dim", t("（生效 = 内置默认 ← 项目级 ← 用户级；CLI 参数运行时再叠加）")))
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
            print(c("red", t("会话不存在: {id}", id=args.session_id)), file=sys.stderr)
            return 1
        msgs = sess.load_messages()
        print(f"{DIVIDER}\n  "
              + t("会话 {id}（{n} 条消息）", id=c('cyan', args.session_id), n=len(msgs))
              + f"\n{DIVIDER}")
        for m in msgs:
            role = m.role.value.upper()
            content = (m.content or "").replace("\n", " ")
            content = content[:500] + "..." if len(content) > 500 else content
            print(f"  {c('cyan', f'[{role:<9}]')} {c('dim', content)}")
        return 0

    sessions = Session.list_sessions(base)
    if not sessions:
        print(c("dim", t("暂无历史会话（{base}）", base=base)))
        return 0
    print(f"{DIVIDER}\n  " + t("历史会话 ({n} 个)", n=len(sessions)) + f"  {c('dim', str(base))}\n{DIVIDER}")
    for s in sessions:
        updated = s.get("updated_at", 0)
        time_str = datetime.fromtimestamp(updated).strftime("%Y-%m-%d %H:%M") if updated else "?"
        model_str = c("dim", f" ({s.get('model')})") if s.get("model") else ""
        print(f"  {c('cyan', s.get('session_id', '?'))}  {c('dim', time_str)}  "
              + t("{n} 条消息", n=s.get('message_count', 0)) + model_str)
    return 0


def _cmd_trace(args) -> int:
    from milu.cli.trace_cmd import cmd_trace
    return cmd_trace(args)


def _cmd_providers(_args) -> int:
    ensure_dotenv_loaded()
    cfg = load_config()
    default_models = cfg.default_models
    default_provider = cfg.llm.get("provider") or DEFAULT_PROVIDER
    providers = ModelRegistry.list_providers()
    print(f"{DIVIDER}\n  " + t("支持的厂商 ({n} 个)", n=len(providers)) + f"\n{DIVIDER}")
    for name in providers:
        env_name = env_key_name(name)
        has_env = bool(os.environ.get(env_name))
        if has_env:
            status = c("green", t("已配置 (env {env})", env=env_name))
        else:
            status = c("dim", t("未配置（在 .env 设置 {env}）", env=env_name))
        default_model = default_models.get(name, "—")
        mark = c("bold", t(" *默认")) if name == default_provider else ""
        print(f"  {c('cyan', name):<22} {t('默认模型')} {c('dim', default_model):<32} {status}{mark}")
    print(c("dim", t("\n  提示：运行 `milu setup` 可交互式配置厂商、模型与 API Key。")))
    return 0


def _cmd_version(_args) -> int:
    try:
        from importlib.metadata import version
        print(f"milu {version('milu')}")
    except Exception:
        print(t("milu (版本未知)"))
    return 0


def _cmd_schedule(args) -> int:
    """定时任务 CRUD 管理（list/run/delete/enable/disable）。"""
    from milu.resources import user_data_dir
    from milu.scheduler.store import ScheduleStore

    action = getattr(args, "schedule_action", None) or "list"
    user = getattr(args, "user", "default")
    store = ScheduleStore(user_data_dir())

    if action == "list":
        show_all = getattr(args, "all", False)
        tasks = store.list_all() if show_all else store.list_user(user)
        if not tasks:
            print(c("dim", t("暂无定时任务。可在对话中让 Agent 调用 schedule_create 工具创建。")))
            return 0
        print(f"{DIVIDER}\n  " + t("定时任务 ({n} 个)", n=len(tasks)) + f"\n{DIVIDER}")
        for tk in tasks:
            status_color = "green" if tk.enabled else "dim"
            status_str = t("启用") if tk.enabled else t("禁用")
            last = tk.last_run[:16].replace("T", " ") if tk.last_run else t("从未")
            nxt = tk.next_run[:16].replace("T", " ") if tk.next_run else t("待计算")
            user_tag = f"  {c('dim', f'@{tk.user_id}')}" if show_all else ""
            print(
                f"  {c('cyan', tk.name)}{user_tag}  {c(status_color, f'[{status_str}]')}  "
                + t("运行 {n} 次", n=tk.run_count) + "\n"
                f"    {t('触发: ')}{c('yellow', tk.trigger_desc())}\n"
                f"    {t('说明: ')}{tk.description or tk.prompt[:60]}\n"
                f"    {t('上次: ')}{c('dim', last)}  {t('下次: ')}{c('dim', nxt)}"
            )
        print(DIVIDER)
        return 0

    if action == "run":
        from milu._env import ensure_dotenv_loaded
        from milu.scheduler.engine import ScheduleEngine

        ensure_dotenv_loaded()
        task = store.get(args.name, user)
        if not task:
            print(c("red", t("错误：任务 '{name}' 不存在", name=args.name)), file=sys.stderr)
            return 1
        print(c("cyan", t("  正在执行任务 '{name}'...", name=args.name)))
        engine = ScheduleEngine(store)
        try:
            result = asyncio.run(engine.run_task_now(args.name, user))
            print(f"\n{DIVIDER}")
            print(c("green", t("  任务 '{name}' 执行完成", name=args.name)))
            print(DIVIDER)
            print(result)
            print(DIVIDER)
        except Exception as e:
            print(c("red", t("  执行失败: {e}", e=e)), file=sys.stderr)
            return 1
        return 0

    if action == "delete":
        if not store.remove(args.name, user):
            print(c("red", t("错误：任务 '{name}' 不存在", name=args.name)), file=sys.stderr)
            return 1
        print(c("green", t("  任务 '{name}' 已删除", name=args.name)))
        return 0

    if action in ("enable", "disable"):
        task = store.get(args.name, user)
        if not task:
            print(c("red", t("错误：任务 '{name}' 不存在", name=args.name)), file=sys.stderr)
            return 1
        task.enabled = action == "enable"
        store.update(task)
        verb = t("已启用") if task.enabled else t("已禁用")
        print(c("green", t("  任务 '{name}' {verb}", name=args.name, verb=verb)))
        return 0

    print(c("red", t("未知操作: {action}", action=action)), file=sys.stderr)
    return 2


def _cmd_scheduler(args) -> int:
    """调度守护进程（start）。"""
    from milu._env import ensure_dotenv_loaded
    from milu.cli.builder import build_scheduler_engine
    from milu.scheduler.lock import SchedulerLock

    action = getattr(args, "scheduler_action", None) or "start"

    if action == "start":
        ensure_dotenv_loaded()
        engine, store, data_dir = build_scheduler_engine(echo=True)
        log_dir = data_dir / "scheduler_logs"

        # 单实例锁：防止多个调度引擎并存导致任务被重复执行
        lock = SchedulerLock(data_dir)
        if not lock.try_acquire():
            print(c("red", t("调度器已在运行（PID {pid}），同一时间只能有一个引擎。",
                             pid=lock.holder_pid())), file=sys.stderr)
            print(c("dim", t("如确认它已不存在，请删除锁文件后重试: {path}", path=lock.path)), file=sys.stderr)
            return 1

        tasks = store.list_all()
        enabled = [tk for tk in tasks if tk.enabled]
        users = sorted({tk.user_id for tk in tasks})
        print(f"{DIVIDER}")
        print(c("bold", c("cyan", t("  milu 调度守护进程"))))
        print(DIVIDER)
        print(t("  任务目录: {p}", p=c('dim', str(data_dir / 'schedules'))))
        print(t("  日志目录: {p}", p=c('dim', str(log_dir))))
        print(t("  已启用任务: {n} / {tot} 个（{u} 个用户）",
                n=c('yellow', str(len(enabled))), tot=len(tasks), u=len(users)))
        if enabled:
            for tk in enabled:
                print(f"    {c('cyan', tk.name):<20} @{tk.user_id}  {tk.trigger_desc()}")
        print(DIVIDER + "\n")

        try:
            asyncio.run(engine.start())
        except KeyboardInterrupt:
            print(c("dim", t("\n  调度器已停止。")))
        finally:
            lock.release()
        return 0

    print(c("red", t("未知操作: {action}（可用: start）", action=action)), file=sys.stderr)
    return 2


def _cmd_serve(args) -> int:
    """启动内置 Web 服务（FastAPI + AgentPool + 可选嵌入调度 + 演示前端）。"""
    try:
        from milu.serving.web import find_available_port, run_server
    except ImportError as e:  # 理论上不会（懒导入），保险起见
        print(c("red", t("加载 Web 服务失败：{e}", e=e)), file=sys.stderr)
        return 1

    config = load_config()
    settings = resolve_settings(config, args)

    # 临时 API Key（一般走 .env；这里显式设进环境供 ModelRegistry.create 读取）
    if getattr(args, "api_key", None):
        os.environ[env_key_name(settings.provider)] = args.api_key

    opts = dict(
        provider=settings.provider,
        model=settings.model,
        mode=settings.mode,
        session_enabled=settings.session_enabled,
        web_search=settings.web_search,
        enable_thinking=settings.enable_thinking,
        use_mcp=not getattr(args, "no_mcp", False),
        use_subagents=not getattr(args, "no_subagents", False),
        use_scheduler=not getattr(args, "no_scheduler", False),
        selfguard_enabled=settings.selfguard_enabled,
        show_subagent_events=settings.show_subagent_events,
    )

    host, port = args.host, args.port
    # 端口被占用时向后顺延，使横幅打印的访问地址与实际监听端口一致
    try:
        actual_port = find_available_port(host, port)
    except OSError as e:
        print(c("red", str(e)), file=sys.stderr)
        return 1
    if actual_port != port:
        print(c("yellow", t("  端口 {old} 被占用，自动改用端口 {new}。", old=port, new=actual_port)))
        port = actual_port

    print(f"{DIVIDER}")
    print(c("bold", c("cyan", t("  milu Web 服务"))))
    print(DIVIDER)
    print(t("  访问地址: {url}", url=c('cyan', f'http://{host}:{port}')))
    _admin_tok = os.environ.get("MILU_ADMIN_TOKEN")
    _dash_url = f'http://{host}:{port}/dashboard'
    if _admin_tok:
        _dash_url += '?token=<MILU_ADMIN_TOKEN>'
    print(t("  观测大屏: {url}", url=c('cyan', _dash_url))
          + ("" if _admin_tok else c('dim', t("（未设 MILU_ADMIN_TOKEN，仅本机可访问）"))))
    print(t("  默认厂商: {p}  模型: {m}", p=c('yellow', settings.provider), m=c('dim', settings.model)))
    print(t("  操作模式: {mode}  调度: {sch}  MCP: {mcp}",
            mode=c('yellow', settings.mode),
            sch=t('开') if opts['use_scheduler'] else t('关'),
            mcp=t('开') if opts['use_mcp'] else t('关')))
    # 部署策略与隔离状态（多用户服务尤其关键；docker daemon 未就绪时红色告警）
    from milu.config import deployment_lines
    _be = config.sandbox.get("backend", "subprocess")
    _docker_ok = None
    if config.multiuser == "strict" and _be == "docker":
        from milu.sandbox.docker import docker_available_sync
        _docker_ok = docker_available_sync()
    for _line in deployment_lines(
        config.multiuser, _be,
        bool(config.agent.get("workspace_jail", False)),
        docker_ok=_docker_ok,
    ):
        _col = 'red' if _line.startswith('⚠') else ('yellow' if config.multiuser == 'strict' else 'dim')
        print(f"  {c(_col, _line)}")
    print(c("dim", t("  不同浏览器标签用不同「用户ID」即可演示多用户隔离。Ctrl+C 停止。")))
    print(DIVIDER + "\n")

    try:
        run_server(host=host, port=port, reload=getattr(args, "reload", False), **opts)
    except ImportError:
        # Web 服务依赖（fastapi/uvicorn/sse-starlette）已是核心依赖，通常随 milu 一并安装；
        # 仅当被手动卸载时才会到这里。
        print(c("red", t("缺少 Web 服务依赖（通常已随 milu 安装）。请修复：")), file=sys.stderr)
        print(c("yellow", "  pip install --force-reinstall milu"), file=sys.stderr)
        print(c("dim", "  或: pip install fastapi uvicorn sse-starlette"), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(c("dim", t("\n  Web 服务已停止。")))
    return 0


def _build_gateway_channels(requested: list[str] | None, persist: bool):
    """按环境变量凭证（或显式 --channel）构建启用的渠道列表。

    :param requested: 显式指定的渠道名集合；None 表示按凭证自动探测。
    :param persist: True 用文件持久化 StateStore（去重/游标重启不丢），False 用内存版。
    :return: (channels, store, warnings)
    """
    from milu.channels import FileStateStore, InMemoryStateStore

    store = FileStateStore() if persist else InMemoryStateStore()
    channels = []
    warnings: list[str] = []

    def _want(name: str) -> bool:
        return requested is None or name in requested

    # 微信客服
    if _want("wechat_kf"):
        if os.environ.get("WECHAT_KF_CORP_ID"):
            from milu.channels.wechat_kf import WeChatKfChannel, WeChatKfConfig
            channels.append(WeChatKfChannel(WeChatKfConfig.from_env(), state=store))
        elif requested is not None:
            warnings.append(t("渠道 wechat_kf 缺少环境变量（WECHAT_KF_*），已跳过"))
    # 飞书
    if _want("feishu"):
        if os.environ.get("FEISHU_APP_ID"):
            from milu.channels.feishu import FeishuChannel, FeishuConfig
            channels.append(FeishuChannel(FeishuConfig.from_env(), state=store))
        elif requested is not None:
            warnings.append(t("渠道 feishu 缺少环境变量（FEISHU_*），已跳过"))
    # Telegram
    if _want("telegram"):
        if os.environ.get("TELEGRAM_BOT_TOKEN"):
            from milu.channels.telegram import TelegramChannel, TelegramConfig
            channels.append(TelegramChannel(TelegramConfig.from_env(), state=store))
        elif requested is not None:
            warnings.append(t("渠道 telegram 缺少环境变量（TELEGRAM_BOT_TOKEN），已跳过"))

    return channels, store, warnings


# 各渠道的回调路径取值（用于启动横幅展示）
def _channel_endpoint(ch) -> str:
    cfg = getattr(ch, "_config", None)
    path = getattr(cfg, "callback_path", None)
    if path:
        return path
    if ch.name == "telegram":
        return t("（长轮询，无回调路径）")
    return "-"


def _cmd_gateway(args) -> int:
    """启动多渠道接入网关：webhook（微信客服/飞书）+ 长轮询（Telegram）→ milu Agent。"""
    ensure_dotenv_loaded()

    config = load_config()
    settings = resolve_settings(config, args)
    # gateway 默认 auto（无人值守自主决策）；显式 --mode 优先
    if not getattr(args, "mode", None):
        settings.mode = "auto"

    if getattr(args, "api_key", None):
        os.environ[env_key_name(settings.provider)] = args.api_key

    requested = None
    if getattr(args, "channel", None):
        requested = [x.strip() for x in args.channel.split(",") if x.strip()]
    persist = not getattr(args, "no_persist", False)

    channels, store, warnings = _build_gateway_channels(requested, persist)
    for w in warnings:
        print(c("yellow", f"  {w}"))
    if not channels:
        print(c("red", t("没有可用渠道。请至少配置一个渠道的环境变量：")), file=sys.stderr)
        print(c("dim", "  微信客服: WECHAT_KF_CORP_ID/SECRET/TOKEN/AESKEY"), file=sys.stderr)
        print(c("dim", "  飞书:     FEISHU_APP_ID/APP_SECRET（+ VERIFY_TOKEN/ENCRYPT_KEY）"), file=sys.stderr)
        print(c("dim", "  Telegram: TELEGRAM_BOT_TOKEN"), file=sys.stderr)
        return 1

    from milu.cli.builder import build_gateway_pool
    from milu.channels import AgentRunner, Gateway

    pool, mc = build_gateway_pool(settings)
    runner = AgentRunner(pool)
    gateway = Gateway.from_runner(runner, channels, title="milu Gateway")

    host, port = args.host, args.port
    print(f"{DIVIDER}")
    print(c("bold", c("cyan", t("  milu 多渠道网关"))))
    print(DIVIDER)
    print(t("  监听地址: {url}", url=c('cyan', f'http://{host}:{port}')))
    print(t("  默认厂商: {p}  模型: {m}",
            p=c('yellow', settings.provider), m=c('dim', settings.model)))
    print(t("  操作模式: {mode}  去重/游标: {persist}",
            mode=c('yellow', settings.mode),
            persist=t('文件持久化') if persist else t('内存版')))
    print(t("  启用渠道（{n} 个）:", n=len(channels)))
    for ch in channels:
        print(f"    - {c('green', ch.name)}  {c('dim', _channel_endpoint(ch))}")
    print(c("dim", t(
        "  webhook 渠道请把平台回调 URL 指向 https://<公网域名><回调路径>（需公网 HTTPS）。")))
    # 部署/隔离状态（多用户网关尤其关键；docker daemon 未就绪时红色告警）
    from milu.config import deployment_lines
    try:
        from milu.config import docker_available_sync
    except ImportError:
        docker_available_sync = None  # 老版本兜底
    _be = mc.sandbox.get("backend", "subprocess")
    _docker_ok = None
    if mc.multiuser == "strict" and _be == "docker" and docker_available_sync:
        _docker_ok = docker_available_sync()
    for _line in deployment_lines(
        mc.multiuser, _be, bool(mc.agent.get("workspace_jail", False)),
        docker_ok=_docker_ok,
    ):
        _col = 'red' if _line.startswith('⚠') else ('yellow' if mc.multiuser == 'strict' else 'dim')
        print(f"  {c(_col, _line)}")
    print(c("dim", t("  Ctrl+C 停止。")))
    print(DIVIDER + "\n")

    try:
        gateway.run(host=host, port=port)
    except KeyboardInterrupt:
        print(c("dim", t("\n  网关已停止。")))
    return 0


# ── main ─────────────────────────────────────────────────

def _resolve_lang(argv_list: list[str]) -> str:
    """解析界面语言：--lang > MILU_LANG 环境变量 > config.json 的 lang > 默认 zh。"""
    for i, a in enumerate(argv_list):
        if a == "--lang" and i + 1 < len(argv_list):
            return argv_list[i + 1]
        if a.startswith("--lang="):
            return a.split("=", 1)[1]
    env = os.environ.get("MILU_LANG")
    if env:
        return env
    try:
        return str(load_config().data.get("lang") or "zh")
    except Exception:
        return "zh"


def main(argv: list[str] | None = None) -> int:
    # Windows 下启用 ANSI 颜色
    if sys.platform == "win32":
        os.system("")
    # 进程内加载一次 .env（可被 MILU_NO_DOTENV 关闭）
    ensure_dotenv_loaded()

    argv_list = sys.argv[1:] if argv is None else list(argv)
    # 在构建解析器前先定语言，使 argparse 帮助文本也随之中/英切换
    set_lang(_resolve_lang(argv_list))

    parser = build_parser()
    args = parser.parse_args(argv_list)
    command = args.command or "chat"  # 无子命令 → 进入交互对话

    handlers = {
        "chat": _cmd_chat,
        "run": _cmd_run,
        "setup": _cmd_setup,
        "config": _cmd_config,
        "sessions": _cmd_sessions,
        "providers": _cmd_providers,
        "version": _cmd_version,
        "schedule": _cmd_schedule,
        "scheduler": _cmd_scheduler,
        "serve": _cmd_serve,
        "gateway": _cmd_gateway,
        "trace": _cmd_trace,
    }
    handler = handlers.get(command)
    if handler is None:
        print(c("red", t("未知子命令: {c}", c=command)), file=sys.stderr)
        return 2

    try:
        return handler(args) or 0  # type: ignore[misc]
    except KeyboardInterrupt:
        print(c("dim", t("\n已中断。")))
        return 130
    except AuthenticationError as e:
        provider = (getattr(args, "provider", None)
                    or load_config().llm.get("provider") or DEFAULT_PROVIDER)
        print(c("red", t("\n鉴权失败：{e}", e=e)), file=sys.stderr)
        print(c("dim", t("可运行 `milu setup` 进行初始化引导，或在 .env / 环境变量中设置 {env}。",
                         env=env_key_name(provider))),
              file=sys.stderr)
        return 1
    except ValueError as e:
        print(c("red", t("\n错误：{e}", e=e)), file=sys.stderr)
        return 2
    except MiluError as e:
        print(c("red", t("\n运行失败：{e}", e=e)), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
