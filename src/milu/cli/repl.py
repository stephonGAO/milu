"""交互式聊天 REPL：主循环 + 全部 / 命令。

迁移并整理自 examples/multi_turn_chat.py 的 main() 与 handle_command()。
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path

from milu.agent.config import AgentMode
from milu.i18n import t
from milu.cli.config import Settings
from milu.cli.render import BANNER, DIVIDER, c, render_turn
from milu.tools.builtin import BUILTIN_TOOLS

# 内置子代理名（Agent 全配默认注入的三件套，用于 /tools 展示分类）
_SUBAGENT_NAMES = {"researcher", "reader", "coder"}
_META_NAMES = {"list_catalog", "search_tools", "activate_tools", "load_skill"}

# 欢迎横幅模板：各标签为具名占位，运行时用 t() 填中/英，模板本身中英共用
_HEADER_TMPL = (
    "\n"
    "  {l_pm}: {prov} / {model}    {l_mode}: {mode}\n"
    "  {l_tools}:  {names}\n"
    "  {l_sub}:    {sub}\n"
    "  {l_meta}:    list_catalog, search_tools, activate_tools{meta_note}\n"
    "  {l_skills}:      {skills_note}\n\n"
    "  {cmd_line}\n"
)


# ── ESC 键中断 ────────────────────────────────────────────

def _start_esc_watcher(
    loop: asyncio.AbstractEventLoop,
    task: "asyncio.Task[str]",
    stop_event: threading.Event,
) -> threading.Thread:
    """后台线程：监听 ESC 键，检测到后取消指定 asyncio Task。

    Windows 用 msvcrt（无需修改终端模式）。
    Unix 用 termios 关闭 ICANON/ECHO（保留 OPOST，print 输出不受影响）。
    """

    def _watch_win32() -> None:
        import msvcrt
        import time
        while not stop_event.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch == b"\x1b":
                    loop.call_soon_threadsafe(task.cancel)
                    return
            time.sleep(0.05)

    def _watch_unix() -> None:
        import select
        try:
            import termios
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            new = list(old)
            # 只关 ICANON/ECHO，不动 OPOST（保证 print 换行正常）
            new[3] = new[3] & ~(termios.ICANON | termios.ECHO)
            new[6][termios.VMIN] = 1   # type: ignore[index]
            new[6][termios.VTIME] = 0  # type: ignore[index]
            termios.tcsetattr(fd, termios.TCSAFLUSH, new)
            try:
                while not stop_event.is_set():
                    r, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if r:
                        ch = sys.stdin.buffer.read(1)
                        if ch == b"\x1b":
                            loop.call_soon_threadsafe(task.cancel)
                            return
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            # 非 tty 场景（如 stdin 重定向）静默退出
            import time
            while not stop_event.is_set():
                time.sleep(0.1)

    target = _watch_win32 if sys.platform == "win32" else _watch_unix
    th = threading.Thread(target=target, daemon=True)
    th.start()
    return th


async def _render_with_esc(agent, user_input: str, *, show_subagent: bool = True) -> str:
    """运行 render_turn，同时监听 ESC 键中断。

    检测到 ESC 时立即取消当前轮次，并回滚历史到本轮开始前的状态，
    使用户可以继续输入新消息。
    """
    loop = asyncio.get_running_loop()
    # 快照历史：中断后回滚，避免半截的 assistant/tool 消息破坏下一次 LLM 调用
    snapshot = list(agent.history._messages)

    task: asyncio.Task[str] = asyncio.create_task(
        render_turn(agent, user_input, show_subagent=show_subagent)
    )
    stop_event = threading.Event()
    _start_esc_watcher(loop, task, stop_event)

    try:
        return await task
    except asyncio.CancelledError:
        agent.history.replace_all(snapshot)
        print(c("yellow", t("\n  [已中断]")), flush=True)
        return ""
    finally:
        stop_event.set()


def _read_plan_items(agent) -> list[dict]:
    """读取当前会话的计划条目（todo 工具持久化在 session 目录的 plan.json）。"""
    if not agent.session:
        return []
    plan_file = agent.session.dir_path / "plan.json"
    if not plan_file.exists():
        return []
    try:
        return json.loads(plan_file.read_text(encoding="utf-8")).get("items", [])
    except (OSError, ValueError):
        return []


def print_header(agent, settings: Settings) -> None:
    """打印欢迎横幅与可用能力概览。"""
    print(f"\n{BANNER}")
    print(c("bold", c("cyan", t("  milu — 交互式对话"))))
    print(BANNER)
    all_names = [tf._tool_wrapper.name for tf in BUILTIN_TOOLS]
    sub = "researcher / reader / coder" if settings.use_subagents else t("（已禁用 --no-subagents）")
    print(_HEADER_TMPL.format(
        l_pm=t("厂商/模型"), prov=c('cyan', settings.provider), model=c('cyan', settings.model),
        l_mode=t("模式"), mode=c('yellow', settings.mode),
        l_tools=t("内置工具"), names=', '.join(all_names),
        l_sub=t("子代理"), sub=sub,
        l_meta=t("元工具"), meta_note=t("（发现/激活 MCP 工具）"),
        l_skills=t("技能"), skills_note=t("内置技能元数据已注入，LLM 按需 load_skill 加载"),
        cmd_line=t("命令: /help 查看全部命令  ·  /quit 退出"),
    ))
    # 部署策略与隔离状态（strict 醒目；docker daemon 未就绪时运行时告警）
    from milu.config import deployment_lines
    _be = settings.sandbox.get("backend", "subprocess")
    _docker_ok = None
    if settings.multiuser == "strict" and _be == "docker":
        from milu.sandbox.docker import docker_available_sync
        _docker_ok = docker_available_sync()
    for _line in deployment_lines(
        settings.multiuser, _be,
        bool(settings.agent.get("workspace_jail", False)),
        docker_ok=_docker_ok,
    ):
        _col = 'red' if _line.startswith('⚠') else (
            'yellow' if settings.multiuser == 'strict' else 'dim')
        print(f"  {c(_col, _line)}")


# ── / 命令处理 ────────────────────────────────────────────

async def handle_command(agent, cmd: str) -> bool:
    """处理 / 命令。返回 True 继续循环，False 表示退出。"""
    if cmd == "/quit" or cmd == "/exit":
        print(c("dim", t("\n再见!")))
        return False

    elif cmd == "/reset":
        await agent.reset()
        msg = t("\n  对话已重置，上下文和计划已清空。")
        if agent.session:
            msg += t("\n  新日志段: {name}", name=agent.session.conversation_path.name)
        print(c("yellow", msg + "\n"))

    elif cmd == "/history":
        messages = agent.history.all_messages
        session = agent.history.session
        print(f"\n{DIVIDER}")
        print(c("bold", t("  对话历史")) + c("dim", t(" ({n} 条消息)", n=len(messages))))
        if session:
            print(c("dim", t("  会话 ID: {id}", id=session.session_id)))
            print(c("dim", t("  日志文件: {p}", p=session.conversation_path)))
        print(DIVIDER)
        for msg in messages:
            role = msg.role.value.upper()
            raw = msg.content or ""
            if isinstance(raw, list):
                # 多模态消息：图片块显示摘要，文本块拼接
                images = sum(1 for b in raw if isinstance(b, dict)
                             and b.get("type") in ("image_path", "image_url"))
                texts = " ".join(b.get("text", "") for b in raw
                                 if isinstance(b, dict) and b.get("type") == "text")
                raw = (t("[图片×{n}] ", n=images) if images else "") + texts
            content = (raw[:500] + "...") if len(raw) > 500 else raw
            content = content.replace("\n", " ")
            print(f"  {c('cyan', f'[{role:<9}]')} {c('dim', content)}")
        print(DIVIDER + "\n")

    elif cmd == "/tools":
        active_tools = agent.tools.list_tools()
        dormant_tools = agent.tools.list_dormant_tools()
        print(f"\n{DIVIDER}")
        print(c("bold", t("  活跃工具")) + c("dim", t(" ({n} 个)", n=len(active_tools))))
        print(DIVIDER)
        builtin_names = {w._tool_wrapper.name for w in BUILTIN_TOOLS}
        for tf in BUILTIN_TOOLS:
            w = tf._tool_wrapper
            if w.name in active_tools:
                safe = c("green", " [S]") if w.is_safe else ""
                print(f"  {c('yellow', w.name):<30} {w.description[:40]}{safe}")
        for name in _META_NAMES:
            if name in active_tools:
                wrapper = agent.tools.get_tool(name)
                print(f"  {c('cyan', name):<30} {(wrapper.description[:40] if wrapper else '')}")
        for name in _SUBAGENT_NAMES:
            if name in active_tools:
                wrapper = agent.tools.get_tool(name)
                print(f"  {c('magenta', name + ' [SubAgent]'):<30} {(wrapper.description[:40] if wrapper else '')}")
        special = builtin_names | _META_NAMES | _SUBAGENT_NAMES
        activated_mcp = [n for n in active_tools if n not in special]
        if activated_mcp:
            print(DIVIDER)
            print(c("bold", t("  已激活 MCP 工具")))
            print(DIVIDER)
            for name in activated_mcp:
                wrapper = agent.tools.get_tool(name)
                print(f"  {c('green', name):<30} {(wrapper.description[:40] if wrapper else '')}")
        if dormant_tools:
            grouped: dict[str, list] = {}
            for dt in dormant_tools:
                grouped.setdefault(dt["category"] or t("未分类"), []).append(dt)
            print(DIVIDER)
            print(c("bold", t("  休眠工具（search_tools / activate_tools 激活）"))
                  + c("dim", t(" ({n} 个)", n=len(dormant_tools))))
            print(DIVIDER)
            for cat, items in grouped.items():
                print(f"  {c('magenta', f'【{cat}】')}")
                for item in items:
                    print(f"    {c('dim', item['name']):<32} {item['description'][:38]}")
        print(DIVIDER + "\n")

    elif cmd == "/skills":
        names = agent.skill_registry.list_names()
        if not names:
            print(f"\n  {c('dim', t('暂无可用技能。'))}\n")
            return True
        print(f"\n{DIVIDER}")
        print(c("bold", t("  可用技能")) + c("dim", t(" ({n} 个)", n=len(names))))
        print(DIVIDER)
        for name in names:
            cfg = agent.skill_registry.get(name)
            triggers = f"  {c('dim', '[' + ', '.join(cfg.triggers) + ']')}" if cfg.triggers else ""
            print(f"  {c('blue', name):<30} {cfg.description}{triggers}")
        print(f"\n  {c('dim', t('LLM 会自动调用 load_skill 按需加载技能正文'))}")
        print(DIVIDER + "\n")

    elif cmd == "/plan":
        items = _read_plan_items(agent)
        if not items:
            print(f"\n  {c('dim', t('暂无会话计划。'))}\n")
            return True
        print(f"\n{DIVIDER}")
        print(c("bold", t("  当前会话计划")) + c("dim", t(" ({n} 个条目)", n=len(items))))
        print(DIVIDER)
        markers = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
        colors = {"pending": "yellow", "in_progress": "green", "completed": "dim"}
        completed = 0
        for item in items:
            status = item.get("status", "pending")
            line = f"  {markers.get(status, '[ ]')} {item.get('content', '')}"
            if status == "in_progress" and item.get("activeForm"):
                line += f"  ({c('cyan', item['activeForm'])})"
            if status == "completed":
                completed += 1
            print(f"  {c(colors.get(status, 'yellow'), line)}")
        print(f"\n  {c('dim', t('({c}/{n} 已完成)', c=completed, n=len(items)))}")
        print(DIVIDER + "\n")

    elif cmd == "/mode":
        mode = agent.mode
        mode_colors = {
            AgentMode.TALK: "cyan",
            AgentMode.MANUAL: "yellow",
            AgentMode.AUTO: "green",
            AgentMode.SUPERWORK: "red",
        }
        mode_desc = {
            AgentMode.TALK: t("只读模式（仅允许安全操作）"),
            AgentMode.MANUAL: t("人工审批模式（不安全操作逐一确认）"),
            AgentMode.AUTO: t("自主模式（不安全操作自动执行，AI 判定兜底）"),
            AgentMode.SUPERWORK: t("全权限模式（跳过所有安全检查）"),
        }
        print(f"\n{DIVIDER}")
        print(c("bold", t("  操作模式")))
        print(DIVIDER)
        for m in AgentMode:
            marker = c("bold", t(" → 当前")) if m == mode else ""
            print(f"  {c(mode_colors.get(m, 'reset'), m.value):<20} {c('dim', mode_desc.get(m, ''))}{marker}")
        print(DIVIDER)
        print(c("dim", t("  用法: /mode <talk|manual|auto|superwork>\n")))

    elif cmd.startswith("/mode "):
        new_mode = cmd.split(maxsplit=1)[1].strip()
        try:
            agent.set_mode(new_mode)
            color = {"talk": "cyan", "manual": "yellow", "auto": "green",
                     "superwork": "red"}.get(new_mode, "reset")
            print(f"\n  {c('green', t('模式已切换为:'))} {c(color, c('bold', new_mode))}\n")
        except ValueError:
            print(f"\n  {c('red', t('无效模式: {m}', m=new_mode))}{t('（可选: talk, manual, auto, superwork）')}\n")

    elif cmd == "/prompt":
        agent._build_system_prompt()
        system_msg = agent.history.all_messages[0] if agent.history.all_messages else None
        content = system_msg.content if system_msg else ""
        if not content:
            print(f"\n  {c('dim', t('当前系统提示词为空。'))}\n")
            return True
        print(f"\n{DIVIDER}")
        print(c("bold", t("  当前系统提示词"))
              + c("dim", t("  ({n} 行, {c} 字符)", n=content.count(chr(10)) + 1, c=len(content))))
        print(DIVIDER)
        print(content)
        print(DIVIDER + "\n")

    elif cmd == "/compact":
        if not agent._history.compact_enabled:
            print(f"\n  {c('dim', t('上下文压缩未启用。'))}\n")
            return True
        original = len(agent.history._messages)
        compacted, summary = await agent._history.manual_compact()
        agent.history.replace_all(compacted)
        agent._history._log_compaction(compacted)
        print(f"\n{DIVIDER}")
        print(c("bold", t("  手动压缩完成"))
              + c("dim", t("  ({a} → {b} 条消息)", a=original, b=len(compacted))))
        print(DIVIDER)
        preview = (summary[:500] + "...") if len(summary) > 500 else summary
        print(f"  {c('dim', preview)}")
        print(DIVIDER + "\n")

    elif cmd == "/save":
        agent.save_session()
        if agent.session:
            print(f"\n  {c('green', t('会话已保存'))}")
            print(f"  ID: {c('cyan', agent.session.session_id)}")
            print(t("  路径: {p}", p=c('dim', str(agent.session.dir_path))) + "\n")
        else:
            print(f"\n  {c('dim', t('会话功能未启用。'))}\n")

    elif cmd == "/sessions":
        from milu.agent.session import Session as SessionClass
        from milu.resources import default_session_dir
        base_dir = Path(agent.session.base_dir) if agent.session else default_session_dir()
        sessions = SessionClass.list_sessions(base_dir)
        if not sessions:
            print(f"\n  {c('dim', t('暂无历史会话。'))}\n")
            return True
        print(f"\n{DIVIDER}")
        print(c("bold", t("  历史会话")) + c("dim", t(" ({n} 个)", n=len(sessions))))
        print(DIVIDER)
        current_id = agent.session.session_id if agent.session else None
        for s in sessions:
            sid = s.get("session_id", "?")
            updated = s.get("updated_at", 0)
            time_str = datetime.fromtimestamp(updated).strftime("%m-%d %H:%M") if updated else "?"
            marker = c("green", t(" ← 当前")) if sid == current_id else ""
            model_str = c("dim", f" ({s.get('model')})") if s.get("model") else ""
            print(f"  {c('cyan', sid)}  {c('dim', time_str)}  "
                  + t("{n} 条消息", n=s.get('message_count', 0)) + f"{model_str}{marker}")
        print(DIVIDER + "\n")

    elif cmd == "/new":
        agent.new_session()
        print(f"\n  {c('green', t('新会话已创建'))}")
        if agent.session:
            print(f"  ID: {c('cyan', agent.session.session_id)}\n")

    elif cmd.startswith("/load"):
        parts = cmd.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            print(f"\n  {c('yellow', t('用法: /load <session_id>'))}\n")
            return True
        session_id = parts[1].strip()
        try:
            count = agent.load_session(session_id)
            print(f"\n  {c('green', t('会话已加载'))}")
            print(t("  ID: {id}  消息数: {n}", id=c('cyan', session_id), n=count) + "\n")
        except FileNotFoundError:
            print(f"\n  {c('red', t('会话不存在: {id}', id=session_id))}\n")
        except Exception as e:
            print(f"\n  {c('red', t('加载失败: {e}', e=e))}\n")

    elif cmd == "/schedule" or cmd.startswith("/schedule "):
        from milu.resources import user_data_dir
        from milu.scheduler.store import ScheduleStore

        store = ScheduleStore(user_data_dir())
        tasks = store.list_user("default")  # REPL 为 CLI 单人场景，列默认用户
        if not tasks:
            print(f"\n  {c('dim', t('暂无定时任务。可在对话中让 Agent 调用 schedule_create 工具创建。'))}\n")
            return True
        print(f"\n{DIVIDER}")
        print(c("bold", t("  定时任务")) + c("dim", t(" ({n} 个)", n=len(tasks))))
        print(DIVIDER)
        for tk in tasks:
            status_color = "green" if tk.enabled else "dim"
            status_str = t("启用") if tk.enabled else t("禁用")
            last = tk.last_run[:16].replace("T", " ") if tk.last_run else t("从未")
            nxt = tk.next_run[:16].replace("T", " ") if tk.next_run else t("待计算")
            print(
                f"  {c('cyan', tk.name)}  {c(status_color, f'[{status_str}]')}  "
                f"{c('dim', t('运行 {n} 次', n=tk.run_count))}\n"
                f"    {t('触发: ')}{c('yellow', tk.trigger_desc())}\n"
                f"    {t('下次: ')}{c('dim', nxt)}  {t('上次: ')}{c('dim', last)}"
            )
        print(DIVIDER)
        print(c("dim", t("  milu scheduler start  — 启动调度守护进程\n")))

    elif cmd == "/help":
        print(t(
            "\n"
            "  /history    — 查看对话历史\n"
            "  /reset      — 重置对话（清空上下文）\n"
            "  /tools      — 查看可用工具（含休眠工具）\n"
            "  /skills     — 查看可用技能\n"
            "  /plan       — 查看当前会话计划\n"
            "  /schedule   — 查看定时任务列表\n"
            "  /mode       — 查看/切换操作模式（talk/manual/auto/superwork）\n"
            "  /prompt     — 查看当前系统提示词\n"
            "  /compact    — 手动压缩对话历史\n"
            "  /save       — 保存当前会话\n"
            "  /sessions   — 查看所有会话\n"
            "  /new        — 新建会话（自动保存当前）\n"
            "  /load <id>  — 加载历史会话\n"
            "  /help       — 显示帮助\n"
            "  /quit       — 退出\n"
        ))

    else:
        print(c("red", t("\n  未知命令: {cmd}  (输入 /help 查看帮助)\n", cmd=cmd)))

    return True


# ── 主循环 ────────────────────────────────────────────────

async def run_chat(agent, settings: Settings) -> None:
    """运行交互式聊天 REPL。"""
    from milu.tools.mcp.connection import suppress_mcp_asyncgen_errors
    suppress_mcp_asyncgen_errors()

    print_header(agent, settings)

    # 嵌入式调度引擎：对话期间自动执行定时任务，退出即停。
    # 单实例锁防重复执行（与 daemon / web serve 共用）；echo=False 不污染
    # REPL，结果走系统弹窗 + outbox + 日志文件，/schedule 可查。
    # ⚠️ 必须独立线程跑专属事件循环：REPL 主循环在 input() 上同步阻塞主线程
    # 的事件循环（停在 You> 提示符时即冻结），start_background() 挂在主循环
    # 上的话 tick 永远得不到调度。daemon 线程随进程退出，无需 join。
    sched_engine = sched_lock = None
    if settings.use_scheduler:
        from milu.cli.builder import build_scheduler_engine
        from milu.scheduler.lock import SchedulerLock

        engine, store, data_dir = build_scheduler_engine(echo=False)
        lock = SchedulerLock(data_dir)
        sched_engine, sched_lock = engine, lock
        holder = lock.holder_pid()  # 起线程前探测（线程可能立刻抢到锁，避免把自己误报为他人）

        def _sched_main() -> None:
            try:
                # start_with_lock：抢到锁即运行；被占用则等待，持锁者退出后自动接管
                asyncio.run(engine.start_with_lock(lock))
            except Exception:
                logging.getLogger(__name__).exception("嵌入式调度线程异常退出")

        threading.Thread(
            target=_sched_main, name="milu-scheduler", daemon=True
        ).start()
        if holder:
            print(c("dim", t("  检测到调度器已在运行（PID {pid}），任务由它执行；其退出后本会话自动接管\n",
                             pid=holder)))
        else:
            enabled = sum(1 for tk in store.list_all() if tk.enabled)
            print(c("cyan", t("  定时任务调度已启动：{n} 个启用任务，对话期间自动执行", n=enabled))
                  + c("dim", t("（--no-scheduler 可关）\n")))

    # 连接 MCP（如启用且存在配置文件）
    if settings.use_mcp:
        await agent.connect_mcp()
        dormant = agent.tools.list_dormant_tools()
        if dormant:
            cats: dict[str, list] = {}
            for dt in dormant:
                cats.setdefault(dt["category"] or "MCP", []).append(dt["name"])
            summary = ", ".join(t("{cat}({n}个)", cat=cat, n=len(v)) for cat, v in cats.items())
            print(c("cyan", t("  MCP 工具已加载（休眠态）: {summary}", summary=summary)))
            print(c("dim", t("  输入 /tools 查看，或 search_tools / activate_tools 按需激活\n")))

    # 续接历史会话（如指定）
    if settings.session_id and agent.session:
        try:
            count = agent.load_session(settings.session_id)
            print(c("green", t("  已加载会话 {id}（{n} 条消息）\n", id=settings.session_id, n=count)))
        except FileNotFoundError:
            print(c("yellow", t("  会话 {id} 不存在，使用新会话\n", id=settings.session_id)))

    if agent.session:
        print(c("cyan", t("  会话 ID: {id}", id=agent.session.session_id)))
        print(c("dim", t("  日志路径: {p}\n", p=agent.session.dir_path)))

    # 恢复的计划状态
    items = _read_plan_items(agent)
    if items:
        completed = sum(1 for i in items if i.get("status") == "completed")
        in_progress = next((i.get("content") for i in items if i.get("status") == "in_progress"), None)
        line = t("  计划已恢复: {c}/{n} 已完成", c=completed, n=len(items))
        if in_progress:
            line += t(" | 当前: {x}", x=c('green', in_progress))
        print(c("yellow", line))
        print(c("dim", t("  输入 /plan 查看详情\n")))

    turn_count = 0
    try:
        while True:
            try:
                user_input = input(c("green", "\nYou> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print(c("dim", t("\n再见!")))
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                if not await handle_command(agent, user_input):
                    break
                continue

            turn_count += 1
            print(c("blue", f"\n  [Turn {turn_count}]") + c("dim", t("  ESC 中断")))
            print(DIVIDER)
            try:
                await _render_with_esc(
                    agent, user_input, show_subagent=settings.show_subagent_events
                )
            except Exception as e:  # noqa: BLE001 — REPL 兜底，单轮异常不退出
                print(f"\n  {c('red', '[EXCEPTION]')} {c('red', str(e))}")
    finally:
        # 停嵌入式调度：置停止标志 + 释放锁（daemon 线程随进程退出，不 join——
        # 它可能正睡在分钟级 sleep 上）
        if sched_engine is not None:
            sched_engine.stop()
        if sched_lock is not None:
            sched_lock.release()
        agent.save_session()
        await agent.disconnect_mcp()
