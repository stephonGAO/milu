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
from milu.cli.config import Settings
from milu.cli.render import BANNER, DIVIDER, c, render_turn
from milu.tools.builtin import BUILTIN_TOOLS

# 内置子代理名（Agent 全配默认注入的三件套，用于 /tools 展示分类）
_SUBAGENT_NAMES = {"researcher", "reader", "coder"}
_META_NAMES = {"list_catalog", "search_tools", "activate_tools", "load_skill"}


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
    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t


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
        print(c("yellow", "\n  [已中断]"), flush=True)
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
    print(c("bold", c("cyan", "  milu — 交互式对话")))
    print(BANNER)
    all_names = [t._tool_wrapper.name for t in BUILTIN_TOOLS]
    sub = "researcher / reader / coder" if settings.use_subagents else "（已禁用 --no-subagents）"
    print(f"""
  厂商/模型: {c('cyan', settings.provider)} / {c('cyan', settings.model)}    模式: {c('yellow', settings.mode)}
  内置工具:  {', '.join(all_names)}
  子代理:    {sub}
  元工具:    list_catalog, search_tools, activate_tools（发现/激活 MCP 工具）
  技能:      内置技能元数据已注入，LLM 按需 load_skill 加载

  命令: /help 查看全部命令  ·  /quit 退出
""")


# ── / 命令处理 ────────────────────────────────────────────

async def handle_command(agent, cmd: str) -> bool:
    """处理 / 命令。返回 True 继续循环，False 表示退出。"""
    if cmd == "/quit" or cmd == "/exit":
        print(c("dim", "\n再见!"))
        return False

    elif cmd == "/reset":
        await agent.reset()
        msg = "\n  对话已重置，上下文和计划已清空。"
        if agent.session:
            msg += f"\n  新日志段: {agent.session.conversation_path.name}"
        print(c("yellow", msg + "\n"))

    elif cmd == "/history":
        messages = agent.history.all_messages
        session = agent.history.session
        print(f"\n{DIVIDER}")
        print(c("bold", "  对话历史") + c("dim", f" ({len(messages)} 条消息)"))
        if session:
            print(c("dim", f"  会话 ID: {session.session_id}"))
            print(c("dim", f"  日志文件: {session.conversation_path}"))
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
                raw = (f"[图片×{images}] " if images else "") + texts
            content = (raw[:500] + "...") if len(raw) > 500 else raw
            content = content.replace("\n", " ")
            print(f"  {c('cyan', f'[{role:<9}]')} {c('dim', content)}")
        print(DIVIDER + "\n")

    elif cmd == "/tools":
        active_tools = agent.tools.list_tools()
        dormant_tools = agent.tools.list_dormant_tools()
        print(f"\n{DIVIDER}")
        print(c("bold", "  活跃工具") + c("dim", f" ({len(active_tools)} 个)"))
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
            print(c("bold", "  已激活 MCP 工具"))
            print(DIVIDER)
            for name in activated_mcp:
                wrapper = agent.tools.get_tool(name)
                print(f"  {c('green', name):<30} {(wrapper.description[:40] if wrapper else '')}")
        if dormant_tools:
            grouped: dict[str, list] = {}
            for t in dormant_tools:
                grouped.setdefault(t["category"] or "未分类", []).append(t)
            print(DIVIDER)
            print(c("bold", "  休眠工具（search_tools / activate_tools 激活）")
                  + c("dim", f" ({len(dormant_tools)} 个)"))
            print(DIVIDER)
            for cat, items in grouped.items():
                print(f"  {c('magenta', f'【{cat}】')}")
                for item in items:
                    print(f"    {c('dim', item['name']):<32} {item['description'][:38]}")
        print(DIVIDER + "\n")

    elif cmd == "/skills":
        names = agent.skill_registry.list_names()
        if not names:
            print(f"\n  {c('dim', '暂无可用技能。')}\n")
            return True
        print(f"\n{DIVIDER}")
        print(c("bold", "  可用技能") + c("dim", f" ({len(names)} 个)"))
        print(DIVIDER)
        for name in names:
            cfg = agent.skill_registry.get(name)
            triggers = f"  {c('dim', '[' + ', '.join(cfg.triggers) + ']')}" if cfg.triggers else ""
            print(f"  {c('blue', name):<30} {cfg.description}{triggers}")
        print(f"\n  {c('dim', 'LLM 会自动调用 load_skill 按需加载技能正文')}")
        print(DIVIDER + "\n")

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
            line = f"  {markers.get(status, '[ ]')} {item.get('content', '')}"
            if status == "in_progress" and item.get("activeForm"):
                line += f"  ({c('cyan', item['activeForm'])})"
            if status == "completed":
                completed += 1
            print(f"  {c(colors.get(status, 'yellow'), line)}")
        print(f"\n  {c('dim', f'({completed}/{len(items)} 已完成)')}")
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
            AgentMode.TALK: "只读模式（仅允许安全操作）",
            AgentMode.MANUAL: "人工审批模式（不安全操作逐一确认）",
            AgentMode.AUTO: "自主模式（不安全操作自动执行，AI 判定兜底）",
            AgentMode.SUPERWORK: "全权限模式（跳过所有安全检查）",
        }
        print(f"\n{DIVIDER}")
        print(c("bold", "  操作模式"))
        print(DIVIDER)
        for m in AgentMode:
            marker = c("bold", " → 当前") if m == mode else ""
            print(f"  {c(mode_colors.get(m, 'reset'), m.value):<20} {c('dim', mode_desc.get(m, ''))}{marker}")
        print(DIVIDER)
        print(c("dim", "  用法: /mode <talk|manual|auto|superwork>\n"))

    elif cmd.startswith("/mode "):
        new_mode = cmd.split(maxsplit=1)[1].strip()
        try:
            agent.set_mode(new_mode)
            color = {"talk": "cyan", "manual": "yellow", "auto": "green",
                     "superwork": "red"}.get(new_mode, "reset")
            print(f"\n  {c('green', '模式已切换为:')} {c(color, c('bold', new_mode))}\n")
        except ValueError:
            print(f"\n  {c('red', f'无效模式: {new_mode}')}（可选: talk, manual, auto, superwork）\n")

    elif cmd == "/prompt":
        agent._build_system_prompt()
        system_msg = agent.history.all_messages[0] if agent.history.all_messages else None
        content = system_msg.content if system_msg else ""
        if not content:
            print(f"\n  {c('dim', '当前系统提示词为空。')}\n")
            return True
        print(f"\n{DIVIDER}")
        print(c("bold", "  当前系统提示词") + c("dim", f"  ({content.count(chr(10)) + 1} 行, {len(content)} 字符)"))
        print(DIVIDER)
        print(content)
        print(DIVIDER + "\n")

    elif cmd == "/compact":
        if not agent._history.compact_enabled:
            print(f"\n  {c('dim', '上下文压缩未启用。')}\n")
            return True
        original = len(agent.history._messages)
        compacted, summary = await agent._history.manual_compact()
        agent.history.replace_all(compacted)
        agent._history._log_compaction(compacted)
        print(f"\n{DIVIDER}")
        print(c("bold", "  手动压缩完成") + c("dim", f"  ({original} → {len(compacted)} 条消息)"))
        print(DIVIDER)
        preview = (summary[:500] + "...") if len(summary) > 500 else summary
        print(f"  {c('dim', preview)}")
        print(DIVIDER + "\n")

    elif cmd == "/save":
        agent.save_session()
        if agent.session:
            print(f"\n  {c('green', '会话已保存')}")
            print(f"  ID: {c('cyan', agent.session.session_id)}")
            print(f"  路径: {c('dim', str(agent.session.dir_path))}\n")
        else:
            print(f"\n  {c('dim', '会话功能未启用。')}\n")

    elif cmd == "/sessions":
        from milu.agent.session import Session as SessionClass
        from milu.resources import default_session_dir
        base_dir = Path(agent.session.base_dir) if agent.session else default_session_dir()
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
            updated = s.get("updated_at", 0)
            time_str = datetime.fromtimestamp(updated).strftime("%m-%d %H:%M") if updated else "?"
            marker = c("green", " ← 当前") if sid == current_id else ""
            model_str = c("dim", f" ({s.get('model')})") if s.get("model") else ""
            print(f"  {c('cyan', sid)}  {c('dim', time_str)}  {s.get('message_count', 0)} 条消息{model_str}{marker}")
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
            count = agent.load_session(session_id)
            print(f"\n  {c('green', '会话已加载')}")
            print(f"  ID: {c('cyan', session_id)}  消息数: {count}\n")
        except FileNotFoundError:
            print(f"\n  {c('red', f'会话不存在: {session_id}')}\n")
        except Exception as e:
            print(f"\n  {c('red', f'加载失败: {e}')}\n")

    elif cmd == "/schedule" or cmd.startswith("/schedule "):
        from milu.resources import user_data_dir
        from milu.scheduler.store import ScheduleStore

        store = ScheduleStore(user_data_dir())
        tasks = store.list_user("default")  # REPL 为 CLI 单人场景，列默认用户
        if not tasks:
            print(f"\n  {c('dim', '暂无定时任务。可在对话中让 Agent 调用 schedule_create 工具创建。')}\n")
            return True
        print(f"\n{DIVIDER}")
        print(c("bold", "  定时任务") + c("dim", f" ({len(tasks)} 个)"))
        print(DIVIDER)
        for t in tasks:
            status_color = "green" if t.enabled else "dim"
            status_str = "启用" if t.enabled else "禁用"
            last = t.last_run[:16].replace("T", " ") if t.last_run else "从未"
            nxt = t.next_run[:16].replace("T", " ") if t.next_run else "待计算"
            print(
                f"  {c('cyan', t.name)}  {c(status_color, f'[{status_str}]')}  "
                f"{c('dim', f'运行 {t.run_count} 次')}\n"
                f"    触发: {c('yellow', t.trigger_desc())}\n"
                f"    下次: {c('dim', nxt)}  上次: {c('dim', last)}"
            )
        print(DIVIDER)
        print(c("dim", "  milu scheduler start  — 启动调度守护进程\n"))

    elif cmd == "/help":
        print("""
  /history    — 查看对话历史
  /reset      — 重置对话（清空上下文）
  /tools      — 查看可用工具（含休眠工具）
  /skills     — 查看可用技能
  /plan       — 查看当前会话计划
  /schedule   — 查看定时任务列表
  /mode       — 查看/切换操作模式（talk/manual/auto/superwork）
  /prompt     — 查看当前系统提示词
  /compact    — 手动压缩对话历史
  /save       — 保存当前会话
  /sessions   — 查看所有会话
  /new        — 新建会话（自动保存当前）
  /load <id>  — 加载历史会话
  /help       — 显示帮助
  /quit       — 退出
""")

    else:
        print(c("red", f"\n  未知命令: {cmd}  (输入 /help 查看帮助)\n"))

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
            print(c("dim", f"  检测到调度器已在运行（PID {holder}），任务由它执行；"
                           f"其退出后本会话自动接管\n"))
        else:
            enabled = sum(1 for t in store.list_all() if t.enabled)
            print(c("cyan", f"  定时任务调度已启动：{enabled} 个启用任务，对话期间自动执行")
                  + c("dim", "（--no-scheduler 可关）\n"))

    # 连接 MCP（如启用且存在配置文件）
    if settings.use_mcp:
        await agent.connect_mcp()
        dormant = agent.tools.list_dormant_tools()
        if dormant:
            cats: dict[str, list] = {}
            for t in dormant:
                cats.setdefault(t["category"] or "MCP", []).append(t["name"])
            summary = ", ".join(f"{cat}({len(v)}个)" for cat, v in cats.items())
            print(c("cyan", f"  MCP 工具已加载（休眠态）: {summary}"))
            print(c("dim", "  输入 /tools 查看，或 search_tools / activate_tools 按需激活\n"))

    # 续接历史会话（如指定）
    if settings.session_id and agent.session:
        try:
            count = agent.load_session(settings.session_id)
            print(c("green", f"  已加载会话 {settings.session_id}（{count} 条消息）\n"))
        except FileNotFoundError:
            print(c("yellow", f"  会话 {settings.session_id} 不存在，使用新会话\n"))

    if agent.session:
        print(c("cyan", f"  会话 ID: {agent.session.session_id}"))
        print(c("dim", f"  日志路径: {agent.session.dir_path}\n"))

    # 恢复的计划状态
    items = _read_plan_items(agent)
    if items:
        completed = sum(1 for i in items if i.get("status") == "completed")
        in_progress = next((i.get("content") for i in items if i.get("status") == "in_progress"), None)
        line = f"  计划已恢复: {completed}/{len(items)} 已完成"
        if in_progress:
            line += f" | 当前: {c('green', in_progress)}"
        print(c("yellow", line))
        print(c("dim", "  输入 /plan 查看详情\n"))

    turn_count = 0
    try:
        while True:
            try:
                user_input = input(c("green", "\nYou> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print(c("dim", "\n再见!"))
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                if not await handle_command(agent, user_input):
                    break
                continue

            turn_count += 1
            print(c("blue", f"\n  [Turn {turn_count}]") + c("dim", "  ESC 中断"))
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
