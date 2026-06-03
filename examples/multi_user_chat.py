"""多用户多轮对话 — AgentPool + FastAPI + 单页 Web 聊天 UI

演示（功能与 examples/multi_turn_chat.py 一一对应）：
  - 多用户隔离：每个 user_id 通过 AgentPool 自动得到独立 Agent 实例
  - 多轮对话记忆：同 user_id+session_id 复用 Agent，history 跨轮保持
  - 全部内置工具：file / python_repl / http_request / datetime / todo_write ...
  - 子代理：researcher / coder / reviewer
  - 全部命令：/history /reset /tools /skills /plan /mode /prompt /compact
              /save /sessions /new /load /help /quit
  - 工具自动调用 + 危险工具确认（Web 弹窗）
  - 流式输出（Server-Sent Events）+ 思考过程 + 子代理可视化
  - MCP 自动连接（config/mcp_servers.json）
  - 会话持久化（每个 user_id+session_id 独立目录）

启动：
    pip install fastapi uvicorn sse-starlette
    .venv/Scripts/python examples/8_multi_user_chat.py

打开浏览器访问：http://localhost:8000

不同浏览器标签页使用不同 user_id 即可看到多用户隔离效果。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

from dotenv import load_dotenv

from agent_framework.agent.config import AgentConfig, AgentMode
from agent_framework.agent.events import (
    AgentDone,
    AgentError,
    ConfirmResponse,
    ToolConfirmRequired,
)
from agent_framework.llm.providers import ModelRegistry
from agent_framework.serving import AgentPool, AgentPoolConfig
from agent_framework.tools.builtin import (
    BUILTIN_TOOLS,
    create_structured_output_tool,
    file_read,
    file_write,
    http_request,
    python_repl,
)
from agent_framework import (
    Agent,
    SubAgentConfig,
    create_subagent_tools,
    SkillConfig,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
logger = logging.getLogger("multi_user_chat")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Agent 工厂（与 multi_turn_chat.py 保持完全一致）
# ─────────────────────────────────────────────────────────────────────────────


def make_agent_factory(llm):
    """返回一个 agent_factory：每次新建 Agent 时由 AgentPool 调用。

    这里把 multi_turn_chat.py:build_agent() 的逻辑原样搬过来，
    并在闭包里持有 LLM，确保每个 user 都拿到带全套工具/子代理/技能的 Agent。

    注意：子代理工具必须在 agent_factory 内部创建（不能用闭包外共享），
    因为 get_parent_mode=lambda: agent.mode 需要引用「正在创建的」agent 自身。
    跨用户共享同一组 subagent_tools 会让所有用户的 mode 锁定为第一个用户。
    """
    prompts_base = Path(__file__).resolve().parent.parent / "config" / "prompts"
    skills_dir = Path(__file__).resolve().parent.parent / "skills"
    so_tool = create_structured_output_tool()

    def _build_subagent_tools(agent):
        return create_subagent_tools(
            llm=llm,
            subagents=[
                SubAgentConfig(
                    name="researcher",
                    description=(
                        "调研助手：擅长搜索和整理信息。"
                        "当需要查找资料、对比分析、总结报告时委派此代理。"
                    ),
                    prompt_dir=prompts_base / "researcher",
                    tools=[file_read, http_request],
                    config=AgentConfig(),
                    skills=[
                        SkillConfig(
                            name="deep-research",
                            description="深度调研专家，擅长多角度分析和系统性总结",
                            triggers=["调研", "研究", "分析"],
                            content=(
                                "你是一位深度调研专家。请按以下流程工作：\n\n"
                                "## 调研流程\n"
                                "1. **明确问题** — 拆解用户问题为 2-3 个子问题\n"
                                "2. **多源搜索** — 从不同角度搜索信息\n"
                                "3. **交叉验证** — 对比多个来源，标注可信度\n"
                                "4. **系统总结** — 按重要性排序，给出结论\n\n"
                                "## 输出格式\n"
                                "- 关键发现：编号列表\n"
                                "- 信息来源：标注 URL 或出处\n"
                                "- 可信度评级：[高/中/低]\n"
                            ),
                        ),
                    ],
                ),
                SubAgentConfig(
                    name="coder",
                    description="编程助手：擅长写代码和调试。",
                    prompt_dir=prompts_base / "coder",
                    tools=[file_read, file_write, python_repl],
                    config=AgentConfig(),
                ),
                SubAgentConfig(
                    name="reviewer",
                    description="代码审查专家：擅长代码审查与建议提出。",
                    prompt_dir=prompts_base / "reviewer",
                    tools=[file_read, python_repl],
                    config=AgentConfig(),
                    skills=[
                        SkillConfig.from_file(str(skills_dir / "code-review.md")),
                    ] if (skills_dir / "code-review.md").exists() else [],
                ),
            ],
            get_parent_mode=lambda: agent.mode,
        )

    def agent_factory(user_id: str, session_id: str, llm_for_user: Any):
        # 每次新建时创建一个新的 Agent + 自己的子代理工具集
        agent = Agent(
            llm=llm_for_user,
            prompt_dir=prompts_base / "main",
            tools=[*BUILTIN_TOOLS, so_tool],
            config=AgentConfig(session_enabled=True, session_dir="./.sessions"),
            on_confirm=lambda tool, args: confirm_unsafe(user_id, tool, args),
        )
        # 子代理工具必须在 Agent 创建后绑定（get_parent_mode 闭包要拿 agent）
        for tool in _build_subagent_tools(agent):
            agent.tools.register(tool)
        return agent

    return agent_factory


# ─────────────────────────────────────────────────────────────────────────────
# 2. 危险工具确认 — per-user 异步 Future 队列
# ─────────────────────────────────────────────────────────────────────────────
#
# 原始 multi_turn_chat.py 用 input() 阻塞读 stdin。
# 在 Web 场景下，confirm_unsafe 是 async 函数，
# 它创建一个 asyncio.Future 并 await，前端通过 HTTP /confirm 解析它。
#
# 关键设计：
#   - 每个 user_id 维护一个「当前待确认」的 Future（同时只允许一个待确认）
#   - Future 由 confirm() 调用方解析（前端 POST /confirm）
#   - 用户离开/刷新页面 → 启动超时（30s）自动拒绝，避免 Agent 永远卡住

_confirmations: dict[str, asyncio.Future] = {}


async def confirm_unsafe(user_id: str, tool_name: str, args_str: str) -> ConfirmResponse:
    """per-user 异步确认：阻塞直到前端响应或 30s 超时。"""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[ConfirmResponse] = loop.create_future()

    # 若已有待确认，先取消旧的（防止堆积）
    old = _confirmations.get(user_id)
    if old is not None and not old.done():
        old.set_result(ConfirmResponse(approved=False, message="被新请求替代"))

    _confirmations[user_id] = future

    try:
        # 等待前端响应 / 超时
        return await asyncio.wait_for(future, timeout=30.0)
    except asyncio.TimeoutError:
        return ConfirmResponse(approved=False, message="确认超时（30s），自动拒绝")


# ─────────────────────────────────────────────────────────────────────────────
# 3. AgentPool 生命周期（FastAPI lifespan）
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app):
    """启动时创建 Pool + 共享 LLM，关闭时优雅停机。"""
    llm = ModelRegistry.create("qwen", model="qwen-plus", web_search=True, enable_thinking=False)
    agent_factory = make_agent_factory(llm)
    pool = AgentPool(
        llm_factory=lambda uid, sid: llm,
        agent_factory=agent_factory,
        config=AgentPoolConfig(
            max_agents=200,
            max_concurrent_runs=50,
            idle_ttl_seconds=300,
            sweep_interval_seconds=60,
        ),
    )
    await pool.start()
    app.state.pool = pool
    app.state.llm = llm
    logger.info("AgentPool 已就绪: %s", pool.get_stats())

    # 启动时为每个 Agent 连接 MCP（共享同一 LLM 子进程是安全的）
    # 注意：实际 MCP 连接延后到首次 acquire（per-user 隔离）
    try:
        yield
    finally:
        await pool.stop()
        logger.info("AgentPool 已关闭")


# ─────────────────────────────────────────────────────────────────────────────
# 4. SSE 事件序列化
# ─────────────────────────────────────────────────────────────────────────────


def _to_jsonable(obj: Any) -> Any:
    """把 dataclass / 对象转为可 JSON 序列化的 dict。"""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return str(obj)


def _event_type(evt: Any) -> str:
    return evt.__class__.__name__


# ─────────────────────────────────────────────────────────────────────────────
# 5. / 命令处理（与 multi_turn_chat.py:handle_command 一一对应）
# ─────────────────────────────────────────────────────────────────────────────
#
# 原始实现是 async def handle_command(agent, cmd) -> bool。
# 这里改为：每条 / 命令生成一段「CommandResult」SSE 事件流，
# 让前端按相同的事件协议渲染（避免两条不同的渲染路径）。


async def _exec_command(agent: Agent, cmd: str) -> AsyncIterator[dict]:
    """执行一个 / 命令，通过 SSE 事件流返回结果。"""
    try:
        if cmd == "/quit":
            yield {"event": "CommandResult", "data": json.dumps(
                {"type": "info", "text": "已断开连接（请关闭标签页）"}, ensure_ascii=False)}

        elif cmd == "/reset":
            await agent.reset()
            yield {"event": "CommandResult", "data": json.dumps(
                {"type": "info", "text": "对话已重置，上下文和计划已清空。"}, ensure_ascii=False)}

        elif cmd == "/history":
            messages = agent.history.all_messages
            session = agent.history.session
            lines = [
                f"=== 对话历史（共 {len(messages)} 条消息）===",
            ]
            if session:
                lines.append(f"会话 ID: {session.session_id}")
                lines.append(f"日志文件: {session.conversation_path}")
                lines.append(f"已记录消息: {session.message_count} 条 | 当前内存: {len(messages)} 条")
            lines.append("-" * 50)
            for msg in messages:
                role = msg.role.value.upper()
                raw = msg.content or ""
                content = raw[:500] + "..." if len(raw) > 500 else raw
                content = content.replace("\n", " ")
                lines.append(f"  [{role:<9}] {content}")
            lines.append("-" * 50)
            yield {"event": "CommandResult", "data": json.dumps(
                {"type": "text", "text": "\n".join(lines)}, ensure_ascii=False)}

        elif cmd == "/tools":
            active_tools = agent.tools.list_tools()
            dormant_tools = agent.tools.list_dormant_tools()
            lines = [f"=== 活跃工具（共 {len(active_tools)} 个）==="]
            builtin_factory = [*BUILTIN_TOOLS, create_structured_output_tool()]
            builtin_names = {w._tool_wrapper.name for w in builtin_factory}
            for tool_func in builtin_factory:
                w = tool_func._tool_wrapper
                if w.name in active_tools:
                    safe = " [S]" if w.is_safe else ""
                    lines.append(f"  {w.name:<30} {w.description[:40]}{safe}")
            meta_names = {"list_catalog", "search_tools", "activate_tools", "load_skill"}
            for name in meta_names:
                if name in active_tools:
                    wrapper = agent.tools.get_tool(name)
                    desc = wrapper.description[:40] if wrapper else ""
                    lines.append(f"  {name:<30} {desc}")
            for name in ("researcher", "coder", "reviewer"):
                if name in active_tools:
                    wrapper = agent.tools.get_tool(name)
                    desc = wrapper.description[:40] if wrapper else ""
                    lines.append(f"  {name + ' [SubAgent]':<30} {desc}")
            all_special = builtin_names | meta_names | {"researcher", "coder", "reviewer"}
            activated_mcp = [n for n in active_tools if n not in all_special]
            if activated_mcp:
                lines.append("-" * 50)
                lines.append("  已激活 MCP 工具")
                lines.append("-" * 50)
                for name in activated_mcp:
                    wrapper = agent.tools.get_tool(name)
                    desc = wrapper.description[:40] if wrapper else ""
                    lines.append(f"  {name:<30} {desc}")
            if dormant_tools:
                lines.append("-" * 50)
                lines.append(f"  休眠工具（共 {len(dormant_tools)} 个）")
                lines.append("-" * 50)
                grouped: dict[str, list] = {}
                for t in dormant_tools:
                    cat = t["category"] or "未分类"
                    grouped.setdefault(cat, []).append(t)
                for cat, items in grouped.items():
                    lines.append(f"  【{cat}】")
                    for item in items:
                        lines.append(f"    {item['name']:<32} {item['description'][:38]}")
            yield {"event": "CommandResult", "data": json.dumps(
                {"type": "text", "text": "\n".join(lines)}, ensure_ascii=False)}

        elif cmd == "/skills":
            skill_names = agent.skill_registry.list_names()
            if not skill_names:
                yield {"event": "CommandResult", "data": json.dumps(
                    {"type": "info", "text": "暂无可用技能。"}, ensure_ascii=False)}
            else:
                lines = [f"=== 可用技能（共 {len(skill_names)} 个）===", "-" * 50]
                for name in skill_names:
                    cfg = agent.skill_registry.get(name)
                    triggers = f"  [{', '.join(cfg.triggers)}]" if cfg.triggers else ""
                    lines.append(f"  {name:<30} {cfg.description}{triggers}")
                lines.append("-" * 50)
                lines.append("  LLM 会自动调用 load_skill 按需加载技能正文")
                yield {"event": "CommandResult", "data": json.dumps(
                    {"type": "text", "text": "\n".join(lines)}, ensure_ascii=False)}

        elif cmd == "/plan":
            mgr = agent._todo_manager
            if not mgr or not mgr.state.items:
                yield {"event": "CommandResult", "data": json.dumps(
                    {"type": "info", "text": "暂无会话计划。"}, ensure_ascii=False)}
            else:
                markers = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
                lines = [f"=== 当前会话计划（共 {len(mgr.state.items)} 个条目）===", "-" * 50]
                completed = 0
                for item in mgr.state.items:
                    marker = markers.get(item.status, "[ ]")
                    line = f"  {marker} {item.content}"
                    if item.status == "in_progress" and item.active_form:
                        line += f"  ({item.active_form})"
                    if item.status == "completed":
                        completed += 1
                    lines.append(line)
                lines.append(f"\n  ({completed}/{len(mgr.state.items)} 已完成)")
                if mgr.plan_file:
                    lines.append(f"  文件: {mgr.plan_file}")
                lines.append("-" * 50)
                yield {"event": "CommandResult", "data": json.dumps(
                    {"type": "text", "text": "\n".join(lines)}, ensure_ascii=False)}

        elif cmd == "/mode" or cmd.startswith("/mode "):
            if cmd == "/mode":
                mode = agent.mode
                mode_desc = {
                    AgentMode.TALK: "只读模式（仅允许安全操作）",
                    AgentMode.AUTO: "标准模式（不安全操作需确认）",
                    AgentMode.SUPERWORK: "全权限模式（无安全检查）",
                }
                lines = ["=== 操作模式 ===", "-" * 50]
                for m in AgentMode:
                    marker = " → 当前" if m == mode else ""
                    lines.append(f"  {m.value:<20} {mode_desc.get(m, '')}{marker}")
                lines.append("-" * 50)
                lines.append("  用法: /mode <talk|auto|superwork>")
                yield {"event": "CommandResult", "data": json.dumps(
                    {"type": "text", "text": "\n".join(lines)}, ensure_ascii=False)}
            else:
                new_mode = cmd.split(maxsplit=1)[1].strip()
                try:
                    agent.set_mode(new_mode)
                    yield {"event": "CommandResult", "data": json.dumps(
                        {"type": "info", "text": f"模式已切换为: {new_mode}"}, ensure_ascii=False)}
                except ValueError:
                    yield {"event": "CommandResult", "data": json.dumps(
                        {"type": "error", "text": f"无效模式: {new_mode}（可选: talk, auto, superwork）"},
                        ensure_ascii=False)}

        elif cmd == "/prompt":
            agent._build_system_prompt()
            system_msg = agent.history.all_messages[0] if agent.history.all_messages else None
            content = system_msg.content if system_msg else ""
            if not content:
                yield {"event": "CommandResult", "data": json.dumps(
                    {"type": "info", "text": "当前系统提示词为空。"}, ensure_ascii=False)}
            else:
                line_count = content.count("\n") + 1
                char_count = len(content)
                file_list = []
                if agent.prompt_builder:
                    file_list = agent.prompt_builder.list_files()
                header = f"=== 当前系统提示词（{line_count} 行, {char_count} 字符）===\n"
                if file_list:
                    header += f"来源: {', '.join(f['file'] for f in file_list)}\n"
                header += "-" * 50 + "\n"
                yield {"event": "CommandResult", "data": json.dumps(
                    {"type": "text", "text": header + content + "\n" + "-" * 50},
                    ensure_ascii=False)}

        elif cmd == "/compact":
            if not agent._history.compact_enabled:
                yield {"event": "CommandResult", "data": json.dumps(
                    {"type": "info", "text": "上下文压缩未启用。"}, ensure_ascii=False)}
            else:
                original_count = len(agent.history._messages)
                compacted, summary = await agent._history.manual_compact()
                agent.history.replace_all(compacted)
                agent._history._log_compaction(compacted)
                preview = summary[:500] + "..." if len(summary) > 500 else summary
                yield {"event": "CommandResult", "data": json.dumps(
                    {"type": "text",
                     "text": f"=== 手动压缩完成（{original_count} → {len(compacted)} 条消息）===\n{preview}"},
                    ensure_ascii=False)}

        elif cmd == "/save":
            agent.save_session()
            if agent.session:
                yield {"event": "CommandResult", "data": json.dumps(
                    {"type": "info",
                     "text": f"会话已保存\nID: {agent.session.session_id}\n路径: {agent.session.dir_path}\n消息数: {agent.session.message_count}"},
                    ensure_ascii=False)}
            else:
                yield {"event": "CommandResult", "data": json.dumps(
                    {"type": "info", "text": "会话功能未启用。"}, ensure_ascii=False)}

        elif cmd == "/sessions":
            from agent_framework.agent.session import Session as SessionClass
            base_dir = Path(agent.session.base_dir) if agent.session else Path(".sessions")
            sessions = SessionClass.list_sessions(base_dir)
            if not sessions:
                yield {"event": "CommandResult", "data": json.dumps(
                    {"type": "info", "text": "暂无历史会话。"}, ensure_ascii=False)}
            else:
                lines = [f"=== 历史会话（共 {len(sessions)} 个）===", "-" * 50]
                current_id = agent.session.session_id if agent.session else None
                for s in sessions:
                    sid = s.get("session_id", "?")
                    model = s.get("model", "")
                    msg_count = s.get("message_count", 0)
                    updated = s.get("updated_at", 0)
                    time_str = datetime.fromtimestamp(updated).strftime("%m-%d %H:%M") if updated else "?"
                    marker = " ← 当前" if sid == current_id else ""
                    model_str = f" ({model})" if model else ""
                    lines.append(f"  {sid}  {time_str}  {msg_count} 条消息{model_str}{marker}")
                lines.append("-" * 50)
                yield {"event": "CommandResult", "data": json.dumps(
                    {"type": "text", "text": "\n".join(lines)}, ensure_ascii=False)}

        elif cmd == "/new":
            agent.new_session()
            if agent.session:
                yield {"event": "CommandResult", "data": json.dumps(
                    {"type": "info", "text": f"新会话已创建\nID: {agent.session.session_id}"},
                    ensure_ascii=False)}

        elif cmd.startswith("/load "):
            session_id = cmd.split(maxsplit=1)[1].strip()
            try:
                msg_count = agent.load_session(session_id)
                yield {"event": "CommandResult", "data": json.dumps(
                    {"type": "info", "text": f"会话已加载\nID: {session_id}\n消息数: {msg_count}"},
                    ensure_ascii=False)}
            except FileNotFoundError:
                yield {"event": "CommandResult", "data": json.dumps(
                    {"type": "error", "text": f"会话不存在: {session_id}"}, ensure_ascii=False)}
            except Exception as e:
                yield {"event": "CommandResult", "data": json.dumps(
                    {"type": "error", "text": f"加载失败: {e}"}, ensure_ascii=False)}

        elif cmd == "/help":
            yield {"event": "CommandResult", "data": json.dumps(
                {"type": "text", "text": "\n".join([
                    "可用命令:",
                    "  /history   — 查看对话历史",
                    "  /reset     — 重置对话（清空上下文）",
                    "  /tools     — 查看可用工具",
                    "  /skills    — 查看可用技能",
                    "  /plan      — 查看当前会话计划",
                    "  /mode      — 查看/切换操作模式（talk/auto/superwork）",
                    "  /prompt    — 查看当前系统提示词",
                    "  /compact   — 手动压缩对话历史",
                    "  /save      — 保存当前会话",
                    "  /sessions  — 查看所有会话",
                    "  /new       — 新建会话（自动保存当前）",
                    "  /load <id> — 加载历史会话",
                    "  /help      — 显示帮助",
                ])}, ensure_ascii=False)}

        else:
            yield {"event": "CommandResult", "data": json.dumps(
                {"type": "error", "text": f"未知命令: {cmd}  (输入 /help 查看帮助)"},
                ensure_ascii=False)}

    except Exception as e:
        logger.exception("命令执行异常: %s", e)
        yield {"event": "CommandResult", "data": json.dumps(
            {"type": "error", "text": f"命令执行异常: {e}"}, ensure_ascii=False)}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Agent 事件流（带流式检测客户端断开 + ToolConfirmRequired 触发弹窗）
# ─────────────────────────────────────────────────────────────────────────────

# MCP 连接是 lazy 的：每个 user 第一次 acquire 时由第一个到达的请求触发连接。
# 用 per-user 锁防并发：同一 user 多请求同时到达时只连接一次。
_mcp_locks: dict[str, asyncio.Lock] = {}


def _lock_for_user(user_id: str) -> asyncio.Lock:
    if user_id not in _mcp_locks:
        _mcp_locks[user_id] = asyncio.Lock()
    return _mcp_locks[user_id]


async def _ensure_mcp(agent) -> None:
    """若该 Agent 尚未连接 MCP，则连接（同一 user 多次调用只连一次）。"""
    if agent._mcp_manager is not None:
        return
    async with _lock_for_user(agent.session.session_id if agent.session else "default"):
        if agent._mcp_manager is None:
            await agent.connect_mcp()


async def _stream_agent(
    user_id: str, session_id: str, user_input: str, request
) -> AsyncIterator[dict]:
    """从 AgentPool 取出 Agent，逐事件 yield 为 SSE 消息。"""
    pool: AgentPool = request.app.state.pool

    try:
        async with pool.acquire(user_id, session_id) as handle:
            agent = handle.agent
            await _ensure_mcp(agent)
            try:
                async for evt in agent.run(user_input):
                    if await request.is_disconnected():
                        logger.info("客户端断开 (user=%s)", user_id)
                        return

                    # 危险工具确认事件：转发给前端，前端弹窗 + 调 /confirm
                    if isinstance(evt, ToolConfirmRequired):
                        yield {
                            "event": "ConfirmationRequest",
                            "data": json.dumps({
                                "tool_name": evt.tool_name,
                                "tool_call_id": evt.tool_call_id,
                                "arguments": evt.arguments,
                            }, ensure_ascii=False),
                        }
                        continue

                    yield {
                        "event": _event_type(evt),
                        "data": json.dumps(_to_jsonable(evt), ensure_ascii=False),
                    }
                    if isinstance(evt, (AgentDone, AgentError)):
                        return
            except Exception as e:
                logger.exception("Agent run 异常 (user=%s): %s", user_id, e)
                yield {
                    "event": "AgentError",
                    "data": json.dumps({"error_type": "exception", "message": str(e)},
                                       ensure_ascii=False),
                }
    except Exception as e:
        logger.exception("acquire 失败 (user=%s): %s", user_id, e)
        yield {
            "event": "ServerError",
            "data": json.dumps({"message": str(e)}, ensure_ascii=False),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 7. FastAPI 应用 + 端点
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, Header, HTTPException, Request  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from sse_starlette.sse import EventSourceResponse  # noqa: E402

app = FastAPI(title="agent-framework multi-user chat", lifespan=lifespan)


@app.post("/chat")
async def chat(
    request: Request,
    x_user_id: str = Header(..., alias="X-User-Id"),
    x_session_id: str = Header("default", alias="X-Session-Id"),
):
    """统一入口：消息 + / 命令。

    Headers:
        X-User-Id: 必填。
        X-Session-Id: 可选。默认 "default"。
    Body:
        {"input": "用户输入 / 或 / 命令"}

    Returns:
        SSE 事件流：AgentEvent（TextDelta/ToolCallStart/...）+ CommandResult
    """
    body = await request.json()
    user_input = body.get("input", "").strip()
    if not user_input:
        raise HTTPException(400, "input 不能为空")

    # / 命令走特殊路径（不进 LLM）
    if user_input.startswith("/"):
        async def _cmd_stream() -> AsyncIterator[dict]:
            pool: AgentPool = request.app.state.pool
            try:
                async with pool.acquire(x_user_id, x_session_id) as h:
                    await _ensure_mcp(h.agent)
                    async for sse in _exec_command(h.agent, user_input):
                        if await request.is_disconnected():
                            return
                        yield sse
            except Exception as e:
                yield {"event": "ServerError", "data": json.dumps(
                    {"message": str(e)}, ensure_ascii=False)}
        return EventSourceResponse(_cmd_stream(), ping=15)

    # 普通消息走 Agent.run
    return EventSourceResponse(
        _stream_agent(x_user_id, x_session_id, user_input, request),
        ping=15,
    )


@app.post("/confirm")
async def confirm(
    request: Request,
    x_user_id: str = Header(..., alias="X-User-Id"),
):
    """前端响应 ToolConfirmRequired 时调用。

    Body: {"approved": true|false, "message": "可选，自定义指示"}
    """
    body = await request.json()
    approved = bool(body.get("approved", False))
    message = str(body.get("message", ""))
    future = _confirmations.get(x_user_id)
    if future is None or future.done():
        raise HTTPException(404, "没有待确认的工具调用（可能已超时）")
    future.set_result(ConfirmResponse(approved=approved, message=message))
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    return app.state.pool.get_stats()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回单页聊天 UI（HTML 内嵌）"""
    return HTML_PAGE


# ─────────────────────────────────────────────────────────────────────────────
# 8. 前端 HTML（vanilla JS + SSE + EventSource）
# ─────────────────────────────────────────────────────────────────────────────
#
# 设计要点：
#   - 用户ID 用 localStorage 持久化（标签页之间隔离）
#   - 消息流通过 EventSource 接收，渲染到主区域
#   - 工具确认弹窗：收到 ConfirmationRequest 事件 → 弹模态框 → POST /confirm
#   - 命令按钮 / 自动补全
#   - 简易 Markdown 渲染（保留换行 + 代码块）

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>多用户 Agent 聊天 — AgentPool</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
    background: #f5f5f7; height: 100vh; display: flex; flex-direction: column;
  }
  header {
    background: #1d1d1f; color: #fff; padding: 10px 16px;
    display: flex; align-items: center; gap: 12px; font-size: 14px;
  }
  header strong { color: #0af; }
  header input {
    background: #2c2c2e; color: #fff; border: 1px solid #3a3a3c;
    padding: 5px 10px; border-radius: 6px; font-size: 13px; width: 140px;
  }
  header .stat {
    margin-left: auto; font-size: 12px; color: #aaa;
  }
  header .stat span { color: #4cd964; }
  #chat {
    flex: 1; overflow-y: auto; padding: 16px;
    background: #fff; margin: 8px; border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }
  .msg { margin: 8px 0; line-height: 1.6; white-space: pre-wrap; word-wrap: break-word; }
  .msg.user { color: #1d1d1f; }
  .msg.user::before { content: "🧑 "; }
  .msg.assistant { color: #1d1d1f; }
  .msg.assistant::before { content: "🤖 "; }
  .msg.thinking { color: #888; font-style: italic; }
  .msg.thinking::before { content: "💭 "; }
  .msg.tool { color: #af52de; font-family: "SF Mono", Consolas, monospace; font-size: 13px; }
  .msg.tool::before { content: "🔧 "; }
  .msg.tool-result { color: #34c759; font-family: "SF Mono", Consolas, monospace; font-size: 13px; }
  .msg.tool-result::before { content: "✅ "; }
  .msg.tool-error { color: #ff3b30; font-family: "SF Mono", Consolas, monospace; font-size: 13px; }
  .msg.tool-error::before { content: "❌ "; }
  .msg.subagent { color: #5856d6; font-size: 13px; margin-left: 24px; }
  .msg.subagent::before { content: "👶 "; }
  .msg.subagent-done { color: #5856d6; font-size: 13px; margin-left: 24px; font-weight: bold; }
  .msg.subagent-done::before { content: "🏁 "; }
  .msg.error { color: #ff3b30; }
  .msg.error::before { content: "⚠️ "; }
  .msg.info { color: #0af; }
  .msg.info::before { content: "ℹ️ "; }
  .msg.compact { color: #ff9500; }
  .msg.compact::before { content: "📦 "; }
  .msg.system { color: #888; font-size: 12px; }
  .msg.system::before { content: "⚙️ "; }
  .msg code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 12px; }
  .msg pre { background: #1d1d1f; color: #f5f5f7; padding: 8px; border-radius: 4px; overflow-x: auto; font-size: 12px; }
  .divider { text-align: center; color: #aaa; margin: 12px 0; font-size: 12px; }
  .divider::before { content: "─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─"; }

  footer {
    padding: 8px 16px 16px; background: #f5f5f7;
    display: flex; flex-direction: column; gap: 6px;
  }
  .input-row { display: flex; gap: 8px; }
  #input {
    flex: 1; padding: 10px 12px; border: 1px solid #ccc; border-radius: 6px;
    font-size: 14px; font-family: inherit; resize: none; min-height: 40px; max-height: 120px;
  }
  #send, #commands {
    padding: 10px 16px; border: none; border-radius: 6px; cursor: pointer;
    font-size: 14px; font-weight: 500;
  }
  #send { background: #007aff; color: #fff; }
  #send:disabled { background: #aaa; cursor: not-allowed; }
  #commands { background: #fff; border: 1px solid #ccc; color: #333; }
  .suggest {
    display: flex; flex-wrap: wrap; gap: 6px; font-size: 12px;
  }
  .suggest button {
    padding: 3px 8px; border: 1px solid #ddd; background: #fff;
    border-radius: 12px; cursor: pointer; font-size: 12px; color: #555;
  }
  .suggest button:hover { background: #f0f0f0; }

  /* 确认弹窗 */
  .modal-bg {
    position: fixed; inset: 0; background: rgba(0,0,0,0.4);
    display: none; align-items: center; justify-content: center; z-index: 100;
  }
  .modal-bg.show { display: flex; }
  .modal {
    background: #fff; border-radius: 8px; padding: 20px; min-width: 400px; max-width: 600px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
  }
  .modal h3 { margin-bottom: 12px; color: #ff3b30; }
  .modal pre { background: #f5f5f7; padding: 8px; border-radius: 4px; font-size: 12px; max-height: 200px; overflow: auto; }
  .modal .btns { display: flex; gap: 8px; margin-top: 16px; }
  .modal button { padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; }
  .modal .approve { background: #34c759; color: #fff; }
  .modal .reject { background: #ff3b30; color: #fff; }
  .modal .cancel { background: #ddd; color: #333; }
  .modal input { width: 100%; padding: 6px 8px; border: 1px solid #ccc; border-radius: 4px; margin-top: 8px; font-size: 13px; }
</style>
</head>
<body>
<header>
  <strong>多用户 Agent 聊天</strong>
  <span style="color:#888;font-size:12px">|</span>
  <span>用户ID：</span>
  <input id="userId" placeholder="user_id" value="">
  <span>会话ID：</span>
  <input id="sessionId" placeholder="session_id" value="default">
  <button id="commands" onclick="toggleCmds()">/ 命令</button>
  <div class="stat">Pool: <span id="stat">--</span></div>
</header>

<div id="chat"></div>

<footer>
  <div class="suggest" id="suggest" style="display:none">
    <span style="color:#888;margin-right:4px">命令：</span>
    <button onclick="sendCmd('/history')">/history</button>
    <button onclick="sendCmd('/reset')">/reset</button>
    <button onclick="sendCmd('/tools')">/tools</button>
    <button onclick="sendCmd('/skills')">/skills</button>
    <button onclick="sendCmd('/plan')">/plan</button>
    <button onclick="sendCmd('/mode')">/mode</button>
    <button onclick="sendCmd('/prompt')">/prompt</button>
    <button onclick="sendCmd('/compact')">/compact</button>
    <button onclick="sendCmd('/save')">/save</button>
    <button onclick="sendCmd('/sessions')">/sessions</button>
    <button onclick="sendCmd('/new')">/new</button>
    <button onclick="sendCmd('/help')">/help</button>
  </div>
  <div class="input-row">
    <textarea id="input" placeholder="输入消息或 / 命令（Enter 发送，Shift+Enter 换行）" rows="1"></textarea>
    <button id="send" onclick="sendMsg()">发送</button>
  </div>
</footer>

<!-- 确认弹窗 -->
<div class="modal-bg" id="modalBg">
  <div class="modal">
    <h3>🔴 危险工具确认</h3>
    <div>工具：<strong id="mTool"></strong></div>
    <pre id="mArgs"></pre>
    <input id="mMessage" placeholder="拒绝时可填自定义指示（可选）">
    <div class="btns">
      <button class="approve" onclick="respondConfirm(true)">同意执行</button>
      <button class="reject" onclick="respondConfirm(false)">拒绝</button>
      <button class="cancel" onclick="respondConfirm(false,true)">30s 后自动拒绝</button>
    </div>
  </div>
</div>

<script>
const $ = (id) => document.getElementById(id);
const chat = $("chat");
let busy = false;
let currentConfirmTool = null;

// 持久化用户ID
const savedUid = localStorage.getItem("chat_user_id") || "user_" + Math.random().toString(36).slice(2, 8);
$("userId").value = savedUid;
$("userId").addEventListener("change", e => {
  localStorage.setItem("chat_user_id", e.target.value);
  location.reload();  // 切换用户时刷新页面
});

function addMsg(role, text, cls = "") {
  const div = document.createElement("div");
  div.className = "msg " + (cls || role);
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function addDivider() {
  const d = document.createElement("div");
  d.className = "divider";
  chat.appendChild(d);
}

function toggleCmds() {
  const s = $("suggest");
  s.style.display = s.style.display === "none" ? "flex" : "none";
}

function sendCmd(cmd) {
  $("input").value = cmd;
  sendMsg();
}

async function sendMsg() {
  if (busy) return;
  const userId = $("userId").value.trim();
  const sessionId = $("sessionId").value.trim() || "default";
  const input = $("input").value.trim();
  if (!userId) { alert("请填写用户ID"); return; }
  if (!input) return;

  busy = true;
  $("send").disabled = true;
  $("send").textContent = "...";
  addDivider();
  addMsg("user", input);
  $("input").value = "";

  // 创建流式响应的视觉占位
  const assistantBuf = [];
  let thinkingBuf = null;
  let inSub = null;

  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: {
        "X-User-Id": userId,
        "X-Session-Id": sessionId,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ input }),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // SSE 格式：event: X\ndata: Y\n\n
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        const lines = part.split("\n");
        let ev = null, data = null;
        for (const line of lines) {
          if (line.startsWith("event: ")) ev = line.slice(7).trim();
          else if (line.startsWith("data: ")) data = line.slice(6);
        }
        if (!ev || !data) continue;
        handleEvent(ev, JSON.parse(data), { assistantBuf, thinkingBuf, inSub });
      }
    }
  } catch (e) {
    addMsg("error", "请求失败: " + e.message);
  } finally {
    busy = false;
    $("send").disabled = false;
    $("send").textContent = "发送";
  }
}

function handleEvent(ev, data, ctx) {
  if (ev === "TextDelta") {
    ctx.assistantBuf.push(data.text);
    addMsg("assistant", data.text, "assistant");
  } else if (ev === "ReasoningDelta") {
    addMsg("thinking", data.text, "thinking");
  } else if (ev === "ToolCallStart") {
    addMsg("tool", `→ ${data.tool_name}(${data.arguments.slice(0,80)})`);
  } else if (ev === "ToolResult") {
    const cls = data.is_error ? "tool-error" : "tool-result";
    const out = data.output.length > 200 ? data.output.slice(0, 200) + "..." : data.output;
    addMsg(cls, `${data.tool_name}: ${out}`);
  } else if (ev === "SubAgentEvent") {
    const inner = data.event;
    const name = data.subagent_name;
    if (inner.type === "TextDelta") {
      addMsg("subagent", `[${name}] ${inner.text}`);
    } else if (inner.type === "ReasoningDelta") {
      addMsg("subagent", `[${name} 💭] ${inner.text}`);
    } else if (inner.type === "ToolCallStart") {
      addMsg("subagent", `[${name}] → ${inner.tool_name}`);
    } else if (inner.type === "ToolResult") {
      addMsg("subagent", `[${name} ✅] ${inner.tool_name}`);
    }
  } else if (ev === "SubAgentDone") {
    addMsg("subagent-done", `[${data.subagent_name}] 完成 turns=${data.turn_count}`);
  } else if (ev === "AgentDone") {
    addMsg("system", `[turns=${data.turn_count}, tokens=${data.total_usage.total_tokens}]`);
  } else if (ev === "AgentError") {
    addMsg("error", data.message);
  } else if (ev === "HistoryCompacted") {
    addMsg("compact", `对话历史已压缩: ${data.original_count} → ${data.compacted_count} 条`);
  } else if (ev === "SessionLoaded") {
    addMsg("system", `会话已加载: ${data.session_id}（${data.message_count} 条）`);
  } else if (ev === "CommandResult") {
    if (data.type === "info") addMsg("info", data.text);
    else if (data.type === "error") addMsg("error", data.text);
    else addMsg("system", data.text);
  } else if (ev === "ConfirmationRequest") {
    showConfirm(data);
  } else if (ev === "ServerError") {
    addMsg("error", "服务端错误: " + data.message);
  }
}

function showConfirm(data) {
  currentConfirmTool = data;
  $("mTool").textContent = data.tool_name;
  $("mArgs").textContent = data.arguments;
  $("mMessage").value = "";
  $("modalBg").classList.add("show");
}

async function respondConfirm(approved, cancel = false) {
  if (!currentConfirmTool) return;
  const message = $("mMessage").value;
  $("modalBg").classList.remove("show");
  const userId = $("userId").value.trim();
  if (cancel) {
    addMsg("info", "已忽略本次确认（30s 后会自动拒绝）");
    currentConfirmTool = null;
    return;
  }
  addMsg("tool", approved ? "✅ 用户同意执行" : `❌ 用户拒绝${message ? "：" + message : ""}`);
  try {
    await fetch("/confirm", {
      method: "POST",
      headers: { "X-User-Id": userId, "Content-Type": "application/json" },
      body: JSON.stringify({ approved, message }),
    });
  } catch (e) {
    addMsg("error", "确认请求失败: " + e.message);
  }
  currentConfirmTool = null;
}

// 输入框：Enter 发送，Shift+Enter 换行
$("input").addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMsg();
  }
});

// 定期拉取池状态
setInterval(async () => {
  try {
    const r = await fetch("/stats");
    const s = await r.json();
    $("stat").textContent = `active=${s.active_entries}/${s.max_agents}, reused=${s.reused}, evicted=${(s.evicted_lru||0)+(s.evicted_idle||0)}`;
  } catch {}
}, 3000);

// 初始欢迎
addMsg("system", "✨ AgentPool 多用户聊天已就绪。每个 user_id 自动得到独立 Agent 实例。");
addMsg("info", "提示：多个浏览器标签页使用不同 user_id 即可看到多用户隔离效果。");
addMsg("system", "点击右上角「/ 命令」查看全部命令。");
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# 9. 启动入口
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  多用户多轮对话 — AgentPool + FastAPI")
    print("  访问: http://localhost:8000")
    print("  多个浏览器标签页使用不同 user_id 即可演示多用户隔离")
    print("=" * 60)
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
    )
