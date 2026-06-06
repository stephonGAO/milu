"""milu 内置 Web 服务 —— FastAPI 应用工厂 + 全部 HTTP 端点。

把 AgentPool（多用户并发对话）+ ScheduleEngine（嵌入式定时任务）+ 全功能演示
前端，沉淀为包内置的正规服务（CLI 子命令 `milu serve` 一键启动）。

核心设计：
- **多用户隔离**：复用 AgentPool，按 (X-User-Id, X-Session-Id) 派生独立 Agent。
- **运行时切换厂商/模型**：per-(user,session) 偏好 + LLM 缓存 + `pool.remove()` 驱逐重建。
- **确认流队列桥接**：危险工具确认时 `on_confirm` 会**阻塞** agent 生成器，
  因此用后台任务跑 `agent.run()` 喂 `asyncio.Queue`，`on_confirm` 在 await 前把
  「确认请求」推进队列，SSE 协程独立排空队列——保证弹窗在阻塞期间即可显示。
- **库纯净/可选依赖**：fastapi/uvicorn/sse-starlette 为 `[serve]` 可选依赖，本模块
  仅在 `create_app()`/`run_server()` 被调用时才导入它们。

程序化用法：
    from milu.serving.web import create_app
    import uvicorn
    uvicorn.run(create_app(provider="qwen"), host="127.0.0.1", port=8000)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

# fastapi / sse-starlette 为 [serve] 可选依赖。本模块（app.py）只在
# create_app()/run_server() 被调用时才经 milu.serving.web.__init__ 延迟导入，
# 故在此模块顶层导入它们是安全的——未安装时 import 错误会在调用入口被捕获。
# 注意：必须在模块顶层（而非函数内）导入这些类型，否则配合 `from __future__
# import annotations` 时 FastAPI 无法从模块全局解析 `Request` 等字符串注解。
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from milu._env import ensure_dotenv_loaded
from milu.agent import Agent
from milu.agent.config import AgentMode
from milu.agent.events import (
    AgentDone,
    AgentError,
    ConfirmResponse,
    SubAgentDone,
    SubAgentEvent,
    ToolConfirmRequired,
)
from milu.cli.config import DEFAULT_MODELS, DEFAULT_PROVIDER, env_key_name
from milu.llm.providers import ModelRegistry
from milu.resources import default_session_dir, user_data_dir
from milu.serving import AgentPool
from milu.tools.builtin import BUILTIN_TOOLS, create_structured_output_tool

logger = logging.getLogger("milu.serving.web")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# 危险工具确认等待超时（秒）：超时自动拒绝，避免用户关页后 Agent 永远卡住
CONFIRM_TIMEOUT = 120.0


# ── 服务启动默认值 ────────────────────────────────────────

@dataclass
class ServerOptions:
    """服务启动时的默认设置（每个 (user,session) 可经 /api/settings 覆盖）。"""
    provider: str = DEFAULT_PROVIDER
    model: str | None = None
    mode: str = "auto"
    session_enabled: bool = True
    web_search: bool = True
    enable_thinking: bool = False
    use_mcp: bool = True
    use_subagents: bool = True
    use_scheduler: bool = True
    selfguard_enabled: bool = True
    show_subagent_events: bool = False   # 是否向前端转发子代理内部事件（默认隐藏）

    def resolved_model(self) -> str | None:
        return self.model or DEFAULT_MODELS.get(self.provider)


# ── 序列化辅助 ────────────────────────────────────────────

def _to_jsonable(obj: Any) -> Any:
    """把 dataclass / 嵌套对象转为可 JSON 序列化的结构。"""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return str(obj)


def _sse(event: str, data: Any) -> dict:
    """构造一条 SSE 消息（event + JSON data）。"""
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


# ── LLM 解析与缓存 ────────────────────────────────────────

def _get_llm(state, provider: str, model: str | None,
             web_search: bool, enable_thinking: bool):
    """按 (provider, model, web_search, thinking) 取/建 LLM（跨用户共享同型号实例）。

    AsyncOpenAI 协程安全，同一配置的 LLM 可被所有用户安全共享。鉴权懒加载，
    构造时不触网；Key 缺失会在实际 run() 时报 AuthenticationError（前端经
    /api/providers 展示 Key 状态供用户预判）。
    """
    resolved_model = model or DEFAULT_MODELS.get(provider)
    key = (provider, resolved_model, web_search, enable_thinking)
    cache = state.llm_cache
    if key not in cache:
        cache[key] = ModelRegistry.create(
            provider,
            model=resolved_model,
            web_search=web_search,
            enable_thinking=enable_thinking,
        )
    return cache[key]


def _effective_settings(state, user_id: str, session_id: str) -> dict:
    """(user,session) 的生效设置：偏好覆盖服务默认。"""
    d = state.options
    pref = state.prefs.get((user_id, session_id), {})
    provider = pref.get("provider", d.provider)
    return {
        "provider": provider,
        "model": pref.get("model") or (d.model if provider == d.provider else None)
                 or DEFAULT_MODELS.get(provider),
        "mode": pref.get("mode", d.mode),
        "web_search": pref.get("web_search", d.web_search),
        "enable_thinking": pref.get("enable_thinking", d.enable_thinking),
        "show_subagent_events": pref.get("show_subagent_events", d.show_subagent_events),
        "session_enabled": d.session_enabled,
    }


# ── 危险工具确认（队列桥接）──────────────────────────────

async def _confirm_unsafe(state, user_id: str, session_id: str,
                          tool_name: str, args_str: str) -> ConfirmResponse:
    """on_confirm 回调：把确认请求推进当前请求的队列（驱动前端弹窗），再阻塞等响应。

    被 agent.run() 在后台任务里调用（阻塞该生成器）；同请求的 SSE 协程独立排空
    队列，故「确认请求」事件能在阻塞期间即时送达前端。
    """
    key = (user_id, session_id)
    loop = asyncio.get_running_loop()
    future: asyncio.Future[ConfirmResponse] = loop.create_future()

    old = state.confirmations.get(key)
    if old is not None and not old.done():
        old.set_result(ConfirmResponse(approved=False, message="被新的确认请求替代"))
    state.confirmations[key] = future

    queue = state.queues.get(key)
    if queue is not None:
        await queue.put(("event", _sse("ConfirmationRequest", {
            "tool_name": tool_name,
            "arguments": args_str,
        })))

    try:
        return await asyncio.wait_for(future, timeout=CONFIRM_TIMEOUT)
    except asyncio.TimeoutError:
        return ConfirmResponse(
            approved=False, message=f"确认超时（{int(CONFIRM_TIMEOUT)}s），自动拒绝")


# ── Agent 工厂 ────────────────────────────────────────────

def _make_agent_factory(state, shared_mcp):
    """自定义 agent_factory：读 per-(user,session) 偏好 → 取缓存 LLM → 构造 Agent。

    自带「全配默认」之上叠加服务层关注点：会话确定性 id（淘汰/重启可恢复历史）、
    memory/schedule_user 按 user_id 派生隔离、共享 MCP、确认回调接 per-请求队列。
    """
    def factory(user_id: str, session_id: str, _llm):
        eff = _effective_settings(state, user_id, session_id)
        llm = _get_llm(state, eff["provider"], eff["model"],
                       eff["web_search"], eff["enable_thinking"])
        d = state.options
        return Agent(
            llm=llm,
            tools=[*BUILTIN_TOOLS, create_structured_output_tool()],
            mode=eff["mode"],
            session_enabled=d.session_enabled,
            session_id=AgentPool._derive_session_id(user_id, session_id),
            memory=user_id,                 # 多用户长期记忆，按 user_id 隔离
            schedule_user=user_id,          # 定时任务按 user_id 隔离
            subagents=None if d.use_subagents else [],
            mcp_manager=shared_mcp,
            on_confirm=lambda t, a: _confirm_unsafe(state, user_id, session_id, t, a),
        )
    return factory


# ── 应用工厂 ──────────────────────────────────────────────

def create_app(
    *,
    provider: str | None = None,
    model: str | None = None,
    mode: str = "auto",
    session_enabled: bool = True,
    web_search: bool = True,
    enable_thinking: bool = False,
    use_mcp: bool = True,
    use_subagents: bool = True,
    use_scheduler: bool = True,
    selfguard_enabled: bool = True,
    show_subagent_events: bool = False,
):
    """构造 FastAPI 应用（含 AgentPool + 可选嵌入式调度引擎 + 全部端点）。"""
    options = ServerOptions(
        provider=provider or DEFAULT_PROVIDER,
        model=model,
        mode=mode,
        session_enabled=session_enabled,
        web_search=web_search,
        enable_thinking=enable_thinking,
        use_mcp=use_mcp,
        use_subagents=use_subagents,
        use_scheduler=use_scheduler,
        selfguard_enabled=selfguard_enabled,
        show_subagent_events=show_subagent_events,
    )

    @asynccontextmanager
    async def lifespan(app):
        ensure_dotenv_loaded()
        from milu.config import load_config
        from milu.tools._selfguard import set_enabled as _set_selfguard

        _set_selfguard(options.selfguard_enabled)

        st = app.state
        st.options = options
        st.prefs = {}                  # (user,session) -> 偏好覆盖
        st.llm_cache = {}              # (provider,model,...) -> BaseLLM
        st.confirmations = {}          # (user,session) -> asyncio.Future
        st.queues = {}                 # (user,session) -> asyncio.Queue（仅请求期间存在）

        # 预热默认 LLM（也作为 llm_factory 占位返回值；真实 LLM 由 agent_factory 解析）
        default_llm = _get_llm(st, options.provider, options.model,
                               options.web_search, options.enable_thinking)

        # 共享 MCP（整池一组子进程，失败降级为无 MCP）
        shared_mcp = None
        if options.use_mcp:
            try:
                from milu.tools.mcp.config import MCPServerConfig
                from milu.tools.mcp.manager import MCPManager
                configs = MCPServerConfig.load_file(None)
                if configs:
                    shared_mcp = MCPManager(configs)
                    await shared_mcp.connect_all()
                    logger.info("共享 MCP 已连接：%d 个工具", len(shared_mcp.get_tools()))
            except Exception as e:
                logger.warning("共享 MCP 连接失败，无 MCP 运行：%s", e)
                shared_mcp = None
        st.shared_mcp = shared_mcp

        mc = load_config()
        pool = AgentPool(
            llm_factory=lambda uid, sid: default_llm,
            agent_factory=_make_agent_factory(st, shared_mcp),
            config=mc.to_pool_config(),
            agent_config=mc.to_agent_config(),
        )
        await pool.start()
        st.pool = pool

        # 定时任务存储始终可用（REST 管理）；引擎按需嵌入
        from milu.scheduler import ScheduleEngine, ScheduleStore, SchedulerLock
        from milu.scheduler.engine import SchedulerConfig
        st.store = ScheduleStore(user_data_dir())
        st.engine = None
        sched_task = None
        sched_lock = None
        st.recent_results = []  # 内存最近结果（on_result 推送，便于前端即时感知）
        if options.use_scheduler:
            async def _on_result(task, result: str) -> None:
                st.recent_results.append({
                    "task": task.name, "user_id": task.user_id,
                    "time": datetime.now().isoformat(),
                    "result": result[:200],
                })
                st.recent_results[:] = st.recent_results[-100:]

            st.engine = ScheduleEngine(
                st.store,
                log_dir=user_data_dir() / "scheduler_logs",
                config=SchedulerConfig(notify=False),  # 服务端无桌面，关弹窗
                agent_pool=pool,
                on_result=_on_result,
            )
            # 单实例锁（与 daemon / chat 嵌入共用，多引擎并存会重复执行任务）：
            # start_with_lock 抢到即运行，被占用则等待、持锁者退出后自动接管
            sched_lock = SchedulerLock(user_data_dir())
            sched_task = st.engine.start_background(sched_lock)

        logger.info("milu Web 服务就绪：provider=%s model=%s mode=%s",
                    options.provider, options.resolved_model(), options.mode)
        try:
            yield
        finally:
            if st.engine is not None:
                st.engine.stop()
            if sched_task is not None:
                sched_task.cancel()
                try:
                    await sched_task
                except asyncio.CancelledError:
                    pass
            if sched_lock is not None:
                sched_lock.release()
            await pool.stop()
            if shared_mcp is not None:
                try:
                    await shared_mcp.disconnect_all()
                except Exception:
                    pass
            logger.info("milu Web 服务已关闭")

    app = FastAPI(title="milu Web 服务", lifespan=lifespan)

    # ── 页面 ──────────────────────────────────────────────

    @app.get("/")
    async def index():
        # 演示前端随版本迭代；禁用缓存，避免浏览器用旧 HTML（改了前端却看不到变化）
        return FileResponse(
            STATIC_DIR / "index.html",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ── 对话（SSE，队列桥接确认流）────────────────────────

    async def _stream_chat(app, user_id, session_id, user_input, request) -> AsyncIterator[dict]:
        st = app.state
        pool: AgentPool = st.pool
        key = (user_id, session_id)
        try:
            async with pool.acquire(user_id, session_id) as handle:
                agent = handle.agent
                queue: asyncio.Queue = asyncio.Queue()
                st.queues[key] = queue

                # 子代理事件显示：按 per-(user,session) 偏好取（前端可运行时切换），
                # 未设置则回退到服务启动默认。每次对话请求重新读取，切换无需重建实例。
                show_subagent = _effective_settings(st, user_id, session_id)["show_subagent_events"]

                async def _run():
                    try:
                        async for evt in agent.run(user_input):
                            # 子代理内部事件：按配置开关决定是否转发给前端（默认隐藏）
                            if not show_subagent and isinstance(evt, (SubAgentEvent, SubAgentDone)):
                                continue
                            await queue.put(("event", _sse(evt.__class__.__name__,
                                                           _to_jsonable(evt))))
                            if isinstance(evt, (AgentDone, AgentError)):
                                break
                    except Exception as e:  # noqa: BLE001
                        logger.exception("Agent run 异常 (user=%s)", user_id)
                        await queue.put(("event", _sse("AgentError", {
                            "error_type": "exception", "message": str(e)})))
                    finally:
                        await queue.put(("done", None))

                task = asyncio.create_task(_run())
                try:
                    while True:
                        kind, payload = await queue.get()
                        if kind == "done":
                            break
                        if await request.is_disconnected():
                            break
                        yield payload
                finally:
                    st.queues.pop(key, None)
                    fut = st.confirmations.pop(key, None)
                    if fut is not None and not fut.done():
                        fut.set_result(ConfirmResponse(approved=False, message="连接已断开"))
                    if not task.done():
                        task.cancel()
        except Exception as e:  # noqa: BLE001
            logger.exception("acquire 失败 (user=%s)", user_id)
            yield _sse("ServerError", {"message": str(e)})

    @app.post("/api/chat")
    async def chat(
        request: Request,
        x_user_id: str = Header("default", alias="X-User-Id"),
        x_session_id: str = Header("default", alias="X-Session-Id"),
    ):
        body = await request.json()
        user_input = (body.get("input") or "").strip()
        if not user_input:
            raise HTTPException(400, "input 不能为空")

        if user_input.startswith("/"):
            async def _cmd_stream() -> AsyncIterator[dict]:
                pool: AgentPool = request.app.state.pool
                try:
                    async with pool.acquire(x_user_id, x_session_id) as h:
                        async for sse in _exec_command(request.app, h.agent, user_input):
                            if await request.is_disconnected():
                                return
                            yield sse
                except Exception as e:  # noqa: BLE001
                    yield _sse("ServerError", {"message": str(e)})
            return EventSourceResponse(_cmd_stream(), ping=15)

        return EventSourceResponse(
            _stream_chat(request.app, x_user_id, x_session_id, user_input, request),
            ping=15,
        )

    @app.post("/api/confirm")
    async def confirm(
        request: Request,
        x_user_id: str = Header("default", alias="X-User-Id"),
        x_session_id: str = Header("default", alias="X-Session-Id"),
    ):
        body = await request.json()
        fut = request.app.state.confirmations.get((x_user_id, x_session_id))
        if fut is None or fut.done():
            raise HTTPException(404, "没有待确认的工具调用（可能已超时或已响应）")
        fut.set_result(ConfirmResponse(
            approved=bool(body.get("approved", False)),
            message=str(body.get("message", "")),
        ))
        return {"status": "ok"}

    # ── 设置 / 厂商 ───────────────────────────────────────

    @app.get("/api/providers")
    async def providers():
        items = []
        for name in ModelRegistry.list_providers():
            env_name = env_key_name(name)
            items.append({
                "name": name,
                "default_model": DEFAULT_MODELS.get(name, ""),
                "env_key": env_name,
                "key_configured": bool(os.environ.get(env_name)),
            })
        return {"providers": items}

    @app.get("/api/settings")
    async def get_settings(
        x_user_id: str = Header("default", alias="X-User-Id"),
        x_session_id: str = Header("default", alias="X-Session-Id"),
    ):
        return _effective_settings(app.state, x_user_id, x_session_id)

    @app.post("/api/settings")
    async def post_settings(
        request: Request,
        x_user_id: str = Header("default", alias="X-User-Id"),
        x_session_id: str = Header("default", alias="X-Session-Id"),
    ):
        body = await request.json()
        st = request.app.state
        key = (x_user_id, x_session_id)
        pref = dict(st.prefs.get(key, {}))
        # 影响 Agent/LLM 构造的字段：变更需驱逐实例按新配置重建
        needs_rebuild = False
        for field_name in ("provider", "model", "mode"):
            if field_name in body and body[field_name] is not None:
                pref[field_name] = body[field_name]
                needs_rebuild = True
        for field_name in ("web_search", "enable_thinking"):
            if field_name in body and body[field_name] is not None:
                pref[field_name] = bool(body[field_name])
                needs_rebuild = True
        # 纯展示开关：读取发生在对话流式时，切换无需重建实例（不触发驱逐）
        if "show_subagent_events" in body and body["show_subagent_events"] is not None:
            pref["show_subagent_events"] = bool(body["show_subagent_events"])
        st.prefs[key] = pref
        # 仅当影响构造的字段变更时才驱逐该实例 → 下条消息按新厂商/模型/参数重建
        if needs_rebuild:
            await st.pool.remove(x_user_id, x_session_id)
        return {"status": "ok", "settings": _effective_settings(st, x_user_id, x_session_id)}

    @app.post("/api/mode")
    async def set_mode(
        request: Request,
        x_user_id: str = Header("default", alias="X-User-Id"),
        x_session_id: str = Header("default", alias="X-Session-Id"),
    ):
        body = await request.json()
        new_mode = str(body.get("mode", "")).strip()
        if new_mode not in {m.value for m in AgentMode}:
            raise HTTPException(400, f"无效模式: {new_mode}")
        st = request.app.state
        key = (x_user_id, x_session_id)
        pref = dict(st.prefs.get(key, {}))
        pref["mode"] = new_mode
        st.prefs[key] = pref
        # 已有实例则即时切换（零成本）；尚未创建则下次按偏好构造
        agent = st.pool._entries.get(key)
        if agent is not None:
            agent.agent.set_mode(new_mode)
        return {"status": "ok", "mode": new_mode}

    # ── 信息查看 ──────────────────────────────────────────

    @app.get("/api/tools")
    async def list_tools(
        x_user_id: str = Header("default", alias="X-User-Id"),
        x_session_id: str = Header("default", alias="X-Session-Id"),
    ):
        agent = await app.state.pool.get_or_create_agent(x_user_id, x_session_id)
        active = agent.tools.list_tools()
        items = []
        for name in active:
            w = agent.tools.get_tool(name)
            items.append({
                "name": name,
                "description": (w.description[:120] if w else ""),
                "is_safe": (w.is_safe if w else True),
            })
        dormant = agent.tools.list_dormant_tools()
        return {"active": items, "dormant_count": len(dormant), "dormant": dormant[:50]}

    @app.get("/api/skills")
    async def list_skills(
        x_user_id: str = Header("default", alias="X-User-Id"),
        x_session_id: str = Header("default", alias="X-Session-Id"),
    ):
        agent = await app.state.pool.get_or_create_agent(x_user_id, x_session_id)
        out = []
        for name in agent.skill_registry.list_names():
            cfg = agent.skill_registry.get(name)
            out.append({
                "name": name,
                "description": cfg.description,
                "triggers": list(cfg.triggers or []),
            })
        return {"skills": out}

    @app.get("/api/memory")
    async def get_memory(x_user_id: str = Header("default", alias="X-User-Id")):
        from milu.tools.builtin.memory_tool import memory_file_path
        path = memory_file_path(x_user_id)
        if not path.exists():
            return {"items": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"items": []}
        items = data if isinstance(data, list) else data.get("items", [])
        return {"items": items}

    @app.get("/api/stats")
    async def stats():
        return app.state.pool.get_stats()

    # ── 会话 ──────────────────────────────────────────────

    @app.get("/api/sessions")
    async def sessions(x_user_id: str = Header("default", alias="X-User-Id")):
        from milu.agent.session import Session
        # 确定性派生 id 形如 "{user}__{session}"，按 "{user}__" 前缀筛本用户会话
        prefix = AgentPool._derive_session_id(x_user_id, "")
        out = [s for s in Session.list_sessions(default_session_dir())
               if str(s.get("session_id", "")).startswith(prefix)]
        return {"sessions": out}

    @app.post("/api/session/action")
    async def session_action(
        request: Request,
        x_user_id: str = Header("default", alias="X-User-Id"),
        x_session_id: str = Header("default", alias="X-Session-Id"),
    ):
        body = await request.json()
        action = body.get("action", "")
        agent = await app.state.pool.get_or_create_agent(x_user_id, x_session_id)
        if action == "reset":
            await agent.reset()
            return {"status": "ok", "message": "对话已重置"}
        if action == "new":
            agent.new_session()
            return {"status": "ok", "session_id": agent.session.session_id if agent.session else None}
        if action == "save":
            agent.save_session()
            return {"status": "ok"}
        if action == "load":
            sid = body.get("session_id", "")
            try:
                count = agent.load_session(sid)
            except FileNotFoundError:
                raise HTTPException(404, f"会话不存在: {sid}")
            return {"status": "ok", "message_count": count}
        raise HTTPException(400, f"未知 action: {action}")

    # ── 定时任务 ──────────────────────────────────────────

    @app.get("/api/schedule/tasks")
    async def schedule_tasks(x_user_id: str = Header("default", alias="X-User-Id")):
        tasks = app.state.store.list_user(x_user_id)
        return {"tasks": [asdict(t) for t in tasks]}

    @app.post("/api/schedule/create")
    async def schedule_create_api(
        request: Request,
        x_user_id: str = Header("default", alias="X-User-Id"),
    ):
        from milu.scheduler.store import ScheduleTask
        body = await request.json()
        name = (body.get("name") or "").strip()
        prompt = (body.get("prompt") or "").strip()
        trigger_type = (body.get("trigger_type") or "").strip()
        if not name or not prompt:
            raise HTTPException(400, "name 和 prompt 必填")
        if trigger_type not in ("cron", "interval", "once"):
            raise HTTPException(400, "trigger_type 必须是 cron/interval/once")
        # 触发参数严格校验——配置无效的任务永远算不出 next_run，会静默不执行
        run_at = (body.get("run_at") or "").strip()
        cron = (body.get("cron") or "").strip()
        interval_minutes = int(body.get("interval_minutes") or 0)
        if trigger_type == "once":
            if not run_at:
                raise HTTPException(400, "once 任务必须填写执行时间 run_at")
            try:
                if datetime.fromisoformat(run_at) <= datetime.now():
                    raise HTTPException(400, f"run_at '{run_at}' 是过去时间，任务不会触发")
            except ValueError:
                raise HTTPException(400, f"run_at 格式无效: {run_at}")
        if trigger_type == "interval" and interval_minutes < 1:
            raise HTTPException(400, "interval 任务必须填写间隔分钟数（≥1）")
        if trigger_type == "cron" and not cron:
            raise HTTPException(400, "cron 任务必须填写 cron 表达式")
        task = ScheduleTask.create(
            name=name, prompt=prompt, trigger_type=trigger_type,
            cron=cron,
            interval_minutes=interval_minutes,
            run_at=run_at,
            description=(body.get("description") or "").strip(),
            provider=(body.get("provider") or "").strip(),
            model=(body.get("model") or "").strip(),
            user_id=x_user_id,
        )
        try:
            app.state.store.add(task)
        except ValueError as e:
            raise HTTPException(409, str(e))
        return {"status": "ok", "task": asdict(task)}

    @app.post("/api/schedule/action")
    async def schedule_action(
        request: Request,
        x_user_id: str = Header("default", alias="X-User-Id"),
    ):
        body = await request.json()
        action = body.get("action", "")
        name = (body.get("name") or "").strip()
        store = app.state.store
        if not name:
            raise HTTPException(400, "name 必填")
        if action == "delete":
            ok = store.remove(name, x_user_id)
            if not ok:
                raise HTTPException(404, f"任务不存在: {name}")
            return {"status": "ok"}
        if action in ("enable", "disable"):
            task = store.get(name, x_user_id)
            if not task:
                raise HTTPException(404, f"任务不存在: {name}")
            task.enabled = action == "enable"
            store.update(task)
            return {"status": "ok", "enabled": task.enabled}
        if action == "run_now":
            from milu.scheduler import ScheduleEngine
            task = store.get(name, x_user_id)
            if not task:
                raise HTTPException(404, f"任务不存在: {name}")
            engine = app.state.engine or ScheduleEngine(store)
            try:
                result = await engine.run_task_now(name, x_user_id)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(500, f"执行失败: {e}")
            return {"status": "ok", "result": result}
        raise HTTPException(400, f"未知 action: {action}")

    @app.get("/api/schedule/results")
    async def schedule_results(
        x_user_id: str = Header("default", alias="X-User-Id"),
        limit: int = 20,
    ):
        from milu.scheduler.store import _safe_user
        path = user_data_dir() / "scheduler_outbox" / f"{_safe_user(x_user_id)}.jsonl"
        if not path.exists():
            return {"results": []}
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return {"results": []}
        results = [json.loads(ln) for ln in lines[-limit:] if ln.strip()]
        results.reverse()
        return {"results": results}

    return app


# ── / 命令处理（聊天框内的控制面，按事件流返回）──────────────

async def _exec_command(app, agent: Agent, cmd: str) -> AsyncIterator[dict]:
    """执行一个 / 命令，以 CommandResult 事件流返回（与 multi_user_chat 同协议）。"""
    def text(t):
        return _sse("CommandResult", {"type": "text", "text": t})

    def info(t):
        return _sse("CommandResult", {"type": "info", "text": t})

    try:
        parts = cmd.split(maxsplit=1)
        head = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""

        if head == "/help":
            yield text("\n".join([
                "可用命令：",
                "  /reset       重置对话（清空上下文与计划）",
                "  /history     查看对话历史",
                "  /tools       查看可用工具",
                "  /skills      查看可用技能",
                "  /plan        查看当前会话计划",
                "  /mode [m]    查看/切换模式（talk/manual/auto/superwork）",
                "  /prompt      查看当前系统提示词",
                "  /compact     手动压缩对话历史",
                "  /save        保存当前会话",
                "  /sessions    查看历史会话",
                "  /new         新建会话",
                "  /load <id>   加载历史会话",
                "  /memory      查看长期记忆",
            ]))
        elif head == "/reset":
            await agent.reset()
            yield info("对话已重置，上下文与计划已清空。")
        elif head == "/history":
            msgs = agent.history.all_messages
            lines = [f"=== 对话历史（{len(msgs)} 条）==="]
            for m in msgs:
                c = (m.content or "").replace("\n", " ")
                c = c[:300] + "..." if len(c) > 300 else c
                lines.append(f"  [{m.role.value.upper():<9}] {c}")
            yield text("\n".join(lines))
        elif head == "/tools":
            active = agent.tools.list_tools()
            lines = [f"=== 活跃工具（{len(active)} 个）==="]
            for name in active:
                w = agent.tools.get_tool(name)
                tag = "" if (w and w.is_safe) else " [需确认]"
                lines.append(f"  {name:<28}{tag}")
            dormant = agent.tools.list_dormant_tools()
            if dormant:
                lines.append(f"\n休眠工具：{len(dormant)} 个（LLM 可按需激活）")
            yield text("\n".join(lines))
        elif head == "/skills":
            names = agent.skill_registry.list_names()
            lines = [f"=== 可用技能（{len(names)} 个）==="]
            for name in names:
                cfg = agent.skill_registry.get(name)
                lines.append(f"  {name:<22} {cfg.description}")
            yield text("\n".join(lines))
        elif head == "/plan":
            items = _read_plan_items(agent)
            if not items:
                yield info("暂无会话计划。")
            else:
                markers = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
                lines = [f"=== 会话计划（{len(items)} 条）==="]
                for it in items:
                    lines.append(f"  {markers.get(it.get('status'), '[ ]')} {it.get('content', '')}")
                yield text("\n".join(lines))
        elif head == "/mode":
            if not arg:
                cur = agent.mode.value if hasattr(agent.mode, "value") else str(agent.mode)
                yield text(f"当前模式：{cur}\n可选：talk / manual / auto / superwork\n用法：/mode <模式>")
            else:
                try:
                    agent.set_mode(arg)
                    yield info(f"模式已切换为：{arg}")
                except ValueError:
                    yield info(f"无效模式：{arg}（可选 talk/manual/auto/superwork）")
        elif head == "/prompt":
            agent._build_system_prompt()
            msgs = agent.history.all_messages
            content = msgs[0].content if msgs else ""
            yield text(f"=== 系统提示词（{len(content)} 字符）===\n{content}")
        elif head == "/compact":
            if not agent._history.compact_enabled:
                yield info("上下文压缩未启用。")
            else:
                n0 = len(agent.history._messages)
                compacted, summary = await agent._history.manual_compact()
                agent.history.replace_all(compacted)
                agent._history._log_compaction(compacted)
                yield text(f"=== 压缩完成（{n0} → {len(compacted)} 条）===\n{summary[:400]}")
        elif head == "/save":
            agent.save_session()
            if agent.session:
                yield info(f"会话已保存：{agent.session.session_id}")
            else:
                yield info("会话功能未启用。")
        elif head == "/sessions":
            from milu.agent.session import Session
            base = Path(agent.session.base_dir) if agent.session else default_session_dir()
            ss = Session.list_sessions(base)
            if not ss:
                yield info("暂无历史会话。")
            else:
                cur = agent.session.session_id if agent.session else None
                lines = [f"=== 历史会话（{len(ss)} 个）==="]
                for s in ss:
                    mark = " ← 当前" if s.get("session_id") == cur else ""
                    lines.append(f"  {s.get('session_id')}  {s.get('message_count', 0)} 条{mark}")
                yield text("\n".join(lines))
        elif head == "/new":
            agent.new_session()
            if agent.session:
                yield info(f"新会话已创建：{agent.session.session_id}")
        elif head == "/load":
            if not arg:
                yield info("用法：/load <会话ID>")
            else:
                try:
                    n = agent.load_session(arg)
                    yield info(f"会话已加载：{arg}（{n} 条消息）")
                except FileNotFoundError:
                    yield info(f"会话不存在：{arg}")
        elif head == "/memory":
            from milu.tools.builtin.memory_tool import memory_file_path
            path = memory_file_path(getattr(agent, "_schedule_user", None) or "default")
            if not path.exists():
                yield info("暂无长期记忆。")
            else:
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    items = data if isinstance(data, list) else data.get("items", [])
                except (OSError, ValueError):
                    items = []
                lines = [f"=== 长期记忆（{len(items)} 条）==="]
                for it in items:
                    lines.append(f"  · {it.get('content', '')}")
                yield text("\n".join(lines))
        else:
            yield info(f"未知命令：{head}（输入 /help 查看帮助）")
    except Exception as e:  # noqa: BLE001
        logger.exception("命令执行异常")
        yield _sse("CommandResult", {"type": "error", "text": f"命令执行异常：{e}"})


def _read_plan_items(agent) -> list[dict]:
    """读取当前会话计划（todo 工具持久化在 session 目录的 plan.json）。"""
    if not agent.session:
        return []
    plan_file = agent.session.dir_path / "plan.json"
    if not plan_file.exists():
        return []
    try:
        return json.loads(plan_file.read_text(encoding="utf-8")).get("items", [])
    except (OSError, ValueError):
        return []


# ── 运行入口 ──────────────────────────────────────────────

def create_app_from_env():
    """从环境变量 _MILU_SERVE_OPTS 读取标量选项构造 app（供 uvicorn --reload 工厂模式）。"""
    raw = os.environ.get("_MILU_SERVE_OPTS", "{}")
    try:
        opts = json.loads(raw)
    except ValueError:
        opts = {}
    return create_app(**opts)


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False, **opts):
    """启动 uvicorn 服务。

    :param opts: 透传给 create_app 的标量选项（provider/model/mode/各开关）。
        reload=True 时需 import 字符串，故经环境变量把标量选项传给工厂。
    """
    import uvicorn

    if reload:
        os.environ["_MILU_SERVE_OPTS"] = json.dumps(opts)
        uvicorn.run("milu.serving.web.app:create_app_from_env",
                    host=host, port=port, reload=True, factory=True)
    else:
        uvicorn.run(create_app(**opts), host=host, port=port)


if __name__ == "__main__":
    # 支持 `python -m milu.serving.web.app` 直接启动（host/port 经环境变量覆盖）
    run_server(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
    )
