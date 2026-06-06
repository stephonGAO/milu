"""定时任务多用户服务示例 — 调度引擎嵌入 FastAPI 进程。

演示：
- ScheduleEngine 以后台 asyncio 任务嵌入服务进程（start_background），与
  AgentPool 共存于同一 lifespan
- 注入 agent_pool：任务经 pool.get_or_create_agent(user_id, ...) 执行——
  per-user 实例复用、memory/schedule_user 自动派生、不占在线并发许可
- 多用户任务隔离：不同 X-User-Id 的对话里让 Agent 调 schedule_create 创建任务，
  任务落各自的 ~/.milu/schedules/{user_id}.json
- on_result 回调：任务完成时服务端可推送（此处示例为记录到内存，可换成
  SSE/WebSocket 推送或写库）
- outbox 查询端点：用户拉取自己任务的执行结果

启动：
    pip install fastapi uvicorn sse-starlette
    .venv/Scripts/python examples/scheduler_server.py

测试：
    # alice 创建定时任务（对话式，让 Agent 调 schedule_create 工具）
    curl -N -X POST http://localhost:8001/chat \\
         -H "X-User-Id: alice" -H "Content-Type: application/json" \\
         -d '{"input": "创建一个定时任务，每30分钟提醒我喝水"}'

    # 查看 alice 的任务执行结果（outbox）
    curl http://localhost:8001/schedule/results -H "X-User-Id: alice"
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

from milu.agent.config import AgentConfig
from milu.agent.events import AgentDone, AgentError
from milu.llm.providers import ModelRegistry
from milu.resources import user_data_dir
from milu.scheduler import ScheduleEngine, SchedulerConfig, ScheduleStore
from milu.scheduler.store import _safe_user
from milu.serving import AgentPool, AgentPoolConfig

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ── 1. 生命周期：AgentPool + 调度引擎同进程 ──────────────────

@asynccontextmanager
async def lifespan(app):
    llm = ModelRegistry.create("qwen")
    pool = AgentPool(
        llm_factory=lambda uid, sid: llm,
        config=AgentPoolConfig(max_agents=200, max_concurrent_runs=50),
        agent_config=AgentConfig(max_turns=50, timeout=120),
        # schedule_user 不传 → 池工厂自动按 user_id 派生（任务文件按用户隔离）
        agent_kwargs={"mode": "auto", "session_enabled": True},
    )
    await pool.start()

    # 任务完成回调：此处仅记录日志；生产中可改为 SSE/WebSocket 推送或写库
    async def on_result(task, result: str) -> None:
        logger.info("[调度结果] 用户=%s 任务=%s: %s", task.user_id, task.name, result[:80])

    store = ScheduleStore(user_data_dir())
    engine = ScheduleEngine(
        store,
        log_dir=user_data_dir() / "scheduler_logs",
        config=SchedulerConfig(
            max_concurrent_tasks=4,
            task_timeout=300.0,
            notify=False,            # 服务端无桌面，关闭系统弹窗
        ),
        agent_pool=pool,             # 任务经池执行（per-user 隔离 + 不占在线许可）
        on_result=on_result,
    )
    sched_task = engine.start_background()

    app.state.pool = pool
    app.state.store = store
    try:
        yield
    finally:
        engine.stop()
        sched_task.cancel()
        try:
            await sched_task
        except asyncio.CancelledError:
            pass
        await pool.stop()
        logger.info("调度引擎与 AgentPool 已关闭")


# ── 2. FastAPI 应用 ──────────────────────────────────────────

def create_app():
    from fastapi import FastAPI, Header, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="milu 定时任务多用户服务示例", lifespan=lifespan)

    @app.post("/chat")
    async def chat(
        request: Request,
        x_user_id: str = Header("default", alias="X-User-Id"),
        x_session_id: str = Header("s1", alias="X-Session-Id"),
    ):
        """对话端点（精简版，完整 SSE 见 7_server_fastapi.py）。

        在对话中让 Agent 调用 schedule_create 等工具，任务自动归属当前用户。
        """
        body = await request.json()
        pool: AgentPool = request.app.state.pool
        final_text, error = "", None
        async with pool.acquire(x_user_id, x_session_id) as handle:
            async for ev in handle.agent.run(body.get("input", "")):
                if isinstance(ev, AgentDone):
                    final_text = ev.final_text
                elif isinstance(ev, AgentError):
                    error = ev.message
        return JSONResponse({"user": x_user_id, "text": final_text, "error": error})

    @app.get("/schedule/tasks")
    async def list_tasks(
        request: Request,
        x_user_id: str = Header("default", alias="X-User-Id"),
    ):
        """列出当前用户的定时任务。"""
        store: ScheduleStore = request.app.state.store
        from dataclasses import asdict
        return JSONResponse({"tasks": [asdict(t) for t in store.list_user(x_user_id)]})

    @app.get("/schedule/results")
    async def list_results(
        x_user_id: str = Header("default", alias="X-User-Id"),
        limit: int = 20,
    ):
        """拉取当前用户最近的任务执行结果（outbox JSONL）。"""
        path = user_data_dir() / "scheduler_outbox" / f"{_safe_user(x_user_id)}.jsonl"
        if not path.exists():
            return JSONResponse({"results": []})
        lines = path.read_text(encoding="utf-8").splitlines()
        results = [json.loads(line) for line in lines[-limit:] if line.strip()]
        return JSONResponse({"results": results})

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=8001)
