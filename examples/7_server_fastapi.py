"""FastAPI 多用户 Agent 服务示例 — SSE 流式输出。

演示：
- 多用户 HTTP API（不同 X-User-Id 走不同 Agent 实例）
- SSE 流式返回 AgentEvent（TextDelta / ToolCallStart / ToolResult / AgentDone / AgentError）
- AgentPool 自动按 user_id+session_id 隔离 + 限流
- 后台生命周期：startup 创建 pool, shutdown 优雅关闭

启动：
    pip install fastapi uvicorn sse-starlette
    .venv/Scripts/python examples/7_server_fastapi.py

测试：
    # 基础请求
    curl -N -X POST http://localhost:8000/chat \\
         -H "X-User-Id: alice" -H "X-Session-Id: s1" \\
         -H "Content-Type: application/json" \\
         -d '{"input": "你好，介绍下你自己"}'

    # 流式事件会一行行 SSE 输出
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import AsyncIterator, Optional

from dotenv import load_dotenv

from milu.agent.config import AgentConfig
from milu.agent.events import AgentEvent, AgentDone, AgentError
from milu.llm.providers import ModelRegistry
from milu.serving import AgentPool, AgentPoolConfig
from milu.tools.builtin import BUILTIN_TOOLS

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ── 1. Pool 生命周期（FastAPI lifespan） ─────────────────────


@asynccontextmanager
async def lifespan(app):
    """启动时创建 Pool，关闭时优雅停机。"""
    llm = ModelRegistry.create("qwen", model="qwen3.7-max")
    pool = AgentPool(
        llm_factory=lambda uid, sid: llm,  # 所有用户共享同一 LLM（AsyncOpenAI 协程安全）
        config=AgentPoolConfig(
            max_agents=200,
            max_concurrent_runs=50,
            idle_ttl_seconds=300,
            sweep_interval_seconds=60,
        ),
        # AgentConfig 现仅含运行限额；mode/session 等能力参数通过 agent_kwargs 透传
        agent_config=AgentConfig(max_turns=50, timeout=120),
        # session_dir 不传 → 用 Agent 默认 default_session_dir()（~/.milu/sessions，与 CWD 解耦）
        agent_kwargs={"mode": "auto", "session_enabled": True},
    )
    await pool.start()
    app.state.pool = pool
    logger.info("AgentPool 已就绪: %s", pool.get_stats())
    try:
        yield
    finally:
        await pool.stop()
        logger.info("AgentPool 已关闭")


# ── 2. FastAPI 应用 ───────────────────────────────────────


from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse


app = FastAPI(title="milu multi-user server", lifespan=lifespan)


# ── 3. SSE 事件序列化 ───────────────────────────────────────


def _event_to_dict(evt: AgentEvent) -> dict:
    """把 AgentEvent dataclass 序列化为可 JSON 化的 dict。"""
    if hasattr(evt, "__dataclass_fields__"):
        return asdict(evt)
    return {"type": evt.__class__.__name__, "repr": repr(evt)}


def _event_type(evt: AgentEvent) -> str:
    return evt.__class__.__name__


async def _stream_agent(
    user_id: str, session_id: str, user_input: str, request: Request
) -> AsyncIterator[dict]:
    """从 AgentPool 取出 Agent，逐事件 yield 为 SSE 消息。"""
    pool: AgentPool = request.app.state.pool

    # 异常转换
    try:
        async with pool.acquire(user_id, session_id) as handle:
            try:
                async for evt in handle.agent.run(user_input):
                    if await request.is_disconnected():
                        logger.info("客户端断开 (user=%s)", user_id)
                        return
                    yield {
                        "event": _event_type(evt),
                        "data": json.dumps(_event_to_dict(evt), ensure_ascii=False),
                    }
                    # AgentDone / AgentError 后停止
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


# ── 4. HTTP 端点 ───────────────────────────────────────────


@app.post("/chat")
async def chat(
    request: Request,
    x_user_id: str = Header(..., alias="X-User-Id"),
    x_session_id: str = Header("default", alias="X-Session-Id"),
):
    """SSE 流式聊天端点。

    Headers:
        X-User-Id: 必填。不同用户得到不同 Agent 实例。
        X-Session-Id: 可选。同一用户多个会话用不同 session_id 区分。
    Body:
        {"input": "用户输入文本"}

    Returns:
        SSE 事件流，每个事件形如:
            event: TextDelta
            data: {"text": "你"}
    """
    body = await request.json()
    user_input = body.get("input", "").strip()
    if not user_input:
        raise HTTPException(400, "input 不能为空")

    return EventSourceResponse(
        _stream_agent(x_user_id, x_session_id, user_input, request),
        ping=15,  # 15s 心跳，防止代理超时
    )


@app.post("/reset")
async def reset(
    x_user_id: str = Header(..., alias="X-User-Id"),
    x_session_id: str = Header("default", alias="X-Session-Id"),
):
    """重置某用户的对话历史（保留 system 消息）。"""
    pool: AgentPool = app.state.pool
    async with pool.acquire(x_user_id, x_session_id) as h:
        await h.agent.reset()
    return {"status": "ok", "user_id": x_user_id, "session_id": x_session_id}


@app.get("/stats")
async def stats():
    """获取 Pool 状态（监控用）。"""
    return app.state.pool.get_stats()


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── 5. 启动 ──────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
    )
