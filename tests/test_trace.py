"""测试运行追踪（observability）：span 树结构 / 并发隔离 / 子代理嵌套 /
fail-silent / 关闭零产物 / RunReport 聚合 / 成本估算。

conftest 的 _isolate_paths 已把 MILU_HOME 重定向到 tmp_path，
runs.jsonl 与 traces/ 散件均落在隔离目录。
"""
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from milu.agent import Agent
from milu.agent.events import AgentDone, ToolResult
from milu.llm.base.response import StreamChunk, TokenUsage
from milu.observability import (
    JsonlSink,
    TraceConfig,
    Tracer,
    build_run_report,
    estimate_cost,
    load_run_index,
    load_trace,
    resolve_price,
)
from milu.resources import user_data_dir
from milu.tools import tool


# ── 测试工装 ──────────────────────────────────────────────

def make_tool_call(call_id: str, name: str, arguments: str = "{}"):
    """构造流式 tool_call 片段 mock 对象。"""
    return type("obj", (), {
        "index": 0, "id": call_id,
        "function": type("obj", (), {"name": name, "arguments": arguments})(),
    })()


def make_llm_with_tool(tool_name: str = "get_time"):
    """mock LLM：第 1 轮调用工具、第 2 轮输出最终文本。"""
    call_count = 0

    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield StreamChunk(tool_calls=[make_tool_call("call_1", tool_name)])
            yield StreamChunk(finish_reason="tool_calls")
            yield StreamChunk(usage=TokenUsage(
                prompt_tokens=100, completion_tokens=10, total_tokens=110))
        else:
            yield StreamChunk(content="完成", finish_reason="stop")
            yield StreamChunk(usage=TokenUsage(
                prompt_tokens=120, completion_tokens=5, total_tokens=125))

    llm = AsyncMock()
    llm.chat = mock_chat
    return llm


@tool(name="get_time", description="获取时间")
async def get_time() -> str:
    return "2026-06-12 10:00"


@tool(name="dangerous_op", description="危险操作", is_safe=False)
async def dangerous_op() -> str:
    return "已执行"


def _make_agent(llm, trace=True, **kwargs):
    """精简 Agent：不注入内置工具/技能/子代理，session 默认关闭。"""
    kwargs.setdefault("tools", [get_time])
    kwargs.setdefault("subagents", [])
    kwargs.setdefault("session_enabled", False)
    kwargs.setdefault("judge_llm", False)
    return Agent(llm=llm, system_prompt="测试", trace=trace, **kwargs)


async def _consume(agent, text="测试输入"):
    events = []
    async for e in agent.run(text):
        events.append(e)
    return events


def _trace_rows():
    """读 runs 索引并按 trace_path 取各自的 span 行。"""
    rows = load_run_index()
    return rows, {r["trace_id"]: load_trace(r["trace_path"]) for r in rows if r.get("trace_path")}


# ── 基础：span 树结构 / runs 索引 ─────────────────────────

async def test_trace_basic_structure():
    agent = _make_agent(make_llm_with_tool())
    events = await _consume(agent)
    assert any(isinstance(e, AgentDone) for e in events)

    rows = load_run_index()
    assert len(rows) == 1
    report = rows[0]
    assert report["status"] == "ok"
    assert report["turn_count"] == 2
    assert report["input_tokens"] == 220 and report["output_tokens"] == 15
    assert report["tool_calls"] == 1 and report["tool_errors"] == 0
    assert report["trace_path"]

    spans = load_trace(report["trace_path"])
    by_kind = {}
    for s in spans:
        by_kind.setdefault(s["kind"], []).append(s)
    assert len(by_kind["agent"]) == 1
    assert len(by_kind["generation"]) == 2
    assert len(by_kind["tool"]) == 1

    root = by_kind["agent"][0]
    assert root["parent_id"] is None
    assert root["attributes"]["milu.turn_count"] == 2
    assert root["attributes"]["gen_ai.usage.input_tokens"] == 220
    # 所有子 span 直接挂根、同一 trace_id、带 schema 版本
    for s in spans:
        assert s["trace_id"] == report["trace_id"]
        assert s["schema"] == 1
        if s is not root:
            assert s["parent_id"] == root["span_id"]

    gen1 = sorted(by_kind["generation"], key=lambda s: s["start_time"])[0]
    assert gen1["attributes"]["gen_ai.usage.input_tokens"] == 100
    assert gen1["attributes"]["gen_ai.response.finish_reasons"] == ["tool_calls"]
    assert "gen_ai.response.time_to_first_chunk" in gen1["attributes"]

    t = by_kind["tool"][0]
    assert t["attributes"]["gen_ai.tool.name"] == "get_time"
    assert t["status"] == "ok" and t["duration_ms"] is not None


async def test_trace_with_session_writes_into_session_dir():
    agent = _make_agent(make_llm_with_tool(), session_enabled=True)
    await _consume(agent)
    trace_file = agent.session.dir_path / "trace.jsonl"
    assert trace_file.exists()
    rows = load_run_index()
    assert rows[0]["session_id"] == agent.session.session_id


def make_stateless_tool_llm():
    """每次 run 行为一致的 mock：按"最后一条消息角色"决策（而非累积计数），
    用户消息→调工具、工具结果→出最终文本。同一 agent 多轮调用每轮都稳定
    产出 2 generation + 1 tool span。"""
    async def chat(messages, *args, **kwargs):
        last = messages[-1] if messages else None
        role = getattr(getattr(last, "role", None), "value", None)
        if role == "tool":
            yield StreamChunk(content="完成", finish_reason="stop")
            yield StreamChunk(usage=TokenUsage(
                prompt_tokens=120, completion_tokens=5, total_tokens=125))
        else:
            yield StreamChunk(tool_calls=[make_tool_call("call_1", "get_time")])
            yield StreamChunk(finish_reason="tool_calls")
            yield StreamChunk(usage=TokenUsage(
                prompt_tokens=100, completion_tokens=10, total_tokens=110))
    llm = AsyncMock()
    llm.chat = chat
    return llm


async def test_trace_multi_turn_session_isolated_by_trace_id():
    """同一会话多轮：trace.jsonl 累积所有轮次的 span（每轮独立 trace_id），
    但 load_trace(path, trace_id) 必须只返回该轮的 span——否则查看单次运行
    会把历史轮次一并渲染（前端"点进去含全部历史"的 bug）。"""
    agent = _make_agent(make_stateless_tool_llm(), session_enabled=True)
    await _consume(agent, "第一轮")
    await _consume(agent, "第二轮")
    await _consume(agent, "第三轮")

    trace_file = agent.session.dir_path / "trace.jsonl"
    rows = load_run_index()
    assert len(rows) == 3
    # 三轮的 trace_id 互不相同
    tids = {r["trace_id"] for r in rows}
    assert len(tids) == 3
    # 不过滤 → 读到全部三轮的 span（每轮 1 agent+2 generation+1 tool = 4）
    assert len(load_trace(trace_file)) == 12
    # 按 trace_id 过滤 → 每轮只拿回自己的 4 个 span，且 trace_id 全部一致
    for r in rows:
        sub = load_trace(r["trace_path"], r["trace_id"])
        assert len(sub) == 4
        assert {s["trace_id"] for s in sub} == {r["trace_id"]}
        assert sum(1 for s in sub if s["kind"] == "agent") == 1


# ── 关闭：零产物 ──────────────────────────────────────────

async def test_trace_disabled_no_artifacts():
    agent = _make_agent(make_llm_with_tool(), trace=False)
    events = await _consume(agent)
    assert any(isinstance(e, AgentDone) for e in events)
    assert not (user_data_dir() / "runs.jsonl").exists()
    assert not (user_data_dir() / "traces").exists()


# ── fail-silent：sink 异常不影响业务 ─────────────────────

async def test_trace_sink_failure_is_silent():
    class BoomSink:
        def emit(self, span):
            raise RuntimeError("boom")

    cfg = TraceConfig(extra_sinks=[BoomSink()])
    agent = _make_agent(make_llm_with_tool(), trace=cfg)
    events = await _consume(agent)
    assert any(isinstance(e, AgentDone) for e in events)
    # JSONL sink 不受连坐，trace 照常落盘
    assert len(load_run_index()) == 1


# ── 并发隔离 ──────────────────────────────────────────────

async def test_trace_concurrent_isolation():
    a1 = _make_agent(make_llm_with_tool())
    a2 = _make_agent(make_llm_with_tool())
    await asyncio.gather(_consume(a1), _consume(a2))

    rows, traces = _trace_rows()
    assert len(rows) == 2
    assert rows[0]["trace_id"] != rows[1]["trace_id"]
    for tid, spans in traces.items():
        assert spans, "每个 trace 各有自己的文件"
        assert all(s["trace_id"] == tid for s in spans)
        assert sum(1 for s in spans if s["kind"] == "agent") == 1


# ── 子代理：嵌套挂父树 ────────────────────────────────────

async def test_trace_nested_agent_attaches_to_parent_tool_span():
    async def inner_chat(*args, **kwargs):
        yield StreamChunk(content="内层完成", finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(
            prompt_tokens=30, completion_tokens=3, total_tokens=33))

    inner_llm = AsyncMock()
    inner_llm.chat = inner_chat

    @tool(name="delegate", description="委派给内层 Agent")
    async def delegate() -> str:
        inner = Agent(llm=inner_llm, system_prompt="内层", tools=[],
                      subagents=[], session_enabled=False, judge_llm=False)
        async for _ in inner.run("内层任务"):
            pass
        return "委派完成"

    agent = _make_agent(make_llm_with_tool("delegate"), tools=[delegate])
    await _consume(agent)

    rows, traces = _trace_rows()
    assert len(rows) == 1, "内层运行不是 trace 所有者，不入 runs 索引"
    spans = list(traces.values())[0]
    agent_spans = [s for s in spans if s["kind"] == "agent"]
    tool_spans = [s for s in spans if s["kind"] == "tool"]
    assert len(agent_spans) == 2 and len(tool_spans) == 1
    inner_span = next(s for s in agent_spans if s["parent_id"] is not None)
    assert inner_span["parent_id"] == tool_spans[0]["span_id"]
    # 全树 token 合计含内层（report 按 generation span 求和）
    assert rows[0]["input_tokens"] == 220 + 30
    assert rows[0]["subagent_runs"] == 1


# ── 审批等待 span（manual 模式）──────────────────────────

async def test_trace_confirmation_span():
    async def approve(tool_name, args):
        return True

    agent = _make_agent(
        make_llm_with_tool("dangerous_op"),
        tools=[dangerous_op], mode="manual", on_confirm=approve,
    )
    events = await _consume(agent)
    assert any(isinstance(e, ToolResult) and not e.is_error for e in events)

    rows, traces = _trace_rows()
    spans = list(traces.values())[0]
    confirms = [s for s in spans if s["kind"] == "confirmation"]
    assert len(confirms) == 1
    assert confirms[0]["attributes"]["milu.decision"] == "approved"
    assert confirms[0]["name"] == "blocked_on_user dangerous_op"
    assert rows[0]["wait_time_ms"] >= 0


# ── AI 安全判定 span（auto 模式）─────────────────────────

async def test_trace_guardrail_span_with_verdicts():
    async def judge_chat(*args, **kwargs):
        yield StreamChunk(content=json.dumps({"verdicts": [
            {"id": "call_1", "decision": "allow", "reason": "测试操作无风险"},
        ]}))

    judge_llm = AsyncMock()
    judge_llm.chat = judge_chat

    agent = _make_agent(
        make_llm_with_tool("dangerous_op"),
        tools=[dangerous_op], mode="auto", judge_llm=judge_llm,
    )
    await _consume(agent)

    rows, traces = _trace_rows()
    spans = list(traces.values())[0]
    guards = [s for s in spans if s["kind"] == "guardrail"]
    assert len(guards) == 1
    g = guards[0]["attributes"]
    assert g["milu.judge.fail_open"] is False
    assert g["milu.judge.verdicts"][0]["decision"] == "allow"
    assert g["milu.judge.verdicts"][0]["reason"] == "测试操作无风险"
    assert g["milu.judge.tools"] == ["dangerous_op"]


async def test_trace_guardrail_fail_open_marked():
    async def broken_judge(*args, **kwargs):
        raise RuntimeError("judge 挂了")
        yield  # pragma: no cover

    judge_llm = AsyncMock()
    judge_llm.chat = broken_judge

    agent = _make_agent(
        make_llm_with_tool("dangerous_op"),
        tools=[dangerous_op], mode="auto", judge_llm=judge_llm,
    )
    events = await _consume(agent)
    # fail-open：工具仍执行成功
    assert any(isinstance(e, ToolResult) and not e.is_error for e in events)

    rows, traces = _trace_rows()
    spans = list(traces.values())[0]
    g = next(s for s in spans if s["kind"] == "guardrail")
    assert g["attributes"]["milu.judge.fail_open"] is True
    assert g["status"] == "error"
    assert rows[0]["fail_opens"] == 1


# ── 内容捕获分级 ──────────────────────────────────────────

async def test_trace_capture_content_none():
    cfg = TraceConfig(capture_content="none")
    agent = _make_agent(make_llm_with_tool(), trace=cfg)
    await _consume(agent, "敏感输入内容")

    _, traces = _trace_rows()
    spans = list(traces.values())[0]
    root = next(s for s in spans if s["kind"] == "agent")
    assert "milu.user_input" not in root["attributes"]
    t = next(s for s in spans if s["kind"] == "tool")
    assert "gen_ai.tool.call.arguments" not in t["attributes"]


async def test_trace_capture_content_truncated():
    cfg = TraceConfig(capture_content="truncated", max_content_chars=10)
    agent = _make_agent(make_llm_with_tool(), trace=cfg)
    await _consume(agent, "很长的输入" * 20)

    _, traces = _trace_rows()
    root = next(s for s in list(traces.values())[0] if s["kind"] == "agent")
    assert "截断" in root["attributes"]["milu.user_input"]


# ── RunReport 聚合 / 成本估算（纯单元）───────────────────

def test_resolve_price_prefix_and_override():
    assert resolve_price("deepseek-chat") is not None
    assert resolve_price("qwen-plus-latest")["input"] == resolve_price("qwen-plus")["input"]
    assert resolve_price("unknown-model-xyz") is None
    override = {"my-model": {"input": 1.0, "output": 2.0, "currency": "CNY"},
                "deepseek-chat": {"input": 99.0, "output": 99.0, "currency": "CNY"}}
    assert resolve_price("my-model-v2", override)["input"] == 1.0
    assert resolve_price("deepseek-chat", override)["input"] == 99.0  # 用户表优先


def test_estimate_cost():
    cost = estimate_cost("deepseek-chat", 1_000_000, 1_000_000)
    assert cost is not None
    amount, currency = cost
    assert amount == pytest.approx(10.0) and currency == "CNY"
    assert estimate_cost("unknown-model-xyz", 100, 100) is None


def test_run_report_cost_none_for_unknown_model(tmp_path):
    tracer = Tracer(TraceConfig(), [JsonlSink(tmp_path / "t.jsonl")])
    root = tracer.start("invoke_agent", "agent")
    root.set_attrs({"gen_ai.request.model": "unknown-model-xyz",
                    "gen_ai.provider.name": "test", "milu.mode": "auto"})
    gen = tracer.start("chat unknown-model-xyz", "generation")
    gen.set_attrs({"gen_ai.usage.input_tokens": 50, "gen_ai.usage.output_tokens": 5})
    tracer.finish(gen)
    tracer.finish(root)
    report = build_run_report(tracer, root)
    assert report.cost_estimate is None and report.cost_currency is None
    assert report.input_tokens == 50 and report.output_tokens == 5


async def test_run_report_cost_with_price_table():
    cfg = TraceConfig(price_table={
        "test-model": {"input": 1.0, "output": 2.0, "currency": "CNY"},
    })
    llm = make_llm_with_tool()
    llm._model_config = type("C", (), {"model": "test-model"})()
    agent = _make_agent(llm, trace=cfg)
    await _consume(agent)
    report = load_run_index()[0]
    # 220 input + 15 output（每百万单价 1/2 元）
    assert report["cost_estimate"] == pytest.approx(220 / 1e6 * 1.0 + 15 / 1e6 * 2.0)
    assert report["cost_currency"] == "CNY"
