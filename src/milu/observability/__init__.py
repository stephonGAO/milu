"""milu 可观测性 —— Agent 运行追踪（trace/span）、指标聚合与成本估算。

让每一次 Agent.run() 可解释、可测量、可比较、可追溯、可视化：
一次运行 = 一棵 span 树（invoke_agent 根 → chat/execute_tool/guardrail/
blocked_on_user/compact 子级，子代理整树嵌套），落盘 trace.jsonl +
按用户分文件的运行索引 runs/{safe_user_id}.jsonl（各自轮转上限）。
属性命名对齐 OTel GenAI semantic conventions v1.37+，
将来 OTLP 导出零映射。设计与调研结论见 docs/可观测性方案设计.md。

用法：
    Agent(llm)                      # 默认关闭（库纯净，单测 hermetic）
    Agent(llm, trace=True)          # 开启，默认配置
    Agent(llm, trace=TraceConfig(capture_content="full"))
CLI / Web 服务按 config.json observability 分节自动开启（本地落盘默认开）。
"""
from milu.observability.span import (
    NOOP_SPAN,
    SCHEMA_VERSION,
    Span,
    clip_content,
    new_span_id,
    new_trace_id,
)
from milu.observability.tracer import (
    NOOP_TRACER,
    TraceConfig,
    Tracer,
    _current_span,
    _current_tracer,
    extract_provider_name,
)
from milu.observability.sink import CallbackSink, JsonlSink, TraceSink
from milu.observability.report import (
    RunReport,
    append_run_index,
    build_run_report,
    cleanup_old_traces,
    load_run_index,
    load_trace,
)
from milu.observability.pricing import BUILTIN_PRICE_TABLE, estimate_cost, resolve_price

__all__ = [
    "Span", "SCHEMA_VERSION", "NOOP_SPAN", "new_trace_id", "new_span_id", "clip_content",
    "Tracer", "NOOP_TRACER", "TraceConfig", "extract_provider_name",
    "_current_tracer", "_current_span",
    "TraceSink", "JsonlSink", "CallbackSink",
    "RunReport", "build_run_report", "append_run_index", "load_run_index",
    "load_trace", "cleanup_old_traces",
    "BUILTIN_PRICE_TABLE", "estimate_cost", "resolve_price",
]
