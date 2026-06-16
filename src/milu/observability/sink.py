"""TraceSink —— span 落地通道（可插拔，学 OpenAI Agents SDK 的 processor 架构）。

默认 JsonlSink 本地落盘；TraceConfig.extra_sinks 可追加（如 Web 层 CallbackSink
喂 SSE 实时时间线）；P4 的 OtlpSink（milu[otel] 可选 extra）也是一个 sink。
所有 sink 由 Tracer 统一 try/except 调用（fail-silent），自身无需防御。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Protocol

from milu.observability.span import Span


class TraceSink(Protocol):
    """span 完成时的落地协议（同步、轻量；重活请自行异步化）。"""

    def emit(self, span: Span) -> None: ...


class JsonlSink:
    """append-only JSONL 落盘（与 session.py conversation.jsonl 同款形态）。

    span 结束即写一行：崩溃不丢已完成 span，半途运行可追溯到中断点。
    单行 append 的原子性与 scheduler outbox 同档取舍（单机场景足够）。
    """

    def __init__(self, path: "Path | str"):
        self.path = Path(path)
        self._dir_ready = False

    def emit(self, span: Span) -> None:
        if not self._dir_ready:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._dir_ready = True
        line = json.dumps(span.to_dict(), ensure_ascii=False, default=str)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class CallbackSink:
    """把完成的 span 交给回调（Web 层用它把 span 推进 SSE 队列）。"""

    def __init__(self, fn: Callable[[Span], None]):
        self._fn = fn

    def emit(self, span: Span) -> None:
        self._fn(span)
