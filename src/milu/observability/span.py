"""Span 数据模型 —— 运行追踪的最小记录单元。

ID 遵循 W3C Trace Context（trace_id 32 个十六进制字符、span_id 16 个），
将来经 OTLP 导出到 Langfuse / Phoenix / Jaeger 时零映射。
属性命名对齐 OTel GenAI semantic conventions v1.37+ 新名
（gen_ai.provider.name / gen_ai.usage.input_tokens 等），milu 私有信息走
milu.* 命名空间；约定仍在演进，故 JSONL 落盘每行带 schema 版本，
内部存储与导出映射解耦（约定变了只改导出层）。

设计详见 docs/可观测性方案设计.md。
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field

# JSONL 行格式版本（GenAI semconv 为 Development 状态，留好迁移余地）
SCHEMA_VERSION = 1


def new_trace_id() -> str:
    """生成 W3C 格式 trace_id（16 字节 = 32 hex）。"""
    return secrets.token_hex(16)


def new_span_id() -> str:
    """生成 W3C 格式 span_id（8 字节 = 16 hex）。"""
    return secrets.token_hex(8)


def clip_content(text, capture: str, max_chars: int) -> str | None:
    """按捕获级别处理内容：none → None（不记录）；truncated → 截断；full → 原样。

    会话全文已在 conversation.jsonl，trace 默认截断不重复存储（隐私/体积双控）。
    """
    if text is None or capture == "none":
        return None
    s = text if isinstance(text, str) else str(text)
    if capture == "full" or len(s) <= max_chars:
        return s
    return s[:max_chars] + f"…[截断，原文 {len(s)} 字符]"


@dataclass
class Span:
    """一段有起止时间的操作记录，按 parent_id 构成树。

    start_time 为墙钟（epoch 秒，展示用）；duration_ms 由 monotonic 差值计算
    （精确计时，不受系统时间调整影响）。status 取 "ok" / "error" / "interrupted"
    （仅根 span：消费者提前断开）。
    """

    trace_id: str
    span_id: str
    parent_id: str | None
    name: str       # "invoke_agent" | "chat {model}" | "execute_tool {tool}" | ...
    kind: str       # agent | generation | tool | guardrail | confirmation | compaction
    start_time: float
    start_monotonic: float
    end_time: float | None = None
    duration_ms: float | None = None
    status: str = "ok"
    attributes: dict = field(default_factory=dict)
    events: list = field(default_factory=list)

    def set_attr(self, key: str, value) -> None:
        self.attributes[key] = value

    def set_attrs(self, mapping: dict) -> None:
        self.attributes.update(mapping)

    def add_event(self, name: str) -> float:
        """记录 span 内时间点事件（如 first_token），返回相对起点的毫秒偏移。"""
        offset_ms = round((time.monotonic() - self.start_monotonic) * 1000, 1)
        self.events.append({"name": name, "offset_ms": offset_ms})
        return offset_ms

    def to_dict(self) -> dict:
        """序列化为 JSONL 行（start_monotonic 是进程内计时基准，不落盘）。"""
        d = {
            "schema": SCHEMA_VERSION,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
        }
        if self.events:
            d["events"] = self.events
        return d


class _NoopSpan:
    """关闭追踪时的占位 span：所有方法空操作，调用方无需判空。"""

    __slots__ = ()

    def set_attr(self, key, value) -> None:
        pass

    def set_attrs(self, mapping) -> None:
        pass

    def add_event(self, name) -> None:
        return None


NOOP_SPAN = _NoopSpan()
