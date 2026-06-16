"""Tracer —— span 生命周期管理 + ContextVar 注入。

设计要点（详见 docs/可观测性方案设计.md）：
- **ContextVar 注入与 judge/todo 同模式**：`_current_tracer` 沿 asyncio 任务树
  继承——子代理的 run() 检测到已有 tracer，把自己的 invoke_agent span 挂在
  当前活动 span（父的 execute_tool span）之下，委派链路零额外代码完整可追溯；
  `_current_span` 在工具批次的 gather 任务内各自设置，并发隔离天然成立。
- **fail-silent**：sink 写入失败只记 warning（与 judge 的 fail-open 哲学一致），
  观测是增强能力，绝不影响业务运行。
- **关闭零开销**：NOOP_TRACER 所有方法空操作；`Agent(trace=False)`（库默认）
  不注入 ContextVar，单测 hermetic。
"""
from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass, field

from milu.observability.span import (
    NOOP_SPAN,
    Span,
    clip_content,
    new_span_id,
    new_trace_id,
)

logger = logging.getLogger(__name__)

# per-run 追踪上下文（asyncio 任务级隔离；子代理经任务树自动继承父值）
_current_tracer: ContextVar = ContextVar("milu_current_tracer", default=None)
_current_span: ContextVar = ContextVar("milu_current_span", default=None)


@dataclass
class TraceConfig:
    """追踪配置。库内默认不启用（Agent(trace=False)）；CLI/Web 按
    config.json observability 分节构造并传入（应用层默认开）。"""

    enabled: bool = True
    capture_content: str = "truncated"   # full | truncated | none（参数/输入的捕获级别）
    max_content_chars: int = 2000        # truncated 级别的截断长度
    retention_days: int = 30             # ~/.milu/traces 散件保留天数（应用层启动时清理）
    runs_index: bool = True              # 顶层运行结束追加 RunReport 到 ~/.milu/runs.jsonl
    user_id: str | None = None           # 多用户标识（入 runs 索引，Web 层按用户过滤）
    price_table: dict = field(default_factory=dict)   # 覆盖内置价格表（pricing.py）
    extra_sinks: list = field(default_factory=list)   # 默认 JSONL sink 之外追加的 sink

    # config.json observability 分节里属于本 dataclass 的可调键
    _MAPPING_KEYS = ("capture_content", "max_content_chars", "retention_days")

    @classmethod
    def from_mapping(cls, m: dict, user_id: str | None = None) -> "TraceConfig":
        """从 config.json observability 分节构造（未知键忽略；enabled 是应用层
        开关，由调用方自行判断后才调用本方法——与 KnowledgeConfig.from_mapping 同约定）。"""
        kwargs = {k: m[k] for k in cls._MAPPING_KEYS if k in m}
        return cls(
            user_id=user_id,
            price_table=dict(m.get("price_table") or {}),
            **kwargs,
        )


class Tracer:
    """单次 trace（一棵 span 树）的生命周期管理器，per-run 创建，无共享可变状态。"""

    enabled = True

    def __init__(self, config: TraceConfig, sinks: list, trace_id: str | None = None):
        self.config = config
        self.sinks = sinks
        self.trace_id = trace_id or new_trace_id()
        self.trace_path = None   # JSONL 落盘路径（构造方设置；入 RunReport 供下钻定位）
        # 已完成 span（含子代理的），供运行结束时聚合 RunReport；单次运行量级有限
        self.finished: list[Span] = []

    def clip(self, text) -> str | None:
        """按本 trace 的捕获级别处理内容（None 表示不记录该属性）。"""
        return clip_content(text, self.config.capture_content, self.config.max_content_chars)

    def start(self, name: str, kind: str) -> Span:
        """开启一个 span，父子关系取自 _current_span（调用方按需 set 该 ContextVar）。"""
        parent = _current_span.get()
        return Span(
            trace_id=self.trace_id,
            span_id=new_span_id(),
            parent_id=parent.span_id if isinstance(parent, Span) else None,
            name=name,
            kind=kind,
            start_time=time.time(),
            start_monotonic=time.monotonic(),
        )

    def finish(self, span, status: str | None = None, error_type: str | None = None) -> None:
        """结束 span 并发往所有 sink。幂等；NoopSpan / 未开启的 span 安全忽略。

        未调用 finish 的 span（如未实际发生的压缩）不落盘——天然过滤空操作。
        """
        if not isinstance(span, Span) or span.end_time is not None:
            return
        span.end_time = time.time()
        span.duration_ms = round((time.monotonic() - span.start_monotonic) * 1000, 1)
        if status:
            span.status = status
        if error_type:
            span.attributes["error.type"] = error_type
        self.finished.append(span)
        for sink in self.sinks:
            try:
                sink.emit(span)
            except Exception as e:
                # fail-silent：观测不影响业务
                logger.warning("trace sink 写入失败（已忽略）: %s", e)


class _NoopTracer:
    """关闭追踪时的占位实现：零开销、零副作用。"""

    enabled = False

    def clip(self, text) -> None:
        return None

    def start(self, name, kind):
        return NOOP_SPAN

    def finish(self, span, status=None, error_type=None) -> None:
        pass


NOOP_TRACER = _NoopTracer()


def extract_provider_name(llm) -> str:
    """从 LLM 实例反查厂商名（ModelRegistry 注册名；查不到回退类名小写去 llm 后缀）。

    对应 OTel 属性 gen_ai.provider.name：官方枚举含 openai/anthropic/deepseek 等，
    枚举外（qwen/kimi/glm/minimax/doubao）按 spec 允许用自定义小写值。
    """
    try:
        from milu.llm.providers import ModelRegistry
        for name, cls in ModelRegistry._providers.items():
            if type(llm) is cls:
                return name
    except Exception:
        pass
    name = type(llm).__name__.lower()
    return name[:-3] if name.endswith("llm") and len(name) > 3 else name
