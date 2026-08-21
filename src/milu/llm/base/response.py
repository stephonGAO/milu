"""统一响应模型 - 流式输出数据块和Token用量统计"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _usage_value(obj: Any, name: str) -> Any:
    """兼容 SDK 对象与原始字典两种 usage 表示。"""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _usage_int(obj: Any, name: str) -> int:
    value = _usage_value(obj, name)
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


@dataclass
class TokenUsage:
    """统一的Token用量统计。"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    # 缓存命中的输入 token；是 prompt_tokens 的子集，不重复计入 total_tokens。
    cached_tokens: int = 0

    @classmethod
    def from_api_usage(cls, usage: Any) -> "TokenUsage":
        """解析 OpenAI/兼容 API 的 usage，并保留输入缓存命中量。

        Chat Completions 使用 ``prompt_tokens_details.cached_tokens``，Responses API
        使用 ``input_tokens_details.cached_tokens``；DeepSeek 还会在 usage 顶层返回
        ``prompt_cache_hit_tokens``。这里统一兼容，provider 不再各自丢弃明细字段。
        """
        prompt_tokens = _usage_int(usage, "prompt_tokens")
        if not prompt_tokens:
            prompt_tokens = _usage_int(usage, "input_tokens")

        completion_tokens = _usage_int(usage, "completion_tokens")
        if not completion_tokens:
            completion_tokens = _usage_int(usage, "output_tokens")

        total_tokens = _usage_int(usage, "total_tokens")
        if not total_tokens:
            total_tokens = prompt_tokens + completion_tokens

        input_details = (
            _usage_value(usage, "prompt_tokens_details")
            or _usage_value(usage, "input_tokens_details")
        )
        output_details = (
            _usage_value(usage, "completion_tokens_details")
            or _usage_value(usage, "output_tokens_details")
        )
        cached_tokens = _usage_int(input_details, "cached_tokens")
        if not cached_tokens:
            # DeepSeek 的官方字段；cached_tokens 顶层别名兼容少数网关。
            cached_tokens = (
                _usage_int(usage, "prompt_cache_hit_tokens")
                or _usage_int(usage, "cached_tokens")
            )

        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=_usage_int(output_details, "reasoning_tokens"),
            cached_tokens=min(cached_tokens, prompt_tokens) if prompt_tokens else cached_tokens,
        )


@dataclass
class StreamChunk:
    """流式输出的统一数据块。"""
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None
