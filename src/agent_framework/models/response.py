"""统一响应模型 - 流式输出数据块和Token用量统计"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenUsage:
    """统一的Token用量统计。"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass
class StreamChunk:
    """流式输出的统一数据块。"""
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None
