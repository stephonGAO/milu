"""Agent 配置"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Agent 运行配置"""
    max_turns: int = 10            # 最大循环轮次（防无限循环）
    timeout: float = 120.0         # 单次 LLM 调用超时（秒）
    total_timeout: float = 300.0   # 整个 run() 的总超时（秒）
    max_total_tokens: int | None = None   # 总 token 上限（None=不限制）
    tool_call_limit: int = 20      # 单次 run() 中最大工具调用次数
    confirm_dangerous: bool = True  # 危险工具调用前需确认（预留接口）
