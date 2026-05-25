"""Agent 事件类型定义 - 用于 Agent 循环的事件流输出"""

from __future__ import annotations

from dataclasses import dataclass

from agent_framework.llm.base.response import TokenUsage


@dataclass(frozen=True)
class AgentEvent:
    """Agent 事件基类"""
    pass


@dataclass(frozen=True)
class TextDelta(AgentEvent):
    """LLM 正文输出片段"""
    text: str


@dataclass(frozen=True)
class ReasoningDelta(AgentEvent):
    """LLM 思考过程输出片段"""
    text: str


@dataclass(frozen=True)
class ToolCallStart(AgentEvent):
    """LLM 决定调用工具"""
    tool_name: str
    tool_call_id: str
    arguments: str  # JSON 字符串


@dataclass(frozen=True)
class ToolResult(AgentEvent):
    """工具执行完成"""
    tool_name: str
    tool_call_id: str
    output: str
    is_error: bool


@dataclass(frozen=True)
class ToolConfirmRequired(AgentEvent):
    """危险工具执行前请求用户确认"""
    tool_name: str
    tool_call_id: str
    arguments: str  # JSON 字符串
    approved: bool  # True=用户同意执行, False=用户拒绝


@dataclass(frozen=True)
class AgentDone(AgentEvent):
    """Agent 循环正常结束"""
    final_text: str
    total_usage: TokenUsage
    turn_count: int


@dataclass(frozen=True)
class AgentError(AgentEvent):
    """Agent 异常终止"""
    error_type: str  # "max_turns" | "call_timeout" | "total_timeout" | "token_limit" | "tool_limit"
    message: str
