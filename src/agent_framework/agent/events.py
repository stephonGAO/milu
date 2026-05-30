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


@dataclass
class ConfirmResponse:
    """危险工具确认回调的返回值。

    Attributes:
        approved: True=同意执行, False=拒绝
        message: 自定义消息（拒绝时发回给 LLM 作为指导，
                 如 "不要用 rm，改用 mv 到回收站"）
    """
    approved: bool
    message: str = ""


@dataclass(frozen=True)
class ToolConfirmRequired(AgentEvent):
    """危险工具执行前请求用户确认"""
    tool_name: str
    tool_call_id: str
    arguments: str  # JSON 字符串
    approved: bool  # True=用户同意执行, False=用户拒绝
    message: str = ""  # 用户自定义消息（拒绝时可发给 LLM）


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


@dataclass(frozen=True)
class SubAgentEvent(AgentEvent):
    """子代理执行过程中的内部事件。

    父 Agent 转发此事件，让消费者能区分子代理输出和父代理输出。
    内部事件可以是 TextDelta、ReasoningDelta、ToolCallStart、ToolResult。
    """
    subagent_name: str    # 哪个子代理产生的事件
    event: AgentEvent     # 实际事件


@dataclass(frozen=True)
class SubAgentDone(AgentEvent):
    """子代理执行完毕。

    在一次 SubAgentEvent 序列结束后发出一次，携带摘要元数据。
    """
    subagent_name: str
    final_text: str         # 子代理最终回答
    turn_count: int         # 子代理使用的 LLM 轮数
    total_usage: TokenUsage  # 子代理的 token 消耗
    is_error: bool          # 是否以错误结束


@dataclass(frozen=True)
class HistoryCompacted(AgentEvent):
    """对话历史被压缩。

    在自动压缩（auto）、手动压缩（manual）或应急压缩（reactive）后发出。
    """
    strategy: str          # "auto" | "manual" | "reactive"
    original_count: int    # 压缩前消息数
    compacted_count: int   # 压缩后消息数
