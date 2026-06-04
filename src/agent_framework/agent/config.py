"""Agent 配置"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AgentMode(str, Enum):
    """Agent 操作模式

    四档权限递增：talk < manual < auto < superwork。
    「高危工具」指 @tool(requires_confirm=True) 标记的工具（支付、删库等不可逆操作）。

    | 模式      | 安全工具 | 不安全工具       | 高危工具(requires_confirm) |
    |-----------|---------|-----------------|---------------------------|
    | talk      | 执行     | 阻止            | 阻止                       |
    | manual    | 执行     | 人工审批         | 人工审批                   |
    | auto(默认) | 执行     | 自动执行(自主)   | 人工审批                   |
    | superwork | 执行     | 自动执行         | 自动执行                   |
    """
    TALK = "talk"             # 只读模式：仅允许安全工具，不可修改/执行
    MANUAL = "manual"         # 人工审批模式：安全工具直接执行，不安全工具需人工审批
    AUTO = "auto"             # 自主模式（默认）：自主决策执行不安全工具，仅高危工具需人工审批
    SUPERWORK = "superwork"   # 全权限模式：跳过所有安全检查（含高危工具）


@dataclass
class CompactConfig:
    """上下文压缩配置"""
    enabled: bool = True                # 自动压缩总开关
    trigger_ratio: float = 0.7          # L4 触发：prompt_tokens / max_context_window > 此比例
    recent_rounds: int = 5              # 保留最近 N 轮工具结果完整（动态：超 30% 时降为 0）
    max_messages: int = 300              # L1 消息数量上限


@dataclass
class AgentConfig:
    """Agent 运行限额配置（纯调参，运行期不被原地修改）。

    注意：以下「能力/身份」参数已上移为 Agent.__init__ 的直接参数，不再属于 AgentConfig：
      - mode（操作模式 talk/manual/auto/superwork）
      - session_enabled / session_dir（会话持久化）
      - mcp_tools_active_by_default（MCP 工具是否默认激活）
    它们更属于 Agent 实例能力，且 mode 运行期可变；放回实例字段后天然无跨用户串扰风险。
    """
    max_turns: int = 100            # 最大循环轮次（防无限循环）
    timeout: float = 300.0         # 单次 LLM 调用超时（秒）
    total_timeout: float = 60 * 60 * 1   # 整个 run() 的总超时（秒）
    max_total_tokens: int | None = None   # 总 token 上限（None=不限制）
    tool_call_limit: int = 100      # 单次 run() 中最大工具调用次数
