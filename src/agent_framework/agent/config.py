"""Agent 配置"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AgentMode(str, Enum):
    """Agent 操作模式"""
    TALK = "talk"             # 只读模式：仅允许安全工具，不可修改/执行
    AUTO = "auto"             # 标准模式：安全工具直接执行，不安全工具需审批
    SUPERWORK = "superwork"   # 全权限模式：跳过所有安全检查


@dataclass
class CompactConfig:
    """上下文压缩配置"""
    enabled: bool = True                # 自动压缩总开关
    trigger_ratio: float = 0.7          # L4 触发：prompt_tokens / max_context_window > 此比例
    recent_rounds: int = 5              # 保留最近 N 轮工具结果完整（动态：超 30% 时降为 0）
    max_messages: int = 300              # L1 消息数量上限


@dataclass
class AgentConfig:
    """Agent 运行配置"""
    max_turns: int = 100            # 最大循环轮次（防无限循环）
    timeout: float = 300.0         # 单次 LLM 调用超时（秒）
    total_timeout: float = 60 * 60 * 1   # 整个 run() 的总超时（秒）
    max_total_tokens: int | None = None   # 总 token 上限（None=不限制）
    tool_call_limit: int = 100      # 单次 run() 中最大工具调用次数
    mcp_tools_active_by_default: bool = False  # MCP 工具默认激活（False=进入休眠池，需手动激活）
    mode: AgentMode = AgentMode.AUTO           # 操作模式：talk（只读）/ auto（标准）/ superwork（全权限）

    # Session 会话配置
    session_enabled: bool = True           # 自动创建会话（对话日志持久化）
    session_dir: str | None = None         # 会话目录（默认 .sessions/）
