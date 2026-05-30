"""Agent 配置"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Agent 运行配置"""
    max_turns: int = 100            # 最大循环轮次（防无限循环）
    timeout: float = 120.0         # 单次 LLM 调用超时（秒）
    total_timeout: float = 60 * 60 * 1   # 整个 run() 的总超时（秒）
    max_total_tokens: int | None = None   # 总 token 上限（None=不限制）
    tool_call_limit: int = 100      # 单次 run() 中最大工具调用次数
    confirm_dangerous: bool = True  # 危险工具调用前需确认（预留接口）
    mcp_tools_active_by_default: bool = False  # MCP 工具默认激活（False=进入休眠池，需手动激活）
    compact_enabled: bool = True       # 自动上下文压缩总开关
    compact_threshold: int = 80000     # L4 触发阈值（字符数），超过时调用 LLM 生成摘要
    compact_keep_recent: int = 3       # L2 保留最近 N 个工具结果不压缩
    compact_max_messages: int = 50     # L1 消息数量上限，超过时裁剪中间消息

    # Session 会话配置
    session_enabled: bool = True           # 自动创建会话（对话日志持久化）
    session_dir: str | None = None         # 会话目录（默认 .sessions/）
