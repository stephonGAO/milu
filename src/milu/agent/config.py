"""Agent 配置"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AgentMode(str, Enum):
    """Agent 操作模式

    四档权限递增：talk < manual < auto < superwork。

    | 模式      | 安全工具 | 不安全工具                                      |
    |-----------|---------|------------------------------------------------|
    | talk      | 执行     | 阻止                                           |
    | manual    | 执行     | 人工审批                                        |
    | auto(默认) | 执行     | 自动执行；配 judge_llm 时由 AI 判定兜底         |
    | superwork | 执行     | 自动执行（不经 AI 判定）                        |

    auto 模式可选配 Agent(judge_llm=...) 启用 AI 安全判定器（见 judge.py）：
    不安全工具调用交由小模型判定 allow（执行）/ confirm（转人工）/ deny（拒绝）。
    """
    TALK = "talk"             # 只读模式：仅允许安全工具，不可修改/执行
    MANUAL = "manual"         # 人工审批模式：安全工具直接执行，不安全工具需人工审批
    AUTO = "auto"             # 自主模式（默认）：自主决策执行不安全工具，可选 AI 判定兜底
    SUPERWORK = "superwork"   # 全权限模式：跳过所有安全检查（含 AI 判定）


@dataclass
class CompactConfig:
    """上下文压缩配置。

    压缩按「上下文实际用量 / 模型最大上下文窗口」分阶段触发，全部与窗口大小挂钩，
    避免大窗口模型（如 131K 的 qwen、1M 的 minimax）在窗口还很空时就过早压缩：
      用量 < round_trigger_ratio        → 不压缩（保留完整工具结果）
      round_trigger_ratio ~ trigger_ratio → 轮次分层截断旧轮工具结果，保留最近 recent_rounds 轮
      用量 >= trigger_ratio              → 收紧最近轮为 0 + L4 LLM 摘要
    """
    enabled: bool = True                # 自动压缩总开关
    round_trigger_ratio: float = 0.5    # 轮次分层压缩启动阈值：prompt_tokens / max_context_window >= 此比例才压缩
    trigger_ratio: float = 0.7          # L4 LLM 摘要触发：prompt_tokens / max_context_window >= 此比例
    recent_rounds: int = 5              # 保留最近 N 轮工具结果完整（动态：用量达 trigger_ratio 时降为 0）
    max_messages: int = 300              # L1 消息数量硬上限（条数兜底，与窗口无关）


@dataclass
class AgentConfig:
    """Agent 运行限额配置（纯调参，运行期不被原地修改）。

    注意：以下「能力/身份」参数已上移为 Agent.__init__ 的直接参数，不再属于 AgentConfig：
      - mode（操作模式 talk/manual/auto/superwork）
      - session_enabled / session_dir（会话持久化）
      - mcp_tools_active_by_default（MCP 工具是否默认激活）
    它们更属于 Agent 实例能力，且 mode 运行期可变；放回实例字段后天然无跨用户串扰风险。
    """
    # 以下 max_turns / tool_call_limit 是「防死循环」兜底，不是正常工作预算。
    # 真正的成本闸是 total_timeout（默认 1 小时）。复杂编码任务（多文件 CRUD 等）
    # 轻松超过百次工具调用，故兜底放宽，避免任务做一半就 AgentError 中断。
    max_turns: int = 200            # 最大循环轮次（防无限循环）
    timeout: float = 300.0         # 单次 LLM 调用超时（秒）
    total_timeout: float = 60 * 60 * 1   # 整个 run() 的总超时（秒）
    max_total_tokens: int | None = None   # 总 token 上限（None=不限制）
    tool_call_limit: int = 1000     # 单次 run() 中最大工具调用次数
