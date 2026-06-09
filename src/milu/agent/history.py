"""对话历史管理 - 支持多种截断策略和自动压缩

strategy 可选值：
  "auto_compact"    — 自动压缩（默认）。需要 llm 参数，不可用时降级为 sliding_window
  "sliding_window"  — 滑动窗口截断，保留最近 max_turns 条消息
  "token_limit"     — 按 token 数截断
  "head_tail"       — 保留头尾消息
  "none"            — 不截断
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from milu.llm.base.message import Message, MessageRole

if TYPE_CHECKING:
    from milu.agent.config import CompactConfig
    from milu.agent.session import Session
    from milu.llm.providers.base import BaseLLM


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数：中文约 1.5 字/token，英文约 4 字符/token"""
    if not isinstance(text, str):
        return 0
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


# 被中断（达到上限/超时/报错）时，补给孤儿 tool_call 的占位结果文本。
_INTERRUPTED_TOOL_RESULT = (
    "[工具未执行：上一轮在工具执行前被中断（达到轮次/调用上限、超时或出错）。"
    "如仍需该结果，请重新发起调用。]"
)


def repair_tool_call_sequence(messages: list[Message]) -> list[Message]:
    """修复工具调用/结果配对，保证发送给 LLM 的消息序列合法（不改入参，返回新列表）。

    OpenAI 及兼容协议（含 MiniMax）要求：每条 assistant(tool_calls) 之后必须紧跟
    与各 tool_call_id 一一对应的 tool 结果消息。当上一轮在工具执行【前】被中断
    （达到 max_turns / tool_call_limit、超时或报错），历史会留下"孤儿"——
    assistant(tool_calls) 后没有 tool 结果；下一轮再追加 user 消息（如输入"继续"）
    发送时，MiniMax 报 400 "tool call result does not follow tool call (2013)"，
    且每次重发都失败，无法恢复。

    本函数修复两类破坏：
      1. assistant(tool_calls) 缺失对应 tool 结果 → 补一条占位 tool 结果；
      2. 找不到前置 tool_call 的孤儿 tool 结果（如压缩截断切断了配对）→ 丢弃。
    幂等：合法序列原样返回。在每次发送前调用，可同时修好「历史已损坏的旧会话」
    与「未来任何中断源」，无需改写 append-only 会话日志。
    """
    result: list[Message] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
            result.append(msg)
            declared = [tc.get("id") for tc in msg.tool_calls if tc.get("id")]
            # 收集紧随其后的 tool 结果（直到遇到非 tool 消息），按 id 去重
            seen: dict[str, Message] = {}
            j = i + 1
            while j < n and messages[j].role == MessageRole.TOOL:
                tmsg = messages[j]
                if tmsg.tool_call_id in declared and tmsg.tool_call_id not in seen:
                    seen[tmsg.tool_call_id] = tmsg
                # 不匹配已声明 id 的 tool 结果（孤儿）直接跳过丢弃
                j += 1
            # 按声明顺序输出结果；缺失的补占位（满足"每个 tool_call 都有结果"）
            for cid in declared:
                if cid in seen:
                    result.append(seen[cid])
                else:
                    name = next(
                        (tc.get("function", {}).get("name")
                         for tc in msg.tool_calls if tc.get("id") == cid),
                        None,
                    )
                    result.append(Message(
                        role=MessageRole.TOOL,
                        content=_INTERRUPTED_TOOL_RESULT,
                        tool_call_id=cid,
                        name=name,
                    ))
            i = j
        elif msg.role == MessageRole.TOOL:
            # 走到这里说明该 tool 结果前面没有 assistant(tool_calls)（孤儿）→ 丢弃
            i += 1
        else:
            result.append(msg)
            i += 1
    return result


class ConversationHistory:
    """对话历史管理 - 支持多种截断策略和上下文压缩"""

    def __init__(
        self,
        strategy: str = "auto_compact",
        max_turns: int = 50,
        max_tokens: int | None = None,
        preserve_system: bool = True,
        head_turns: int = 4,
        tail_turns: int = 10,
        # 压缩配置（新增）
        llm: "BaseLLM | None" = None,
        compact_config: "CompactConfig | None" = None,
    ):
        self._messages: list[Message] = []
        self._strategy = strategy
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._preserve_system = preserve_system
        self._head_turns = head_turns
        self._tail_turns = tail_turns
        self._session: "Session | None" = None  # 会话日志（可选）

        # 压缩器：strategy="auto_compact" 且有 llm 时启用
        self._compactor = None
        if compact_config is None:
            from milu.agent.config import CompactConfig
            compact_config = CompactConfig()
        if strategy == "auto_compact" and compact_config.enabled and llm:
            from milu.agent.compactor import Compactor
            self._compactor = Compactor(llm, compact_config=compact_config)
        elif strategy == "auto_compact":
            # auto_compact 不可用（缺少 llm 或被禁用），降级为滑动窗口
            self._strategy = "sliding_window"

    def set_system(self, message: Message) -> None:
        """设置 system 消息（始终作为第一条）"""
        if self._messages and self._messages[0].role == MessageRole.SYSTEM:
            self._messages[0] = message
        else:
            self._messages.insert(0, message)

    def add(self, message: Message) -> None:
        """追加一条消息，同时写入会话日志（如有）"""
        self._messages.append(message)
        if self._session:
            self._session.log_message(message)

    def attach_session(self, session: "Session") -> None:
        """绑定会话日志，后续 add() 会自动写入 JSONL，同时绑定压缩器用于生成文件指针"""
        self._session = session
        if self._compactor:
            self._compactor._session = session

    def detach_session(self) -> None:
        """解绑会话日志"""
        self._session = None

    @property
    def compact_enabled(self) -> bool:
        """压缩是否启用"""
        return self._compactor is not None

    def update_prompt_tokens(self, tokens: int) -> None:
        """更新 prompt token 数（委托给压缩器）"""
        if self._compactor:
            self._compactor.update_prompt_tokens(tokens)

    async def auto_compact(self) -> list[Message]:
        """自动压缩流水线（委托给压缩器）"""
        if not self._compactor:
            return self._messages
        compacted = await self._compactor.auto_compact(self._messages)
        if compacted is not self._messages:
            self._messages = compacted
            self._log_compaction(compacted)
        return compacted

    async def reactive_compact(self) -> list[Message]:
        """应急压缩（委托给压缩器）"""
        if not self._compactor:
            return self._messages
        compacted = await self._compactor.reactive_compact(self._messages)
        if compacted is not self._messages:
            self._messages = compacted
            self._log_compaction(compacted)
        return compacted

    async def manual_compact(self, focus: str = "") -> tuple[list[Message], str]:
        """手动压缩（委托给压缩器）。

        调用方需要在 replace_all() 后调用 _log_compaction() 持久化压缩状态。
        """
        if not self._compactor:
            return self._messages, ""
        return await self._compactor.manual_compact(self._messages, focus)

    def _log_compaction(self, messages: list[Message]) -> None:
        """将压缩后的消息快照写入 session JSONL。"""
        if self._session:
            self._session.log_compaction(messages)

    def load_from_session(self, session: "Session") -> None:
        """从会话日志批量加载消息（不重复写入 JSONL）"""
        self._session = session
        self._messages = session.load_messages()

    def get_messages(self) -> list[Message]:
        """获取截断后的消息列表（用于传给 LLM）"""
        if self._strategy == "none":
            return list(self._messages)
        elif self._strategy == "sliding_window":
            return self._apply_sliding_window()
        elif self._strategy == "token_limit":
            return self._apply_token_limit()
        elif self._strategy == "head_tail":
            return self._apply_head_tail()
        else:
            return list(self._messages)

    def clear(self) -> None:
        """清空历史（保留 system 消息）"""
        if self._preserve_system and self._messages and self._messages[0].role == MessageRole.SYSTEM:
            self._messages = [self._messages[0]]
        else:
            self._messages = []

    @property
    def session(self) -> "Session | None":
        """当前绑定的会话（None 表示未启用持久化）"""
        return self._session

    @property
    def all_messages(self) -> list[Message]:
        """获取完整未截断的历史（用于调试/日志）"""
        return list(self._messages)

    def replace_all(self, messages: list[Message]) -> None:
        """替换全部消息（压缩后调用）。

        system 消息在下一轮 _build_system_prompt 中会自动重建。
        """
        self._messages = messages

    def _apply_sliding_window(self) -> list[Message]:
        """滑动窗口截断：保留 system + 最近 max_turns 条消息"""
        non_system = [m for m in self._messages if m.role != MessageRole.SYSTEM]
        if len(non_system) <= self._max_turns:
            return list(self._messages)

        if self._preserve_system and self._messages and self._messages[0].role == MessageRole.SYSTEM:
            system = self._messages[0]
            kept = non_system[-self._max_turns:]
            return [system] + kept
        else:
            return self._messages[-self._max_turns:]

    def _apply_token_limit(self) -> list[Message]:
        """按 token 数截断"""
        if not self._max_tokens:
            return list(self._messages)

        total_tokens = 0
        keep_from_idx = len(self._messages)

        for i in range(len(self._messages) - 1, -1, -1):
            msg = self._messages[i]
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            tokens = _estimate_tokens(content)

            if total_tokens + tokens > self._max_tokens:
                break

            total_tokens += tokens
            keep_from_idx = i

        if self._preserve_system and self._messages and self._messages[0].role == MessageRole.SYSTEM:
            if keep_from_idx == 0:
                return list(self._messages)
            else:
                return [self._messages[0]] + self._messages[max(1, keep_from_idx):]
        else:
            return self._messages[keep_from_idx:]

    def _apply_head_tail(self) -> list[Message]:
        """头尾截断：保留前 head_turns 条和后 tail_turns 条非 system 消息"""
        non_system = [m for m in self._messages if m.role != MessageRole.SYSTEM]
        if len(non_system) <= self._head_turns + self._tail_turns:
            return list(self._messages)

        head = non_system[:self._head_turns]
        tail = non_system[-self._tail_turns:]

        separator = Message(
            role=MessageRole.SYSTEM,
            content="... [中间消息已省略] ..."
        )

        result = head + [separator] + tail

        # 保留 system 消息
        if self._preserve_system and self._messages and self._messages[0].role == MessageRole.SYSTEM:
            result = [self._messages[0]] + result

        return result
