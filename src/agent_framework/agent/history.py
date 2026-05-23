"""对话历史管理 - 支持多种截断策略"""
from __future__ import annotations

import re

from agent_framework.models.message import Message, MessageRole


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数：中文约 1.5 字/token，英文约 4 字符/token"""
    if not isinstance(text, str):
        return 0
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


class ConversationHistory:
    """对话历史管理 - 支持多种截断策略"""

    def __init__(
        self,
        strategy: str = "sliding_window",
        max_turns: int = 50,
        max_tokens: int | None = None,
        preserve_system: bool = True,
        head_turns: int = 4,
        tail_turns: int = 10,
    ):
        self._messages: list[Message] = []
        self._strategy = strategy
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._preserve_system = preserve_system
        self._head_turns = head_turns
        self._tail_turns = tail_turns

    def set_system(self, message: Message) -> None:
        """设置 system 消息（始终作为第一条）"""
        if self._messages and self._messages[0].role == MessageRole.SYSTEM:
            self._messages[0] = message
        else:
            self._messages.insert(0, message)

    def add(self, message: Message) -> None:
        """追加一条消息"""
        self._messages.append(message)

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
    def all_messages(self) -> list[Message]:
        """获取完整未截断的历史（用于调试/日志）"""
        return list(self._messages)

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
