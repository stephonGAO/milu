"""上下文压缩流水线 — 四层压缩，便宜优先、昂贵兜底

执行顺序（每轮 LLM 调用前）：
    L3 budget → L1 snip → L2 micro → [超阈值?] L4 compact

层级说明：
  L1 snip_compact:     消息数 > max_messages 时裁剪中间消息（0 API 调用）
  L2 micro_compact:    旧工具结果替换为占位符（0 API 调用）
  L3 budget:           旧轮次超大工具结果截断并持久化到磁盘（保护最新一轮，0 API 调用）
  L4 compact_history:  调用 LLM 生成对话摘要，智能保留尾部 0-3 条（1 API 调用）

使用方式：
    compactor = Compactor(llm, config)
    compacted = await compactor.auto_compact(messages)

参考：Claude Code compaction pipeline (query.ts / autoCompact.ts)
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

from agent_framework.agent.events import HistoryCompacted
from agent_framework.llm.base.message import Message, MessageRole
from agent_framework.tools.decorator import tool

if TYPE_CHECKING:
    from agent_framework.agent.agent import Agent
    from agent_framework.agent.config import AgentConfig
    from agent_framework.llm.providers.base import BaseLLM

logger = logging.getLogger(__name__)

# 大结果持久化目录
_COMPACT_OUTPUTS_DIR = Path.cwd() / ".compact_outputs"

# 大结果持久化阈值（字符数）
_PERSIST_THRESHOLD = 30000

# L4 序列化时每条消息的最大字符数
_MAX_MSG_CHARS = 80000

# 连续压缩失败上限
_MAX_CONSECUTIVE_FAILURES = 3


class Compactor:
    """四层上下文压缩器。

    用法：
        compactor = Compactor(llm, config)
        compacted = await compactor.auto_compact(messages)
        # compacted 可能与 messages 相同（无需压缩），也可能是新列表
    """

    def __init__(self, llm: "BaseLLM", config: "AgentConfig") -> None:
        self._llm = llm
        self._config = config
        self._consecutive_failures = 0

    async def auto_compact(self, messages: list[Message]) -> list[Message]:
        """自动压缩流水线：L3 → L1 → L2 → [超阈值?] L4。

        :param messages: 当前历史消息列表（会被原地修改或替换）
        :return: 压缩后的消息列表（可能与传入的是同一对象）
        """
        if len(messages) <= 1:
            return messages

        # L3: 大结果持久化（必须在 L2 之前）
        messages = self._tool_result_budget(messages)

        # L1: 消息数量裁剪
        messages = self._snip_compact(messages) #静默执行

        # L2: 旧工具结果占位
        messages = self._micro_compact(messages) #静默执行

        # L4: 超阈值时 LLM 摘要（连续失败时跳过）
        if (
            self._estimate_size(messages) > self._config.compact_threshold
            and self._consecutive_failures < _MAX_CONSECUTIVE_FAILURES
        ):
            try:
                messages = await self._compact_history(messages)
                self._consecutive_failures = 0
            except Exception as e:
                self._consecutive_failures += 1
                logger.warning(
                    "L4 压缩失败 (%d/%d): %s",
                    self._consecutive_failures, _MAX_CONSECUTIVE_FAILURES, e,
                )

        return messages

    async def manual_compact(
        self, messages: list[Message], focus: str = ""
    ) -> tuple[list[Message], str]:
        """手动压缩（LLM 调用 compact 工具时触发）。

        :param messages: 当前历史消息
        :param focus: 用户指定的关注主题
        :return: (压缩后消息列表, 摘要文本)
        """
        summary = await self._summarize(messages, focus)
        compacted = [Message(role=MessageRole.USER, content=f"[Compacted]\n\n{summary}")]
        return compacted, summary

    async def reactive_compact(self, messages: list[Message]) -> list[Message]:
        """应急压缩（API 返回 context too long 时触发）。

        压缩历史为摘要，保留最近 3 条消息原样不动。
        """
        try:
            # 只对前面的消息生成摘要，尾部 3 条保留完整内容
            keep_recent = 3
            to_summarize = messages[:-keep_recent] if len(messages) > keep_recent else messages
            summary = await self._summarize(to_summarize)
        except Exception as e:
            logger.warning("应急压缩失败: %s", e)
            return messages

        recent = messages[-keep_recent:] if len(messages) > keep_recent else []
        return [
            Message(role=MessageRole.USER, content=f"[Reactive compact]\n\n{summary}"),
            *recent,
        ]

    # ── L1: snip_compact ──────────────────────────────────────

    def _snip_compact(self, messages: list[Message]) -> list[Message]:
        """L1: 消息数超限时，保留头尾、裁剪中间。"""
        max_msgs = self._config.compact_max_messages
        if len(messages) <= max_msgs:
            return messages

        # 分离 system 和非 system
        system_msgs = [m for m in messages if m.role == MessageRole.SYSTEM]
        non_system = [m for m in messages if m.role != MessageRole.SYSTEM]

        if len(non_system) <= max_msgs:
            return messages

        keep_head = 3
        keep_tail = max_msgs - keep_head
        if keep_tail < 1:
            keep_tail = 1

        snipped_count = len(non_system) - keep_head - keep_tail
        snip_marker = Message(
            role=MessageRole.USER,
            content=f"[snipped {snipped_count} messages]",
        )

        result = system_msgs + non_system[:keep_head] + [snip_marker] + non_system[-keep_tail:]
        return result

    # ── L2: micro_compact ─────────────────────────────────────

    def _micro_compact(self, messages: list[Message]) -> list[Message]:
        """L2: 保留最近 N 个工具结果，旧结果替换为占位符。"""
        keep_recent = self._config.compact_keep_recent

        # 收集所有 tool 角色消息的索引
        tool_indices = [
            i for i, m in enumerate(messages) if m.role == MessageRole.TOOL
        ]

        if len(tool_indices) <= keep_recent:
            return messages

        # 替换除最近 keep_recent 个之外的工具结果
        for idx in tool_indices[:-keep_recent]:
            msg = messages[idx]
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(content) > 120:
                messages[idx] = Message(
                    role=MessageRole.TOOL,
                    content="[Earlier tool result compacted. Re-run if needed.]",
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )

        return messages

    # ── L3: tool_result_budget ────────────────────────────────

    def _tool_result_budget(self, messages: list[Message]) -> list[Message]:
        """L3: 仅对旧轮次的 tool 结果做截断持久化，保护最新一轮结果完整性。

        最新一轮（最后一条 assistant 之后的 tool 结果）保持原样，
        因为 LLM 主动请求了这些数据，还没来得及消化。
        """
        if not messages:
            return messages

        # 找到最后一条 assistant 消息的位置
        last_assistant_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == MessageRole.ASSISTANT:
                last_assistant_idx = i
                break

        if last_assistant_idx <= 0:
            return messages

        # 只处理最后一条 assistant 之前的 tool 消息（旧轮次）
        old_tool_blocks = []
        for i in range(last_assistant_idx):
            if messages[i].role == MessageRole.TOOL:
                old_tool_blocks.append(i)

        if not old_tool_blocks:
            return messages

        for idx in old_tool_blocks:
            msg = messages[idx]
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(content) <= _PERSIST_THRESHOLD:
                continue

            # 持久化到磁盘
            tool_id = msg.tool_call_id or msg.name or "unknown"
            persisted = self._persist_large_output(tool_id, content)

            messages[idx] = Message(
                role=MessageRole.TOOL,
                content=persisted,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            )

        return messages

    def _persist_large_output(self, tool_id: str, output: str) -> str:
        """将大输出持久化到磁盘，返回截断版本。"""
        _COMPACT_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r'[^\w\-]', '_', tool_id)
        path = _COMPACT_OUTPUTS_DIR / f"{safe_id}_{int(time.time())}.txt"
        path.write_text(output, encoding="utf-8")

        preview = output[:2000]
        return (
            f"<persisted-output>\n"
            f"Full output: {path}\n"
            f"Preview:\n{preview}\n"
            f"</persisted-output>"
        )

    # ── L4: compact_history ───────────────────────────────────

    async def _compact_history(self, messages: list[Message], preserve_tail: bool = True) -> list[Message]:
        """L4: 调用 LLM 生成对话摘要。

        :param messages: 当前历史消息
        :param preserve_tail: 是否保留尾部消息（自动压缩时为 True，保护最新数据）
        :return: 压缩后的消息列表
        """
        if not preserve_tail:
            # 全量压缩（手动模式）
            summary = await self._summarize(messages)
            return [Message(role=MessageRole.USER, content=f"[Compacted]\n\n{summary}")]

        # 智能保留尾部：优先保留 3 条，但尾部总大小不超过阈值的 30%
        threshold = self._config.compact_threshold
        max_tail_ratio = 0.3

        keep_count = 0
        for try_count in (3, 2, 1, 0):
            if try_count == 0:
                keep_count = 0
                break
            if len(messages) <= try_count:
                # 消息数不足，保留全部
                keep_count = len(messages)
                break
            # 检查尾部是否在 30% 阈值内
            tail_messages = messages[-try_count:]
            tail_size = self._estimate_size(tail_messages)
            if tail_size <= threshold * max_tail_ratio:
                keep_count = try_count
                break

        # 分离要摘要的部分和要保留的尾部
        if keep_count > 0:
            to_summarize = messages[:-keep_count]
            recent = messages[-keep_count:]
        else:
            to_summarize = messages
            recent = []

        if not to_summarize:
            return messages

        summary = await self._summarize(to_summarize)
        return [
            Message(role=MessageRole.USER, content=f"[Compacted]\n\n{summary}"),
            *recent,
        ]

    async def _summarize(
        self, messages: list[Message], focus: str = ""
    ) -> str:
        """调用 LLM 生成对话摘要。"""
        conversation = self._serialize_messages(messages)
        # 截断序列化结果，防止摘要请求本身超限
        if len(conversation) > _MAX_MSG_CHARS:
            conversation = conversation[:_MAX_MSG_CHARS] + "\n\n[... truncated ...]"

        focus_hint = f"\nFocus especially on: {focus}" if focus else ""
        prompt = (
            "Summarize this agent conversation so work can continue.\n"
            "Preserve:\n"
            "1. Current goal and task\n"
            "2. Key findings and decisions\n"
            "3. Files read or changed\n"
            "4. Remaining work\n"
            "5. User constraints and preferences\n"
            f"{focus_hint}\n"
            "Be compact but concrete. Include specific file paths, variable names, "
            "and decisions where relevant.\n\n"
            f"Conversation:\n{conversation}"
        )

        summary_msg = Message(role=MessageRole.USER, content=prompt)
        parts = []
        try:
            async for chunk in self._llm.chat([summary_msg]):
                if chunk.content:
                    parts.append(chunk.content)
                if chunk.finish_reason == "stop":
                    break
        except Exception as e:
            logger.warning("摘要 LLM 调用失败: %s", e)
            raise

        summary = "".join(parts).strip()
        return summary or "(empty summary)"

    # ── 辅助方法 ──────────────────────────────────────────────

    def _serialize_messages(self, messages: list[Message]) -> str:
        """将消息列表序列化为可读文本。"""
        lines = []
        for msg in messages:
            role = msg.role.value.upper()
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if not content:
                content = "(no content)"

            # 截断过长的单条消息
            if len(content) > 2000:
                content = content[:2000] + "... [truncated]"

            line = f"[{role}]"
            if msg.name:
                line += f" ({msg.name})"
            line += f": {content}"

            # 工具调用信息
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    fn = tc.get("function", {})
                    line += f"\n  → tool_call: {fn.get('name', '?')}({fn.get('arguments', '')[:100]})"

            lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def _estimate_size(messages: list[Message]) -> int:
        """估算消息列表的总字符数。"""
        total = 0
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            total += len(content)
            if msg.tool_calls:
                total += len(json.dumps(msg.tool_calls, ensure_ascii=False))
        return total


# ── compact 元工具工厂 ────────────────────────────────────────


def create_compact_tool(agent: "Agent"):
    """创建 compact 元工具，让 LLM 主动触发上下文压缩。

    遵循 catalog.py 的模式：通过闭包持有 Agent 引用。
    """

    @tool(
        name="compact",
        description=(
            "压缩对话历史以释放上下文空间。"
            "当对话过长、上下文即将溢出、或需要整理当前进度时调用此工具。"
            "压缩后对话历史会被替换为一段摘要，之前的详细对话内容将丢失。"
        ),
    )
    async def compact(focus: str = "") -> str:
        """
        :param focus: 压缩时特别关注的主题（如 "当前调试进度"、"文件修改记录"）
        """
        compactor = agent._compactor
        if not compactor:
            return "上下文压缩未启用。"

        original_count = len(agent.history._messages)
        compacted, summary = await compactor.manual_compact(
            agent.history._messages, focus
        )
        agent.history.replace_all(compacted)

        return (
            f"对话历史已压缩: {original_count} → {len(compacted)} 条消息。\n"
            f"摘要:\n{summary}"
        )

    # 标记为元工具
    compact._tool_wrapper.meta = True
    return compact
