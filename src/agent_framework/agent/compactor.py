"""上下文压缩流水线 — 轮次分层 + Token 动态阈值

执行顺序（每轮 LLM 调用前）：
    L1 snip → 轮次分层工具压缩 → [超阈值?] L4 LLM 摘要

层级说明：
  L1 snip_compact:     消息数 > max_messages 时裁剪中间消息（0 API 调用）
  轮次分层工具压缩:     按 assistant 消息计轮次，分层处理工具结果（0 API 调用）
    age > old_round_threshold:  ALL tool results → 占位符 + session 文件指针
    age recent~old:             tool results > truncate_threshold → 截断 + 文件指针
    age 0~recent:               保持不变（动态：上下文超 30% 时 recent 降为 0）

  动态阈值（根据 max_context_window 自动计算）：
    old_round_threshold = max(5, min(max_context_window // 1500 // 2, 30))
    truncate_threshold  = max(500, min(4000, max_context_window // 20))

  L4 compact_history:  调用 LLM 生成对话摘要（1 API 调用）
    触发条件: prompt_tokens / max_context_window > compact_trigger_ratio

使用方式：
    compactor = Compactor(llm, config, session)
    compacted = await compactor.auto_compact(messages)

参考：Claude Code compaction pipeline (query.ts / autoCompact.ts)
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from agent_framework.agent.events import HistoryCompacted
from agent_framework.llm.base.message import Message, MessageRole
from agent_framework.tools.decorator import tool

if TYPE_CHECKING:
    from agent_framework.agent.agent import Agent
    from agent_framework.agent.config import AgentConfig
    from agent_framework.agent.session import Session
    from agent_framework.llm.providers.base import BaseLLM

logger = logging.getLogger(__name__)

# L4 序列化时每条消息的最大字符数
_MAX_MSG_CHARS = 80000

# 连续压缩失败上限
_MAX_CONSECUTIVE_FAILURES = 3

# 每轮平均消耗的 token 数（经验值：1 轮 ≈ assistant + tools + results）
_AVG_TOKENS_PER_ROUND = 1500


class Compactor:
    """上下文压缩器 — 轮次分层 + Token 动态阈值。

    用法：
        compactor = Compactor(llm, config, session)
        compacted = await compactor.auto_compact(messages)
    """

    def __init__(
        self,
        llm: "BaseLLM",
        config: "AgentConfig",
        session: "Session | None" = None,
    ) -> None:
        self._llm = llm
        self._config = config
        self._session = session
        self._last_prompt_tokens = 0
        self._consecutive_failures = 0

        # 从 LLM 获取最大上下文窗口
        try:
            val = llm.capabilities.max_context_window
            if isinstance(val, int) and val > 0:
                self._max_context_window = val
            else:
                self._max_context_window = 8192
        except (AttributeError, TypeError):
            self._max_context_window = 8192  # 默认回退值

        # 动态计算阈值（基于上下文窗口大小）
        self._calc_dynamic_thresholds()

    def _calc_dynamic_thresholds(self) -> None:
        """根据 max_context_window 动态计算压缩阈值。

        占位符轮次：窗口能装下的轮数的一半（下限 5，上限 30）
        截断字符数：随窗口缩放（下限 500，上限 4000）
        """
        # 占位符轮次阈值
        max_keepable_rounds = self._max_context_window // _AVG_TOKENS_PER_ROUND
        self._old_round_threshold = max(5, min(max_keepable_rounds // 2, 30))

        # 截断字符数阈值
        # 8K → 500, 32K → 1600, 128K → 4000（上限）
        self._truncate_threshold = max(500, min(4000, self._max_context_window // 20))

    def update_prompt_tokens(self, tokens: int) -> None:
        """Agent 每次 LLM 调用后更新 prompt_tokens。"""
        if tokens > 0:
            self._last_prompt_tokens = tokens

    async def auto_compact(self, messages: list[Message]) -> list[Message]:
        """自动压缩流水线：L1 snip → 轮次分层 → [超阈值?] L4。

        :param messages: 当前历史消息列表
        :return: 压缩后的消息列表（可能与传入的是同一对象）
        """
        if len(messages) <= 1:
            return messages

        # L1: 消息数量裁剪
        messages = self._snip_compact(messages)

        # 轮次分层工具压缩
        messages = self._round_based_compact(messages)

        # L4: Token 比例触发 LLM 摘要
        if self._consecutive_failures < _MAX_CONSECUTIVE_FAILURES:
            ratio = self._calc_usage_ratio(messages)
            if ratio >= self._config.compact_trigger_ratio:
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

    # ── Token 比例计算 ────────────────────────────────────────

    def _calc_usage_ratio(self, messages: list[Message]) -> float:
        """计算当前上下文使用率。

        优先使用 API 返回的 prompt_tokens，回退到字符估算。
        """
        if self._last_prompt_tokens > 0:
            return self._last_prompt_tokens / self._max_context_window

        # 首次调用无 token 数据，回退到字符估算（约 2 字符/token）
        estimated_tokens = self._estimate_size(messages) // 2
        return estimated_tokens / self._max_context_window

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

    # ── 轮次分层工具压缩 ──────────────────────────────────────

    @staticmethod
    def _group_into_rounds(messages: list[Message]) -> list[list[Message]]:
        """将消息按轮次分组。

        一轮 = 一个 assistant 消息 + 其后的 tool/user 消息，直到下一个 assistant。
        首个 assistant 之前的消息（system, user）归入 round 0。
        """
        rounds: list[list[Message]] = []
        current_round: list[Message] = []
        has_seen_assistant = False

        for msg in messages:
            if msg.role == MessageRole.ASSISTANT:
                if has_seen_assistant and current_round:
                    # 遇到新的 assistant，把之前的轮次存起来
                    rounds.append(current_round)
                    current_round = []
                has_seen_assistant = True
                current_round.append(msg)
            else:
                current_round.append(msg)

        if current_round:
            rounds.append(current_round)

        return rounds

    def _round_based_compact(self, messages: list[Message]) -> list[Message]:
        """按轮次分层处理工具结果。

        age > old_round_threshold: ALL tool results → 占位符
        age recent~old:           tool results > truncate_threshold chars → 截断
        age 0~recent:             保持不变（动态：上下文超 30% 时 recent 降为 0）
        """
        # 检查是否有任何 tool 消息
        has_tool = any(m.role == MessageRole.TOOL for m in messages)
        if not has_tool:
            return messages

        rounds = self._group_into_rounds(messages)
        if len(rounds) <= 1:
            return messages

        total_rounds = len(rounds)

        # 动态计算 recent 保留轮数
        recent_rounds = self._config.compact_recent_rounds
        usage_ratio = self._calc_usage_ratio(messages)
        if usage_ratio > 0.3:
            recent_rounds = 0

        changed = False
        new_rounds = []

        for i, round_msgs in enumerate(rounds):
            age = total_rounds - 1 - i  # 距离最新轮的距离

            if age <= recent_rounds:
                # 最近轮：保持不变
                new_rounds.append(round_msgs)
            elif age <= self._old_round_threshold:
                # 中间轮：截断长工具结果
                processed = self._process_mid_round(round_msgs, age, truncate=True)
                if processed is not round_msgs:
                    changed = True
                new_rounds.append(processed)
            else:
                # 旧轮：全部占位符
                processed = self._process_mid_round(round_msgs, age, truncate=False)
                if processed is not round_msgs:
                    changed = True
                new_rounds.append(processed)

        if not changed:
            return messages

        # 展平
        result = []
        for r in new_rounds:
            result.extend(r)
        return result

    def _process_mid_round(
        self, round_msgs: list[Message], age: int, truncate: bool
    ) -> list[Message]:
        """处理中间/旧轮次的工具结果。

        :param truncate: True=截断（保留前 N 字符），False=全部占位符
        """
        result = []
        changed = False
        threshold = self._truncate_threshold

        for msg in round_msgs:
            if msg.role != MessageRole.TOOL:
                result.append(msg)
                continue

            content = msg.content if isinstance(msg.content, str) else str(msg.content or "")

            # 已压缩（占位符）的消息不再处理（已经是最简形式）
            if content.startswith("[工具结果已压缩:"):
                result.append(msg)
                continue

            # 已截断的消息：中间轮次跳过（避免重复截断），旧轮次升级为占位符
            if content.startswith("[工具结果已截断:") and truncate:
                result.append(msg)
                continue

            tool_name = msg.name or "unknown"
            tool_id = msg.tool_call_id or ""

            # session 文件指针
            session_ref = self._get_session_ref(msg)

            if truncate and len(content) <= threshold:
                # 短内容不需要截断
                result.append(msg)
                continue

            if truncate:
                # 截断模式：保留前 N 字符
                preview = content[:threshold]
                new_content = (
                    f"[工具结果已截断: {tool_name}"
                    + (f" ({tool_id})" if tool_id else "")
                    + f" → 前{threshold}字符如下]\n"
                    f"{preview}\n"
                    f"{session_ref}"
                )
            else:
                # 占位符模式
                new_content = (
                    f"[工具结果已压缩: {tool_name}"
                    + (f" ({tool_id})" if tool_id else "")
                    + f"]\n{session_ref}"
                )

            result.append(Message(
                role=MessageRole.TOOL,
                content=new_content,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            ))
            changed = True

        if not changed:
            return round_msgs
        return result

    def _get_session_ref(self, msg: Message) -> str:
        """生成 session 文件指针字符串。"""
        if not self._session:
            return "[原始内容未被持久化]"

        # 尝试从 JSONL 中找到对应的行号
        # 通过 tool_call_id 匹配
        line_ref = ""
        if msg.tool_call_id and self._session.conversation_path.exists():
            try:
                with open(self._session.conversation_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if msg.tool_call_id in line:
                            line_ref = f" 第{i}行"
                            break
            except OSError:
                pass

        conv_path = self._session.conversation_path
        return f"[完整内容 → {conv_path}{line_ref}]"

    # ── L4: compact_history ───────────────────────────────────

    async def _compact_history(self, messages: list[Message], preserve_tail: bool = True) -> list[Message]:
        """L4: 调用 LLM 生成对话摘要。

        :param messages: 当前历史消息
        :param preserve_tail: 是否保留尾部消息
        :return: 压缩后的消息列表
        """
        if not preserve_tail:
            # 全量压缩（手动模式）
            summary = await self._summarize(messages)
            return [Message(role=MessageRole.USER, content=f"[Compacted]\n\n{summary}")]

        # 智能保留尾部：基于 token 预算
        max_tail_tokens = int(self._max_context_window * 0.30)  # 尾部占 30% 上下文

        keep_count = 0
        for try_count in (3, 2, 1, 0):
            if try_count == 0:
                keep_count = 0
                break
            if len(messages) <= try_count:
                # 消息数不足，保留全部
                keep_count = len(messages)
                break
            # 检查尾部 token 是否在预算内
            tail_messages = messages[-try_count:]
            tail_size = self._estimate_size(tail_messages) // 2  # 粗略 token 估算
            if tail_size <= max_tail_tokens:
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
        # if len(conversation) > _MAX_MSG_CHARS:
        #     conversation = conversation[:_MAX_MSG_CHARS] + "\n\n[... truncated ...]"

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
            # if len(content) > 2000:
            #     content = content[:2000] + "... [truncated]"

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
