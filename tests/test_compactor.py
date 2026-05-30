"""测试 Compactor — 轮次分层 + Token 动态阈值"""
import pytest
from unittest.mock import AsyncMock

from agent_framework.agent.compactor import Compactor, create_compact_tool
from agent_framework.agent.config import AgentConfig
from agent_framework.agent.session import Session
from agent_framework.llm.base.message import Message, MessageRole
from agent_framework.llm.base.response import StreamChunk, TokenUsage

import tempfile
from pathlib import Path


# ── 辅助函数 ──────────────────────────────────────────────

def _make_msg(role: MessageRole, content: str, **kwargs) -> Message:
    return Message(role=role, content=content, **kwargs)


def _make_system(text: str) -> Message:
    return _make_msg(MessageRole.SYSTEM, text)


def _make_user(text: str) -> Message:
    return _make_msg(MessageRole.USER, text)


def _make_assistant(text: str, tool_calls=None) -> Message:
    return _make_msg(MessageRole.ASSISTANT, text, tool_calls=tool_calls)


def _make_tool(text: str, tool_call_id: str = "call_1", name: str = "test") -> Message:
    return _make_msg(MessageRole.TOOL, text, tool_call_id=tool_call_id, name=name)


def _make_llm(response_text: str = "这是摘要"):
    """创建 mock LLM"""
    async def mock_chat(*args, **kwargs):
        yield StreamChunk(content=response_text, finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120))

    llm = AsyncMock()
    llm.chat = mock_chat
    llm.capabilities = AsyncMock()
    llm.capabilities.max_context_window = 8192
    return llm


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ── _group_into_rounds 测试 ──────────────────────────────

class TestGroupIntoRounds:
    """轮次分组"""

    def test_empty_messages(self):
        """空消息列表"""
        assert Compactor._group_into_rounds([]) == []

    def test_no_assistant(self):
        """无 assistant 消息 → 全部归入 round 0"""
        messages = [_make_user("u1"), _make_user("u2")]
        rounds = Compactor._group_into_rounds(messages)
        assert len(rounds) == 1
        assert len(rounds[0]) == 2

    def test_single_round(self):
        """单轮对话"""
        messages = [
            _make_system("sys"),
            _make_user("q"),
            _make_assistant("a", tool_calls=[{"id": "1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]),
            _make_tool("result"),
        ]
        rounds = Compactor._group_into_rounds(messages)
        assert len(rounds) == 1
        assert len(rounds[0]) == 4

    def test_multiple_rounds(self):
        """多轮对话"""
        messages = [
            _make_system("sys"),
            _make_user("q1"),
            _make_assistant("a1"),
            _make_user("q2"),
            _make_assistant("a2"),
            _make_tool("t2", "call_2"),
            _make_user("q3"),
            _make_assistant("a3"),
        ]
        rounds = Compactor._group_into_rounds(messages)
        # round 0: [sys, user(q1), assistant(a1), user(q2)]  ← 首个 assistant 不切轮
        # round 1: [assistant(a2), tool(t2), user(q3)]
        # round 2: [assistant(a3)]
        assert len(rounds) == 3
        assert rounds[0][0].role == MessageRole.SYSTEM
        assert any(m.content == "a1" for m in rounds[0])
        assert rounds[1][0].content == "a2"
        assert rounds[2][0].content == "a3"

    def test_multi_tool_round(self):
        """单轮多工具调用"""
        messages = [
            _make_user("q"),
            _make_assistant("a1", tool_calls=[{"id": "1", "type": "function", "function": {"name": "t1", "arguments": "{}"}}]),
            _make_tool("r1", "call_1"),
            _make_tool("r2", "call_2"),
            _make_assistant("a2"),
        ]
        rounds = Compactor._group_into_rounds(messages)
        # round 0: [user(q), assistant(a1), tool(r1), tool(r2)]
        # round 1: [assistant(a2)]
        assert len(rounds) == 2
        assert len(rounds[0]) == 4  # user + assistant + 2 tools
        assert len(rounds[1]) == 1  # assistant only


# ── L1: snip_compact 测试 ────────────────────────────────

class TestSnipCompact:
    """L1: 消息数量裁剪"""

    def test_no_snip_when_under_limit(self):
        """消息数未超限时不裁剪"""
        config = AgentConfig(compact_max_messages=10)
        compactor = Compactor(_make_llm(), config)
        messages = [_make_user(f"msg {i}") for i in range(8)]
        result = compactor._snip_compact(messages)
        assert result is messages

    def test_snip_when_over_limit(self):
        """消息数超限时裁剪中间消息"""
        config = AgentConfig(compact_max_messages=10)
        compactor = Compactor(_make_llm(), config)
        messages = [_make_user(f"msg {i}") for i in range(20)]
        result = compactor._snip_compact(messages)
        assert result is not messages
        assert len(result) == 11  # 3 + 1 marker + 7
        snip_msgs = [m for m in result if "snipped" in m.content]
        assert len(snip_msgs) == 1

    def test_snip_preserves_system(self):
        """裁剪时保留 system 消息"""
        config = AgentConfig(compact_max_messages=5)
        compactor = Compactor(_make_llm(), config)
        messages = [_make_system("系统")] + [_make_user(f"msg {i}") for i in range(15)]
        result = compactor._snip_compact(messages)
        assert result[0].role == MessageRole.SYSTEM


# ── 轮次分层工具压缩 测试 ─────────────────────────────────

class TestRoundBasedCompact:
    """轮次分层工具压缩"""

    def test_no_tool_messages(self):
        """无 tool 消息时不处理"""
        compactor = Compactor(_make_llm(), AgentConfig())
        messages = [_make_user("hello"), _make_assistant("hi")]
        result = compactor._round_based_compact(messages)
        assert result is messages

    def test_recent_rounds_unchanged(self, tmp_dir):
        """最近轮次工具结果保持不变"""
        session = Session("test", tmp_dir)
        config = AgentConfig(compact_recent_rounds=3)
        compactor = Compactor(_make_llm(), config, session=session)

        # 创建 3 轮对话（全部在 recent 范围内）
        messages = []
        for i in range(3):
            messages.append(_make_user(f"q{i}"))
            messages.append(_make_assistant(f"a{i}", tool_calls=[
                {"id": f"call_{i}", "type": "function", "function": {"name": "file", "arguments": "{}"}}
            ]))
            messages.append(_make_tool("x" * 1000, tool_call_id=f"call_{i}", name="file"))

        result = compactor._round_based_compact(messages)
        # 最近 3 轮不应被修改
        tool_msgs = [m for m in result if m.role == MessageRole.TOOL]
        for tm in tool_msgs:
            assert len(tm.content) == 1000  # 未被截断

    def test_old_rounds_placeholder(self, tmp_dir):
        """旧轮次（>10）工具结果替换为占位符"""
        session = Session("test", tmp_dir)
        config = AgentConfig(compact_recent_rounds=2)
        compactor = Compactor(_make_llm(), config, session=session)

        # 创建 15 轮对话
        messages = []
        for i in range(15):
            messages.append(_make_user(f"q{i}"))
            messages.append(_make_assistant(f"a{i}", tool_calls=[
                {"id": f"call_{i}", "type": "function", "function": {"name": "file", "arguments": "{}"}}
            ]))
            messages.append(_make_tool("x" * 1000, tool_call_id=f"call_{i}", name="file"))

        result = compactor._round_based_compact(messages)

        # 找到旧轮次的 tool 消息（rounds 0-4 的 age > 10）
        tool_msgs = [m for m in result if m.role == MessageRole.TOOL]
        old_tools = [m for m in tool_msgs if "已压缩" in m.content]
        recent_tools = [m for m in tool_msgs if "已压缩" not in m.content]

        assert len(old_tools) > 0  # 有旧轮次被压缩
        assert len(recent_tools) > 0  # 有最近轮次保留

    def test_mid_rounds_truncation(self, tmp_dir):
        """中间轮次（3-10）长工具结果被截断"""
        session = Session("test", tmp_dir)
        config = AgentConfig(compact_recent_rounds=1)
        compactor = Compactor(_make_llm(), config, session=session)

        # 创建 8 轮对话
        messages = []
        for i in range(8):
            messages.append(_make_user(f"q{i}"))
            messages.append(_make_assistant(f"a{i}", tool_calls=[
                {"id": f"call_{i}", "type": "function", "function": {"name": "file", "arguments": "{}"}}
            ]))
            messages.append(_make_tool("x" * 1000, tool_call_id=f"call_{i}", name="file"))

        result = compactor._round_based_compact(messages)

        tool_msgs = [m for m in result if m.role == MessageRole.TOOL]
        truncated = [m for m in tool_msgs if "已截断" in m.content]
        assert len(truncated) > 0

    def test_short_tool_results_not_truncated(self, tmp_dir):
        """短工具结果（<=500字符）在中间轮次不截断"""
        session = Session("test", tmp_dir)
        config = AgentConfig(compact_recent_rounds=1)
        compactor = Compactor(_make_llm(), config, session=session)

        messages = []
        for i in range(5):
            messages.append(_make_user(f"q{i}"))
            messages.append(_make_assistant(f"a{i}", tool_calls=[
                {"id": f"call_{i}", "type": "function", "function": {"name": "file", "arguments": "{}"}}
            ]))
            # 短内容
            messages.append(_make_tool("short", tool_call_id=f"call_{i}", name="file"))

        result = compactor._round_based_compact(messages)
        tool_msgs = [m for m in result if m.role == MessageRole.TOOL]
        # 短内容即使在中间轮也不应截断（age 3-10 但 <=500 字符）
        for tm in tool_msgs:
            if "已压缩" not in tm.content:  # 非旧轮次占位符
                assert tm.content == "short" or "已截断" not in tm.content

    def test_dynamic_recent_reduction(self, tmp_dir):
        """上下文超 30% 时 recent 降为 0"""
        session = Session("test", tmp_dir)
        config = AgentConfig(compact_recent_rounds=3)
        llm = _make_llm()
        compactor = Compactor(llm, config, session=session)

        # 模拟高使用率
        compactor.update_prompt_tokens(3000)  # 3000/8192 > 0.3

        messages = []
        for i in range(5):
            messages.append(_make_user(f"q{i}"))
            messages.append(_make_assistant(f"a{i}", tool_calls=[
                {"id": f"call_{i}", "type": "function", "function": {"name": "file", "arguments": "{}"}}
            ]))
            messages.append(_make_tool("x" * 1000, tool_call_id=f"call_{i}", name="file"))

        result = compactor._round_based_compact(messages)
        tool_msgs = [m for m in result if m.role == MessageRole.TOOL]
        # 由于 recent=0，所有轮次都应被处理（截断或占位）
        processed = [m for m in tool_msgs if "已截断" in m.content or "已压缩" in m.content]
        assert len(processed) > 0

    def test_no_nested_header_on_repeated_compact(self, tmp_dir):
        """多次调用 auto_compact 不应产生嵌套截断 header"""
        session = Session("test", tmp_dir)
        config = AgentConfig(compact_recent_rounds=0)  # 所有轮次都处理
        llm = _make_llm()
        compactor = Compactor(llm, config, session=session)

        # 构造 2 轮对话（至少需要 2 个 assistant 消息）
        messages = [
            _make_user("q0"),
            _make_assistant("a0", tool_calls=[
                {"id": "call_0", "type": "function", "function": {"name": "file", "arguments": "{}"}}
            ]),
            _make_tool("x" * 1000, tool_call_id="call_0", name="file"),
            _make_assistant("a1"),
        ]

        # 第一次调用：截断
        result1 = compactor._round_based_compact(messages)
        tool_msg1 = [m for m in result1 if m.role == MessageRole.TOOL][0]
        header_count1 = tool_msg1.content.count("[工具结果已截断:")
        assert header_count1 == 1

        # 第二次调用（模拟下一轮 auto_compact）：不应产生嵌套 header
        result2 = compactor._round_based_compact(result1)
        tool_msg2 = [m for m in result2 if m.role == MessageRole.TOOL][0]
        header_count2 = tool_msg2.content.count("[工具结果已截断:")
        assert header_count2 == 1, f"嵌套 header: {tool_msg2.content[:150]}"
        # 内容应与第一次完全相同
        assert tool_msg2.content == tool_msg1.content


# ── Token 阈值测试 ────────────────────────────────────────

class TestTokenThreshold:
    """L4 Token 比例触发"""

    def test_calc_usage_ratio_with_tokens(self):
        """有 prompt_tokens 时使用真实数据"""
        llm = _make_llm()
        compactor = Compactor(llm, AgentConfig())
        compactor.update_prompt_tokens(4096)  # 4096/8192 = 0.5
        ratio = compactor._calc_usage_ratio([])
        assert abs(ratio - 0.5) < 0.01

    def test_calc_usage_ratio_fallback(self):
        """无 prompt_tokens 时回退到字符估算"""
        llm = _make_llm()
        compactor = Compactor(llm, AgentConfig())
        # _last_prompt_tokens = 0
        messages = [_make_user("x" * 1000)]  # ~250 tokens
        ratio = compactor._calc_usage_ratio(messages)
        assert ratio > 0  # 应有非零比例

    @pytest.mark.asyncio
    async def test_no_l4_below_threshold(self):
        """低于 70% 时不触发 L4"""
        llm = _make_llm()
        config = AgentConfig(compact_trigger_ratio=0.7)
        compactor = Compactor(llm, config)
        compactor.update_prompt_tokens(100)  # 100/8192 << 0.7

        messages = [_make_user("short"), _make_assistant("reply")]
        result = await compactor.auto_compact(messages)
        # 不应有 LLM 摘要调用，消息不应变为 Compacted
        has_compacted = any("Compacted" in str(m.content) for m in result)
        assert not has_compacted

    @pytest.mark.asyncio
    async def test_l4_above_threshold(self):
        """超过 70% 时触发 L4"""
        llm = _make_llm("LLM 生成的摘要")
        config = AgentConfig(compact_trigger_ratio=0.5)
        compactor = Compactor(llm, config)
        compactor.update_prompt_tokens(5000)  # 5000/8192 > 0.5

        # 需要足够多消息，使尾部保留（最多 3 条）后仍有待压缩内容
        messages = [
            _make_user("很长" * 100),
            _make_assistant("也很长" * 100),
            _make_user("继续" * 100),
            _make_assistant("回复" * 100),
            _make_user("再说" * 50),
        ]
        result = await compactor.auto_compact(messages)
        has_compacted = any("Compacted" in str(m.content) for m in result)
        assert has_compacted

    @pytest.mark.asyncio
    async def test_l4_failure_circuit_breaker(self):
        """L4 连续失败不超过上限"""
        async def failing_chat(*args, **kwargs):
            raise RuntimeError("LLM 调用失败")
            yield

        llm = AsyncMock()
        llm.chat = failing_chat
        llm.capabilities = AsyncMock()
        llm.capabilities.max_context_window = 8192

        config = AgentConfig(compact_trigger_ratio=0.1)
        compactor = Compactor(llm, config)
        compactor.update_prompt_tokens(5000)

        messages = [
            _make_user("x" * 100), _make_assistant("y" * 100),
            _make_user("x" * 100), _make_assistant("y" * 100),
            _make_user("z" * 100),
        ]

        for _ in range(5):
            result = await compactor.auto_compact(messages)
            assert len(result) >= 1

        assert compactor._consecutive_failures == 3


# ── auto_compact 流水线测试 ───────────────────────────────

class TestAutoCompact:
    """自动压缩流水线"""

    @pytest.mark.asyncio
    async def test_single_message_no_compact(self):
        """单条消息不压缩"""
        compactor = Compactor(_make_llm(), AgentConfig())
        messages = [_make_user("hello")]
        result = await compactor.auto_compact(messages)
        assert result is messages

    @pytest.mark.asyncio
    async def test_pipeline_triggers_l4(self):
        """超过阈值时触发 L4"""
        config = AgentConfig(compact_trigger_ratio=0.1)
        compactor = Compactor(_make_llm("流水线摘要"), config)
        compactor.update_prompt_tokens(5000)

        messages = [
            _make_user("很长" * 100),
            _make_assistant("也很长" * 100),
            _make_user("更多" * 100),
            _make_assistant("继续" * 100),
        ]
        result = await compactor.auto_compact(messages)
        has_compacted = any("流水线摘要" in str(m.content) for m in result)
        assert has_compacted


# ── manual_compact 测试 ──────────────────────────────────

class TestManualCompact:
    """手动压缩"""

    @pytest.mark.asyncio
    async def test_manual_compact_returns_summary(self):
        """手动压缩返回 (消息列表, 摘要文本)"""
        llm = _make_llm("手动摘要内容")
        compactor = Compactor(llm, AgentConfig())
        messages = [_make_user("hello"), _make_assistant("world")]

        compacted, summary = await compactor.manual_compact(messages)
        assert len(compacted) == 1
        assert "手动摘要内容" in summary
        assert "Compacted" in compacted[0].content

    @pytest.mark.asyncio
    async def test_manual_compact_with_focus(self):
        """手动压缩支持 focus 参数"""
        llm = _make_llm("带焦点的摘要")
        compactor = Compactor(llm, AgentConfig())
        messages = [_make_user("hello")]

        compacted, summary = await compactor.manual_compact(messages, focus="调试进度")
        assert "带焦点的摘要" in summary


# ── reactive_compact 测试 ────────────────────────────────

class TestReactiveCompact:
    """应急压缩"""

    @pytest.mark.asyncio
    async def test_reactive_keeps_recent_messages(self):
        """应急压缩保留最近 3 条消息"""
        llm = _make_llm("应急摘要")
        compactor = Compactor(llm, AgentConfig())
        messages = [_make_user(f"msg {i}") for i in range(10)]

        result = await compactor.reactive_compact(messages)
        assert len(result) == 4
        assert "Reactive compact" in result[0].content

    @pytest.mark.asyncio
    async def test_reactive_few_messages(self):
        """消息数不足时仅返回摘要"""
        llm = _make_llm("应急摘要")
        compactor = Compactor(llm, AgentConfig())
        messages = [_make_user("msg 1"), _make_assistant("msg 2")]

        result = await compactor.reactive_compact(messages)
        assert len(result) == 1
        assert "Reactive compact" in result[0].content

    @pytest.mark.asyncio
    async def test_reactive_failure_returns_original(self):
        """应急压缩失败时返回原消息"""
        async def failing_chat(*args, **kwargs):
            raise RuntimeError("LLM 失败")
            yield

        llm = AsyncMock()
        llm.chat = failing_chat
        llm.capabilities = AsyncMock()
        llm.capabilities.max_context_window = 8192
        compactor = Compactor(llm, AgentConfig())
        messages = [_make_user("hello")]

        result = await compactor.reactive_compact(messages)
        assert result is messages


# ── 辅助方法测试 ──────────────────────────────────────────

class TestHelpers:
    """辅助方法"""

    def test_estimate_size(self):
        """估算大小应大于零"""
        messages = [_make_user("hello"), _make_assistant("world")]
        size = Compactor._estimate_size(messages)
        assert size == 10

    def test_estimate_size_with_tool_calls(self):
        """包含工具调用时应计入 tool_calls 大小"""
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="thinking",
            tool_calls=[{"id": "1", "type": "function", "function": {"name": "test", "arguments": "{}"}}],
        )
        size = Compactor._estimate_size([msg])
        assert size > len("thinking")

    def test_serialize_messages(self):
        """序列化消息应包含角色和内容"""
        compactor = Compactor(_make_llm(), AgentConfig())
        messages = [
            _make_user("用户输入"),
            _make_assistant("助手回复"),
        ]
        text = compactor._serialize_messages(messages)
        assert "[USER]" in text
        assert "[ASSISTANT]" in text
        assert "用户输入" in text

    def test_serialize_long_content_preserved(self):
        """序列化时保留完整内容（不再自动截断单条消息）"""
        compactor = Compactor(_make_llm(), AgentConfig())
        messages = [_make_user("x" * 5000)]
        text = compactor._serialize_messages(messages)
        # 序列化保留完整内容，L4 摘要调用时有 _MAX_MSG_CHARS 上限
        assert "x" * 5000 in text


# ── create_compact_tool 测试 ─────────────────────────────

class TestCreateCompactTool:
    """compact 元工具工厂"""

    def test_tool_creation(self):
        """应成功创建 compact 工具"""
        from agent_framework.agent import Agent
        llm = _make_llm()
        config = AgentConfig(session_enabled=False)
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_")
        compact_tool = create_compact_tool(agent)
        assert compact_tool is not None
        assert compact_tool._tool_wrapper.meta is True

    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """compact 工具执行应替换历史"""
        from agent_framework.agent import Agent
        llm = _make_llm("工具摘要")
        config = AgentConfig(session_enabled=False)
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_")

        agent.history.add(_make_user("msg 1"))
        agent.history.add(_make_assistant("msg 2"))
        agent.history.add(_make_user("msg 3"))

        original_count = len(agent.history._messages)
        assert original_count >= 3

        compact_tool = create_compact_tool(agent)
        result = await compact_tool(focus="")
        assert "已压缩" in result
        assert len(agent.history._messages) < original_count
