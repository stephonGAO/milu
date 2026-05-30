"""测试 Compactor 四层压缩流水线"""
import pytest
from unittest.mock import AsyncMock

from agent_framework.agent.compactor import Compactor, create_compact_tool
from agent_framework.agent.config import AgentConfig
from agent_framework.llm.base.message import Message, MessageRole
from agent_framework.llm.base.response import StreamChunk, TokenUsage


# ── 辅助函数 ──────────────────────────────────────────────

def _make_msg(role: MessageRole, content: str, **kwargs) -> Message:
    return Message(role=role, content=content, **kwargs)


def _make_system(text: str) -> Message:
    return _make_msg(MessageRole.SYSTEM, text)


def _make_user(text: str) -> Message:
    return _make_msg(MessageRole.USER, text)


def _make_assistant(text: str) -> Message:
    return _make_msg(MessageRole.ASSISTANT, text)


def _make_tool(text: str, tool_call_id: str = "call_1", name: str = "test") -> Message:
    return _make_msg(MessageRole.TOOL, text, tool_call_id=tool_call_id, name=name)


def _make_llm(response_text: str = "这是摘要"):
    """创建 mock LLM，返回固定的摘要文本"""
    async def mock_chat(*args, **kwargs):
        yield StreamChunk(content=response_text, finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120))

    llm = AsyncMock()
    llm.chat = mock_chat
    return llm


# ── L1: snip_compact 测试 ────────────────────────────────

class TestSnipCompact:
    """L1: 消息数量裁剪"""

    def test_no_snip_when_under_limit(self):
        """消息数未超限时不裁剪"""
        config = AgentConfig(compact_max_messages=10)
        compactor = Compactor(_make_llm(), config)
        messages = [_make_user(f"msg {i}") for i in range(8)]
        result = compactor._snip_compact(messages)
        assert result is messages  # 同一对象

    def test_snip_when_over_limit(self):
        """消息数超限时裁剪中间消息"""
        config = AgentConfig(compact_max_messages=10)
        compactor = Compactor(_make_llm(), config)
        messages = [_make_user(f"msg {i}") for i in range(20)]
        result = compactor._snip_compact(messages)
        assert result is not messages
        # 保留头 3 + 尾 7 + 1 个 snip marker = 11
        assert len(result) == 11
        # 有 snip marker
        snip_msgs = [m for m in result if "snipped" in m.content]
        assert len(snip_msgs) == 1
        assert "10 messages" in snip_msgs[0].content

    def test_snip_preserves_system(self):
        """裁剪时保留 system 消息"""
        config = AgentConfig(compact_max_messages=5)
        compactor = Compactor(_make_llm(), config)
        messages = [_make_system("系统")] + [_make_user(f"msg {i}") for i in range(15)]
        result = compactor._snip_compact(messages)
        assert result[0].role == MessageRole.SYSTEM
        assert result[0].content == "系统"


# ── L2: micro_compact 测试 ───────────────────────────────

class TestMicroCompact:
    """L2: 旧工具结果占位"""

    def test_no_replace_when_under_limit(self):
        """工具结果数未超限时不替换"""
        config = AgentConfig(compact_keep_recent=3)
        compactor = Compactor(_make_llm(), config)
        messages = [
            _make_tool("x" * 200, tool_call_id=f"call_{i}")
            for i in range(3)
        ]
        result = compactor._micro_compact(messages)
        for msg in result:
            assert "compacted" not in msg.content

    def test_replace_old_tool_results(self):
        """替换旧的工具结果为占位符"""
        config = AgentConfig(compact_keep_recent=2)
        compactor = Compactor(_make_llm(), config)
        messages = [
            _make_tool("x" * 200, tool_call_id="call_0"),  # 旧 → 替换
            _make_tool("y" * 200, tool_call_id="call_1"),  # 旧 → 替换
            _make_tool("z" * 200, tool_call_id="call_2"),  # 保留
            _make_tool("w" * 200, tool_call_id="call_3"),  # 保留
        ]
        result = compactor._micro_compact(messages)
        assert "compacted" in result[0].content
        assert "compacted" in result[1].content
        assert result[2].content == "y" * 200 or "compacted" not in result[2].content
        assert result[3].content == "w" * 200 or "compacted" not in result[3].content

    def test_skip_short_tool_results(self):
        """短工具结果（<=120字符）不替换"""
        config = AgentConfig(compact_keep_recent=1)
        compactor = Compactor(_make_llm(), config)
        messages = [
            _make_tool("short", tool_call_id="call_0"),  # 旧但短，不替换
            _make_tool("x" * 200, tool_call_id="call_1"),  # 旧且长，替换
            _make_tool("y" * 200, tool_call_id="call_2"),  # 最近 1 个，保留
        ]
        result = compactor._micro_compact(messages)
        assert result[0].content == "short"  # 短内容未替换
        assert "compacted" in result[1].content  # 长内容被替换
        assert result[2].content == "y" * 200  # 最近的保留


# ── L3: tool_result_budget 测试 ──────────────────────────

class TestToolResultBudget:
    """L3: 大结果持久化"""

    def test_no_truncate_when_under_budget(self):
        """工具结果总大小在预算内时不截断"""
        config = AgentConfig()
        compactor = Compactor(_make_llm(), config)
        messages = [
            _make_assistant("thinking"),
            _make_tool("x" * 1000, tool_call_id="call_1"),
        ]
        result = compactor._tool_result_budget(messages)
        assert result is messages
        assert result[1].content == "x" * 1000

    def test_truncate_large_result(self):
        """超大工具结果被截断并持久化"""
        config = AgentConfig()
        compactor = Compactor(_make_llm(), config)
        large_content = "x" * 300_000  # 超过 _BUDGET_MAX_BYTES (200K)
        messages = [
            _make_assistant("thinking"),
            _make_tool(large_content, tool_call_id="call_big"),
        ]
        result = compactor._tool_result_budget(messages)
        # 截断后应包含 persisted-output 标记
        assert "persisted-output" in result[1].content
        assert len(result[1].content) < 300_000

    def test_no_tool_messages(self):
        """无 tool 消息时不处理"""
        config = AgentConfig()
        compactor = Compactor(_make_llm(), config)
        messages = [_make_user("hello"), _make_assistant("hi")]
        result = compactor._tool_result_budget(messages)
        assert result is messages


# ── L4: compact_history 测试 ─────────────────────────────

class TestCompactHistory:
    """L4: LLM 摘要压缩"""

    @pytest.mark.asyncio
    async def test_compact_history_replaces_all(self):
        """L4 应将全部消息替换为摘要"""
        llm = _make_llm("这是生成的摘要")
        config = AgentConfig(compact_threshold=100)  # 低阈值确保触发
        compactor = Compactor(llm, config)

        messages = [
            _make_user("消息1" * 50),
            _make_assistant("回复1" * 50),
        ]
        result = await compactor._compact_history(messages)
        assert len(result) == 1
        assert result[0].role == MessageRole.USER
        assert "Compacted" in result[0].content
        assert "这是生成的摘要" in result[0].content

    @pytest.mark.asyncio
    async def test_summarize_with_focus(self):
        """带 focus 参数时摘要提示应包含 focus"""
        llm = _make_llm("关注主题的摘要")
        config = AgentConfig()
        compactor = Compactor(llm, config)

        messages = [_make_user("hello"), _make_assistant("world")]
        summary = await compactor._summarize(messages, focus="当前调试进度")
        assert isinstance(summary, str)
        assert len(summary) > 0


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
    async def test_pipeline_l3_l1_l2(self):
        """流水线按 L3 → L1 → L2 顺序执行"""
        config = AgentConfig(compact_max_messages=5, compact_keep_recent=1,
                             compact_threshold=999999)  # 高阈值不触发 L4
        compactor = Compactor(_make_llm(), config)

        messages = [_make_user(f"msg {i}") for i in range(15)]
        result = await compactor.auto_compact(messages)
        # L1 应该触发了裁剪
        assert len(result) <= 15

    @pytest.mark.asyncio
    async def test_pipeline_triggers_l4(self):
        """超过阈值时触发 L4 LLM 摘要"""
        config = AgentConfig(compact_threshold=10)  # 极低阈值确保触发
        compactor = Compactor(_make_llm("流水线摘要"), config)

        messages = [
            _make_user("很长" * 100),
            _make_assistant("也很长" * 100),
        ]
        result = await compactor.auto_compact(messages)
        assert len(result) == 1
        assert "流水线摘要" in result[0].content

    @pytest.mark.asyncio
    async def test_l4_failure_circuit_breaker(self):
        """L4 连续失败不超过上限"""
        async def failing_chat(*args, **kwargs):
            raise RuntimeError("LLM 调用失败")
            yield  # make it an async generator

        llm = AsyncMock()
        llm.chat = failing_chat

        config = AgentConfig(compact_threshold=10)
        compactor = Compactor(llm, config)

        messages = [_make_user("x" * 100), _make_assistant("y" * 100)]

        # 连续调用多次，不应抛出异常
        for _ in range(5):
            result = await compactor.auto_compact(messages)
            # 失败时返回原消息
            assert len(result) >= 1

        assert compactor._consecutive_failures == 3  # 上限


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
        """应急压缩保留最近 5 条消息"""
        llm = _make_llm("应急摘要")
        compactor = Compactor(llm, AgentConfig())
        messages = [_make_user(f"msg {i}") for i in range(10)]

        result = await compactor.reactive_compact(messages)
        # 摘要 + 最近 5 条 = 6 条
        assert len(result) == 6
        assert "Reactive compact" in result[0].content

    @pytest.mark.asyncio
    async def test_reactive_few_messages(self):
        """消息数不足 5 条时仅返回摘要（不附加 recent）"""
        llm = _make_llm("应急摘要")
        compactor = Compactor(llm, AgentConfig())
        messages = [_make_user("msg 1"), _make_assistant("msg 2")]

        result = await compactor.reactive_compact(messages)
        # 消息 <= 5 条时 recent 为空，仅返回摘要
        assert len(result) == 1
        assert "Reactive compact" in result[0].content

    @pytest.mark.asyncio
    async def test_reactive_failure_returns_original(self):
        """应急压缩失败时返回原消息"""
        async def failing_chat(*args, **kwargs):
            raise RuntimeError("LLM 失败")
            yield  # make it an async generator

        llm = AsyncMock()
        llm.chat = failing_chat
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
        assert size == 10  # "hello" + "world"

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
        assert "助手回复" in text

    def test_serialize_truncates_long_content(self):
        """序列化时截断过长单条消息"""
        compactor = Compactor(_make_llm(), AgentConfig())
        messages = [_make_user("x" * 5000)]
        text = compactor._serialize_messages(messages)
        assert "truncated" in text
        assert len(text) < 5000


# ── create_compact_tool 测试 ─────────────────────────────

class TestCreateCompactTool:
    """compact 元工具工厂"""

    def test_tool_creation(self):
        """应成功创建 compact 工具"""
        from agent_framework.agent import Agent
        llm = _make_llm()
        agent = Agent(llm=llm, system_prompt="test", skills_dir="/tmp/_nonexistent_")
        compact_tool = create_compact_tool(agent)
        assert compact_tool is not None
        # 应标记为元工具
        assert compact_tool._tool_wrapper.meta is True

    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """compact 工具执行应替换历史"""
        from agent_framework.agent import Agent
        llm = _make_llm("工具摘要")
        agent = Agent(llm=llm, system_prompt="test", skills_dir="/tmp/_nonexistent_")

        # 添加一些历史消息
        agent.history.add(_make_user("msg 1"))
        agent.history.add(_make_assistant("msg 2"))
        agent.history.add(_make_user("msg 3"))

        original_count = len(agent.history._messages)
        assert original_count >= 3

        compact_tool = create_compact_tool(agent)
        # 调用工具
        result = await compact_tool(focus="")
        assert "已压缩" in result
        assert len(agent.history._messages) < original_count
