"""测试 Agent + Compactor 集成 — Token 动态阈值"""
import pytest
from unittest.mock import AsyncMock

from milu.agent import Agent, AgentConfig
from milu.agent.events import (
    AgentDone, AgentError, HistoryCompacted, TextDelta,
)
from milu.agent.compactor import Compactor
from milu.agent.config import CompactConfig
from milu.agent.history import ConversationHistory
from milu.llm.base.message import Message, MessageRole
from milu.llm.base.response import StreamChunk, TokenUsage


def _make_llm(response_text: str = "回复"):
    """创建返回固定文本的 mock LLM"""
    async def mock_chat(*args, **kwargs):
        yield StreamChunk(content=response_text, finish_reason="stop",
                          usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    llm = AsyncMock()
    llm.chat = mock_chat
    llm._model_config = None
    llm.capabilities = AsyncMock()
    llm.capabilities.max_context_window = 8192
    return llm


def _make_agent(llm, compact_config=None, **kwargs):
    """创建带压缩配置的 Agent"""
    config = AgentConfig(**kwargs)
    history = ConversationHistory(llm=llm, compact_config=compact_config)
    return Agent(llm=llm, system_prompt="test", config=config,
                 session_enabled=False, history=history, skills_dir="/tmp/_nonexistent_")


# ── 初始化测试 ────────────────────────────────────────────

class TestCompactorInit:
    """Compactor 初始化"""

    def test_compactor_enabled_by_default(self):
        """默认启用压缩"""
        llm = _make_llm()
        agent = _make_agent(llm)
        assert agent._history._compactor is not None
        assert isinstance(agent._history._compactor, Compactor)

    def test_compactor_disabled(self):
        """CompactConfig.enabled=False 时不创建压缩器"""
        llm = _make_llm()
        cc = CompactConfig(enabled=False)
        agent = _make_agent(llm, compact_config=cc)
        assert agent._history._compactor is None

    def test_compact_tool_registered(self):
        """启用压缩时注册 compact 元工具"""
        llm = _make_llm()
        agent = _make_agent(llm)
        wrapper = agent.tools.get_tool("compact")
        assert wrapper is not None

    def test_compact_tool_not_registered_when_disabled(self):
        """禁用压缩时不注册 compact 工具"""
        llm = _make_llm()
        cc = CompactConfig(enabled=False)
        agent = _make_agent(llm, compact_config=cc)
        wrapper = agent.tools.get_tool("compact")
        assert wrapper is None

    def test_compactor_receives_session(self):
        """Compactor 接收 session 引用"""
        llm = _make_llm()
        config = AgentConfig()
        history = ConversationHistory(llm=llm)
        # session_dir 不传 → 默认 default_session_dir()（conftest 已把 MILU_HOME 重定向到 tmp）
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      history=history, skills_dir="/tmp/_nonexistent_")
        if agent._history._compactor:
            assert agent._history._compactor._session is agent.session

    def test_compactor_reads_context_window(self):
        """Compactor 读取 max_context_window"""
        llm = _make_llm()
        agent = _make_agent(llm)
        assert agent._history._compactor._max_context_window == 8192

    def test_compactor_dynamic_thresholds_8k(self):
        """8K 窗口的动态阈值"""
        llm = _make_llm()
        agent = _make_agent(llm)
        c = agent._history._compactor
        # 8192 // 1500 = 5, // 2 = 2, max(5, 2) = 5
        assert c._old_round_threshold == 5
        # 8192 // 20 = 409, max(500, 409) = 500
        assert c._truncate_threshold == 500

    def test_compactor_dynamic_thresholds_128k(self):
        """128K 窗口的动态阈值"""
        async def mock_chat(*args, **kwargs):
            yield StreamChunk(content="回复", finish_reason="stop")

        llm = AsyncMock()
        llm.chat = mock_chat
        llm._model_config = None
        llm.capabilities = AsyncMock()
        llm.capabilities.max_context_window = 131072

        history = ConversationHistory(llm=llm)
        config = AgentConfig()
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      session_enabled=False, history=history, skills_dir="/tmp/_nonexistent_")
        c = agent._history._compactor
        # 131072 // 1500 = 87, // 2 = 43, min(43, 30) = 30
        assert c._old_round_threshold == 30
        # 131072 // 20 = 6553, min(6553, 4000) = 4000
        assert c._truncate_threshold == 4000

    def test_compactor_dynamic_thresholds_32k(self):
        """32K 窗口的动态阈值"""
        async def mock_chat(*args, **kwargs):
            yield StreamChunk(content="回复", finish_reason="stop")

        llm = AsyncMock()
        llm.chat = mock_chat
        llm._model_config = None
        llm.capabilities = AsyncMock()
        llm.capabilities.max_context_window = 32768

        history = ConversationHistory(llm=llm)
        config = AgentConfig()
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      session_enabled=False, history=history, skills_dir="/tmp/_nonexistent_")
        c = agent._history._compactor
        # 32768 // 1500 = 21, // 2 = 10, max(5, min(10, 30)) = 10
        assert c._old_round_threshold == 10
        # 32768 // 20 = 1638, max(500, min(1638, 4000)) = 1638
        assert c._truncate_threshold == 1638


# ── 自动压缩集成 ──────────────────────────────────────────

class TestAutoCompactIntegration:
    """自动压缩在 Agent 循环中的集成"""

    @pytest.mark.asyncio
    async def test_auto_compact_triggers_in_loop(self):
        """低阈值时自动压缩在 run() 循环中触发"""
        call_count = 0

        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # 第一次：摘要调用
                yield StreamChunk(content="摘要内容", finish_reason="stop")
                yield StreamChunk(usage=TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120))
            else:
                # 第二次：正式回复
                yield StreamChunk(content="最终回复", finish_reason="stop")
                yield StreamChunk(usage=TokenUsage(prompt_tokens=7000, completion_tokens=5, total_tokens=7005))

        llm = AsyncMock()
        llm.chat = mock_chat
        llm._model_config = None
        llm.capabilities = AsyncMock()
        llm.capabilities.max_context_window = 8192

        cc = CompactConfig(trigger_ratio=0.5)
        history = ConversationHistory(llm=llm, compact_config=cc)
        config = AgentConfig()
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      session_enabled=False, history=history, skills_dir="/tmp/_nonexistent_")

        # 预填充历史（模拟之前的高 token 使用）
        agent._history.update_prompt_tokens(7000)  # 7000/8192 > 0.5

        for i in range(6):
            agent.history._messages.append(
                Message(role=MessageRole.USER, content=f"历史消息{i}" * 50)
            )
            agent.history._messages.append(
                Message(role=MessageRole.ASSISTANT, content=f"历史回复{i}" * 50)
            )

        events = []
        async for event in agent.run("很长" * 100):
            events.append(event)

        compact_events = [e for e in events if isinstance(e, HistoryCompacted)]
        assert len(compact_events) >= 1
        assert compact_events[0].strategy == "auto"

    @pytest.mark.asyncio
    async def test_no_compact_when_under_threshold(self):
        """高阈值时不触发自动压缩"""
        llm = _make_llm("正常回复")
        cc = CompactConfig(trigger_ratio=0.99)
        agent = _make_agent(llm, compact_config=cc)

        events = []
        async for event in agent.run("短消息"):
            events.append(event)

        compact_events = [e for e in events if isinstance(e, HistoryCompacted)]
        assert len(compact_events) == 0

    @pytest.mark.asyncio
    async def test_no_compact_when_disabled(self):
        """CompactConfig.enabled=False 时不触发任何压缩"""
        llm = _make_llm("正常回复")
        cc = CompactConfig(enabled=False)
        agent = _make_agent(llm, compact_config=cc)

        events = []
        async for event in agent.run("消息"):
            events.append(event)

        compact_events = [e for e in events if isinstance(e, HistoryCompacted)]
        assert len(compact_events) == 0


# ── 应急压缩集成 ──────────────────────────────────────────

class TestReactiveCompactIntegration:
    """API 错误触发的应急压缩"""

    @pytest.mark.asyncio
    async def test_reactive_compact_on_context_error(self):
        """context_length 错误触发应急压缩后重试"""
        call_count = 0

        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                raise RuntimeError("context_length_exceeded: please reduce input")
            elif call_count == 2:
                yield StreamChunk(content="应急摘要", finish_reason="stop")
                yield StreamChunk(usage=TokenUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60))
            else:
                yield StreamChunk(content="重试成功", finish_reason="stop")
                yield StreamChunk(usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

        llm = AsyncMock()
        llm.chat = mock_chat
        llm._model_config = None
        llm.capabilities = AsyncMock()
        llm.capabilities.max_context_window = 8192

        cc = CompactConfig(trigger_ratio=0.99)
        history = ConversationHistory(llm=llm, compact_config=cc)
        config = AgentConfig()
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      session_enabled=False, history=history, skills_dir="/tmp/_nonexistent_")

        events = []
        async for event in agent.run("消息"):
            events.append(event)

        compact_events = [e for e in events if isinstance(e, HistoryCompacted)]
        assert len(compact_events) == 1
        assert compact_events[0].strategy == "reactive"

        text_events = [e for e in events if isinstance(e, TextDelta)]
        assert any("重试成功" in e.text for e in text_events)

    @pytest.mark.asyncio
    async def test_non_context_error_reraises(self):
        """非上下文错误应重新抛出"""
        async def mock_chat(*args, **kwargs):
            raise ValueError("其他错误")
            yield

        llm = AsyncMock()
        llm.chat = mock_chat
        llm._model_config = None
        llm.capabilities = AsyncMock()
        llm.capabilities.max_context_window = 8192

        agent = _make_agent(llm)

        with pytest.raises(ValueError, match="其他错误"):
            async for event in agent.run("消息"):
                pass


# ── Token 跟踪测试 ────────────────────────────────────────

class TestTokenTracking:
    """Token 跟踪在 run() 中的集成"""

    @pytest.mark.asyncio
    async def test_prompt_tokens_updated_during_run(self):
        """run() 过程中 prompt_tokens 被更新到 compactor"""
        llm = _make_llm("回复")
        agent = _make_agent(llm)

        async for _ in agent.run("hello"):
            pass

        # prompt_tokens 应被更新（mock 返回 10）
        assert agent._history._compactor._last_prompt_tokens == 10


# ── History 状态测试 ──────────────────────────────────────

class TestHistoryState:
    """压缩后历史状态"""

    @pytest.mark.asyncio
    async def test_history_replaced_after_compact(self):
        """压缩后历史被替换为摘要"""
        call_count = 0

        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield StreamChunk(content="摘要", finish_reason="stop")
                yield StreamChunk(usage=TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120))
            else:
                yield StreamChunk(content="回复", finish_reason="stop")
                yield StreamChunk(usage=TokenUsage(prompt_tokens=7000, completion_tokens=5, total_tokens=7005))

        llm = AsyncMock()
        llm.chat = mock_chat
        llm._model_config = None
        llm.capabilities = AsyncMock()
        llm.capabilities.max_context_window = 8192

        cc = CompactConfig(trigger_ratio=0.5)
        history = ConversationHistory(llm=llm, compact_config=cc)
        config = AgentConfig()
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      session_enabled=False, history=history, skills_dir="/tmp/_nonexistent_")

        agent._history.update_prompt_tokens(7000)

        for i in range(6):
            agent.history._messages.append(
                Message(role=MessageRole.USER, content=f"历史消息{i}" * 50)
            )
            agent.history._messages.append(
                Message(role=MessageRole.ASSISTANT, content=f"历史回复{i}" * 50)
            )

        async for event in agent.run("很长" * 100):
            pass

        all_msgs = agent.history.all_messages
        has_compacted = any("Compacted" in str(m.content) for m in all_msgs)
        assert has_compacted
