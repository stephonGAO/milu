"""测试 Agent + Compactor 集成"""
import pytest
from unittest.mock import AsyncMock

from agent_framework.agent import Agent, AgentConfig
from agent_framework.agent.events import (
    AgentDone, AgentError, HistoryCompacted, TextDelta,
)
from agent_framework.agent.compactor import Compactor
from agent_framework.llm.base.message import Message, MessageRole
from agent_framework.llm.base.response import StreamChunk, TokenUsage


def _make_llm(response_text: str = "回复"):
    """创建返回固定文本的 mock LLM"""
    async def mock_chat(*args, **kwargs):
        yield StreamChunk(content=response_text, finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    llm = AsyncMock()
    llm.chat = mock_chat
    return llm


# ── 初始化测试 ────────────────────────────────────────────

class TestCompactorInit:
    """Compactor 初始化"""

    def test_compactor_enabled_by_default(self):
        """默认启用压缩"""
        llm = _make_llm()
        agent = Agent(llm=llm, system_prompt="test", skills_dir="/tmp/_nonexistent_")
        assert agent._compactor is not None
        assert isinstance(agent._compactor, Compactor)

    def test_compactor_disabled(self):
        """compact_enabled=False 时不创建压缩器"""
        llm = _make_llm()
        config = AgentConfig(compact_enabled=False)
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_")
        assert agent._compactor is None

    def test_compact_tool_registered(self):
        """启用压缩时注册 compact 元工具"""
        llm = _make_llm()
        agent = Agent(llm=llm, system_prompt="test", skills_dir="/tmp/_nonexistent_")
        wrapper = agent.tools.get_tool("compact")
        assert wrapper is not None

    def test_compact_tool_not_registered_when_disabled(self):
        """禁用压缩时不注册 compact 工具"""
        llm = _make_llm()
        config = AgentConfig(compact_enabled=False)
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_")
        wrapper = agent.tools.get_tool("compact")
        assert wrapper is None


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
                # 第一次：摘要调用（auto_compact L4 触发）
                yield StreamChunk(content="摘要内容", finish_reason="stop")
                yield StreamChunk(usage=TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120))
            else:
                # 第二次：正式回复
                yield StreamChunk(content="最终回复", finish_reason="stop")
                yield StreamChunk(usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

        llm = AsyncMock()
        llm.chat = mock_chat

        config = AgentConfig(compact_threshold=10)  # 极低阈值确保触发
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_")

        events = []
        async for event in agent.run("很长" * 100):
            events.append(event)

        # 应有 HistoryCompacted 事件
        compact_events = [e for e in events if isinstance(e, HistoryCompacted)]
        assert len(compact_events) >= 1
        assert compact_events[0].strategy == "auto"

    @pytest.mark.asyncio
    async def test_no_compact_when_under_threshold(self):
        """高阈值时不触发自动压缩"""
        llm = _make_llm("正常回复")
        config = AgentConfig(compact_threshold=999999999)  # 极高阈值
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_")

        events = []
        async for event in agent.run("短消息"):
            events.append(event)

        compact_events = [e for e in events if isinstance(e, HistoryCompacted)]
        assert len(compact_events) == 0

    @pytest.mark.asyncio
    async def test_no_compact_when_disabled(self):
        """compact_enabled=False 时不触发任何压缩"""
        llm = _make_llm("正常回复")
        config = AgentConfig(compact_enabled=False)
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_")

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
                # 第一次：正式回复触发上下文过长错误
                raise RuntimeError("context_length_exceeded: please reduce input")
            elif call_count == 2:
                # 第二次：应急压缩的摘要调用
                yield StreamChunk(content="应急摘要", finish_reason="stop")
                yield StreamChunk(usage=TokenUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60))
            else:
                # 第三次：重试成功
                yield StreamChunk(content="重试成功", finish_reason="stop")
                yield StreamChunk(usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

        llm = AsyncMock()
        llm.chat = mock_chat

        config = AgentConfig(compact_threshold=999999999)
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_")

        events = []
        async for event in agent.run("消息"):
            events.append(event)

        # 应有 HistoryCompacted(reactive) 事件
        compact_events = [e for e in events if isinstance(e, HistoryCompacted)]
        assert len(compact_events) == 1
        assert compact_events[0].strategy == "reactive"

        # 最终应有正常回复
        text_events = [e for e in events if isinstance(e, TextDelta)]
        assert any("重试成功" in e.text for e in text_events)

    @pytest.mark.asyncio
    async def test_non_context_error_reraises(self):
        """非上下文错误应重新抛出"""
        async def mock_chat(*args, **kwargs):
            raise ValueError("其他错误")
            yield  # make it an async generator

        llm = AsyncMock()
        llm.chat = mock_chat

        config = AgentConfig()
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_")

        with pytest.raises(ValueError, match="其他错误"):
            async for event in agent.run("消息"):
                pass


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
                yield StreamChunk(usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

        llm = AsyncMock()
        llm.chat = mock_chat

        config = AgentConfig(compact_threshold=10)
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_")

        async for event in agent.run("很长" * 100):
            pass

        # 历史应包含压缩标记
        all_msgs = agent.history.all_messages
        has_compacted = any("Compacted" in str(m.content) for m in all_msgs)
        assert has_compacted
