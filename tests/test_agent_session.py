"""测试 Agent + Session 集成"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

from milu.agent import Agent, AgentConfig
from milu.agent.events import AgentDone, SessionLoaded
from milu.agent.session import Session
from milu.llm.base.message import Message, MessageRole
from milu.llm.base.response import StreamChunk, TokenUsage


def _make_llm(response_text: str = "回复"):
    """创建返回固定文本的 mock LLM"""
    async def mock_chat(*args, **kwargs):
        yield StreamChunk(content=response_text, finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    llm = AsyncMock()
    llm.chat = mock_chat
    llm._model_config = None
    llm.capabilities = AsyncMock()
    llm.capabilities.max_context_window = 8192
    return llm


@pytest.fixture
def tmp_dir():
    """临时目录"""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ── Session 创建 ──────────────────────────────────────────

class TestAgentSessionCreation:
    """Agent 初始化时创建 Session"""

    def test_session_created_by_default(self, tmp_dir):
        """默认启用 session"""
        config = AgentConfig()
        _sess = dict(session_dir=str(tmp_dir))
        llm = _make_llm()
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_", **_sess)
        assert agent.session is not None
        assert isinstance(agent.session, Session)
        assert agent.session.dir_path.exists()

    def test_session_disabled(self, tmp_dir):
        """session_enabled=False 时不创建"""
        config = AgentConfig()
        _sess = dict(session_enabled=False, session_dir=str(tmp_dir))
        llm = _make_llm()
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_", **_sess)
        assert agent.session is None

    def test_session_dir_created(self, tmp_dir):
        """会话目录被正确创建"""
        config = AgentConfig()
        _sess = dict(session_dir=str(tmp_dir))
        llm = _make_llm()
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_", **_sess)
        assert agent.session.dir_path.parent == tmp_dir


# ── 会话日志写入 ──────────────────────────────────────────

class TestAgentSessionLogging:
    """Agent run() 过程中消息写入 JSONL"""

    @pytest.mark.asyncio
    async def test_messages_logged_during_run(self, tmp_dir):
        """run() 过程中 user 和 assistant 消息被记录"""
        config = AgentConfig()
        _sess = dict(session_dir=str(tmp_dir))
        llm = _make_llm("你好！")
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_", **_sess)

        events = []
        async for event in agent.run("你好"):
            events.append(event)

        # 应有 AgentDone 事件
        assert any(isinstance(e, AgentDone) for e in events)

        # JSONL 应包含 user 和 assistant 消息
        assert agent.session.message_count >= 2
        loaded = agent.session.load_messages()
        roles = [m.role for m in loaded]
        assert MessageRole.USER in roles
        assert MessageRole.ASSISTANT in roles


# ── Session 生命周期 ──────────────────────────────────────

class TestAgentSessionLifecycle:
    """Agent session 生命周期方法"""

    def test_save_session(self, tmp_dir):
        """save_session 写入 session.json"""
        config = AgentConfig()
        _sess = dict(session_dir=str(tmp_dir))
        llm = _make_llm()
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_", **_sess)

        agent.save_session()
        assert agent.session.metadata_path.exists()

    def test_new_session(self, tmp_dir):
        """new_session 保存旧会话并创建新的"""
        config = AgentConfig()
        _sess = dict(session_dir=str(tmp_dir))
        llm = _make_llm()
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_", **_sess)

        old_id = agent.session.session_id
        agent.new_session()

        # 旧会话元数据已保存
        assert (tmp_dir / old_id / "session.json").exists()

        # 新会话 ID 不同
        assert agent.session.session_id != old_id

        # 历史已清空
        assert len(agent.history.all_messages) == 0

    @pytest.mark.asyncio
    async def test_load_session(self, tmp_dir):
        """load_session 恢复历史消息"""
        config = AgentConfig()
        _sess = dict(session_dir=str(tmp_dir))
        llm = _make_llm("回复")
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_", **_sess)

        # 先有一些对话
        async for _ in agent.run("问题1"):
            pass
        async for _ in agent.run("问题2"):
            pass

        original_id = agent.session.session_id
        original_count = agent.session.message_count
        assert original_count >= 4  # 2 user + 2 assistant

        # 新建会话（清空历史）
        agent.new_session()
        # clear() 保留 system 消息，所以最多 1 条
        non_system = [m for m in agent.history.all_messages if m.role != MessageRole.SYSTEM]
        assert len(non_system) == 0

        # 加载旧会话
        loaded_count = agent.load_session(original_id)
        assert loaded_count == original_count
        assert len(agent.history.all_messages) >= 4

    def test_load_session_nonexistent(self, tmp_dir):
        """加载不存在的会话"""
        config = AgentConfig()
        _sess = dict(session_dir=str(tmp_dir))
        llm = _make_llm()
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_", **_sess)

        # 加载不存在的 ID 不应崩溃（目录会被创建）
        count = agent.load_session("nonexistent_id")
        assert count == 0

    @pytest.mark.asyncio
    async def test_reset_switches_log_segment(self, tmp_dir):
        """reset() 切换到新日志段：session id/目录不变，旧段保留，历史清空"""
        config = AgentConfig()
        _sess = dict(session_dir=str(tmp_dir))
        llm = _make_llm("回复")
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_", **_sess)

        async for _ in agent.run("问题1"):
            pass

        old_id = agent.session.session_id
        old_dir = agent.session.dir_path
        old_path = agent.session.conversation_path
        assert old_path.name == "conversation.jsonl"

        await agent.reset()

        # 切到新段，id 与目录不变，旧段仍在磁盘
        assert agent.session.session_id == old_id
        assert agent.session.dir_path == old_dir
        assert agent.session.conversation_path.name == "conversation.2.jsonl"
        assert old_path.exists()
        # 内存历史清空（保留 system）
        non_system = [m for m in agent.history.all_messages if m.role != MessageRole.SYSTEM]
        assert len(non_system) == 0

    @pytest.mark.asyncio
    async def test_reset_then_load_only_new_segment(self, tmp_dir):
        """reset 后继续对话，load_session 只恢复 reset 后的消息（原 bug 修复验证）"""
        config = AgentConfig()
        _sess = dict(session_dir=str(tmp_dir))
        llm = _make_llm("回复")
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_", **_sess)

        async for _ in agent.run("问题1"):
            pass

        await agent.reset()

        async for _ in agent.run("问题2"):
            pass

        session_id = agent.session.session_id
        # 重新加载同一会话：只应有 reset 后的 2 条（user + assistant）
        count = agent.load_session(session_id)
        assert count == 2
        contents = [m.content for m in agent.history.all_messages
                    if m.role != MessageRole.SYSTEM]
        assert contents == ["问题2", "回复"]


# ── History + Session 集成 ────────────────────────────────

class TestHistorySessionIntegration:
    """ConversationHistory 与 Session 集成"""

    def test_attach_and_detach_session(self, tmp_dir):
        """attach/detach session"""
        config = AgentConfig()
        _sess = dict(session_enabled=False)
        llm = _make_llm()
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_", **_sess)

        session = Session("manual", tmp_dir)
        agent.history.attach_session(session)
        assert agent.history._session is session

        agent.history.detach_session()
        assert agent.history._session is None

    def test_replace_all_does_not_log(self, tmp_dir):
        """replace_all 不写入 JSONL（append-only 审计日志）"""
        config = AgentConfig()
        _sess = dict(session_dir=str(tmp_dir))
        llm = _make_llm()
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_", **_sess)

        # 添加一些消息
        agent.history.add(Message(role=MessageRole.USER, content="hello"))
        agent.history.add(Message(role=MessageRole.ASSISTANT, content="world"))

        count_before = agent.session.message_count

        # replace_all（模拟压缩）
        compacted = [Message(role=MessageRole.USER, content="[Compacted] summary")]
        agent.history.replace_all(compacted)

        # JSONL 消息数不应增加
        assert agent.session.message_count == count_before

    def test_load_from_session(self, tmp_dir):
        """load_from_session 批量加载"""
        session = Session("bulk_load", tmp_dir)
        session.log_message(Message(role=MessageRole.USER, content="q1"))
        session.log_message(Message(role=MessageRole.ASSISTANT, content="a1"))
        session.log_message(Message(role=MessageRole.USER, content="q2"))

        config = AgentConfig()
        _sess = dict(session_enabled=False)
        llm = _make_llm()
        agent = Agent(llm=llm, system_prompt="test", config=config,
                      skills_dir="/tmp/_nonexistent_", **_sess)

        agent.history.load_from_session(session)
        assert len(agent.history.all_messages) == 3
        assert agent.history._session is session
