"""测试 Session 会话持久化"""
import json
import pytest
import tempfile
from pathlib import Path

from agent_framework.agent.session import Session
from agent_framework.llm.base.message import Message, MessageRole


@pytest.fixture
def tmp_dir():
    """临时目录"""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _make_msg(role: MessageRole, content: str, **kwargs) -> Message:
    return Message(role=role, content=content, **kwargs)


# ── 基本功能 ──────────────────────────────────────────────

class TestSessionBasics:
    """Session 基本功能"""

    def test_create_session(self, tmp_dir):
        """创建会话应生成目录"""
        session = Session("test_001", tmp_dir)
        assert session.dir_path.exists()
        assert session.session_id == "test_001"

    def test_generate_id_format(self):
        """generate_id 应返回 YYYYMMDD_HHMMSS_{16hex} 格式"""
        sid = Session.generate_id()
        parts = sid.split("_")
        assert len(parts) == 3
        assert len(parts[0]) == 8   # YYYYMMDD
        assert len(parts[1]) == 6   # HHMMSS
        assert len(parts[2]) == 16  # 8 字节 = 16 hex（防碰撞）
        int(parts[2], 16)           # 应为合法 16 进制

    def test_generate_id_no_collision_under_burst(self):
        """同一时刻批量生成大量 id 不应碰撞（64 bit 随机）。"""
        ids = {Session.generate_id() for _ in range(20000)}
        assert len(ids) == 20000, "generate_id 出现碰撞"

    def test_append_jsonl_writes_are_complete_lines(self, tmp_dir):
        """连续写入应产出 100 行完整、可解析的 JSONL（验证带锁 append 不破坏写入）。

        注：框架本身是 async 单线程，同进程内写入由事件循环串行化、不会交错；
        跨进程（多 worker）写同一文件的整行原子性由 _append_jsonl 的 flock 保证
        （POSIX 生效），此处仅验证写入路径完整性与可恢复性。
        """
        session = Session("lock_test", tmp_dir)
        for i in range(100):
            session.log_message(_make_msg(MessageRole.USER, f"msg-{i}"))

        lines = session.conversation_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 100
        for line in lines:
            json.loads(line)  # 每行都是完整 JSON

        # 重新加载应能还原全部消息
        reloaded = Session("lock_test", tmp_dir).load_messages()
        assert len(reloaded) == 100
        assert reloaded[0].content == "msg-0"
        assert reloaded[-1].content == "msg-99"

    def test_log_message_appends_jsonl(self, tmp_dir):
        """log_message 应追加到 JSONL 文件"""
        session = Session("test_log", tmp_dir)

        msg = _make_msg(MessageRole.USER, "hello world")
        line_no = session.log_message(msg)

        assert line_no == 0
        assert session.conversation_path.exists()

        # 验证 JSONL 内容
        with open(session.conversation_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["role"] == "user"
        assert obj["content"] == "hello world"
        assert obj["line"] == 0
        assert "timestamp" in obj

    def test_system_message_not_logged(self, tmp_dir):
        """SYSTEM 消息不记录（返回 -1）"""
        session = Session("test_sys", tmp_dir)

        msg = _make_msg(MessageRole.SYSTEM, "system prompt")
        result = session.log_message(msg)

        assert result == -1
        # JSONL 文件不应存在（没有成功写入过）
        assert not session.conversation_path.exists() or \
               session.conversation_path.stat().st_size == 0

    def test_round_counter_increments_on_assistant(self, tmp_dir):
        """round 计数器仅在 ASSISTANT 消息时自增"""
        session = Session("test_round", tmp_dir)

        session.log_message(_make_msg(MessageRole.USER, "q1"))
        assert session.round_count == 0

        session.log_message(_make_msg(MessageRole.ASSISTANT, "a1"))
        assert session.round_count == 1

        session.log_message(_make_msg(MessageRole.TOOL, "tool result",
                                       tool_call_id="call_1"))
        assert session.round_count == 1  # TOOL 不增加 round

        session.log_message(_make_msg(MessageRole.USER, "q2"))
        assert session.round_count == 1

        session.log_message(_make_msg(MessageRole.ASSISTANT, "a2"))
        assert session.round_count == 2

    def test_message_count(self, tmp_dir):
        """message_count 跟踪非 SYSTEM 消息数量"""
        session = Session("test_count", tmp_dir)

        session.log_message(_make_msg(MessageRole.USER, "u1"))
        session.log_message(_make_msg(MessageRole.ASSISTANT, "a1"))
        session.log_message(_make_msg(MessageRole.TOOL, "t1", tool_call_id="c1"))
        session.log_message(_make_msg(MessageRole.SYSTEM, "sys"))  # 不计入

        assert session.message_count == 3


# ── 读取 ──────────────────────────────────────────────────

class TestSessionRead:
    """Session 读取功能"""

    def test_load_messages_roundtrip(self, tmp_dir):
        """写入后读取应还原 Message 列表"""
        session = Session("test_rt", tmp_dir)

        original_msgs = [
            _make_msg(MessageRole.USER, "hello"),
            _make_msg(MessageRole.ASSISTANT, "world"),
            _make_msg(MessageRole.TOOL, "result", tool_call_id="call_1", name="test"),
            _make_msg(MessageRole.USER, "follow up"),
        ]

        for msg in original_msgs:
            session.log_message(msg)

        loaded = session.load_messages()
        assert len(loaded) == 4

        for orig, loaded_msg in zip(original_msgs, loaded):
            assert loaded_msg.role == orig.role
            assert loaded_msg.content == orig.content
            if orig.tool_call_id:
                assert loaded_msg.tool_call_id == orig.tool_call_id
            if orig.name:
                assert loaded_msg.name == orig.name

    def test_load_messages_empty(self, tmp_dir):
        """空会话加载返回空列表"""
        session = Session("test_empty", tmp_dir)
        assert session.load_messages() == []

    def test_load_messages_skips_corrupt_lines(self, tmp_dir):
        """损坏的 JSON 行被跳过"""
        session = Session("test_corrupt", tmp_dir)

        # 写入正常消息
        session.log_message(_make_msg(MessageRole.USER, "good"))

        # 手动追加损坏行
        with open(session.conversation_path, "a", encoding="utf-8") as f:
            f.write("this is not json\n")

        # 再写入正常消息
        session.log_message(_make_msg(MessageRole.ASSISTANT, "also good"))

        loaded = session.load_messages()
        # 损坏行被跳过，但 line_counter 仍然增加了（因为手动写入的）
        # 只有 log_message 写入的 2 条会被正确解析
        assert len(loaded) >= 2
        roles = [m.role for m in loaded]
        assert MessageRole.USER in roles
        assert MessageRole.ASSISTANT in roles

    def test_read_line(self, tmp_dir):
        """read_line 读取指定行号"""
        session = Session("test_readline", tmp_dir)

        session.log_message(_make_msg(MessageRole.USER, "first"))
        session.log_message(_make_msg(MessageRole.ASSISTANT, "second"))
        session.log_message(_make_msg(MessageRole.USER, "third"))

        line = session.read_line(1)
        assert line is not None
        assert line["role"] == "assistant"
        assert line["content"] == "second"

    def test_read_line_out_of_range(self, tmp_dir):
        """超出范围的行号返回 None"""
        session = Session("test_oor", tmp_dir)
        session.log_message(_make_msg(MessageRole.USER, "only one"))

        assert session.read_line(99) is None


# ── 元数据 ────────────────────────────────────────────────

class TestSessionMetadata:
    """Session 元数据"""

    def test_save_and_load_metadata(self, tmp_dir):
        """save_metadata 和 load_session 应保持元数据一致"""
        session = Session("test_meta", tmp_dir, model="qwen-max")
        session.log_message(_make_msg(MessageRole.USER, "hello"))
        session.log_message(_make_msg(MessageRole.ASSISTANT, "world"))
        session.save_metadata()

        # 加载
        loaded = Session.load_session("test_meta", tmp_dir)
        assert loaded.session_id == "test_meta"
        assert loaded.message_count == 2
        assert loaded.round_count == 1

        # 验证 session.json 内容
        with open(loaded.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["model"] == "qwen-max"
        assert meta["message_count"] == 2

    def test_save_metadata_with_extra(self, tmp_dir):
        """save_metadata 支持额外字段"""
        session = Session("test_extra", tmp_dir)
        session.save_metadata(custom_field="value")

        with open(session.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["custom_field"] == "value"


# ── 类方法 ────────────────────────────────────────────────

class TestSessionClassMethods:
    """Session 类方法"""

    def test_list_sessions_empty(self, tmp_dir):
        """空目录返回空列表"""
        sessions = Session.list_sessions(tmp_dir)
        assert sessions == []

    def test_list_sessions_nonexistent_dir(self, tmp_dir):
        """不存在的目录返回空列表"""
        sessions = Session.list_sessions(tmp_dir / "nonexistent")
        assert sessions == []

    def test_list_sessions_multiple(self, tmp_dir):
        """列出多个会话，按 updated_at 降序"""
        import time

        s1 = Session("session_a", tmp_dir, model="model-a")
        s1.log_message(_make_msg(MessageRole.USER, "msg1"))
        s1.save_metadata()

        time.sleep(0.05)

        s2 = Session("session_b", tmp_dir, model="model-b")
        s2.log_message(_make_msg(MessageRole.USER, "msg1"))
        s2.log_message(_make_msg(MessageRole.ASSISTANT, "msg2"))
        s2.save_metadata()

        sessions = Session.list_sessions(tmp_dir)
        assert len(sessions) == 2
        # session_b 更新，应排在前面
        assert sessions[0]["session_id"] == "session_b"
        assert sessions[1]["session_id"] == "session_a"

    def test_list_sessions_without_metadata(self, tmp_dir):
        """没有 session.json 的会话也能列出"""
        session = Session("no_meta", tmp_dir)
        session.log_message(_make_msg(MessageRole.USER, "hello"))
        # 不调用 save_metadata

        sessions = Session.list_sessions(tmp_dir)
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "no_meta"
        assert sessions[0]["message_count"] == 1

    def test_load_session_restores_counters(self, tmp_dir):
        """load_session 应从 JSONL 恢复计数器"""
        # 创建并写入
        s1 = Session("restore", tmp_dir)
        for i in range(5):
            s1.log_message(_make_msg(MessageRole.USER, f"q{i}"))
            s1.log_message(_make_msg(MessageRole.ASSISTANT, f"a{i}"))
        s1.save_metadata()

        # 加载
        s2 = Session.load_session("restore", tmp_dir)
        assert s2.message_count == 10
        assert s2.round_count == 5


# ── 属性 ──────────────────────────────────────────────────

class TestSessionProperties:
    """Session 属性"""

    def test_paths(self, tmp_dir):
        """路径属性正确"""
        session = Session("paths", tmp_dir)
        assert session.conversation_path == tmp_dir / "paths" / "conversation.jsonl"
        assert session.metadata_path == tmp_dir / "paths" / "session.json"
        assert session.dir_path == tmp_dir / "paths"
        assert session.base_dir == tmp_dir

    def test_repr(self, tmp_dir):
        """repr 包含关键信息"""
        session = Session("repr_test", tmp_dir)
        session.log_message(_make_msg(MessageRole.USER, "hi"))
        r = repr(session)
        assert "repr_test" in r
        assert "messages=1" in r
