"""测试 memory 长期记忆工具（用户级文件存储 / Agent 开关 / 系统提示词注入）。"""
from __future__ import annotations

import json

from unittest.mock import AsyncMock

import pytest

from milu.agent import Agent
from milu.llm.base.message import MessageRole
from milu.llm.base.response import StreamChunk, TokenUsage
from milu.tools.builtin.memory_tool import (
    _current_memory_path,
    memory_file_path,
    memory_read,
    memory_write,
    render_memory_prompt,
)


async def _write(content: str, category: str = "fact") -> str:
    return await memory_write._tool_wrapper.func(content=content, category=category)


async def _read(category: str = "") -> str:
    return await memory_read._tool_wrapper.func(category=category)


# ── 工具层（直接注入文件路径）────────────────────────────


class TestMemoryToolFile:
    async def test_write_then_read(self, tmp_path):
        path = tmp_path / "memory" / "u1.json"
        token = _current_memory_path.set(path)
        try:
            result = await _write("用户喜欢简洁回复", "preference")
            assert "已记住" in result

            content = await _read()
            assert "用户喜欢简洁回复" in content
            assert "[preference]" in content

            # 落盘验证
            data = json.loads(path.read_text(encoding="utf-8"))
            assert len(data["items"]) == 1
        finally:
            _current_memory_path.reset(token)

    async def test_dedupe(self, tmp_path):
        """完全相同内容不重复记录"""
        token = _current_memory_path.set(tmp_path / "m.json")
        try:
            await _write("同一条信息")
            result = await _write("同一条信息")
            assert "无需重复" in result
            assert "共 1 条" in result
        finally:
            _current_memory_path.reset(token)

    async def test_category_filter(self, tmp_path):
        token = _current_memory_path.set(tmp_path / "m.json")
        try:
            await _write("偏好A", "preference")
            await _write("事实B", "fact")

            prefs = await _read("preference")
            assert "偏好A" in prefs
            assert "事实B" not in prefs
        finally:
            _current_memory_path.reset(token)

    async def test_empty_content_raises(self, tmp_path):
        token = _current_memory_path.set(tmp_path / "m.json")
        try:
            with pytest.raises(ValueError):
                await _write("   ")
        finally:
            _current_memory_path.reset(token)

    async def test_persists_across_reinjection(self, tmp_path):
        """重新注入同一路径仍能读到（模拟重启 / 新会话）"""
        path = tmp_path / "m.json"
        token = _current_memory_path.set(path)
        try:
            await _write("跨重启的记忆")
        finally:
            _current_memory_path.reset(token)

        token2 = _current_memory_path.set(path)
        try:
            assert "跨重启的记忆" in await _read()
        finally:
            _current_memory_path.reset(token2)

    async def test_not_enabled_read_lenient(self):
        token = _current_memory_path.set(None)
        try:
            assert "暂无" in await _read()
        finally:
            _current_memory_path.reset(token)

    async def test_not_enabled_write_raises(self):
        token = _current_memory_path.set(None)
        try:
            with pytest.raises(RuntimeError, match="长期记忆未启用"):
                await _write("无处安放")
        finally:
            _current_memory_path.reset(token)


# ── 路径派生 / 提示词渲染 ─────────────────────────────────


class TestPathAndPrompt:
    def test_memory_file_path_under_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MILU_HOME", str(tmp_path))
        path = memory_file_path("alice")
        assert path == tmp_path / "memory" / "alice.json"

    def test_memory_file_path_sanitized(self, tmp_path, monkeypatch):
        """用户标识做文件系统安全化；空串回退 default"""
        monkeypatch.setenv("MILU_HOME", str(tmp_path))
        assert memory_file_path("a/b:c").name == "a_b_c.json"
        assert memory_file_path("  ").name == "default.json"

    def test_render_prompt_with_items(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(
            json.dumps({"items": [
                {"content": "回复要简短", "category": "preference",
                 "created_at": "2026-06-04T10:00:00"},
            ]}, ensure_ascii=False),
            encoding="utf-8",
        )
        text = render_memory_prompt(path)
        assert "## 长期记忆" in text
        assert "[preference] 回复要简短（2026-06-04）" in text

    def test_render_prompt_empty_has_guidance(self, tmp_path):
        """无条目时仍注入指引，LLM 才知道何时该 memory_write"""
        text = render_memory_prompt(tmp_path / "none.json")
        assert "## 长期记忆" in text
        assert "memory_write" in text
        assert "暂无记忆条目" in text


# ── Agent 集成（开关 / run() 注入 / 提示词）──────────────


def _make_memory_llm():
    """第一轮调用 memory_write，第二轮收尾"""
    call_count = 0

    async def chat(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': 'call_mem',
                    'function': type('obj', (), {
                        'name': 'memory_write',
                        'arguments': json.dumps(
                            {"content": "用户在测试记忆功能", "category": "fact"},
                            ensure_ascii=False,
                        ),
                    })()
                })()
            ])
            yield StreamChunk(finish_reason="tool_calls")
        else:
            yield StreamChunk(content="已记录", finish_reason="stop")
            yield StreamChunk(usage=TokenUsage(1, 1, 2))

    llm = AsyncMock()
    llm.chat = chat
    return llm


def _make_agent(llm, tmp_path, **kwargs) -> Agent:
    return Agent(
        llm=llm,
        tools=[],
        session_dir=str(tmp_path / "sessions"),
        register_catalog=False, register_skills=False,
        subagents=[],
        **kwargs,
    )


class TestAgentIntegration:
    def test_disabled_by_default(self, tmp_path):
        """默认关闭：不注册 memory 工具、提示词不含记忆段"""
        agent = _make_agent(AsyncMock(), tmp_path, system_prompt="你是助手")
        names = [s["function"]["name"] for s in agent.tools.get_schemas()]
        assert "memory_write" not in names
        assert "memory_read" not in names

        agent._build_system_prompt()
        system = agent.history.get_messages()[0]
        assert "长期记忆" not in system.content

    def test_enabled_registers_tools(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MILU_HOME", str(tmp_path))
        agent = _make_agent(AsyncMock(), tmp_path, memory=True)
        names = [s["function"]["name"] for s in agent.tools.get_schemas()]
        assert "memory_write" in names
        assert "memory_read" in names
        assert agent._memory_path == tmp_path / "memory" / "default.json"

    async def test_run_writes_user_level_file(self, tmp_path, monkeypatch):
        """memory="alice"：run() 注入路径，memory_write 落盘用户级文件（与 session 解耦）"""
        monkeypatch.setenv("MILU_HOME", str(tmp_path))
        agent = _make_agent(_make_memory_llm(), tmp_path, memory="alice")
        async for _ in agent.run("记住这个"):
            pass

        mem_file = tmp_path / "memory" / "alice.json"
        assert mem_file.exists()
        data = json.loads(mem_file.read_text(encoding="utf-8"))
        assert data["items"][0]["content"] == "用户在测试记忆功能"
        # 不再写入 session 目录
        assert not (agent.session.dir_path / "memory.json").exists()

    async def test_memory_injected_at_prompt_tail(self, tmp_path, monkeypatch):
        """记忆条目注入 system prompt 最后"""
        monkeypatch.setenv("MILU_HOME", str(tmp_path))
        agent = _make_agent(
            _make_memory_llm(), tmp_path,
            system_prompt="你是助手", memory="alice",
        )
        async for _ in agent.run("记住这个"):
            pass

        agent._build_system_prompt()
        system = agent.history.get_messages()[0]
        assert system.role == MessageRole.SYSTEM
        idx = system.content.find("## 长期记忆")
        assert idx > system.content.find("你是助手")
        assert "用户在测试记忆功能" in system.content[idx:]

    async def test_shared_across_agents_same_user(self, tmp_path, monkeypatch):
        """同一用户标识的另一个 Agent（新会话）能看到已写入的记忆——跨 session 共享"""
        monkeypatch.setenv("MILU_HOME", str(tmp_path))
        writer = _make_agent(_make_memory_llm(), tmp_path, memory="alice")
        async for _ in writer.run("记住这个"):
            pass

        reader = _make_agent(
            AsyncMock(), tmp_path, system_prompt="你是助手", memory="alice",
        )
        reader._build_system_prompt()
        assert "用户在测试记忆功能" in reader.history.get_messages()[0].content

        other = _make_agent(
            AsyncMock(), tmp_path, system_prompt="你是助手", memory="bob",
        )
        other._build_system_prompt()
        assert "用户在测试记忆功能" not in other.history.get_messages()[0].content
