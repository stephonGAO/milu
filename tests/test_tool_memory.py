"""测试 memory 长期记忆工具（文件后端 / 内存后端 / Agent 集成）。"""
from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock

from agent_framework.agent import Agent
from agent_framework.llm.base.response import StreamChunk, TokenUsage
from agent_framework.tools.builtin.memory_tool import (
    _current_memory_items,
    _current_session_dir,
    memory_read,
    memory_write,
)


async def _write(content: str, category: str = "fact") -> str:
    return await memory_write._tool_wrapper.func(content=content, category=category)


async def _read(category: str = "") -> str:
    return await memory_read._tool_wrapper.func(category=category)


# ── 文件后端 ──────────────────────────────────────────────


class TestFileBackend:
    @pytest.mark.asyncio
    async def test_write_then_read(self, tmp_path):
        token = _current_session_dir.set(tmp_path)
        try:
            result = await _write("用户喜欢简洁回复", "preference")
            assert "已记住" in result

            content = await _read()
            assert "用户喜欢简洁回复" in content
            assert "[preference]" in content

            # 落盘验证
            data = json.loads((tmp_path / "memory.json").read_text(encoding="utf-8"))
            assert len(data["items"]) == 1
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_dedupe(self, tmp_path):
        """完全相同内容不重复记录"""
        token = _current_session_dir.set(tmp_path)
        try:
            await _write("同一条信息")
            result = await _write("同一条信息")
            assert "无需重复" in result
            assert "共 1 条" in result
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_category_filter(self, tmp_path):
        token = _current_session_dir.set(tmp_path)
        try:
            await _write("偏好A", "preference")
            await _write("事实B", "fact")

            prefs = await _read("preference")
            assert "偏好A" in prefs
            assert "事实B" not in prefs
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_empty_content_raises(self, tmp_path):
        token = _current_session_dir.set(tmp_path)
        try:
            with pytest.raises(ValueError):
                await _write("   ")
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_persists_across_reads(self, tmp_path):
        """文件后端：重新注入同一目录仍能读到（模拟重启）"""
        token = _current_session_dir.set(tmp_path)
        try:
            await _write("跨重启的记忆")
        finally:
            _current_session_dir.reset(token)

        token2 = _current_session_dir.set(tmp_path)
        try:
            assert "跨重启的记忆" in await _read()
        finally:
            _current_session_dir.reset(token2)


# ── 内存后端 / 无后端 ─────────────────────────────────────


class TestMemoryBackend:
    @pytest.mark.asyncio
    async def test_in_memory_write_read(self):
        store: list[dict] = []
        sess_token = _current_session_dir.set(None)
        mem_token = _current_memory_items.set(store)
        try:
            await _write("内存里的记忆")
            assert len(store) == 1
            assert "内存里的记忆" in await _read()
        finally:
            _current_session_dir.reset(sess_token)
            _current_memory_items.reset(mem_token)

    @pytest.mark.asyncio
    async def test_no_backend_read_lenient(self):
        sess_token = _current_session_dir.set(None)
        mem_token = _current_memory_items.set(None)
        try:
            assert "暂无" in await _read()
        finally:
            _current_session_dir.reset(sess_token)
            _current_memory_items.reset(mem_token)

    @pytest.mark.asyncio
    async def test_no_backend_write_raises(self):
        sess_token = _current_session_dir.set(None)
        mem_token = _current_memory_items.set(None)
        try:
            with pytest.raises(RuntimeError, match="记忆存储未注入"):
                await _write("无处安放")
        finally:
            _current_session_dir.reset(sess_token)
            _current_memory_items.reset(mem_token)


# ── Agent 集成（run() 注入）──────────────────────────────


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


class TestAgentIntegration:
    @pytest.mark.asyncio
    async def test_run_injects_session_backend(self, tmp_path):
        """有 session：run() 注入 session 目录，memory_write 落盘 memory.json"""
        agent = Agent(
            llm=_make_memory_llm(),
            tools=[memory_write, memory_read],
            session_dir=str(tmp_path),
            register_catalog=False, register_skills=False,
            subagents=[],
        )
        async for _ in agent.run("记住这个"):
            pass

        mem_file = agent.session.dir_path / "memory.json"
        assert mem_file.exists()
        data = json.loads(mem_file.read_text(encoding="utf-8"))
        assert data["items"][0]["content"] == "用户在测试记忆功能"

    @pytest.mark.asyncio
    async def test_run_injects_memory_backend_without_session(self):
        """无 session：run() 注入内存后端，memory_write 不报错且跨轮保留"""
        agent = Agent(
            llm=_make_memory_llm(),
            tools=[memory_write, memory_read],
            session_enabled=False,
            register_catalog=False, register_skills=False,
            subagents=[],
        )
        async for _ in agent.run("记住这个"):
            pass

        assert len(agent._in_memory_memories) == 1
        assert agent._in_memory_memories[0]["content"] == "用户在测试记忆功能"
