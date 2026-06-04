"""测试内置子代理三件套 + on_confirm 透传 + AgentPool 默认子代理。

覆盖：
- builtin_subagent_configs()：默认集合、include 参数、未知名称报错、工具/技能/role 装配
- on_confirm 透传：MANUAL 模式下子代理内的不安全工具走父 Agent 的确认回调（批准/拒绝）
- AgentPool 默认工厂：自动注册三件套；agent_kwargs={"subagents": []} 可关闭
"""
from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock

from agent_framework.agent import Agent, AgentMode
from agent_framework.agent.subagent import (
    SubAgentConfig,
    builtin_subagent_configs,
    create_subagent_tools,
)
from agent_framework.agent.events import ConfirmResponse
from agent_framework.llm.base.response import StreamChunk, TokenUsage
from agent_framework.resources import builtin_prompts_dir
from agent_framework.serving import AgentPool, AgentPoolConfig
from agent_framework.tools.decorator import tool


# ── 内置配置工厂 ──────────────────────────────────────────


class TestBuiltinSubagentConfigs:
    def test_default_trio(self):
        """默认返回三件套：researcher / reader / coder（顺序稳定）"""
        configs = builtin_subagent_configs()
        assert [c.name for c in configs] == ["researcher", "reader", "coder"]

    def test_include_optional_reviewer(self):
        """include 可加入可选的 reviewer"""
        configs = builtin_subagent_configs(
            include=("researcher", "reader", "coder", "reviewer")
        )
        assert [c.name for c in configs] == ["researcher", "reader", "coder", "reviewer"]

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="未知的内置子代理"):
            builtin_subagent_configs(include=("researcher", "nonexistent"))

    def test_role_dirs_exist(self):
        """每个预设的 role 都能解析到内置提示词目录"""
        for cfg in builtin_subagent_configs(
            include=("researcher", "reader", "coder", "reviewer")
        ):
            assert cfg.role == cfg.name
            assert builtin_prompts_dir(cfg.role).is_dir()

    def test_tool_assignment(self):
        """工具按选型清单装配：researcher/reader 只读，coder 含写入"""
        by_name = {c.name: c for c in builtin_subagent_configs()}

        def tool_names(cfg: SubAgentConfig) -> set[str]:
            return {t._tool_wrapper.name for t in cfg.tools}

        assert tool_names(by_name["researcher"]) == {
            "web_search", "web_fetch", "datetime_tool",
        }
        assert tool_names(by_name["reader"]) == {"file_read", "web_fetch"}
        assert tool_names(by_name["coder"]) == {
            "python_repl", "file_read", "file_write",
        }

    def test_researcher_has_deep_research_skill(self):
        """researcher 配套 deep-research 内置技能"""
        researcher = builtin_subagent_configs(include=("researcher",))[0]
        assert researcher.skills is not None
        assert any(s.name == "deep-research" for s in researcher.skills)

    def test_create_tools_from_presets(self):
        """预设可直接生成子代理工具（role → prompt_dir 解析正常）"""
        llm = AsyncMock()
        tools = create_subagent_tools(llm, builtin_subagent_configs())
        assert [t._tool_wrapper.name for t in tools] == [
            "researcher", "reader", "coder",
        ]
        assert all(t._tool_wrapper._is_subagent for t in tools)


# ── on_confirm 透传（决策 1-C）──────────────────────────────


def _make_seq_llm(first_tool_name: str, first_tool_args: dict):
    """构造两轮 mock LLM：第一轮发起指定工具调用，第二轮收尾。"""
    call_count = 0

    async def chat(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': f'call_{first_tool_name}',
                    'function': type('obj', (), {
                        'name': first_tool_name,
                        'arguments': json.dumps(first_tool_args),
                    })()
                })()
            ])
            yield StreamChunk(finish_reason="tool_calls")
        else:
            yield StreamChunk(content="done", finish_reason="stop")
            yield StreamChunk(usage=TokenUsage(1, 1, 2))

    llm = AsyncMock()
    llm.chat = chat
    return llm


class TestSubAgentConfirmPassthrough:
    """MANUAL 模式下，子代理内的不安全工具应走父 Agent 的 on_confirm 回调。"""

    def _build(self, approved: bool):
        """构造 父Agent(MANUAL+on_confirm) → 子代理 → 不安全工具 的链路。"""
        executed = []
        confirm_calls = []

        @tool(name="danger_write", description="不安全写入", is_safe=False)
        async def danger_write() -> str:
            executed.append(True)
            return "written"

        async def parent_confirm(tool_name: str, args: str):
            confirm_calls.append(tool_name)
            return ConfirmResponse(approved=approved, message="" if approved else "不允许")

        sub_llm = _make_seq_llm("danger_write", {})
        sub_tools = create_subagent_tools(
            llm=sub_llm,
            subagents=[SubAgentConfig(
                name="worker", description="测试子代理", tools=[danger_write],
            )],
        )

        parent_llm = _make_seq_llm("worker", {"task": "执行写入"})
        parent = Agent(
            llm=parent_llm,
            tools=sub_tools,
            mode=AgentMode.MANUAL,
            on_confirm=parent_confirm,
            session_enabled=False,
            register_catalog=False, register_skills=False,
        )
        return parent, executed, confirm_calls

    @pytest.mark.asyncio
    async def test_approved_executes(self):
        """父回调批准 → 子代理的不安全工具执行"""
        parent, executed, confirm_calls = self._build(approved=True)
        async for _ in parent.run("委派任务"):
            pass
        assert confirm_calls == ["danger_write"], "确认回调应被子代理的不安全工具触发"
        assert executed, "批准后工具应已执行"

    @pytest.mark.asyncio
    async def test_rejected_blocks(self):
        """父回调拒绝 → 子代理的不安全工具被阻止（委派不构成安全旁路）"""
        parent, executed, confirm_calls = self._build(approved=False)
        async for _ in parent.run("委派任务"):
            pass
        assert confirm_calls == ["danger_write"]
        assert not executed, "拒绝后工具不应执行"

    @pytest.mark.asyncio
    async def test_no_parent_confirm_keeps_old_behavior(self):
        """父没有 on_confirm 时维持原行为：AUTO 模式自主执行（不安全工具免审批）"""
        executed = []

        @tool(name="danger_write2", description="不安全写入", is_safe=False)
        async def danger_write() -> str:
            executed.append(True)
            return "written"

        sub_llm = _make_seq_llm("danger_write2", {})
        sub_tools = create_subagent_tools(
            llm=sub_llm,
            subagents=[SubAgentConfig(
                name="worker2", description="测试子代理", tools=[danger_write],
            )],
        )
        parent = Agent(
            llm=_make_seq_llm("worker2", {"task": "执行"}),
            tools=sub_tools,
            mode=AgentMode.AUTO,
            session_enabled=False,
            register_catalog=False, register_skills=False,
        )
        async for _ in parent.run("委派任务"):
            pass
        assert executed


# ── Agent 级全配默认（方案 A：默认策略下沉进 Agent）──────────


class TestAgentLevelDefaults:
    """裸 Agent(llm) 即全配：BUILTIN_TOOLS + 内置子代理三件套；显式传参可覆盖。"""

    def test_bare_agent_fully_equipped(self):
        """tools/subagents 均不传 → 全套内置工具 + 三件套子代理"""
        agent = Agent(llm=AsyncMock(), session_enabled=False)
        tools = agent.tools.list_tools()
        for name in ("file_read", "python_repl", "web_search", "todo_write"):
            assert name in tools, f"默认应含内置工具 {name}"
        for name in ("researcher", "reader", "coder"):
            assert name in tools, f"默认应含子代理 {name}"

    def test_explicit_empty_tools_keeps_subagents(self):
        """tools=[] → 无内置工具；subagents 独立解析，仍为默认三件套"""
        agent = Agent(llm=AsyncMock(), tools=[], session_enabled=False)
        tools = agent.tools.list_tools()
        assert "file_read" not in tools
        for name in ("researcher", "reader", "coder"):
            assert name in tools

    def test_explicit_empty_subagents(self):
        """subagents=[] → 无子代理；工具仍为默认全套"""
        agent = Agent(llm=AsyncMock(), subagents=[], session_enabled=False)
        tools = agent.tools.list_tools()
        assert "file_read" in tools
        for name in ("researcher", "reader", "coder"):
            assert name not in tools

    def test_custom_subagents(self):
        """subagents=[...] → 自定义集合"""
        agent = Agent(
            llm=AsyncMock(),
            subagents=builtin_subagent_configs(include=("reader",)),
            session_enabled=False,
        )
        tools = agent.tools.list_tools()
        assert "reader" in tools
        assert "researcher" not in tools and "coder" not in tools

    def test_subagent_stays_lean(self):
        """register_catalog=False（子代理）→ 不注入内置工具与子代理（结构性不嵌套）"""
        agent = Agent(
            llm=AsyncMock(),
            session_enabled=False,
            register_catalog=False, register_skills=False,
        )
        assert agent.tools.list_tools() == []


# ── AgentPool 默认子代理 ──────────────────────────────────


def _make_quick_llm():
    async def chat(messages, **kwargs):
        yield StreamChunk(content="ok", finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(10, 5, 15))
    llm = AsyncMock()
    llm.chat = chat
    return llm


class TestPoolDefaultSubagents:
    @pytest.mark.asyncio
    async def test_default_factory_registers_trio(self):
        """默认工厂自动注册三件套子代理"""
        llm = _make_quick_llm()
        pool = AgentPool.from_llm(llm, agent_kwargs={"session_enabled": False})
        await pool.start()
        try:
            async with pool.acquire("u1", "s1") as h:
                tools = h.agent.tools.list_tools()
                for name in ("researcher", "reader", "coder"):
                    assert name in tools, f"默认工厂应注册子代理 {name}"
        finally:
            await pool.stop()

    @pytest.mark.asyncio
    async def test_subagents_disabled_via_agent_kwargs(self):
        """agent_kwargs={"subagents": []} 可关闭默认子代理"""
        llm = _make_quick_llm()
        pool = AgentPool.from_llm(
            llm, agent_kwargs={"session_enabled": False, "subagents": []}
        )
        await pool.start()
        try:
            async with pool.acquire("u1", "s1") as h:
                tools = h.agent.tools.list_tools()
                for name in ("researcher", "reader", "coder"):
                    assert name not in tools
        finally:
            await pool.stop()

    @pytest.mark.asyncio
    async def test_custom_subagents_via_agent_kwargs(self):
        """agent_kwargs={"subagents": [...]} 可自定义子代理集合"""
        llm = _make_quick_llm()
        pool = AgentPool.from_llm(
            llm,
            agent_kwargs={
                "session_enabled": False,
                "subagents": builtin_subagent_configs(include=("reader",)),
            },
        )
        await pool.start()
        try:
            async with pool.acquire("u1", "s1") as h:
                tools = h.agent.tools.list_tools()
                assert "reader" in tools
                assert "researcher" not in tools
                assert "coder" not in tools
        finally:
            await pool.stop()
