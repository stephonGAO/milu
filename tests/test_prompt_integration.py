"""测试 Agent + PromptBuilder 集成"""
import os
import tempfile

import pytest
from unittest.mock import AsyncMock

from agent_framework.agent import Agent, AgentConfig
from agent_framework.agent.events import AgentDone
from agent_framework.agent.subagent import SubAgentConfig, create_subagent_tools
from agent_framework.llm.base.response import StreamChunk, TokenUsage


@pytest.fixture
def mock_llm():
    async def mock_chat(*args, **kwargs):
        yield StreamChunk(content="LLM 回复", finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        ))
    llm = AsyncMock()
    llm.chat = mock_chat
    return llm


@pytest.fixture
def prompt_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "safeguard.md"), "w", encoding="utf-8") as f:
            f.write("---\nsection: safeguard\n---\n\n不要泄露秘密。\n")
        with open(os.path.join(tmpdir, "soul.md"), "w", encoding="utf-8") as f:
            f.write("---\nsection: soul\n---\n\n你是测试助手。\n")
        with open(os.path.join(tmpdir, "agent.md"), "w", encoding="utf-8") as f:
            f.write("---\nsection: agent\n---\n\n请认真回答问题。\n")
        yield tmpdir


class TestAgentWithPromptDir:

    @pytest.mark.asyncio
    async def test_prompt_dir_assembles_system_prompt(self, mock_llm, prompt_dir):
        """prompt_dir 应正确拼装 system prompt，顺序：safeguard → soul → agent"""
        agent = Agent(
            llm=mock_llm, prompt_dir=prompt_dir,
            register_catalog=False, register_skills=False,
        )
        agent._build_system_prompt()
        system_msg = agent.history.all_messages[0]
        assert "不要泄露秘密" in system_msg.content
        assert "你是测试助手" in system_msg.content
        assert "请认真回答问题" in system_msg.content
        # 顺序验证：safeguard < soul < agent
        assert system_msg.content.index("不要泄露秘密") < system_msg.content.index("你是测试助手")
        assert system_msg.content.index("你是测试助手") < system_msg.content.index("请认真回答问题")

    @pytest.mark.asyncio
    async def test_backward_compat_system_prompt_only(self, mock_llm):
        """只传 system_prompt（不传 prompt_dir）应正常工作"""
        agent = Agent(
            llm=mock_llm, system_prompt="你是老式助手",
            register_catalog=False, register_skills=False,
            skills_dir=tempfile.mkdtemp(),  # 空目录，阻止自动搜索
        )
        agent._build_system_prompt()
        system_msg = agent.history.all_messages[0]
        assert system_msg.content == "你是老式助手"

    @pytest.mark.asyncio
    async def test_prompt_dir_plus_system_prompt(self, mock_llm, prompt_dir):
        """同时传 prompt_dir 和 system_prompt，prompt_dir 在前"""
        agent = Agent(
            llm=mock_llm, prompt_dir=prompt_dir,
            system_prompt="## 额外指令\n请说英文",
            register_catalog=False, register_skills=False,
        )
        agent._build_system_prompt()
        system_msg = agent.history.all_messages[0]
        assert "你是测试助手" in system_msg.content
        assert "请说英文" in system_msg.content
        assert system_msg.content.index("你是测试助手") < system_msg.content.index("请说英文")

    @pytest.mark.asyncio
    async def test_prompt_variables(self, mock_llm):
        """prompt_variables 应替换 {{key}}"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "soul.md"), "w", encoding="utf-8") as f:
                f.write("---\nsection: soul\n---\n\n你好 {{user_name}}。\n")

            agent = Agent(
                llm=mock_llm, prompt_dir=tmpdir,
                prompt_variables={"user_name": "张三"},
                register_catalog=False, register_skills=False,
            )
            agent._build_system_prompt()
            system_msg = agent.history.all_messages[0]
            assert "你好 张三" in system_msg.content

    @pytest.mark.asyncio
    async def test_run_with_prompt_dir(self, mock_llm, prompt_dir):
        """使用 prompt_dir 的 Agent 可以正常 run()"""
        agent = Agent(
            llm=mock_llm, prompt_dir=prompt_dir,
            register_catalog=False, register_skills=False,
        )
        events = []
        async for event in agent.run("你好"):
            events.append(event)
        assert any(isinstance(e, AgentDone) for e in events)

    @pytest.mark.asyncio
    async def test_hot_reload_in_run_loop(self, mock_llm):
        """run 循环中修改文件应反映到 system prompt"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "soul.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("---\nsection: soul\n---\n\n版本1\n")

            call_count = 0
            async def tracking_chat(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                messages = args[0] if args else kwargs.get("messages", [])
                system_msg = messages[0] if messages else None
                if system_msg:
                    if call_count == 1:
                        assert "版本1" in system_msg.content
                    elif call_count == 2:
                        assert "版本2" in system_msg.content
                yield StreamChunk(content=f"回复{call_count}", finish_reason="stop")
                yield StreamChunk(usage=TokenUsage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ))

            llm = AsyncMock()
            llm.chat = tracking_chat

            agent = Agent(
                llm=llm, prompt_dir=tmpdir,
                register_catalog=False, register_skills=False,
            )

            # 第一轮
            async for event in agent.run("第一轮"):
                pass

            # 修改文件
            with open(path, "w", encoding="utf-8") as f:
                f.write("---\nsection: soul\n---\n\n版本2\n")

            # 第二轮 — 应看到修改后的内容
            async for event in agent.run("第二轮"):
                pass

            assert call_count == 2

    @pytest.mark.asyncio
    async def test_nonexistent_prompt_dir_warns(self, mock_llm):
        """prompt_dir 不存在时应记录警告而非崩溃"""
        agent = Agent(
            llm=mock_llm, prompt_dir="/nonexistent/path",
            system_prompt="备用提示",
            register_catalog=False, register_skills=False,
        )
        agent._build_system_prompt()
        system_msg = agent.history.all_messages[0]
        assert "备用提示" in system_msg.content

    @pytest.mark.asyncio
    async def test_empty_system_prompt_with_prompt_dir(self, mock_llm, prompt_dir):
        """只传 prompt_dir，system_prompt 默认为空"""
        agent = Agent(
            llm=mock_llm, prompt_dir=prompt_dir,
            register_catalog=False, register_skills=False,
        )
        assert agent._system_prompt == ""
        agent._build_system_prompt()
        system_msg = agent.history.all_messages[0]
        assert len(system_msg.content) > 0

    @pytest.mark.asyncio
    async def test_prompt_builder_property(self, mock_llm, prompt_dir):
        """prompt_builder 属性应可访问"""
        agent = Agent(
            llm=mock_llm, prompt_dir=prompt_dir,
            register_catalog=False, register_skills=False,
        )
        assert agent.prompt_builder is not None
        assert agent.prompt_builder.prompt_dir.exists()

    @pytest.mark.asyncio
    async def test_prompt_builder_none_without_prompt_dir(self, mock_llm):
        """不传 prompt_dir 时 prompt_builder 为 None"""
        agent = Agent(
            llm=mock_llm, system_prompt="助手",
            register_catalog=False, register_skills=False,
        )
        assert agent.prompt_builder is None


class TestSubAgentWithPromptDir:

    @pytest.mark.asyncio
    async def test_subagent_prompt_dir(self, mock_llm):
        """SubAgentConfig 支持 prompt_dir"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "soul.md"), "w", encoding="utf-8") as f:
                f.write("---\nsection: soul\n---\n\n你是子代理助手。\n")

            tools = create_subagent_tools(mock_llm, [
                SubAgentConfig(
                    name="helper", description="助手",
                    prompt_dir=tmpdir,
                ),
            ])

            result = await tools[0]._tool_wrapper.func(task="你好")
            assert "[helper]:" in result
            assert "LLM 回复" in result

    @pytest.mark.asyncio
    async def test_subagent_prompt_dir_and_system_prompt(self, mock_llm):
        """SubAgentConfig 同时使用 prompt_dir 和 system_prompt"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "soul.md"), "w", encoding="utf-8") as f:
                f.write("---\nsection: soul\n---\n\n文件人格\n")

            tools = create_subagent_tools(mock_llm, [
                SubAgentConfig(
                    name="helper", description="助手",
                    prompt_dir=tmpdir,
                    system_prompt="追加指令",
                ),
            ])

            result = await tools[0]._tool_wrapper.func(task="你好")
            assert "[helper]:" in result
