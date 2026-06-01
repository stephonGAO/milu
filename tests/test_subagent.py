"""测试子代理（SubAgent）功能"""
import asyncio
import pytest
from unittest.mock import AsyncMock

from agent_framework.agent import Agent, AgentConfig
from agent_framework.agent.events import (
    AgentDone, AgentError, TextDelta, ToolCallStart, ToolResult,
    SubAgentEvent, SubAgentDone,
)
from agent_framework.agent.subagent import SubAgentConfig, create_subagent_tools
from agent_framework.llm.base.response import StreamChunk, TokenUsage
from agent_framework.tools import tool


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def mock_llm():
    """简单的 mock LLM：直接返回文本"""
    async def mock_chat(*args, **kwargs):
        yield StreamChunk(content="LLM 回复", finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    llm = AsyncMock()
    llm.chat = mock_chat
    return llm


@pytest.fixture
def simple_tool():
    @tool(name="add", description="加法")
    async def add(a: int, b: int) -> str:
        return str(a + b)
    return add


# ── SubAgentConfig 测试 ──────────────────────────────────


class TestSubAgentConfig:
    """SubAgentConfig 数据类测试"""

    def test_required_fields(self):
        """必填字段应正确设置"""
        cfg = SubAgentConfig(
            name="researcher",
            description="调研助手",
            system_prompt="你是研究员",
        )
        assert cfg.name == "researcher"
        assert cfg.description == "调研助手"
        assert cfg.system_prompt == "你是研究员"

    def test_defaults(self):
        """可选字段应有合理默认值"""
        cfg = SubAgentConfig(name="t", description="d", system_prompt="s")
        assert cfg.tools == []
        assert cfg.config is None
        assert cfg.history_max_turns == 50
        assert cfg.history_max_tokens is None

    def test_custom_config(self):
        """应能自定义 AgentConfig"""
        custom = AgentConfig(max_turns=5, timeout=30.0)
        cfg = SubAgentConfig(
            name="t", description="d", system_prompt="s",
            config=custom,
        )
        assert cfg.config is custom
        assert cfg.config.max_turns == 5

    def test_custom_tools(self, simple_tool):
        """应能指定工具列表"""
        cfg = SubAgentConfig(
            name="t", description="d", system_prompt="s",
            tools=[simple_tool],
        )
        assert len(cfg.tools) == 1

    def test_custom_history(self):
        """应能自定义历史参数"""
        cfg = SubAgentConfig(
            name="t", description="d", system_prompt="s",
            history_max_turns=5,
            history_max_tokens=1000,
        )
        assert cfg.history_max_turns == 5
        assert cfg.history_max_tokens == 1000


# ── 工厂函数测试 ──────────────────────────────────────────


class TestCreateSubagentTools:
    """create_subagent_tools 工厂函数测试"""

    def test_returns_list(self, mock_llm):
        """应返回工具列表"""
        tools = create_subagent_tools(mock_llm, [
            SubAgentConfig(name="a", description="A", system_prompt="sa"),
            SubAgentConfig(name="b", description="B", system_prompt="sb"),
        ])
        assert len(tools) == 2

    def test_empty_list(self, mock_llm):
        """空配置应返回空列表"""
        tools = create_subagent_tools(mock_llm, [])
        assert tools == []

    def test_tool_wrapper_metadata(self, mock_llm):
        """工具应有正确的元数据"""
        tools = create_subagent_tools(mock_llm, [
            SubAgentConfig(name="researcher", description="调研助手", system_prompt="s"),
        ])
        w = tools[0]._tool_wrapper
        assert w.name == "researcher"
        assert w.description == "调研助手"
        assert w._is_subagent is True
        assert hasattr(w, '_subagent_events')
        assert isinstance(w._subagent_events, list)

    def test_tool_schema(self, mock_llm):
        """工具 schema 应有 task 参数"""
        tools = create_subagent_tools(mock_llm, [
            SubAgentConfig(name="t", description="d", system_prompt="s"),
        ])
        schema = tools[0]._tool_wrapper.parameters_schema
        assert schema["type"] == "object"
        assert "task" in schema["properties"]
        assert schema["properties"]["task"]["type"] == "string"
        assert "task" in schema["required"]

    def test_multiple_tools_independent(self, mock_llm):
        """多个工具应有独立的事件存储"""
        tools = create_subagent_tools(mock_llm, [
            SubAgentConfig(name="a", description="A", system_prompt="sa"),
            SubAgentConfig(name="b", description="B", system_prompt="sb"),
        ])
        events_a = tools[0]._tool_wrapper._subagent_events
        events_b = tools[1]._tool_wrapper._subagent_events
        assert events_a is not events_b


# ── 子代理执行测试 ────────────────────────────────────────


class TestSubagentExecution:
    """子代理执行行为测试"""

    @pytest.mark.asyncio
    async def test_simple_text_response(self, mock_llm):
        """子代理返回纯文本时，结果应有 [name]: 前缀"""
        tools = create_subagent_tools(mock_llm, [
            SubAgentConfig(name="helper", description="助手", system_prompt="你是助手"),
        ])

        result = await tools[0]._tool_wrapper.func(task="你好")
        assert result.startswith("[helper]:")
        assert "LLM 回复" in result

    @pytest.mark.asyncio
    async def test_events_collected(self, mock_llm):
        """子代理执行后应收集事件"""
        tools = create_subagent_tools(mock_llm, [
            SubAgentConfig(name="helper", description="助手", system_prompt="你是助手"),
        ])

        await tools[0]._tool_wrapper.func(task="你好")
        events = tools[0]._tool_wrapper._subagent_events
        assert len(events) > 0
        # 最后一个事件应是 AgentDone
        assert isinstance(events[-1], AgentDone)

    @pytest.mark.asyncio
    async def test_events_cleared_between_calls(self, mock_llm):
        """每次调用应清空事件列表"""
        tools = create_subagent_tools(mock_llm, [
            SubAgentConfig(name="helper", description="助手", system_prompt="你是助手"),
        ])

        await tools[0]._tool_wrapper.func(task="第一次")
        first_count = len(tools[0]._tool_wrapper._subagent_events)

        await tools[0]._tool_wrapper.func(task="第二次")
        second_count = len(tools[0]._tool_wrapper._subagent_events)

        # 两次调用的事件数量应相同（都只是一次简单回复）
        assert first_count == second_count

    @pytest.mark.asyncio
    async def test_subagent_with_tool(self, mock_llm, simple_tool):
        """子代理可以使用工具"""
        call_count = 0

        async def tool_using_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # 第一次：调用工具
                yield StreamChunk(tool_calls=[
                    type('obj', (), {
                        'index': 0, 'id': 'call_1',
                        'function': type('obj', (), {
                            'name': 'add',
                            'arguments': '{"a": 1, "b": 2}'
                        })()
                    })()
                ])
                yield StreamChunk(finish_reason="tool_calls")
            else:
                # 第二次：给出最终回复
                yield StreamChunk(content="1+2=3", finish_reason="stop")
                yield StreamChunk(usage=TokenUsage(
                    prompt_tokens=20, completion_tokens=5, total_tokens=25
                ))

        llm = AsyncMock()
        llm.chat = tool_using_chat

        tools = create_subagent_tools(llm, [
            SubAgentConfig(
                name="calculator",
                description="计算器",
                system_prompt="你是计算器",
                tools=[simple_tool],
            ),
        ])

        result = await tools[0]._tool_wrapper.func(task="计算 1+2")
        assert "[calculator]:" in result
        assert "1+2=3" in result

    @pytest.mark.asyncio
    async def test_subagent_error(self):
        """子代理 LLM 超时时，应返回错误信息"""
        import asyncio

        async def timeout_chat(*args, **kwargs):
            await asyncio.sleep(10)  # 远超超时时间
            yield StreamChunk(content="不会到这", finish_reason="stop")

        llm = AsyncMock()
        llm.chat = timeout_chat

        sub_config = AgentConfig(
            max_turns=5,
            timeout=0.1,  # 100ms 超时
            total_timeout=0.5,
        )
        tools = create_subagent_tools(llm, [
            SubAgentConfig(
                name="slow",
                description="慢代理",
                system_prompt="你是慢代理",
                config=sub_config,
            ),
        ])

        result = await tools[0]._tool_wrapper.func(task="测试超时")
        assert "[slow]" in result
        assert "错误" in result or "失败" in result

    @pytest.mark.asyncio
    async def test_subagent_crash(self):
        """子代理 LLM 抛异常时，应优雅返回错误"""
        async def crash_chat(*args, **kwargs):
            raise ConnectionError("网络连接失败")
            yield  # 使成为 async generator

        llm = AsyncMock()
        llm.chat = crash_chat

        tools = create_subagent_tools(llm, [
            SubAgentConfig(name="crasher", description="崩溃", system_prompt="s"),
        ])

        result = await tools[0]._tool_wrapper.func(task="测试崩溃")
        assert "[crasher]" in result
        # 应当是错误信息而不是抛异常
        assert "失败" in result or "错误" in result

    @pytest.mark.asyncio
    async def test_no_result_edge_case(self):
        """极端情况：LLM 不产生任何事件"""
        async def empty_chat(*args, **kwargs):
            return
            yield  # 使成为 async generator

        llm = AsyncMock()
        llm.chat = empty_chat

        tools = create_subagent_tools(llm, [
            SubAgentConfig(name="empty", description="空", system_prompt="s"),
        ])

        result = await tools[0]._tool_wrapper.func(task="测试空")
        # 应返回某种结果（不是崩溃）
        assert "[empty]" in result


# ── 隔离性测试 ────────────────────────────────────────────


class TestIsolation:
    """子代理隔离性测试"""

    @pytest.mark.asyncio
    async def test_fresh_history_per_call(self, mock_llm):
        """每次调用子代理应有全新的对话历史"""
        call_count = 0

        async def counting_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # 检查消息数量：应只有 system + user（2条）
            messages = args[0] if args else kwargs.get('messages', [])
            non_system = [m for m in messages if m.role.value != 'system']
            assert len(non_system) == 1, f"期望1条非系统消息，实际 {len(non_system)}"
            yield StreamChunk(content=f"回复{call_count}", finish_reason="stop")
            yield StreamChunk(usage=TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7))

        llm = AsyncMock()
        llm.chat = counting_chat

        tools = create_subagent_tools(llm, [
            SubAgentConfig(name="isolated", description="隔离", system_prompt="s"),
        ])

        # 调用两次
        await tools[0]._tool_wrapper.func(task="第一次")
        await tools[0]._tool_wrapper.func(task="第二次")

        assert call_count == 2  # 两次调用都成功执行

    @pytest.mark.asyncio
    async def test_no_recursive_subagents(self, mock_llm):
        """子代理不应有子代理工具（结构性保证）"""
        tools = create_subagent_tools(mock_llm, [
            SubAgentConfig(name="leaf", description="叶节点", system_prompt="s"),
        ])

        # 手动调用子代理，检查其 Agent 的 registry
        # 通过捕获 Agent 实例来验证
        created_agents = []
        original_agent_init = Agent.__init__

        def tracking_init(self, *args, **kwargs):
            created_agents.append(self)
            original_agent_init(self, *args, **kwargs)

        Agent.__init__ = tracking_init
        try:
            await tools[0]._tool_wrapper.func(task="测试")
        finally:
            Agent.__init__ = original_agent_init

        # 应创建了一个子 Agent
        assert len(created_agents) == 1
        sub_agent = created_agents[0]

        # 子 Agent 的工具不应包含任何子代理工具
        sub_tool_names = sub_agent.tools.list_tools()
        assert "leaf" not in sub_tool_names
        # 不应有 _is_subagent 标记的工具
        for name in sub_tool_names:
            w = sub_agent.tools.get_tool(name)
            assert not getattr(w, '_is_subagent', False), \
                f"子代理不应包含子代理工具: {name}"


# ── 父 Agent 集成测试 ────────────────────────────────────


class TestParentAgentIntegration:
    """父 Agent 与子代理集成测试"""

    @pytest.mark.asyncio
    async def test_parent_calls_subagent(self, mock_llm):
        """父 Agent 应能调用子代理并获取结果"""
        parent_call_count = 0

        async def parent_chat(*args, **kwargs):
            nonlocal parent_call_count
            parent_call_count += 1

            if parent_call_count == 1:
                # 第一次：调用子代理
                yield StreamChunk(tool_calls=[
                    type('obj', (), {
                        'index': 0, 'id': 'call_sub',
                        'function': type('obj', (), {
                            'name': 'helper',
                            'arguments': '{"task": "帮忙查资料"}'
                        })()
                    })()
                ])
                yield StreamChunk(finish_reason="tool_calls")
            else:
                # 第二次：基于子代理结果回复
                yield StreamChunk(content="根据调查结果，答案是42", finish_reason="stop")
                yield StreamChunk(usage=TokenUsage(
                    prompt_tokens=50, completion_tokens=10, total_tokens=60
                ))

        parent_llm = AsyncMock()
        parent_llm.chat = parent_chat

        # 创建子代理工具
        sub_tools = create_subagent_tools(mock_llm, [
            SubAgentConfig(name="helper", description="助手", system_prompt="你是助手"),
        ])

        agent = Agent(
            llm=parent_llm,
            system_prompt="你是总指挥",
            tools=sub_tools,
        )

        events = []
        async for event in agent.run("帮我查资料"):
            events.append(event)

        # 应有 ToolCallStart（调用子代理）
        starts = [e for e in events if isinstance(e, ToolCallStart)]
        assert any(s.tool_name == "helper" for s in starts)

        # 应有 SubAgentEvent（子代理内部事件）
        sub_events = [e for e in events if isinstance(e, SubAgentEvent)]
        assert len(sub_events) > 0

        # 应有 SubAgentDone
        sub_dones = [e for e in events if isinstance(e, SubAgentDone)]
        assert len(sub_dones) == 1
        assert sub_dones[0].subagent_name == "helper"
        assert sub_dones[0].is_error is False
        assert sub_dones[0].turn_count > 0

        # 应有 ToolResult
        results = [e for e in events if isinstance(e, ToolResult)]
        assert any(r.tool_name == "helper" for r in results)
        helper_result = next(r for r in results if r.tool_name == "helper")
        assert "[helper]" in helper_result.output
        assert helper_result.is_error is False

        # 应有最终文本
        done = next(e for e in events if isinstance(e, AgentDone))
        assert "42" in done.final_text

    @pytest.mark.asyncio
    async def test_subagent_events_contain_inner_events(self, mock_llm, simple_tool):
        """SubAgentEvent 应包含子代理的内部事件"""
        call_count = 0

        async def sub_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield StreamChunk(tool_calls=[
                    type('obj', (), {
                        'index': 0, 'id': 'call_add',
                        'function': type('obj', (), {
                            'name': 'add',
                            'arguments': '{"a": 3, "b": 4}'
                        })()
                    })()
                ])
                yield StreamChunk(finish_reason="tool_calls")
            else:
                yield StreamChunk(content="3+4=7", finish_reason="stop")
                yield StreamChunk(usage=TokenUsage(
                    prompt_tokens=20, completion_tokens=5, total_tokens=25
                ))

        sub_llm = AsyncMock()
        sub_llm.chat = sub_chat

        parent_call_count = 0

        async def parent_chat(*args, **kwargs):
            nonlocal parent_call_count
            parent_call_count += 1
            if parent_call_count == 1:
                yield StreamChunk(tool_calls=[
                    type('obj', (), {
                        'index': 0, 'id': 'call_sub',
                        'function': type('obj', (), {
                            'name': 'calculator',
                            'arguments': '{"task": "算3+4"}'
                        })()
                    })()
                ])
                yield StreamChunk(finish_reason="tool_calls")
            else:
                yield StreamChunk(content="结果是7", finish_reason="stop")
                yield StreamChunk(usage=TokenUsage(
                    prompt_tokens=40, completion_tokens=5, total_tokens=45
                ))

        parent_llm = AsyncMock()
        parent_llm.chat = parent_chat

        sub_tools = create_subagent_tools(sub_llm, [
            SubAgentConfig(
                name="calculator",
                description="计算器",
                system_prompt="你是计算器",
                tools=[simple_tool],
            ),
        ])

        agent = Agent(
            llm=parent_llm,
            system_prompt="你是总指挥",
            tools=sub_tools,
        )

        events = []
        async for event in agent.run("计算3+4"):
            events.append(event)

        # 检查 SubAgentEvent 包含 ToolCallStart 和 ToolResult
        sub_events = [e for e in events if isinstance(e, SubAgentEvent)]
        inner_event_types = [type(e.event) for e in sub_events]
        assert ToolCallStart in inner_event_types
        assert ToolResult in inner_event_types
        assert TextDelta in inner_event_types

    @pytest.mark.asyncio
    async def test_parent_history_isolation(self, mock_llm):
        """父 Agent 历史不应包含子代理的内部消息"""
        parent_call_count = 0

        async def parent_chat(*args, **kwargs):
            nonlocal parent_call_count
            parent_call_count += 1
            if parent_call_count == 1:
                yield StreamChunk(tool_calls=[
                    type('obj', (), {
                        'index': 0, 'id': 'call_sub',
                        'function': type('obj', (), {
                            'name': 'helper',
                            'arguments': '{"task": "查资料"}'
                        })()
                    })()
                ])
                yield StreamChunk(finish_reason="tool_calls")
            else:
                yield StreamChunk(content="完成", finish_reason="stop")
                yield StreamChunk(usage=TokenUsage(
                    prompt_tokens=30, completion_tokens=5, total_tokens=35
                ))

        parent_llm = AsyncMock()
        parent_llm.chat = parent_chat

        sub_tools = create_subagent_tools(mock_llm, [
            SubAgentConfig(name="helper", description="助手", system_prompt="你是助手"),
        ])

        agent = Agent(
            llm=parent_llm,
            system_prompt="你是总指挥",
            tools=sub_tools,
        )

        async for event in agent.run("帮我查资料"):
            pass

        # 父 Agent 历史应只有：system + user + assistant(tool_call) + tool_result + assistant(完成)
        messages = agent.history.all_messages
        roles = [m.role.value for m in messages]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]

    @pytest.mark.asyncio
    async def test_multiple_subagents(self, mock_llm):
        """多个子代理应独立工作"""
        parent_call_count = 0

        async def parent_chat(*args, **kwargs):
            nonlocal parent_call_count
            parent_call_count += 1
            if parent_call_count == 1:
                # 调用两个子代理
                yield StreamChunk(tool_calls=[
                    type('obj', (), {
                        'index': 0, 'id': 'call_1',
                        'function': type('obj', (), {
                            'name': 'agent_a',
                            'arguments': '{"task": "任务A"}'
                        })()
                    })(),
                    type('obj', (), {
                        'index': 1, 'id': 'call_2',
                        'function': type('obj', (), {
                            'name': 'agent_b',
                            'arguments': '{"task": "任务B"}'
                        })()
                    })(),
                ])
                yield StreamChunk(finish_reason="tool_calls")
            else:
                yield StreamChunk(content="综合结果", finish_reason="stop")
                yield StreamChunk(usage=TokenUsage(
                    prompt_tokens=60, completion_tokens=5, total_tokens=65
                ))

        parent_llm = AsyncMock()
        parent_llm.chat = parent_chat

        sub_tools = create_subagent_tools(mock_llm, [
            SubAgentConfig(name="agent_a", description="A", system_prompt="A"),
            SubAgentConfig(name="agent_b", description="B", system_prompt="B"),
        ])

        agent = Agent(
            llm=parent_llm,
            system_prompt="总指挥",
            tools=sub_tools,
        )

        events = []
        async for event in agent.run("执行两个任务"):
            events.append(event)

        # 应有两个 SubAgentDone
        sub_dones = [e for e in events if isinstance(e, SubAgentDone)]
        assert len(sub_dones) == 2
        names = {d.subagent_name for d in sub_dones}
        assert names == {"agent_a", "agent_b"}

        # 应有两个 ToolResult
        results = [e for e in events if isinstance(e, ToolResult)]
        assert len(results) == 2
        assert any("[agent_a]" in r.output for r in results)
        assert any("[agent_b]" in r.output for r in results)

    @pytest.mark.asyncio
    async def test_subagent_error_propagation(self):
        """子代理错误应传播到父 Agent"""
        async def crash_chat(*args, **kwargs):
            raise RuntimeError("内部错误")
            yield

        crash_llm = AsyncMock()
        crash_llm.chat = crash_chat

        parent_call_count = 0

        async def parent_chat(*args, **kwargs):
            nonlocal parent_call_count
            parent_call_count += 1
            if parent_call_count == 1:
                yield StreamChunk(tool_calls=[
                    type('obj', (), {
                        'index': 0, 'id': 'call_sub',
                        'function': type('obj', (), {
                            'name': 'faulty',
                            'arguments': '{"task": "危险任务"}'
                        })()
                    })()
                ])
                yield StreamChunk(finish_reason="tool_calls")
            else:
                yield StreamChunk(content="子代理出错了，我换种方式", finish_reason="stop")
                yield StreamChunk(usage=TokenUsage(
                    prompt_tokens=40, completion_tokens=10, total_tokens=50
                ))

        parent_llm = AsyncMock()
        parent_llm.chat = parent_chat

        sub_tools = create_subagent_tools(crash_llm, [
            SubAgentConfig(name="faulty", description="不稳定", system_prompt="s"),
        ])

        agent = Agent(
            llm=parent_llm,
            system_prompt="总指挥",
            tools=sub_tools,
        )

        events = []
        async for event in agent.run("执行危险任务"):
            events.append(event)

        # 父 Agent 不应崩溃
        assert any(isinstance(e, AgentDone) for e in events)

        # ToolResult 应包含错误信息
        results = [e for e in events if isinstance(e, ToolResult)]
        faulty_result = next(r for r in results if r.tool_name == "faulty")
        assert "faulty" in faulty_result.output

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """多个子代理应并发执行（总时间接近单个子代理时间，而非累加）"""
        import time

        async def slow_chat(*args, **kwargs):
            # 每个子代理模拟 0.3s 延迟
            await asyncio.sleep(0.3)
            yield StreamChunk(content="完成", finish_reason="stop")
            yield StreamChunk(usage=TokenUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ))

        sub_llm = AsyncMock()
        sub_llm.chat = slow_chat

        parent_call_count = 0

        async def parent_chat(*args, **kwargs):
            nonlocal parent_call_count
            parent_call_count += 1
            if parent_call_count == 1:
                # 一次调用 3 个子代理
                yield StreamChunk(tool_calls=[
                    type('obj', (), {
                        'index': 0, 'id': 'call_1',
                        'function': type('obj', (), {
                            'name': 'agent_a',
                            'arguments': '{"task": "任务A"}'
                        })()
                    })(),
                    type('obj', (), {
                        'index': 1, 'id': 'call_2',
                        'function': type('obj', (), {
                            'name': 'agent_b',
                            'arguments': '{"task": "任务B"}'
                        })()
                    })(),
                    type('obj', (), {
                        'index': 2, 'id': 'call_3',
                        'function': type('obj', (), {
                            'name': 'agent_c',
                            'arguments': '{"task": "任务C"}'
                        })()
                    })(),
                ])
                yield StreamChunk(finish_reason="tool_calls")
            else:
                yield StreamChunk(content="综合结果", finish_reason="stop")
                yield StreamChunk(usage=TokenUsage(
                    prompt_tokens=60, completion_tokens=5, total_tokens=65
                ))

        parent_llm = AsyncMock()
        parent_llm.chat = parent_chat

        sub_tools = create_subagent_tools(sub_llm, [
            SubAgentConfig(name="agent_a", description="A", system_prompt="A"),
            SubAgentConfig(name="agent_b", description="B", system_prompt="B"),
            SubAgentConfig(name="agent_c", description="C", system_prompt="C"),
        ])

        agent = Agent(
            llm=parent_llm,
            system_prompt="总指挥",
            tools=sub_tools,
        )

        start = time.monotonic()
        events = []
        async for event in agent.run("执行三个任务"):
            events.append(event)
        elapsed = time.monotonic() - start

        # 3 个子代理各 0.3s，顺序执行需要 ~0.9s+，并发执行应 < 0.7s
        assert elapsed < 0.7, f"并发执行时间 {elapsed:.2f}s 过长，可能未并发"

        # 3 个子代理都应完成
        sub_dones = [e for e in events if isinstance(e, SubAgentDone)]
        assert len(sub_dones) == 3
        names = {d.subagent_name for d in sub_dones}
        assert names == {"agent_a", "agent_b", "agent_c"}
