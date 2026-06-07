"""测试 Agent 编排器"""
import pytest
from unittest.mock import AsyncMock
from milu.agent import Agent, AgentConfig
from milu.agent.events import (
    AgentDone,
    AgentError,
    SafetyCheckStart,
    TextDelta,
    ToolCallPreparing,
    ToolCallStart,
    ToolResult,
)
from milu.llm.base.message import MessageRole
from milu.llm.base.response import StreamChunk, TokenUsage
from milu.tools import tool


@pytest.fixture
def simple_tool():
    @tool(name="get_time", description="获取时间")
    async def get_time() -> str:
        return "2026-05-23 10:00"
    return get_time


@pytest.mark.asyncio
async def test_simple_text_response():
    """无工具调用时应直接返回文本"""
    async def mock_chat(*args, **kwargs):
        yield StreamChunk(content="你好", finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    llm = AsyncMock()
    llm.chat = mock_chat

    agent = Agent(llm=llm, system_prompt="你是助手")

    events = []
    async for event in agent.run("你好"):
        events.append(event)

    assert any(isinstance(e, TextDelta) for e in events)
    assert any(isinstance(e, AgentDone) for e in events)

    done = next(e for e in events if isinstance(e, AgentDone))
    assert done.final_text == "你好"
    assert done.turn_count == 1


@pytest.mark.asyncio
async def test_tool_call_flow(simple_tool):
    """应正确处理工具调用流程"""
    call_count = 0

    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': 'call_1',
                    'function': type('obj', (), {'name': 'get_time', 'arguments': '{}'})()
                })()
            ])
            yield StreamChunk(finish_reason="tool_calls")
        else:
            yield StreamChunk(content="当前时间是 2026-05-23 10:00", finish_reason="stop")
            yield StreamChunk(usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30))

    llm = AsyncMock()
    llm.chat = mock_chat

    agent = Agent(llm=llm, system_prompt="你是助手", tools=[simple_tool])

    events = []
    async for event in agent.run("现在几点了？"):
        events.append(event)

    assert any(isinstance(e, ToolCallStart) for e in events)
    assert any(isinstance(e, ToolResult) for e in events)
    assert any(isinstance(e, TextDelta) for e in events)
    assert any(isinstance(e, AgentDone) for e in events)

    tool_start = next(e for e in events if isinstance(e, ToolCallStart))
    assert tool_start.tool_name == "get_time"

    tool_result = next(e for e in events if isinstance(e, ToolResult))
    assert tool_result.output == "2026-05-23 10:00"
    assert tool_result.is_error is False


@pytest.mark.asyncio
async def test_tool_call_preparing_event(simple_tool):
    """参数流式生成期发出 ToolCallPreparing：每个调用仅一次、先于 ToolCallStart。

    回归：正文结束后参数流式生成期（长代码参数可达数十秒）原本零事件，
    前端无从显示活动状态（"聊天框静止"）。
    """
    call_count = 0

    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # 名字先到、参数分两片后到（模拟长参数流式生成）
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': 'call_1',
                    'function': type('obj', (), {'name': 'get_time', 'arguments': ''})()
                })()
            ])
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': '',
                    'function': type('obj', (), {'name': '', 'arguments': '{'})()
                })()
            ])
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': '',
                    'function': type('obj', (), {'name': '', 'arguments': '}'})()
                })()
            ])
            yield StreamChunk(finish_reason="tool_calls")
        else:
            yield StreamChunk(content="完成", finish_reason="stop")

    llm = AsyncMock()
    llm.chat = mock_chat
    agent = Agent(llm=llm, system_prompt="你是助手", tools=[simple_tool])

    events = []
    async for event in agent.run("现在几点？"):
        events.append(event)

    preparing = [e for e in events if isinstance(e, ToolCallPreparing)]
    assert len(preparing) == 1                 # 多个参数分片只发一次
    assert preparing[0].tool_name == "get_time"
    first_start = next(i for i, e in enumerate(events)
                       if isinstance(e, ToolCallStart))
    assert events.index(preparing[0]) < first_start
    assert any(isinstance(e, AgentDone) for e in events)


@pytest.mark.asyncio
async def test_safety_check_start_event():
    """auto 模式不安全工具送 AI 判定前发出 SafetyCheckStart。

    判定是阻塞式 LLM 调用、期间无其他事件——此事件让前端能显示
    「安全判定中」。判定返回非法 JSON 时 fail-open，工具仍执行。
    """
    @tool(name="rm_file", description="删除文件", is_safe=False)
    async def rm_file(path: str) -> str:
        return f"已删除 {path}"

    call_count = 0

    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:      # 主循环第 1 轮 → 调不安全工具
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': 'call_1',
                    'function': type('obj', (), {
                        'name': 'rm_file',
                        'arguments': '{"path": "a.txt"}'})()
                })()
            ])
            yield StreamChunk(finish_reason="tool_calls")
        elif call_count == 2:    # 安全判定器调用（非法 JSON → fail-open）
            yield StreamChunk(content="无法判定", finish_reason="stop")
        else:                    # 工具结果回传后的最终回答
            yield StreamChunk(content="完成", finish_reason="stop")

    llm = AsyncMock()
    llm.chat = mock_chat
    # 默认 auto 模式 + judge_llm=None → 判定器复用主 llm
    agent = Agent(llm=llm, system_prompt="你是助手", tools=[rm_file])

    events = []
    async for event in agent.run("删除 a.txt"):
        events.append(event)

    checks = [e for e in events if isinstance(e, SafetyCheckStart)]
    assert len(checks) == 1
    assert checks[0].tool_names == ("rm_file",)
    # 判定先于 ToolCallStart；fail-open 后工具仍执行成功
    first_start = next(i for i, e in enumerate(events)
                       if isinstance(e, ToolCallStart))
    assert events.index(checks[0]) < first_start
    result = next(e for e in events if isinstance(e, ToolResult))
    assert result.output == "已删除 a.txt"


@pytest.mark.asyncio
async def test_max_turns_limit(simple_tool):
    """超出最大轮次应返回 AgentError"""
    async def mock_chat(*args, **kwargs):
        yield StreamChunk(tool_calls=[
            type('obj', (), {
                'index': 0, 'id': 'call_x',
                'function': type('obj', (), {'name': 'get_time', 'arguments': '{}'})()
            })()
        ])

    llm = AsyncMock()
    llm.chat = mock_chat

    agent = Agent(llm=llm, system_prompt="你是助手", tools=[simple_tool], config=AgentConfig(max_turns=3))

    events = []
    async for event in agent.run("测试"):
        events.append(event)

    assert any(isinstance(e, AgentError) for e in events)
    error = next(e for e in events if isinstance(e, AgentError))
    assert error.error_type == "max_turns"


@pytest.mark.asyncio
async def test_conversation_history():
    """多轮对话应保留历史"""
    call_count = 0

    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        yield StreamChunk(content=f"回复{call_count}", finish_reason="stop")

    llm = AsyncMock()
    llm.chat = mock_chat

    agent = Agent(llm=llm, system_prompt="你是助手")

    async for event in agent.run("问题1"):
        pass
    async for event in agent.run("问题2"):
        pass
    async for event in agent.run("问题3"):
        pass

    messages = agent.history.all_messages
    assert messages[0].role == MessageRole.SYSTEM
    assert len(messages) == 7  # system + 3*(user + assistant)


@pytest.mark.asyncio
async def test_reset():
    """reset 应清空历史但保留 system"""
    async def mock_chat(*args, **kwargs):
        yield StreamChunk(content="回复", finish_reason="stop")

    llm = AsyncMock()
    llm.chat = mock_chat

    agent = Agent(llm=llm, system_prompt="你是助手")

    async for event in agent.run("你好"):
        pass

    assert len(agent.history.all_messages) == 3

    await agent.reset()
    assert len(agent.history.all_messages) == 1


@pytest.mark.asyncio
async def test_tool_call_limit(simple_tool):
    """工具调用次数超限应返回 AgentError"""
    async def mock_chat(*args, **kwargs):
        yield StreamChunk(tool_calls=[
            type('obj', (), {
                'index': 0, 'id': 'call_x',
                'function': type('obj', (), {'name': 'get_time', 'arguments': '{}'})()
            })()
        ])

    llm = AsyncMock()
    llm.chat = mock_chat

    agent = Agent(
        llm=llm,
        system_prompt="你是助手",
        tools=[simple_tool],
        config=AgentConfig(max_turns=100, tool_call_limit=2)
    )

    events = []
    async for event in agent.run("测试"):
        events.append(event)

    assert any(isinstance(e, AgentError) for e in events)
    error = next(e for e in events if isinstance(e, AgentError))
    assert error.error_type == "tool_limit"


@pytest.mark.asyncio
async def test_plan_tool_batch_conflict_rejected():
    """计划工具与其他工具同批调用时应被拒绝"""

    @tool(name="todo_write", description="计划工具")
    async def dummy_todo_write(items: list) -> str:
        return "计划已更新"

    @tool(name="researcher", description="调研助手")
    async def dummy_researcher(task: str) -> str:
        return "调研结果"

    async def mock_chat(*args, **kwargs):
        # 第一轮：返回冲突批次
        if not hasattr(mock_chat, "_called"):
            mock_chat._called = True
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': 'call_tw',
                    'function': type('obj', (), {
                        'name': 'todo_write',
                        'arguments': '{"items": [{"content": "任务1", "status": "pending"}]}'
                    })()
                })(),
                type('obj', (), {
                    'index': 1, 'id': 'call_rs',
                    'function': type('obj', (), {
                        'name': 'researcher',
                        'arguments': '{"task": "调研"}'
                    })()
                })()
            ])
            yield StreamChunk(finish_reason="tool_calls")
        else:
            # 第二轮：正常文本结束
            yield StreamChunk(content="已理解，我会分开调用", finish_reason="stop")
            yield StreamChunk(usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    llm = AsyncMock()
    llm.chat = mock_chat

    agent = Agent(
        llm=llm,
        system_prompt="你是助手",
        tools=[dummy_todo_write, dummy_researcher],
        config=AgentConfig(max_turns=3),
    )

    events = []
    async for event in agent.run("帮我规划并调研"):
        events.append(event)

    # 应有 2 个 ToolCallStart + 2 个 ToolResult（均为错误）
    tool_starts = [e for e in events if isinstance(e, ToolCallStart)]
    tool_results = [e for e in events if isinstance(e, ToolResult)]

    assert len(tool_starts) == 2
    assert len(tool_results) == 2
    assert all(r.is_error for r in tool_results)
    assert "计划工具" in tool_results[0].output
    assert "不能与其他工具" in tool_results[0].output


@pytest.mark.asyncio
async def test_plan_tool_alone_works():
    """计划工具单独调用时应正常执行"""

    @tool(name="todo_write", description="计划工具")
    async def dummy_todo_write(items: list) -> str:
        return "计划已更新: 1 个条目"

    call_count = 0

    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': 'call_tw',
                    'function': type('obj', (), {
                        'name': 'todo_write',
                        'arguments': '{"items": [{"content": "任务1", "status": "pending"}]}'
                    })()
                })()
            ])
            yield StreamChunk(finish_reason="tool_calls")
        else:
            yield StreamChunk(content="计划已创建", finish_reason="stop")
            yield StreamChunk(usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    llm = AsyncMock()
    llm.chat = mock_chat

    agent = Agent(
        llm=llm,
        system_prompt="你是助手",
        tools=[dummy_todo_write],
        config=AgentConfig(max_turns=3),
    )

    events = []
    async for event in agent.run("帮我规划"):
        events.append(event)

    tool_results = [e for e in events if isinstance(e, ToolResult)]
    assert len(tool_results) == 1
    assert tool_results[0].is_error is False
    assert "计划已更新" in tool_results[0].output


@pytest.mark.asyncio
async def test_plan_tools_only_batch_allowed():
    """纯计划工具批次（todo_write + todo_read）不应被拦截"""

    @tool(name="todo_write", description="计划工具")
    async def dummy_todo_write(items: list) -> str:
        return "计划已更新"

    @tool(name="todo_read", description="查看计划")
    async def dummy_todo_read() -> str:
        return "[ ] 任务1"

    call_count = 0

    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': 'call_tw',
                    'function': type('obj', (), {
                        'name': 'todo_write',
                        'arguments': '{"items": [{"content": "任务1", "status": "pending"}]}'
                    })()
                })(),
                type('obj', (), {
                    'index': 1, 'id': 'call_tr',
                    'function': type('obj', (), {
                        'name': 'todo_read',
                        'arguments': '{}'
                    })()
                })()
            ])
            yield StreamChunk(finish_reason="tool_calls")
        else:
            yield StreamChunk(content="计划完成", finish_reason="stop")
            yield StreamChunk(usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    llm = AsyncMock()
    llm.chat = mock_chat

    agent = Agent(
        llm=llm,
        system_prompt="你是助手",
        tools=[dummy_todo_write, dummy_todo_read],
        config=AgentConfig(max_turns=3),
    )

    events = []
    async for event in agent.run("更新并查看计划"):
        events.append(event)

    tool_results = [e for e in events if isinstance(e, ToolResult)]
    assert len(tool_results) == 2
    assert all(not r.is_error for r in tool_results)


@pytest.mark.asyncio
async def test_plan_blocked_after_work_started():
    """已经开始执行非计划工具后，再调用 todo_write 应被拒绝"""

    @tool(name="todo_write", description="计划工具")
    async def dummy_todo_write(items: list) -> str:
        return "计划已更新"

    @tool(name="get_time", description="获取时间")
    async def get_time() -> str:
        return "2026-05-28 10:00"

    call_count = 0

    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # 第一轮：调用普通工具（非计划工具）
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': 'call_gt',
                    'function': type('obj', (), {
                        'name': 'get_time',
                        'arguments': '{}'
                    })()
                })()
            ])
            yield StreamChunk(finish_reason="tool_calls")
        elif call_count == 2:
            # 第二轮：尝试创建计划（应被拦截）
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': 'call_tw',
                    'function': type('obj', (), {
                        'name': 'todo_write',
                        'arguments': '{"items": [{"content": "任务1", "status": "pending"}]}'
                    })()
                })()
            ])
            yield StreamChunk(finish_reason="tool_calls")
        else:
            # 第三轮：正常结束
            yield StreamChunk(content="好的，直接继续工作", finish_reason="stop")
            yield StreamChunk(usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    llm = AsyncMock()
    llm.chat = mock_chat

    agent = Agent(
        llm=llm,
        system_prompt="你是助手",
        tools=[dummy_todo_write, get_time],
        config=AgentConfig(max_turns=5),
    )

    events = []
    async for event in agent.run("帮我做几件事"):
        events.append(event)

    tool_results = [e for e in events if isinstance(e, ToolResult)]
    # 应有 2 个 ToolResult：get_time（成功）+ todo_write（被拒绝）
    assert len(tool_results) == 2

    # get_time 正常执行
    assert tool_results[0].tool_name == "get_time"
    assert tool_results[0].is_error is False

    # todo_write 被拦截
    assert tool_results[1].tool_name == "todo_write"
    assert tool_results[1].is_error is True
    assert "流程约束" in tool_results[1].output
    assert "已经开始执行" in tool_results[1].output


@pytest.mark.asyncio
async def test_plan_first_then_work_allowed():
    """先创建计划再执行工具，后续更新计划应正常允许"""

    @tool(name="todo_write", description="计划工具")
    async def dummy_todo_write(items: list) -> str:
        return "计划已更新"

    @tool(name="get_time", description="获取时间")
    async def get_time() -> str:
        return "2026-05-28 10:00"

    call_count = 0

    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # 第一轮：创建计划
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': 'call_tw1',
                    'function': type('obj', (), {
                        'name': 'todo_write',
                        'arguments': '{"items": [{"content": "任务1", "status": "pending"}]}'
                    })()
                })()
            ])
            yield StreamChunk(finish_reason="tool_calls")
        elif call_count == 2:
            # 第二轮：执行普通工具
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': 'call_gt',
                    'function': type('obj', (), {
                        'name': 'get_time',
                        'arguments': '{}'
                    })()
                })()
            ])
            yield StreamChunk(finish_reason="tool_calls")
        elif call_count == 3:
            # 第三轮：更新计划（应正常允许，因为 _plan_created=True）
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': 'call_tw2',
                    'function': type('obj', (), {
                        'name': 'todo_write',
                        'arguments': '{"items": [{"content": "任务1", "status": "completed"}]}'
                    })()
                })()
            ])
            yield StreamChunk(finish_reason="tool_calls")
        else:
            # 第四轮：正常结束
            yield StreamChunk(content="全部完成", finish_reason="stop")
            yield StreamChunk(usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    llm = AsyncMock()
    llm.chat = mock_chat

    agent = Agent(
        llm=llm,
        system_prompt="你是助手",
        tools=[dummy_todo_write, get_time],
        config=AgentConfig(max_turns=5),
    )

    events = []
    async for event in agent.run("帮我规划并执行"):
        events.append(event)

    tool_results = [e for e in events if isinstance(e, ToolResult)]
    # 应有 3 个 ToolResult，全部成功
    assert len(tool_results) == 3
    assert all(not r.is_error for r in tool_results)

    # 验证工具名顺序
    assert tool_results[0].tool_name == "todo_write"
    assert tool_results[1].tool_name == "get_time"
    assert tool_results[2].tool_name == "todo_write"


@pytest.mark.asyncio
async def test_run_early_break_finalize_in_other_context():
    """回归：消费方提前 break 后，run() 生成器在其它 Context 中 finalize 不应报错。

    场景：SSE 客户端断连 / 消费方收到所需事件后 break——异步生成器之后由
    别的任务（新 Context）执行 finally，ContextVar.reset(token) 曾抛
    ValueError: Token was created in a different Context（已用 _safe_reset 容错）。
    """
    import asyncio

    async def mock_chat(*args, **kwargs):
        yield StreamChunk(content="你好", finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))

    llm = AsyncMock()
    llm.chat = mock_chat
    agent = Agent(llm=llm, session_enabled=False)

    gen = agent.run("hi")

    async def consume_one():
        # 在子任务的 Context 中启动生成器（ContextVar token 在该 Context 创建）
        async for _ in gen:
            break

    await asyncio.create_task(consume_one())
    # 在主任务（不同 Context）中 finalize —— 修复前此处抛 ValueError
    await gen.aclose()
