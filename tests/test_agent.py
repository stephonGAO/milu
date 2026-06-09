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
async def test_tool_call_preserves_extra_content(simple_tool):
    """provider 透传字段（tool_call.extra_content，如 Gemini thought_signature）
    应被捕获进 assistant 历史消息，并在下一轮回传给 LLM。

    回归：Gemini 思考模型多轮工具调用要求带回 thought_signature，否则
    400 "Function call is missing a thought_signature"。
    """
    sig = {"google": {"thought_signature": "SIG-123"}}
    seen_messages = []
    call_count = 0

    async def mock_chat(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        seen_messages.append(list(messages))
        if call_count == 1:
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': 'call_1',
                    'function': type('obj', (), {'name': 'get_time', 'arguments': '{}'})(),
                    'extra_content': sig,
                })()
            ])
            yield StreamChunk(finish_reason="tool_calls")
        else:
            yield StreamChunk(content="完成", finish_reason="stop")
            yield StreamChunk(usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))

    llm = AsyncMock()
    llm.chat = mock_chat

    agent = Agent(llm=llm, system_prompt="你是助手", tools=[simple_tool], session_enabled=False)

    async for _ in agent.run("现在几点了？"):
        pass

    assert call_count == 2  # 工具调用后进入第二轮
    # 第二轮发给 LLM 的历史里，带 tool_calls 的 assistant 消息应保留 extra_content
    second_turn = seen_messages[1]
    assistant_msgs = [
        m for m in second_turn
        if m.role == MessageRole.ASSISTANT and m.tool_calls
    ]
    assert assistant_msgs, "第二轮历史缺少带工具调用的 assistant 消息"
    tc = assistant_msgs[0].tool_calls[0]
    assert tc.get("extra_content") == sig
    # to_dict() 也应原样透传（真实 provider 据此发往 API）
    assert assistant_msgs[0].to_dict()["tool_calls"][0]["extra_content"] == sig


def _fake_tc(idx, cid, name, args):
    """构造一个仿 OpenAI SDK 的流式 tool_call delta 对象。"""
    return type('obj', (), {
        'index': idx,
        'id': cid,
        'function': type('obj', (), {'name': name, 'arguments': args})(),
    })()


def test_merge_tool_calls_parallel_missing_index():
    """并行工具调用即便 index 缺失（都为 None），也应按唯一 id 拆成多个调用，
    不能合并成名字拼接的单个调用。

    回归：Gemini 兼容层并行调用 index 缺失/为 0，旧逻辑纯按 index 合并，
    导致 researcher+coder 被并成 "researchercoder"（工具不存在）。
    """
    from milu.agent.agent import _merge_tool_calls

    buffer = []
    _merge_tool_calls(buffer, [_fake_tc(None, "c1", "researcher", '{"task":"a"}')])
    _merge_tool_calls(buffer, [_fake_tc(None, "c2", "coder", '{"task":"b"}')])

    assert len(buffer) == 2
    assert {b["function"]["name"] for b in buffer} == {"researcher", "coder"}
    assert {b["id"] for b in buffer} == {"c1", "c2"}
    assert {b["function"]["arguments"] for b in buffer} == {'{"task":"a"}', '{"task":"b"}'}


def test_merge_tool_calls_fragmented_args_by_index():
    """单个调用参数分多片到达（续片无 id），按 index 正确拼接，不误拆。"""
    from milu.agent.agent import _merge_tool_calls

    buffer = []
    _merge_tool_calls(buffer, [_fake_tc(0, "c1", "get_time", "")])
    _merge_tool_calls(buffer, [_fake_tc(0, "", "", '{"tz":')])
    _merge_tool_calls(buffer, [_fake_tc(0, "", "", '"utc"}')])

    assert len(buffer) == 1
    assert buffer[0]["id"] == "c1"
    assert buffer[0]["function"]["name"] == "get_time"
    assert buffer[0]["function"]["arguments"] == '{"tz":"utc"}'


def test_merge_tool_calls_parallel_distinct_index():
    """并行调用带正确的不同 index（qwen 等）：仍正确拆为多个调用。"""
    from milu.agent.agent import _merge_tool_calls

    buffer = []
    _merge_tool_calls(buffer, [_fake_tc(0, "c1", "a", '{}')])
    _merge_tool_calls(buffer, [_fake_tc(1, "c2", "b", '{}')])

    assert len(buffer) == 2
    assert [b["function"]["name"] for b in buffer] == ["a", "b"]


@pytest.mark.asyncio
async def test_rate_limit_retry(monkeypatch):
    """429 / 服务过载属瞬时错误，应退避重试本轮而非直接判失败。

    回归：子代理（如 coder）撞到 Kimi 429 engine_overloaded 时曾直接执行失败；
    现在主/子代理共用的 run() 循环会指数退避重试。
    """
    from milu.llm.base.exceptions import StreamError

    call_count = 0

    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise StreamError(
                "Kimi 流式调用异常: Error code: 429 - {'error': {'message': "
                "'The engine is currently overloaded, please try again later', "
                "'type': 'engine_overloaded_error'}}"
            )
        yield StreamChunk(content="恢复成功", finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))

    # 跳过真实退避等待，加速测试并验证确实退避了
    slept = []

    async def fake_sleep(delay):
        slept.append(delay)

    monkeypatch.setattr("milu.agent.agent.asyncio.sleep", fake_sleep)

    llm = AsyncMock()
    llm.chat = mock_chat

    agent = Agent(llm=llm, system_prompt="你是助手")

    events = []
    async for event in agent.run("你好"):
        events.append(event)

    assert call_count == 2          # 第一次 429 失败、退避后第二次成功
    assert slept                    # 确实做了退避等待
    assert not any(isinstance(e, AgentError) for e in events)
    done = next(e for e in events if isinstance(e, AgentDone))
    assert done.final_text == "恢复成功"


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
    """已经开始执行"有副作用"的工具后，再调用 todo_write 应被拒绝。

    注意：守卫只由不安全（有副作用）工具触发；只读调查类工具不触发
    （见 test_plan_allowed_after_readonly_research）。故这里用 is_safe=False 的
    写文件工具代表"开始干活"。
    """

    @tool(name="todo_write", description="计划工具")
    async def dummy_todo_write(items: list) -> str:
        return "计划已更新"

    @tool(name="write_file", description="写文件（有副作用）", is_safe=False)
    async def write_file(path: str) -> str:
        return f"已写入 {path}"

    call_count = 0

    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # 第一轮：调用有副作用的工具（开始干活）
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': 'call_wf',
                    'function': type('obj', (), {
                        'name': 'write_file',
                        'arguments': '{"path": "a.txt"}'
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
        tools=[dummy_todo_write, write_file],
        config=AgentConfig(max_turns=5),
        judge_llm=False,   # 关闭 AI 判定，保持 mock_chat 调用序列干净
    )

    events = []
    async for event in agent.run("帮我做几件事"):
        events.append(event)

    tool_results = [e for e in events if isinstance(e, ToolResult)]
    # 应有 2 个 ToolResult：write_file（成功）+ todo_write（被拒绝）
    assert len(tool_results) == 2

    # write_file 正常执行
    assert tool_results[0].tool_name == "write_file"
    assert tool_results[0].is_error is False

    # todo_write 被拦截
    assert tool_results[1].tool_name == "todo_write"
    assert tool_results[1].is_error is True
    assert "流程约束" in tool_results[1].output
    assert "已经开始执行" in tool_results[1].output


@pytest.mark.asyncio
async def test_plan_allowed_after_readonly_research():
    """只读调查类工具（is_safe=True）执行后，仍可创建 todo 计划。

    回归用例：「先读代码理解 → 再列 todo 计划 → 执行」是自然工作流，
    研究阶段（只读工具）不应触发「已开始工作后禁止建计划」的守卫。
    """

    @tool(name="todo_write", description="计划工具")
    async def dummy_todo_write(items: list) -> str:
        return "计划已更新"

    @tool(name="read_file", description="读文件（只读调查）")  # 默认 is_safe=True
    async def read_file(path: str) -> str:
        return f"文件 {path} 的内容……"

    call_count = 0

    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # 第一轮：只读调查（读代码理解任务）
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': 'call_rf',
                    'function': type('obj', (), {
                        'name': 'read_file',
                        'arguments': '{"path": "a.txt"}'
                    })()
                })()
            ])
            yield StreamChunk(finish_reason="tool_calls")
        elif call_count == 2:
            # 第二轮：研究完成后创建计划（应被允许）
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
            yield StreamChunk(content="计划已就绪，开始执行", finish_reason="stop")
            yield StreamChunk(usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    llm = AsyncMock()
    llm.chat = mock_chat

    agent = Agent(
        llm=llm,
        system_prompt="你是助手",
        tools=[dummy_todo_write, read_file],
        config=AgentConfig(max_turns=5),
    )

    events = []
    async for event in agent.run("帮我做几件事"):
        events.append(event)

    tool_results = [e for e in events if isinstance(e, ToolResult)]
    # 应有 2 个 ToolResult：read_file（成功）+ todo_write（成功，未被拦截）
    assert len(tool_results) == 2

    assert tool_results[0].tool_name == "read_file"
    assert tool_results[0].is_error is False

    # 关键断言：研究之后的 todo_write 未被守卫拦截
    assert tool_results[1].tool_name == "todo_write"
    assert tool_results[1].is_error is False
    assert "计划已更新" in tool_results[1].output


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


@pytest.mark.asyncio
async def test_continue_after_orphan_tool_calls_repairs_sequence():
    """回归：上一轮在工具执行前被中断留下孤儿 assistant(tool_calls)，再输入"继续"
    时，发送给 LLM 的消息序列必须被修复为合法配对（否则 MiniMax 报 400
    "tool call result does not follow tool call" 且无法恢复）。
    """
    from milu.llm.base.message import Message

    captured = {}

    async def mock_chat(messages, *args, **kwargs):
        # 捕获本次发送给 LLM 的消息序列
        captured["messages"] = list(messages)
        yield StreamChunk(content="好的，继续", finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))

    llm = AsyncMock()
    llm.chat = mock_chat
    agent = Agent(llm=llm, system_prompt="你是助手", session_enabled=False, tools=[])

    # 手工构造"已损坏"历史：assistant(tool_calls) 后没有 tool 结果
    agent.history.add(Message(role=MessageRole.USER, content="做点事"))
    agent.history.add(Message(role=MessageRole.ASSISTANT, content="调用工具", tool_calls=[
        {"id": "call_1", "type": "function", "function": {"name": "shell", "arguments": "{}"}},
    ]))

    events = []
    async for event in agent.run("继续"):
        events.append(event)

    # 不应产生 AgentError（修复前会因 LLM 400 报错）
    assert not any(isinstance(e, AgentError) for e in events)

    # 发送给 LLM 的序列里，assistant(tool_calls) 后必须紧跟匹配的 tool 结果
    sent = captured["messages"]
    idx = next(i for i, m in enumerate(sent)
               if m.role == MessageRole.ASSISTANT and m.tool_calls)
    assert sent[idx + 1].role == MessageRole.TOOL
    assert sent[idx + 1].tool_call_id == "call_1"
