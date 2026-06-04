"""测试非安全工具确认机制（MANUAL 人工审批模式，原 AUTO 行为）"""
import pytest
from unittest.mock import AsyncMock
from agent_framework.agent import Agent, AgentConfig
from agent_framework.agent.events import (
    ToolCallStart,
    ToolConfirmRequired,
    ToolResult,
    AgentDone,
)
from agent_framework.llm.base.response import StreamChunk
from agent_framework.tools import tool


# ── 测试用工具 ────────────────────────────────────────────


@pytest.fixture
def unsafe_tool():
    """非安全工具（显式 is_safe=False）"""
    @tool(name="unsafe_op", description="非安全操作", is_safe=False)
    async def unsafe_op(cmd: str) -> str:
        return f"executed: {cmd}"
    return unsafe_op


@pytest.fixture
def safe_tool():
    """安全工具（is_safe=True）"""
    @tool(name="safe_op", description="安全操作", is_safe=True)
    async def safe_op(msg: str) -> str:
        return f"safe: {msg}"
    return safe_op


def _make_llm_with_tool_call(tool_name: str, args: dict | None = None):
    """创建一个会调用指定工具的 mock LLM，第二轮返回文本"""
    import json
    call_count = 0
    args_str = json.dumps(args or {})

    async def mock_chat(*a, **kw):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # 第一轮：发起工具调用
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': 'call_1',
                    'function': type('obj', (), {
                        'name': tool_name,
                        'arguments': args_str,
                    })()
                })()
            ])
            yield StreamChunk(finish_reason="tool_calls")
        else:
            # 第二轮：基于工具结果回复
            yield StreamChunk(content="完成", finish_reason="stop")

    llm = AsyncMock()
    llm.chat = mock_chat
    return llm


# ── 测试用例 ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unsafe_tool_triggers_confirm(unsafe_tool):
    """MANUAL 模式 + 非安全工具 + on_confirm 回调 → 触发确认"""
    confirm_called = False

    async def on_confirm(tool_name, args_str):
        nonlocal confirm_called
        confirm_called = True
        return True

    llm = _make_llm_with_tool_call("unsafe_op", {"cmd": "rm -rf /"})
    agent = Agent(
        llm=llm,
        system_prompt="你是助手",
        mode="manual",  # 人工审批模式（原 auto 行为）
        tools=[unsafe_tool],
        on_confirm=on_confirm,
    )

    events = []
    async for event in agent.run("执行非安全操作"):
        events.append(event)

    assert confirm_called, "确认回调应被调用"
    assert any(isinstance(e, ToolConfirmRequired) for e in events)


@pytest.mark.asyncio
async def test_confirm_approved_executes(unsafe_tool):
    """回调返回 True 时工具应正常执行"""
    async def on_confirm(tool_name, args_str):
        return True

    llm = _make_llm_with_tool_call("unsafe_op", {"cmd": "ls"})
    agent = Agent(
        llm=llm,
        system_prompt="你是助手",
        mode="manual",  # 人工审批模式（原 auto 行为）
        tools=[unsafe_tool],
        on_confirm=on_confirm,
    )

    events = []
    async for event in agent.run("执行"):
        events.append(event)

    confirm_events = [e for e in events if isinstance(e, ToolConfirmRequired)]
    assert len(confirm_events) == 1
    assert confirm_events[0].approved is True

    results = [e for e in events if isinstance(e, ToolResult)]
    assert len(results) == 1
    assert results[0].is_error is False
    assert "executed: ls" in results[0].output


@pytest.mark.asyncio
async def test_confirm_rejected_skips_execution(unsafe_tool):
    """回调返回 False 时工具不应执行，返回拒绝信息"""
    tool_executed = False

    # 重新定义一个会记录是否被调用的非安全工具
    @tool(name="unsafe_op", description="非安全操作", is_safe=False)
    async def tracked_unsafe_op(cmd: str) -> str:
        nonlocal tool_executed
        tool_executed = True
        return f"executed: {cmd}"

    async def on_confirm(tool_name, args_str):
        return False  # 用户拒绝

    llm = _make_llm_with_tool_call("unsafe_op", {"cmd": "rm -rf /"})
    agent = Agent(
        llm=llm,
        system_prompt="你是助手",
        mode="manual",  # 人工审批模式（原 auto 行为）
        tools=[tracked_unsafe_op],
        on_confirm=on_confirm,
    )

    events = []
    async for event in agent.run("执行非安全操作"):
        events.append(event)

    assert not tool_executed, "工具函数不应被执行"

    confirm_events = [e for e in events if isinstance(e, ToolConfirmRequired)]
    assert len(confirm_events) == 1
    assert confirm_events[0].approved is False

    results = [e for e in events if isinstance(e, ToolResult)]
    assert len(results) == 1
    assert results[0].is_error is True
    assert "拒绝" in results[0].output


@pytest.mark.asyncio
async def test_safe_tool_no_confirm(safe_tool):
    """安全工具（is_safe=True）不应触发确认回调"""
    confirm_called = False

    async def on_confirm(tool_name, args_str):
        nonlocal confirm_called
        confirm_called = True
        return True

    llm = _make_llm_with_tool_call("safe_op", {"msg": "hello"})
    agent = Agent(
        llm=llm,
        system_prompt="你是助手",
        mode="manual",  # 人工审批模式（原 auto 行为）
        tools=[safe_tool],
        on_confirm=on_confirm,
    )

    events = []
    async for event in agent.run("执行安全操作"):
        events.append(event)

    assert not confirm_called, "安全工具不应触发确认回调"
    assert not any(isinstance(e, ToolConfirmRequired) for e in events)

    results = [e for e in events if isinstance(e, ToolResult)]
    assert len(results) == 1
    assert results[0].is_error is False


@pytest.mark.asyncio
async def test_no_callback_executes_directly(unsafe_tool):
    """on_confirm=None 时非安全工具也直接执行"""
    llm = _make_llm_with_tool_call("unsafe_op", {"cmd": "ls"})
    agent = Agent(
        llm=llm,
        system_prompt="你是助手",
        mode="manual",  # 人工审批模式（原 auto 行为）
        tools=[unsafe_tool],
        on_confirm=None,  # 没有回调
    )

    events = []
    async for event in agent.run("执行"):
        events.append(event)

    assert not any(isinstance(e, ToolConfirmRequired) for e in events)

    results = [e for e in events if isinstance(e, ToolResult)]
    assert len(results) == 1
    assert results[0].is_error is False
    assert "executed: ls" in results[0].output


@pytest.mark.asyncio
async def test_confirm_rejected_yields_correct_event(unsafe_tool):
    """拒绝时应 yield ToolConfirmRequired(approved=False)"""
    async def on_confirm(tool_name, args_str):
        return False

    llm = _make_llm_with_tool_call("unsafe_op", {"cmd": "rm -rf /"})
    agent = Agent(
        llm=llm,
        system_prompt="你是助手",
        mode="manual",  # 人工审批模式（原 auto 行为）
        tools=[unsafe_tool],
        on_confirm=on_confirm,
    )

    events = []
    async for event in agent.run("执行"):
        events.append(event)

    confirm_events = [e for e in events if isinstance(e, ToolConfirmRequired)]
    assert len(confirm_events) == 1
    ce = confirm_events[0]
    assert ce.tool_name == "unsafe_op"
    assert ce.approved is False
    assert "rm -rf" in ce.arguments


@pytest.mark.asyncio
async def test_confirm_rejected_history_updated(unsafe_tool):
    """拒绝后应将拒绝信息写入对话历史，供 LLM 下一轮看到"""
    async def on_confirm(tool_name, args_str):
        return False

    llm = _make_llm_with_tool_call("unsafe_op", {"cmd": "rm"})
    agent = Agent(
        llm=llm,
        system_prompt="你是助手",
        mode="manual",  # 人工审批模式（原 auto 行为）
        tools=[unsafe_tool],
        on_confirm=on_confirm,
    )

    async for event in agent.run("执行"):
        pass

    messages = agent.history.all_messages
    # 找到最后一条 TOOL 消息
    tool_messages = [m for m in messages if m.role.value == "tool"]
    assert len(tool_messages) >= 1
    assert "拒绝" in tool_messages[-1].content


# ── ConfirmResponse 自定义消息测试 ────────────────────────


@pytest.mark.asyncio
async def test_confirm_response_approved(unsafe_tool):
    """返回 ConfirmResponse(approved=True) 时工具正常执行"""
    from agent_framework.agent.events import ConfirmResponse

    async def on_confirm(tool_name, args_str):
        return ConfirmResponse(approved=True)

    llm = _make_llm_with_tool_call("unsafe_op", {"cmd": "ls"})
    agent = Agent(
        llm=llm,
        system_prompt="你是助手",
        mode="manual",  # 人工审批模式（原 auto 行为）
        tools=[unsafe_tool],
        on_confirm=on_confirm,
    )

    events = []
    async for event in agent.run("执行"):
        events.append(event)

    confirm_events = [e for e in events if isinstance(e, ToolConfirmRequired)]
    assert confirm_events[0].approved is True

    results = [e for e in events if isinstance(e, ToolResult)]
    assert results[0].is_error is False


@pytest.mark.asyncio
async def test_confirm_response_rejected_with_message(unsafe_tool):
    """返回 ConfirmResponse(approved=False, message="...") 时自定义消息发回 LLM"""
    from agent_framework.agent.events import ConfirmResponse

    custom_msg = "不要用 rm，改用 mv 到回收站"

    async def on_confirm(tool_name, args_str):
        return ConfirmResponse(approved=False, message=custom_msg)

    llm = _make_llm_with_tool_call("unsafe_op", {"cmd": "rm -rf /"})
    agent = Agent(
        llm=llm,
        system_prompt="你是助手",
        mode="manual",  # 人工审批模式（原 auto 行为）
        tools=[unsafe_tool],
        on_confirm=on_confirm,
    )

    events = []
    async for event in agent.run("执行"):
        events.append(event)

    # ToolConfirmRequired 应包含自定义消息
    confirm_events = [e for e in events if isinstance(e, ToolConfirmRequired)]
    assert len(confirm_events) == 1
    assert confirm_events[0].approved is False
    assert confirm_events[0].message == custom_msg

    # ToolResult 应包含自定义消息
    results = [e for e in events if isinstance(e, ToolResult)]
    assert len(results) == 1
    assert results[0].is_error is True
    assert custom_msg in results[0].output
    assert "指示" in results[0].output


@pytest.mark.asyncio
async def test_confirm_response_custom_message_in_history(unsafe_tool):
    """自定义消息应写入对话历史，供 LLM 下一轮看到"""
    from agent_framework.agent.events import ConfirmResponse

    custom_msg = "请先备份再删除"

    async def on_confirm(tool_name, args_str):
        return ConfirmResponse(approved=False, message=custom_msg)

    llm = _make_llm_with_tool_call("unsafe_op", {"cmd": "rm"})
    agent = Agent(
        llm=llm,
        system_prompt="你是助手",
        mode="manual",  # 人工审批模式（原 auto 行为）
        tools=[unsafe_tool],
        on_confirm=on_confirm,
    )

    async for event in agent.run("执行"):
        pass

    messages = agent.history.all_messages
    tool_messages = [m for m in messages if m.role.value == "tool"]
    assert len(tool_messages) >= 1
    assert custom_msg in tool_messages[-1].content
    assert "指示" in tool_messages[-1].content


@pytest.mark.asyncio
async def test_confirm_response_rejected_no_message(unsafe_tool):
    """ConfirmResponse(approved=False) 无消息时使用默认拒绝文案"""
    from agent_framework.agent.events import ConfirmResponse

    async def on_confirm(tool_name, args_str):
        return ConfirmResponse(approved=False)

    llm = _make_llm_with_tool_call("unsafe_op", {"cmd": "rm"})
    agent = Agent(
        llm=llm,
        system_prompt="你是助手",
        mode="manual",  # 人工审批模式（原 auto 行为）
        tools=[unsafe_tool],
        on_confirm=on_confirm,
    )

    events = []
    async for event in agent.run("执行"):
        events.append(event)

    results = [e for e in events if isinstance(e, ToolResult)]
    assert results[0].is_error is True
    assert "拒绝" in results[0].output


@pytest.mark.asyncio
async def test_backward_compat_bool_true(unsafe_tool):
    """向后兼容：回调返回 True 仍然正常工作"""
    async def on_confirm(tool_name, args_str):
        return True  # 旧式 bool 返回

    llm = _make_llm_with_tool_call("unsafe_op", {"cmd": "ls"})
    agent = Agent(
        llm=llm,
        system_prompt="你是助手",
        mode="manual",  # 人工审批模式（原 auto 行为）
        tools=[unsafe_tool],
        on_confirm=on_confirm,
    )

    events = []
    async for event in agent.run("执行"):
        events.append(event)

    confirm_events = [e for e in events if isinstance(e, ToolConfirmRequired)]
    assert confirm_events[0].approved is True
    assert confirm_events[0].message == ""  # bool 返回时 message 为空


@pytest.mark.asyncio
async def test_backward_compat_bool_false(unsafe_tool):
    """向后兼容：回调返回 False 仍然使用默认拒绝文案"""
    async def on_confirm(tool_name, args_str):
        return False  # 旧式 bool 返回

    llm = _make_llm_with_tool_call("unsafe_op", {"cmd": "rm"})
    agent = Agent(
        llm=llm,
        system_prompt="你是助手",
        mode="manual",  # 人工审批模式（原 auto 行为）
        tools=[unsafe_tool],
        on_confirm=on_confirm,
    )

    events = []
    async for event in agent.run("执行"):
        events.append(event)

    results = [e for e in events if isinstance(e, ToolResult)]
    assert results[0].is_error is True
    assert "拒绝" in results[0].output
