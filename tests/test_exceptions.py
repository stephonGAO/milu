"""测试 Agent 相关异常类"""
import pytest
from milu.exceptions import MiluError
from milu.agent.exceptions import (
    AgentLoopError,
    MaxTurnsExceeded,
    AgentTimeout,
    TokenLimitExceeded,
    ToolCallLimitExceeded,
    ToolExecutionError,
)


def test_agent_loop_error_inherits_from_base():
    """AgentLoopError 应继承 MiluError"""
    assert issubclass(AgentLoopError, MiluError)

    error = AgentLoopError("测试错误")
    assert isinstance(error, MiluError)
    assert str(error) == "测试错误"


def test_max_turns_exceeded():
    """MaxTurnsExceeded 应继承 AgentLoopError"""
    assert issubclass(MaxTurnsExceeded, AgentLoopError)

    error = MaxTurnsExceeded("超出最大轮次")
    assert isinstance(error, AgentLoopError)


def test_agent_timeout():
    """AgentTimeout 应继承 AgentLoopError"""
    assert issubclass(AgentTimeout, AgentLoopError)


def test_token_limit_exceeded():
    """TokenLimitExceeded 应继承 AgentLoopError"""
    assert issubclass(TokenLimitExceeded, AgentLoopError)


def test_tool_call_limit_exceeded():
    """ToolCallLimitExceeded 应继承 AgentLoopError"""
    assert issubclass(ToolCallLimitExceeded, AgentLoopError)


def test_tool_execution_error():
    """ToolExecutionError 应继承 AgentLoopError"""
    assert issubclass(ToolExecutionError, AgentLoopError)
