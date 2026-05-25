"""测试 AgentConfig"""
import pytest
from agent_framework.agent.config import AgentConfig


def test_default_config():
    """应有合理的默认值"""
    config = AgentConfig()
    assert config.max_turns == 10
    assert config.timeout == 120.0
    assert config.total_timeout == 300.0
    assert config.max_total_tokens is None
    assert config.tool_call_limit == 20
    assert config.confirm_dangerous is True


def test_custom_config():
    """应能自定义配置"""
    config = AgentConfig(
        max_turns=5,
        timeout=60.0,
        total_timeout=180.0,
        max_total_tokens=10000,
        tool_call_limit=10,
        confirm_dangerous=False,
    )
    assert config.max_turns == 5
    assert config.timeout == 60.0
    assert config.total_timeout == 180.0
    assert config.max_total_tokens == 10000
    assert config.tool_call_limit == 10
    assert config.confirm_dangerous is False
