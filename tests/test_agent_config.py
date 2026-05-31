"""测试 AgentConfig 和 CompactConfig"""
import pytest
from agent_framework.agent.config import AgentConfig, CompactConfig


def test_default_config():
    """应有合理的默认值"""
    config = AgentConfig()
    assert config.max_turns == 100
    assert config.timeout == 300.0
    assert config.total_timeout == 3600.0
    assert config.max_total_tokens is None
    assert config.tool_call_limit == 100
    assert config.confirm_dangerous is True
    assert config.mcp_tools_active_by_default is False


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


def test_compact_config_defaults():
    """CompactConfig 应有合理的默认值"""
    config = CompactConfig()
    assert config.enabled is True
    assert config.trigger_ratio == 0.7
    assert config.recent_rounds == 3
    assert config.max_messages == 50


def test_compact_config_custom():
    """CompactConfig 应能自定义"""
    config = CompactConfig(
        enabled=False,
        trigger_ratio=0.5,
        recent_rounds=5,
        max_messages=100,
    )
    assert config.enabled is False
    assert config.trigger_ratio == 0.5
    assert config.recent_rounds == 5
    assert config.max_messages == 100
