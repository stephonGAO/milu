"""测试 AgentConfig 和 CompactConfig"""
from unittest.mock import AsyncMock

import pytest
from agent_framework.agent import Agent, AgentMode
from agent_framework.agent.config import AgentConfig, CompactConfig


def test_default_config():
    """应有合理的默认值"""
    config = AgentConfig()
    assert config.max_turns == 100
    assert config.timeout == 300.0
    assert config.total_timeout == 3600.0
    assert config.max_total_tokens is None
    assert config.tool_call_limit == 100
    assert config.mcp_tools_active_by_default is False


def test_custom_config():
    """应能自定义配置"""
    config = AgentConfig(
        max_turns=5,
        timeout=60.0,
        total_timeout=180.0,
        max_total_tokens=10000,
        tool_call_limit=10,
    )
    assert config.max_turns == 5
    assert config.timeout == 60.0
    assert config.total_timeout == 180.0
    assert config.max_total_tokens == 10000
    assert config.tool_call_limit == 10


def test_compact_config_defaults():
    """CompactConfig 应有合理的默认值"""
    config = CompactConfig()
    assert config.enabled is True
    assert config.trigger_ratio == 0.7
    assert config.recent_rounds == 5
    assert config.max_messages == 300


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


def _make_agent(shared_config):
    """用最小依赖构造一个 Agent（关闭 session/元工具，避免落盘和外部调用）。"""
    llm = AsyncMock()
    return Agent(
        llm=llm,
        config=shared_config,
        register_catalog=False,
        register_skills=False,
    )


def test_agent_copies_config_no_cross_contamination():
    """P0-2：多个 Agent 共享同一 config 模板时，set_mode 不应互相污染。

    Agent.__init__ 应对传入 config 做拷贝，否则 set_mode() 原地修改共享对象，
    会导致多用户场景下模式跨用户串扰（越权风险）。
    """
    shared = AgentConfig(mode=AgentMode.AUTO, session_enabled=False)

    a1 = _make_agent(shared)
    a2 = _make_agent(shared)

    a1.set_mode("superwork")

    assert a1.mode == AgentMode.SUPERWORK
    assert a2.mode == AgentMode.AUTO, "set_mode 不应影响其他 Agent"
    assert shared.mode == AgentMode.AUTO, "set_mode 不应修改共享的原始 config"


def test_agent_config_is_independent_instance():
    """Agent 持有的 config 应是独立实例，而非传入对象本身。"""
    shared = AgentConfig(session_enabled=False)
    agent = _make_agent(shared)
    assert agent._config is not shared
