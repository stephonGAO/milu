"""CLI 单元测试（全程离线，不触网——LLM 构造不鉴权、不调用 run）。"""
from __future__ import annotations

import argparse

import pytest

from milu import AgentMode
from milu.cli.app import _cmd_providers, build_parser
from milu.cli.builder import build_agent
from milu.cli.config import (
    DEFAULT_MODELS,
    DEFAULT_PROVIDER,
    Settings,
    env_key_name,
    resolve_settings,
)
from milu.config import MiluConfig, _builtin_defaults, _deep_merge, load_config, set_user_value
from milu.resources import user_config_path


@pytest.fixture
def home(tmp_path, monkeypatch):
    """把用户数据目录指向 tmp_path（隔离 config.json）。"""
    monkeypatch.setenv("MILU_HOME", str(tmp_path))
    return tmp_path


def _args(**kw):
    """构造一个带默认 None 的 argparse.Namespace。"""
    base = dict(provider=None, model=None, api_key=None, mode=None,
                no_session=False, no_mcp=False, no_subagents=False, session=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _cfg(**overrides):
    """内置默认 + 分节覆盖，构造一个 MiluConfig（overrides 形如 llm={...}, agent={...}）。"""
    return MiluConfig(_deep_merge(_builtin_defaults(), overrides))


# ── 配置读写：用户级 set + 分层合并 ─────────────────────────

def test_user_set_and_merge(home):
    assert not user_config_path().exists()
    set_user_value("agent.llm.provider", "deepseek")
    set_user_value("agent.max_turns", "7")
    cfg = load_config()
    assert cfg.llm["provider"] == "deepseek"    # cfg.llm 便捷访问 agent.llm
    assert cfg.agent["max_turns"] == 7          # 按当前 int 类型转换
    assert user_config_path().exists()


def test_load_missing_returns_builtin_defaults(home):
    cfg = load_config()
    assert cfg.llm["provider"] == DEFAULT_PROVIDER
    assert cfg.agent["mode"] == "auto"
    assert cfg.agent["max_turns"] == 100


def test_load_ignores_legacy_api_keys(home):
    user_config_path().parent.mkdir(parents=True, exist_ok=True)
    user_config_path().write_text(
        '{"agent": {"llm": {"provider": "kimi"}}, "api_keys": {"qwen": "x"}}',
        encoding="utf-8")
    cfg = load_config()
    assert cfg.llm["provider"] == "kimi"
    assert "api_keys" not in cfg.data


def test_coerce_via_set(home):
    set_user_value("agent.session_enabled", "off")   # bool
    set_user_value("compact.trigger_ratio", "0.5")   # float
    cfg = load_config()
    assert cfg.agent["session_enabled"] is False
    assert cfg.compact["trigger_ratio"] == 0.5


def test_set_unknown_key_raises(home):
    with pytest.raises(ValueError):
        set_user_value("nope.nope", "1")


def test_set_section_raises(home):
    # 不能直接设置一个分节（必须是具体标量项）
    with pytest.raises(ValueError):
        set_user_value("agent", "x")


# ── 设置解析优先级 ─────────────────────────────────────────

def test_resolve_defaults(home, monkeypatch):
    monkeypatch.delenv(env_key_name(DEFAULT_PROVIDER), raising=False)
    s = resolve_settings(_cfg(), _args())
    assert s.provider == DEFAULT_PROVIDER
    assert s.model == DEFAULT_MODELS[DEFAULT_PROVIDER]
    assert s.mode == "auto"
    assert s.api_key is None


def test_resolve_flag_over_config(home):
    cfg = _cfg(agent={"llm": {"provider": "qwen", "model": "qwen-plus"}})
    s = resolve_settings(cfg, _args(provider="deepseek"))
    assert s.provider == "deepseek"
    # 未给 model → 用 deepseek 的默认模型，而非配置里的 qwen-plus
    assert s.model == DEFAULT_MODELS["deepseek"]


def test_resolve_config_provider_when_no_flag(home):
    cfg = _cfg(agent={"llm": {"provider": "glm"}})
    s = resolve_settings(cfg, _args())
    assert s.provider == "glm"
    assert s.model == DEFAULT_MODELS["glm"]


def test_resolve_api_key_priority(home, monkeypatch):
    # env > 无（不再有 config.json 兜底）
    monkeypatch.setenv("QWEN_API_KEY", "env-key")
    cfg = _cfg()
    assert resolve_settings(cfg, _args()).api_key == "env-key"
    # flag > env
    assert resolve_settings(cfg, _args(api_key="flag-key")).api_key == "flag-key"
    # 都没有 → None
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    assert resolve_settings(cfg, _args()).api_key is None


def test_resolve_agent_section_carried(home):
    # 文件配置的运行限额/压缩分节随 Settings 透传给 builder
    cfg = _cfg(agent={"max_turns": 9}, compact={"recent_rounds": 2})
    s = resolve_settings(cfg, _args())
    assert s.agent["max_turns"] == 9
    assert s.compact["recent_rounds"] == 2


def test_resolve_missing_model_raises(home):
    with pytest.raises(ValueError):
        resolve_settings(_cfg(), _args(provider="no-such-provider"))


def test_resolve_no_session_flag(home):
    s = resolve_settings(_cfg(), _args(no_session=True))
    assert s.session_enabled is False


# ── 参数解析 ──────────────────────────────────────────────

def test_parser_no_command():
    args = build_parser().parse_args([])
    assert args.command is None  # main() 会兜底为 chat


def test_parser_run():
    args = build_parser().parse_args(["run", "你好", "-q", "-p", "deepseek"])
    assert args.command == "run"
    assert args.prompt == "你好"
    assert args.quiet is True
    assert args.provider == "deepseek"


def test_parser_chat_options():
    args = build_parser().parse_args(["chat", "--mode", "superwork", "--no-mcp"])
    assert args.command == "chat"
    assert args.mode == "superwork"
    assert args.no_mcp is True


def test_parser_mode_manual():
    args = build_parser().parse_args(["chat", "--mode", "manual"])
    assert args.mode == "manual"


def test_parser_config_set_dotted():
    args = build_parser().parse_args(["config", "set", "agent.max_turns", "5"])
    assert args.command == "config"
    assert args.config_action == "set"
    assert args.key == "agent.max_turns"
    assert args.value == "5"


def test_parser_config_init():
    args = build_parser().parse_args(["config", "init"])
    assert args.config_action == "init"


# ── builder（构造 Agent，不调用 run）────────────────────────

def test_build_agent_full(home):
    s = Settings(
        provider="qwen", model="qwen-plus", api_key="dummy",
        mode="auto", session_enabled=False,
        web_search=False, enable_thinking=False,
        use_mcp=False, use_subagents=True,
    )
    agent = build_agent(s)
    assert agent.mode == AgentMode.AUTO
    names = agent.tools.list_tools()
    # 全配默认：内置工具 + 内置子代理三件套（researcher/reader/coder）
    for expected in ("file_read", "file_write", "python_repl", "shell_command",
                     "researcher", "reader", "coder"):
        assert expected in names, f"缺少工具 {expected}"


def test_build_agent_no_subagents(home):
    s = Settings(
        provider="qwen", model="qwen-plus", api_key="dummy",
        mode="talk", session_enabled=False,
        web_search=False, enable_thinking=False,
        use_mcp=False, use_subagents=False,
    )
    agent = build_agent(s)
    assert agent.mode == AgentMode.TALK
    names = agent.tools.list_tools()
    assert "researcher" not in names
    assert "file_read" in names


def test_build_agent_applies_agent_config(home):
    # 分层配置的运行限额经 Settings.agent 落到 AgentConfig
    s = Settings(
        provider="qwen", model="qwen-plus", api_key="dummy",
        mode="auto", session_enabled=False,
        web_search=False, enable_thinking=False,
        use_mcp=False, use_subagents=False,
        agent={"max_turns": 13, "timeout": 99.0, "total_timeout": 100.0,
               "max_total_tokens": None, "tool_call_limit": 7},
    )
    agent = build_agent(s)
    assert agent._config.max_turns == 13
    assert agent._config.tool_call_limit == 7


# ── providers 子命令 ──────────────────────────────────────

def test_providers_command_lists_qwen(home, capsys):
    rc = _cmd_providers(_args())
    assert rc == 0
    out = capsys.readouterr().out
    assert "qwen" in out
