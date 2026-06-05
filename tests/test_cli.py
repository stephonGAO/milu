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
    CLIConfig,
    Settings,
    coerce_scalar,
    env_key_name,
    resolve_settings,
)


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


# ── 配置读写 ──────────────────────────────────────────────

def test_config_roundtrip(home):
    assert not CLIConfig.path().exists()
    cfg = CLIConfig(provider="deepseek", model="deepseek-chat", mode="superwork")
    cfg.api_keys["qwen"] = "sk-test"
    p = cfg.save()
    assert p.exists()
    assert p == home / "config.json"

    loaded = CLIConfig.load()
    assert loaded.provider == "deepseek"
    assert loaded.model == "deepseek-chat"
    assert loaded.mode == "superwork"
    assert loaded.api_keys["qwen"] == "sk-test"


def test_config_load_missing_returns_default(home):
    cfg = CLIConfig.load()
    assert cfg.provider is None
    assert cfg.mode == "auto"


def test_config_ignores_unknown_keys(home):
    CLIConfig.path().parent.mkdir(parents=True, exist_ok=True)
    CLIConfig.path().write_text('{"provider": "kimi", "bogus": 123}', encoding="utf-8")
    cfg = CLIConfig.load()
    assert cfg.provider == "kimi"
    assert not hasattr(cfg, "bogus")


def test_coerce_scalar_bool():
    assert coerce_scalar("web_search", "true") is True
    assert coerce_scalar("session_enabled", "off") is False
    assert coerce_scalar("model", "x-1") == "x-1"
    with pytest.raises(ValueError):
        coerce_scalar("web_search", "maybe")


# ── 设置解析优先级 ─────────────────────────────────────────

def test_resolve_defaults(home, monkeypatch):
    monkeypatch.delenv(env_key_name(DEFAULT_PROVIDER), raising=False)
    s = resolve_settings(CLIConfig(), _args())
    assert s.provider == DEFAULT_PROVIDER
    assert s.model == DEFAULT_MODELS[DEFAULT_PROVIDER]
    assert s.mode == "auto"
    assert s.api_key is None


def test_resolve_flag_over_config(home):
    cfg = CLIConfig(provider="qwen", model="qwen-plus")
    s = resolve_settings(cfg, _args(provider="deepseek"))
    assert s.provider == "deepseek"
    # 未给 model → 用 deepseek 的默认模型，而非配置里的 qwen-plus
    assert s.model == DEFAULT_MODELS["deepseek"]


def test_resolve_config_provider_when_no_flag(home):
    cfg = CLIConfig(provider="glm")
    s = resolve_settings(cfg, _args())
    assert s.provider == "glm"
    assert s.model == DEFAULT_MODELS["glm"]


def test_resolve_api_key_priority(home, monkeypatch):
    # env > config
    monkeypatch.setenv("QWEN_API_KEY", "env-key")
    cfg = CLIConfig(api_keys={"qwen": "cfg-key"})
    assert resolve_settings(cfg, _args()).api_key == "env-key"
    # flag > env
    assert resolve_settings(cfg, _args(api_key="flag-key")).api_key == "flag-key"
    # config when no env/flag
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    assert resolve_settings(cfg, _args()).api_key == "cfg-key"


def test_resolve_missing_model_raises(home):
    with pytest.raises(ValueError):
        resolve_settings(CLIConfig(), _args(provider="no-such-provider"))


def test_resolve_no_session_flag(home):
    s = resolve_settings(CLIConfig(session_enabled=True), _args(no_session=True))
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
    # manual 人工审批模式（四档之一）可被解析
    args = build_parser().parse_args(["chat", "--mode", "manual"])
    assert args.mode == "manual"


def test_parser_config_set():
    args = build_parser().parse_args(["config", "set", "mode", "talk"])
    assert args.command == "config"
    assert args.config_action == "set"
    assert args.key == "mode"
    assert args.value == "talk"


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


# ── providers 子命令 ──────────────────────────────────────

def test_providers_command_lists_qwen(home, capsys):
    rc = _cmd_providers(_args())
    assert rc == 0
    out = capsys.readouterr().out
    assert "qwen" in out
