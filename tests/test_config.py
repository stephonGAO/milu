"""统一分层配置体系测试（milu.config）。"""
from __future__ import annotations

import json

import pytest

from milu.config import (
    DEFAULT_MODELS,
    DEFAULT_PROVIDER,
    _builtin_defaults,
    _deep_merge,
    load_config,
    set_user_value,
    write_project_template,
)
from milu.resources import project_config_path, user_config_path


@pytest.fixture
def paths(tmp_path, monkeypatch):
    """隔离项目级（MILU_PROJECT_DIR）与用户级（MILU_HOME）配置路径到 tmp。"""
    proj = tmp_path / "proj"
    home = tmp_path / "home"
    proj.mkdir()
    home.mkdir()
    monkeypatch.setenv("MILU_PROJECT_DIR", str(proj))
    monkeypatch.setenv("MILU_HOME", str(home))
    return proj, home


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ── 基线派生（不重复默认值）────────────────────────────────

def test_builtin_defaults_derived_from_dataclasses():
    from milu.agent.config import AgentConfig, CompactConfig

    d = _builtin_defaults()
    assert d["agent"]["max_turns"] == AgentConfig().max_turns
    assert d["agent"]["tool_call_limit"] == AgentConfig().tool_call_limit
    assert d["compact"]["trigger_ratio"] == CompactConfig().trigger_ratio
    assert d["agent"]["llm"]["provider"] == DEFAULT_PROVIDER  # llm 嵌套在 agent 下
    assert "llm" not in d                                     # 顶层不再单列 llm
    assert d["default_models"] == DEFAULT_MODELS
    # pool 仅暴露可序列化子集
    assert "max_agents" in d["pool"] and "mcp_config_path" not in d["pool"]
    # scheduler 分节从 SchedulerConfig 派生
    from milu.scheduler.engine import SchedulerConfig
    assert d["scheduler"]["max_concurrent_tasks"] == SchedulerConfig().max_concurrent_tasks
    assert d["scheduler"]["task_timeout"] == SchedulerConfig().task_timeout
    assert d["scheduler"]["notify"] == SchedulerConfig().notify


# ── 分层合并优先级 ────────────────────────────────────────

def test_load_no_files_is_builtin(paths):
    cfg = load_config()
    assert cfg.agent["max_turns"] == 200
    assert cfg.llm["provider"] == DEFAULT_PROVIDER


def test_layering_user_over_project(paths):
    _write(project_config_path(),
           {"agent": {"max_turns": 50, "llm": {"provider": "glm"}}})
    _write(user_config_path(), {"agent": {"max_turns": 5}})
    cfg = load_config()
    assert cfg.agent["max_turns"] == 5            # 用户级覆盖项目级
    assert cfg.llm["provider"] == "glm"           # 项目级覆盖内置默认（agent.llm）
    assert cfg.compact["max_messages"] == 300     # 未覆盖 → 内置默认


def test_deep_merge_only_touches_present_keys():
    base = _builtin_defaults()
    merged = _deep_merge(base, {"agent": {"max_turns": 1}})
    assert merged["agent"]["max_turns"] == 1
    assert merged["agent"]["tool_call_limit"] == base["agent"]["tool_call_limit"]


def test_corrupted_file_falls_back(paths):
    project_config_path().parent.mkdir(parents=True, exist_ok=True)
    project_config_path().write_text("{ not json", encoding="utf-8")
    cfg = load_config()  # 不抛异常
    assert cfg.agent["max_turns"] == 200


def test_legacy_api_keys_ignored(paths):
    _write(user_config_path(),
           {"agent": {"llm": {"provider": "kimi"}}, "api_keys": {"qwen": "x"}})
    cfg = load_config()
    assert cfg.llm["provider"] == "kimi"
    assert "api_keys" not in cfg.data


# ── dotted get / set ──────────────────────────────────────

def test_dotted_get(paths):
    cfg = load_config()
    assert cfg.get("agent.max_turns") == 200
    with pytest.raises(KeyError):
        cfg.get("agent.nope")


def test_set_user_value_sparse(paths):
    set_user_value("agent.max_turns", "8")
    # 用户文件只存覆盖项（稀疏）
    raw = json.loads(user_config_path().read_text(encoding="utf-8"))
    assert raw == {"agent": {"max_turns": 8}}
    assert load_config().agent["max_turns"] == 8


def test_set_user_value_types(paths):
    set_user_value("agent.session_enabled", "off")    # bool
    set_user_value("compact.trigger_ratio", "0.4")    # float
    set_user_value("agent.llm.model", "gpt-4o")       # None → str
    set_user_value("agent.max_total_tokens", "2048")  # None → int
    cfg = load_config()
    assert cfg.agent["session_enabled"] is False
    assert cfg.compact["trigger_ratio"] == 0.4
    assert cfg.llm["model"] == "gpt-4o"
    assert cfg.agent["max_total_tokens"] == 2048


def test_set_unknown_or_section_raises(paths):
    with pytest.raises(ValueError):
        set_user_value("nope.nope", "1")
    with pytest.raises(ValueError):
        set_user_value("agent", "x")        # 分节不可直接设置


# ── 转运行期 dataclass ────────────────────────────────────

def test_to_dataclasses(paths):
    cfg = load_config()
    ac = cfg.to_agent_config()
    assert ac.max_turns == 200 and ac.tool_call_limit == 1000
    cc = cfg.to_compact_config()
    assert cc.recent_rounds == 5
    assert cc.round_trigger_ratio == 0.5
    pc = cfg.to_pool_config()
    assert pc.max_agents == 200 and pc.max_concurrent_runs == 50
    sc = cfg.to_scheduler_config()
    assert sc.max_concurrent_tasks == 4 and sc.notify is True


def test_scheduler_section_overridable(paths):
    _write(user_config_path(), {"scheduler": {"max_concurrent_tasks": 8, "notify": False}})
    sc = load_config().to_scheduler_config()
    assert sc.max_concurrent_tasks == 8
    assert sc.notify is False
    assert sc.task_timeout == 300.0  # 未覆盖 → dataclass 默认


# ── config init 模板 ──────────────────────────────────────

def test_write_project_template(paths):
    p = write_project_template()
    assert p == project_config_path()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["agent"]["max_turns"] == 200
    assert data["agent"]["llm"]["provider"] == "qwen"
    assert data["agent"]["workspace"] == ""   # 工作区默认空串（→ ~/.milu/workspace）
    assert set(data.keys()) == {
        "agent", "compact", "pool", "scheduler", "knowledge", "observability",
        "default_models", "security", "sandbox", "display", "lang",
    }
    assert data["lang"] == "zh"   # CLI 界面语言默认中文
    assert data["display"]["show_subagent_events"] is False
    assert data["knowledge"]["enabled"] is True   # 知识库默认开启（空库零开销）
    assert data["sandbox"]["backend"] == "subprocess"  # 代码执行后端默认子进程隔离
    assert data["sandbox"]["timeout"] == 60
    assert data["sandbox"]["ephemeral_workdir"] is False
    assert data["observability"]["enabled"] is True   # 运行追踪默认开启（仅本地落盘）
    assert data["observability"]["capture_content"] == "truncated"
