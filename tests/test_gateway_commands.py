"""网关 / 命令单测——命令分层执行（commands.py）+ 管理员判定/拦截（runner.py）。

不依赖真实 LLM/平台：用同形假 Agent 驱动 `run_command`，用假 Pool 驱动 `AgentRunner`
的命令拦截与权限判定。
"""
from __future__ import annotations

import pytest

from milu.channels import AgentRunner, InboundMessage, OutboundMessage
from milu.channels.commands import is_command, run_command
from milu.channels.runner import _normalize_admins


# ── 假 Agent（仅实现命令用到的属性/方法）──────────────────────────────
class _Mode:
    def __init__(self, v: str):
        self.value = v


class _FakeAgent:
    def __init__(self):
        self.mode = _Mode("auto")
        self._memory_path = None
        self.session = None
        self.reset_called = False
        self.set_mode_arg = None
        self.new_called = False

    async def reset(self):
        self.reset_called = True

    def set_mode(self, m: str):
        if m not in ("talk", "manual", "auto", "superwork"):
            raise ValueError(m)
        self.set_mode_arg = m
        self.mode = _Mode(m)

    def new_session(self):
        self.new_called = True


# ── is_command ────────────────────────────────────────────────────────
def test_is_command():
    assert is_command("/help") is True
    assert is_command("  /reset") is True
    assert is_command("你好") is False
    assert is_command("") is False


# ── 信息类命令：所有人可用 ─────────────────────────────────────────────
async def test_help_lists_info_and_admin():
    out = await run_command(_FakeAgent(), "/help")
    assert "/whoami" in out and "/reset" in out
    assert "/new" in out and "/sessions" in out      # 已下放到信息类
    assert "管理员命令" in out and "/prompt" in out   # 敏感类仍在


async def test_whoami_shows_identity_and_role():
    a = _FakeAgent()
    normal = await run_command(a, "/whoami", is_admin=False, identity="feishu:ou_1")
    assert "feishu:ou_1" in normal and "普通用户" in normal
    admin = await run_command(a, "/whoami", is_admin=True, identity="telegram:42")
    assert "telegram:42" in admin and "管理员" in admin


async def test_mode_view_is_open_to_all():
    out = await run_command(_FakeAgent(), "/mode", is_admin=False)
    assert "当前模式：auto" in out
    assert "talk" in out and "superwork" in out


async def test_reset_open_to_all():
    a = _FakeAgent()
    out = await run_command(a, "/reset", is_admin=False)
    assert a.reset_called is True
    assert "重置" in out


async def test_memory_disabled_message():
    out = await run_command(_FakeAgent(), "/memory")
    assert "未启用" in out


async def test_unknown_command():
    out = await run_command(_FakeAgent(), "/nope")
    assert "未知命令" in out


# ── 敏感类命令：仅管理员 ───────────────────────────────────────────────
async def test_mode_switch_denied_for_non_admin():
    a = _FakeAgent()
    out = await run_command(a, "/mode superwork", is_admin=False)
    assert "管理员权限" in out
    assert a.set_mode_arg is None          # 未真正切换
    assert a.mode.value == "auto"


async def test_mode_switch_allowed_for_admin():
    a = _FakeAgent()
    out = await run_command(a, "/mode superwork", is_admin=True)
    assert a.set_mode_arg == "superwork"
    assert "已切换" in out


async def test_mode_switch_invalid_for_admin():
    out = await run_command(_FakeAgent(), "/mode bogus", is_admin=True)
    assert "无效模式" in out


async def test_admin_only_commands_denied_for_non_admin():
    a = _FakeAgent()
    for cmd in ("/save", "/prompt", "/load x"):
        out = await run_command(a, cmd, is_admin=False)
        assert "管理员权限" in out, cmd


async def test_new_open_to_all():
    # /new 自助操作，所有人可用（非管理员也能新建自己的会话）
    a = _FakeAgent()
    out = await run_command(a, "/new", is_admin=False)
    assert a.new_called is True
    assert "管理员权限" not in out


async def test_sessions_open_to_all_and_self_scoped():
    # /sessions 对所有人开放，但只列自己名下会话（非管理员不被拒绝）
    a = _FakeAgent()
    out = await run_command(a, "/sessions", is_admin=False, identity="feishu:ou_x")
    assert "管理员权限" not in out


def test_own_session_prefix():
    from milu.channels.commands import _own_session_prefix
    # "channel:user" → sanitize → "channel_user__"；"__" 边界防前缀粘连
    assert _own_session_prefix("feishu:ou_a") == "feishu_ou_a__"
    assert _own_session_prefix("") == ""
    p = _own_session_prefix("feishu:ou_a")
    assert "feishu_ou_a__feishu_ou_a".startswith(p)       # 自己的会话命中
    assert not "feishu_ou_ab__feishu_ou_ab".startswith(p)  # 他人(前缀粘连)不命中


async def test_long_output_is_clipped():
    from milu.channels import commands as cmds
    a = _FakeAgent()
    # 伪造超长技能列表，确认整体被截断到上限内
    a.skill_registry = type("R", (), {
        "list_names": lambda self: ["s"] * 2000,
        "get": lambda self, n: type("C", (), {"description": "x" * 50})(),
    })()
    a.tools = None
    out = await run_command(a, "/skills")
    assert len(out) <= cmds._MAX_LEN
    assert "已截断" in out


# ── 管理员白名单规整 + AgentRunner 判定 ────────────────────────────────
def test_normalize_admins():
    assert _normalize_admins(None) == set()
    assert _normalize_admins("") == set()
    assert _normalize_admins("a:1, b:2 ,，c") == {"a:1", "b:2", "c"}
    assert _normalize_admins(["x", " y "]) == {"x", "y"}


class _FakeHandle:
    def __init__(self, agent):
        self.agent = agent


class _FakeAcquire:
    def __init__(self, h):
        self._h = h

    async def __aenter__(self):
        return self._h

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, agent):
        self._agent = agent
        self.run_called = False

    def acquire(self, user_id, session_id):
        return _FakeAcquire(_FakeHandle(self._agent))


def _inbound(text, user="ou_1", channel="feishu"):
    return InboundMessage(channel=channel, user_id=user, text=text)


async def test_runner_is_admin_matches_both_forms():
    r = AgentRunner(_FakePool(_FakeAgent()), admins={"feishu:ou_1", "bareuser"})
    assert r._is_admin(_inbound("x", user="ou_1", channel="feishu")) is True   # 渠道:用户
    assert r._is_admin(_inbound("x", user="bareuser", channel="telegram")) is True  # 裸用户跨渠道
    assert r._is_admin(_inbound("x", user="ou_2", channel="feishu")) is False
    # 无白名单 → 任何人非管理员
    r2 = AgentRunner(_FakePool(_FakeAgent()), admins=set())
    assert r2._is_admin(_inbound("x")) is False


async def test_runner_intercepts_command_for_non_admin():
    a = _FakeAgent()
    r = AgentRunner(_FakePool(a), commands=True, admins=set())
    out = await r(_inbound("/mode superwork"))
    assert isinstance(out, OutboundMessage)
    assert "管理员权限" in out.text
    assert a.set_mode_arg is None          # 命令未提权


async def test_runner_intercepts_command_for_admin():
    a = _FakeAgent()
    r = AgentRunner(_FakePool(a), commands=True, admins={"feishu:ou_1"})
    out = await r(_inbound("/mode talk"))
    assert a.set_mode_arg == "talk"
    assert "已切换" in out.text


async def test_runner_commands_disabled_by_default_passthrough():
    # 默认 commands=False → / 消息不拦截，照常喂给 Agent.run
    # （这里假 Agent 无 run 方法 → 触发异常兜底，证明走的是正常 run 路径而非命令路径）
    a = _FakeAgent()
    r = AgentRunner(_FakePool(a), fallback_text="FB")   # 不传 commands → 默认关
    out = await r(_inbound("/help"))
    assert out.text == "FB"


async def test_runner_env_admins(monkeypatch):
    monkeypatch.setenv("GATEWAY_ADMINS", "feishu:ou_9, telegram:7")
    r = AgentRunner(_FakePool(_FakeAgent()))    # admins=None → 读环境变量
    assert r.admins == {"feishu:ou_9", "telegram:7"}
    assert r._is_admin(_inbound("x", user="ou_9")) is True
