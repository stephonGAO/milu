"""部署策略 multiuser（normal/strict）+ 文件工具工作区围栏 + 部署横幅测试。

strict 作为「比内置默认更低的安全基线层」：一键打包 sandbox=docker + workspace_jail=on
+ 断网，单项仍可显式覆盖（降级时横幅告警）。围栏让 file_read/file_write/doc_read/
image_read 只能在工作区内读写。
"""
from __future__ import annotations

import json

import pytest

from milu.config import deployment_lines, load_config
from milu.resources import _current_workspace, _current_workspace_jail
from milu.tools.builtin.file_tool import file_read, file_write


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """把项目级/用户级配置路径隔离到 tmp，返回写用户配置的函数。"""
    proj = tmp_path / "proj"
    home = tmp_path / "home"
    proj.mkdir()
    home.mkdir()
    monkeypatch.setenv("MILU_PROJECT_DIR", str(proj))
    monkeypatch.setenv("MILU_HOME", str(home))

    def write_user(cfg: dict):
        (home / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    return write_user


# ── strict 策略分层 ───────────────────────────────────────
class TestStrictProfile:
    def test_strict_applies_isolation_baseline(self, isolated_config):
        isolated_config({"multiuser": "strict"})
        c = load_config()
        assert c.multiuser == "strict"
        assert c.sandbox.get("backend") == "docker"
        assert c.agent.get("workspace_jail") is True
        assert c.sandbox.get("network") is False

    def test_strict_forces_over_config(self, isolated_config):
        # 即便配置显式 backend=subprocess / jail=false，strict 仍强制 docker + jail
        # （strict 作为最高层覆盖，不被全量模板/用户配置架空）
        isolated_config({"multiuser": "strict",
                         "sandbox": {"backend": "subprocess"},
                         "agent": {"workspace_jail": False}})
        c = load_config()
        assert c.sandbox.get("backend") == "docker"
        assert c.agent.get("workspace_jail") is True

    def test_strict_leaves_non_security_keys(self, isolated_config):
        # strict 只强制安全键；其余键（如 docker_image）仍按配置
        isolated_config({"multiuser": "strict",
                         "sandbox": {"docker_image": "python:3.13-slim"}})
        c = load_config()
        assert c.sandbox.get("backend") == "docker"
        assert c.sandbox.get("docker_image") == "python:3.13-slim"

    def test_normal_unaffected(self, isolated_config):
        isolated_config({"multiuser": "normal"})
        c = load_config()
        assert c.sandbox.get("backend") == "subprocess"
        assert c.agent.get("workspace_jail") is False

    def test_default_is_normal(self, isolated_config):
        isolated_config({})
        c = load_config()
        assert c.multiuser == "normal"


# ── 文件工具工作区围栏 ────────────────────────────────────
class TestWorkspaceJail:
    def _enter(self, ws: str):
        return _current_workspace.set(ws), _current_workspace_jail.set(True)

    async def test_in_bounds_allowed(self, tmp_path):
        wt, jt = self._enter(str(tmp_path))
        try:
            r = json.loads(await file_write(action="write", path="ok.txt", content="hi"))
            assert r["success"]
            assert (tmp_path / "ok.txt").read_text(encoding="utf-8") == "hi"
        finally:
            _current_workspace.reset(wt)
            _current_workspace_jail.reset(jt)

    async def test_absolute_escape_rejected(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        wt, jt = self._enter(str(ws))
        try:
            target = str(outside / "evil.txt")
            r = json.loads(await file_write(action="write", path=target, content="x"))
            assert not r["success"] and "围栏" in r["error"]
            assert not (outside / "evil.txt").exists()
            r = json.loads(await file_read(action="read", path=str(outside / "x.txt")))
            assert not r["success"] and "围栏" in r["error"]
        finally:
            _current_workspace.reset(wt)
            _current_workspace_jail.reset(jt)

    async def test_dotdot_traversal_rejected(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        wt, jt = self._enter(str(ws))
        try:
            r = json.loads(await file_write(action="write", path="../escape.txt", content="x"))
            assert not r["success"] and "围栏" in r["error"]
            assert not (tmp_path / "escape.txt").exists()
        finally:
            _current_workspace.reset(wt)
            _current_workspace_jail.reset(jt)

    async def test_jail_off_allows_outside(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        # 只注入 workspace、不开围栏 → 越界放行（普通模式行为）
        wt = _current_workspace.set(str(ws))
        try:
            r = json.loads(await file_write(action="write", path=str(outside / "fine.txt"), content="x"))
            assert r["success"]
            assert (outside / "fine.txt").exists()
        finally:
            _current_workspace.reset(wt)


# ── 部署横幅 ──────────────────────────────────────────────
class TestDeploymentLines:
    def test_normal_single_line(self):
        lines = deployment_lines("normal", "subprocess", False)
        assert len(lines) == 1
        assert "normal" in lines[0]

    def test_strict_shows_enforced(self):
        lines = deployment_lines("strict", "docker", True)
        assert any("强制" in ln for ln in lines)

    def test_strict_docker_down_warns(self):
        lines = deployment_lines("strict", "docker", True, docker_ok=False)
        assert any("Docker daemon 未运行" in ln for ln in lines)

    def test_strict_docker_ok_no_warn(self):
        lines = deployment_lines("strict", "docker", True, docker_ok=True)
        assert not any("⚠" in ln for ln in lines)
