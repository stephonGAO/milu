"""Agent 工作区（workspace）测试：路径解析、file 工具相对/绝对、沙箱 CWD、提示词变量。

工作区让 agent 产出文件默认落在用户目录下的工作区（~/.milu/workspace），不污染
milu 启动目录/项目；绝对路径不受影响；未注入时回退进程 CWD（hermetic）。
"""
from __future__ import annotations

import json
import os

from milu.resources import (
    _current_workspace,
    current_workspace,
    user_data_dir,
    workspace_dir,
)
from milu.tools.builtin.file_tool import file_read, file_write


# ── 工作区路径解析 ────────────────────────────────────────
class TestWorkspaceDir:
    def test_default_under_user_data_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MILU_HOME", str(tmp_path))
        monkeypatch.delenv("MILU_WORKSPACE", raising=False)
        assert workspace_dir() == user_data_dir() / "workspace"

    def test_per_user_subdir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MILU_HOME", str(tmp_path))
        monkeypatch.delenv("MILU_WORKSPACE", raising=False)
        assert workspace_dir(user_id="alice") == user_data_dir() / "workspace" / "alice"
        # default 用户不加子目录
        assert workspace_dir(user_id="default") == user_data_dir() / "workspace"

    def test_unsafe_user_id_sanitized(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MILU_HOME", str(tmp_path))
        monkeypatch.delenv("MILU_WORKSPACE", raising=False)
        root = user_data_dir() / "workspace"
        # 路径分隔符被替换 → 始终是 root 的直接子目录，无法穿越
        p = workspace_dir(user_id="../../etc")
        assert os.sep not in p.name and "/" not in p.name
        assert p.parent == root
        # 纯点号组件（. / ..）回退 default，杜绝穿越到工作区父目录
        assert workspace_dir(user_id="..") == root / "default"
        assert workspace_dir(user_id=".") == root / "default"

    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MILU_WORKSPACE", str(tmp_path / "envws"))
        assert workspace_dir() == (tmp_path / "envws")

    def test_explicit_base_wins_over_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MILU_WORKSPACE", str(tmp_path / "envws"))
        assert workspace_dir(base=str(tmp_path / "cfgws")) == (tmp_path / "cfgws")


# ── file 工具相对/绝对路径 ────────────────────────────────
class TestFileToolWorkspace:
    async def test_relative_write_lands_in_workspace(self, tmp_path):
        token = _current_workspace.set(str(tmp_path))
        try:
            r = json.loads(await file_write(action="write", path="report.md", content="hi-ws"))
            assert r["success"]
            assert os.path.normcase(r["path"]) == os.path.normcase(str(tmp_path / "report.md"))
            assert (tmp_path / "report.md").read_text(encoding="utf-8") == "hi-ws"
        finally:
            _current_workspace.reset(token)

    async def test_relative_read_back_consistent(self, tmp_path):
        token = _current_workspace.set(str(tmp_path))
        try:
            await file_write(action="write", path="a.txt", content="consistent")
            r = json.loads(await file_read(action="read", path="a.txt"))
            assert "consistent" in r["content"]
        finally:
            _current_workspace.reset(token)

    async def test_absolute_path_not_redirected(self, tmp_path):
        ws = tmp_path / "ws"
        other = tmp_path / "other"
        other.mkdir()
        token = _current_workspace.set(str(ws))
        try:
            abs_target = str(other / "abs.txt")
            r = json.loads(await file_write(action="write", path=abs_target, content="x"))
            assert os.path.normcase(r["path"]) == os.path.normcase(abs_target)
            assert (other / "abs.txt").exists()
            assert not (ws / "abs.txt").exists()
        finally:
            _current_workspace.reset(token)

    async def test_no_workspace_falls_back_to_cwd(self, tmp_path, monkeypatch):
        # 未注入工作区 → 相对路径按进程 CWD（hermetic，行为同改造前）
        assert current_workspace() is None
        monkeypatch.chdir(tmp_path)
        r = json.loads(await file_write(action="write", path="local.txt", content="y"))
        assert (tmp_path / "local.txt").exists()
        assert os.path.normcase(
            os.path.dirname(os.path.abspath(r["path"]))
        ) == os.path.normcase(str(tmp_path))


# ── 沙箱 CWD 跟随工作区 ───────────────────────────────────
class TestSandboxWorkspace:
    async def test_subprocess_cwd_is_workspace(self, tmp_path):
        from milu.sandbox import SubprocessBackend, SandboxConfig
        token = _current_workspace.set(str(tmp_path))
        try:
            b = SubprocessBackend(SandboxConfig(backend="subprocess"))
            res = await b.run_python("import os; print(os.getcwd())")
            assert os.path.normcase(res.stdout.strip()) == os.path.normcase(str(tmp_path))
        finally:
            _current_workspace.reset(token)

    async def test_explicit_workdir_overrides_workspace(self, tmp_path):
        from milu.sandbox import SubprocessBackend, SandboxConfig
        fixed = tmp_path / "fixed"
        ws = tmp_path / "ws"
        token = _current_workspace.set(str(ws))
        try:
            b = SubprocessBackend(SandboxConfig(backend="subprocess", workdir=str(fixed)))
            res = await b.run_python("import os; print(os.getcwd())")
            assert os.path.normcase(res.stdout.strip()) == os.path.normcase(str(fixed))
        finally:
            _current_workspace.reset(token)


# ── 提示词变量 {{workspace}} ──────────────────────────────
class TestPromptWorkspaceVariable:
    def test_main_prompt_interpolates_workspace(self):
        from milu.prompts.builder import PromptBuilder
        from milu.resources import builtin_prompts_dir
        pb = PromptBuilder(str(builtin_prompts_dir("main")))
        out = pb.build(
            workspace="WS_MARKER", current_date="2026-06-20",
            platform="Windows", cwd="C:/x",
        )
        assert "WS_MARKER" in out
