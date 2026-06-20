"""沙箱执行后端测试：后端分发、LocalBackend 行为对齐、SubprocessBackend 隔离、
ContextVar 注入与工具委派。

Docker 后端本版本未实现（build_backend 对 "docker" 抛错），相关隔离测试待后续版本。
"""
from __future__ import annotations

import os
import shutil
import sys

import pytest

from milu.resources import _current_workspace

from milu.sandbox import (
    ExecResult,
    LocalBackend,
    SandboxBackend,
    SandboxConfig,
    SubprocessBackend,
    _current_sandbox,
    build_backend,
    current_backend,
    default_backend,
)


# ── 后端工厂分发 ──────────────────────────────────────────
class TestBuildBackend:
    def test_none_returns_default_subprocess(self):
        # SandboxConfig 默认后端 = subprocess，故 build_backend(None) 走子进程后端
        assert isinstance(build_backend(None), SubprocessBackend)

    def test_str_local(self):
        assert isinstance(build_backend("local"), LocalBackend)

    def test_str_subprocess(self):
        assert isinstance(build_backend("subprocess"), SubprocessBackend)

    def test_config_dispatch(self):
        b = build_backend(SandboxConfig(backend="subprocess", memory_mb=256))
        assert isinstance(b, SubprocessBackend)
        assert b.config.memory_mb == 256

    def test_instance_passthrough(self):
        inst = LocalBackend(SandboxConfig())
        assert build_backend(inst) is inst

    def test_docker_backend_constructs(self):
        from milu.sandbox import DockerBackend
        assert isinstance(build_backend("docker"), DockerBackend)

    def test_unknown_backend(self):
        with pytest.raises(ValueError):
            build_backend("nonsense")

    def test_config_from_mapping_ignores_unknown_keys(self):
        cfg = SandboxConfig.from_mapping({"backend": "subprocess", "//": "注释", "timeout": 5})
        assert cfg.backend == "subprocess"
        assert cfg.timeout == 5


# ── current_backend / 默认后端 ────────────────────────────
class TestCurrentBackend:
    def test_default_is_local_singleton(self):
        assert isinstance(default_backend(), LocalBackend)
        assert default_backend() is default_backend()

    def test_current_falls_back_to_default(self):
        # 未注入时回退默认 LocalBackend
        assert _current_sandbox.get() is None
        assert isinstance(current_backend(), LocalBackend)

    def test_injected_backend_takes_priority(self):
        sp = SubprocessBackend(SandboxConfig(backend="subprocess"))
        token = _current_sandbox.set(sp)
        try:
            assert current_backend() is sp
        finally:
            _current_sandbox.reset(token)


# ── LocalBackend 行为（与改造前对齐）──────────────────────
class TestLocalBackend:
    async def test_python_success(self):
        b = LocalBackend(SandboxConfig())
        res = await b.run_python("print(2 + 3)")
        assert res.exit_code == 0
        assert "5" in res.stdout
        assert res.timed_out is False

    async def test_python_exception_to_stderr(self):
        b = LocalBackend(SandboxConfig())
        res = await b.run_python("1 / 0")
        assert res.exit_code == 1
        assert "ZeroDivisionError" in res.stderr

    async def test_python_guarded_open_blocks_protected(self, monkeypatch):
        # 强制把目标路径判定为受保护，验证 guarded open 拦截读取
        import milu.sandbox.local as local_mod
        monkeypatch.setattr(local_mod, "is_protected_path", lambda p: True)
        b = LocalBackend(SandboxConfig())
        res = await b.run_python("open('whatever.txt').read()")
        assert res.exit_code == 1
        assert "受保护" in res.stderr or "PermissionError" in res.stderr

    async def test_shell_echo(self):
        b = LocalBackend(SandboxConfig())
        res = await b.run_shell("echo hello_local")
        assert "hello_local" in res.stdout
        assert res.exit_code == 0

    async def test_shell_timeout(self):
        b = LocalBackend(SandboxConfig())
        cmd = "ping -n 30 localhost" if sys.platform == "win32" else "sleep 30"
        res = await b.run_shell(cmd, timeout=1)
        assert res.timed_out is True


# ── SubprocessBackend 隔离（本轮核心）──────────────────────
class TestSubprocessBackend:
    async def test_python_success(self):
        b = SubprocessBackend(SandboxConfig(backend="subprocess"))
        res = await b.run_python("print(40 + 2)")
        assert res.exit_code == 0
        assert "42" in res.stdout

    async def test_timeout_really_kills(self):
        """死循环到点被真杀（local 后端做不到）。"""
        b = SubprocessBackend(SandboxConfig(backend="subprocess"))
        res = await b.run_python("while True:\n    pass", timeout=2)
        assert res.timed_out is True

    async def test_env_scrubbed_hides_secrets(self, monkeypatch):
        """子进程看不到父进程的敏感环境变量（如 *_API_KEY）。"""
        monkeypatch.setenv("MILU_SBX_SECRET_KEY", "TOP_SECRET")
        b = SubprocessBackend(SandboxConfig(backend="subprocess"))
        res = await b.run_python(
            "import os; print(os.environ.get('MILU_SBX_SECRET_KEY', 'ABSENT'))"
        )
        assert "TOP_SECRET" not in res.stdout
        assert "ABSENT" in res.stdout

    async def test_nonzero_exit_code(self):
        b = SubprocessBackend(SandboxConfig(backend="subprocess"))
        res = await b.run_python("raise SystemExit(7)")
        assert res.exit_code == 7

    async def test_shell_runs(self):
        b = SubprocessBackend(SandboxConfig(backend="subprocess"))
        res = await b.run_shell("echo hello_subproc")
        assert "hello_subproc" in res.stdout
        assert res.exit_code == 0

    async def test_cannot_import_milu_project_source(self):
        """隔离模式（-I）下 sys.path 不含项目目录，import 不到 milu 自身源码。

        说明：子进程用的是同一解释器的 venv，若 milu 已 pip install 到 venv 则仍可
        import（这是预期——隔离的是「项目工作目录注入」，不是已安装的包）。此处只验证
        当前工作目录不在子进程 sys.path 中。
        """
        b = SubprocessBackend(SandboxConfig(backend="subprocess"))
        res = await b.run_python("import sys; print('' in sys.path or '.' in sys.path)")
        assert "False" in res.stdout

    @pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT 仅 POSIX 支持")
    async def test_memory_limit_posix(self):
        """POSIX 下内存限额生效：超额分配触发 MemoryError / 非零退出。"""
        b = SubprocessBackend(SandboxConfig(backend="subprocess", memory_mb=256))
        # 尝试分配约 1GB，应被 RLIMIT_AS 拦下
        res = await b.run_python("x = bytearray(1024 * 1024 * 1024)")
        assert res.exit_code != 0 or "MemoryError" in res.stderr

    async def test_default_workdir_inherits_cwd(self):
        """默认工作目录 = 当前目录（与 local 一致，编码工作流照常）。"""
        b = SubprocessBackend(SandboxConfig(backend="subprocess"))
        res = await b.run_python("import os; print(os.getcwd())")
        assert os.path.normcase(res.stdout.strip()) == os.path.normcase(os.getcwd())

    async def test_ephemeral_workdir_uses_tempdir(self):
        """ephemeral_workdir=True → 每次在一次性临时目录执行（不落当前目录）。"""
        b = SubprocessBackend(SandboxConfig(backend="subprocess", ephemeral_workdir=True))
        res = await b.run_python("import os; print(os.getcwd())")
        assert "milu_sbx_" in res.stdout
        assert os.path.normcase(res.stdout.strip()) != os.path.normcase(os.getcwd())

    async def test_guarded_open_blocks_protected_read(self, monkeypatch, tmp_path):
        """guarded-open 拦截读受保护文件（堵住「env 已清洗但 .env 仍可 open 读密钥」旁路）。"""
        secret = tmp_path / "secret.env"
        secret.write_text("QWEN_API_KEY=LEAKED", encoding="utf-8")
        norm = os.path.normcase(os.path.normpath(str(secret)))
        import milu.sandbox.process as proc_mod
        monkeypatch.setattr(proc_mod, "get_protected_paths", lambda: ([], [norm]))
        b = SubprocessBackend(SandboxConfig(backend="subprocess"))
        res = await b.run_python(f"print(open(r'''{secret}''').read())")
        assert res.exit_code != 0
        assert "受保护" in res.stderr or "PermissionError" in res.stderr
        assert "LEAKED" not in res.stdout

    async def test_guarded_open_blocks_protected_write(self, monkeypatch, tmp_path):
        """guarded-open 拦截写受保护目录（防止改 milu 源码）。"""
        guarded_dir = tmp_path / "pkg"
        guarded_dir.mkdir()
        norm_dir = os.path.normcase(os.path.normpath(str(guarded_dir)))
        target = guarded_dir / "x.py"
        import milu.sandbox.process as proc_mod
        monkeypatch.setattr(proc_mod, "get_protected_paths", lambda: ([norm_dir], []))
        b = SubprocessBackend(SandboxConfig(backend="subprocess"))
        res = await b.run_python(f"open(r'''{target}''', 'w').write('x')")
        assert res.exit_code != 0
        assert "受保护" in res.stderr or "PermissionError" in res.stderr
        assert not target.exists()

    async def test_guard_disabled_allows_open(self, monkeypatch, tmp_path):
        """selfguard 关闭（清单为空）时不拦截，正常读写。"""
        f = tmp_path / "ok.txt"
        f.write_text("hello", encoding="utf-8")
        import milu.sandbox.process as proc_mod
        monkeypatch.setattr(proc_mod, "get_protected_paths", lambda: ([], []))
        b = SubprocessBackend(SandboxConfig(backend="subprocess"))
        res = await b.run_python(f"print(open(r'''{f}''').read())")
        assert res.exit_code == 0
        assert "hello" in res.stdout

    async def test_traceback_line_number_preserved(self):
        """以 <repl> 名编译用户代码 → 异常行号从 1 开始（不受 guarded-open 前导偏移）。"""
        b = SubprocessBackend(SandboxConfig(backend="subprocess"))
        res = await b.run_python("x = 1\nraise ValueError('boom')\n")
        assert res.exit_code != 0
        assert "line 2" in res.stderr
        assert "ValueError" in res.stderr


# ── DockerBackend（真隔离；hermetic 部分不需要本机 docker）────────
class TestDockerBackend:
    def _backend(self, **kw):
        from milu.sandbox import DockerBackend
        return DockerBackend(SandboxConfig(backend="docker", **kw))

    def test_base_args_isolation_flags(self):
        args = self._backend(memory_mb=512)._base_args("milu_sbx_test")
        assert "--rm" in args and "--read-only" in args
        assert args[args.index("--network") + 1] == "none"     # 默认断网
        assert args[args.index("--memory") + 1] == "512m"
        assert "--pids-limit" in args
        assert args[args.index("-w") + 1] == "/work"
        assert args[args.index("--name") + 1] == "milu_sbx_test"

    def test_base_args_network_enabled(self):
        args = self._backend(network=True)._base_args("n")
        assert args[args.index("--network") + 1] == "bridge"

    def test_base_args_no_workspace_uses_tmpfs(self):
        # 未注入工作区 → /work 用 tmpfs（全隔离，无宿主挂载）
        assert _current_workspace.get() is None
        args = self._backend()._base_args("n")
        assert any(a.startswith("/work:") for a in args)        # --tmpfs /work:...
        assert not any(a.endswith(":/work") for a in args)      # 无 -v 挂载

    def test_base_args_mounts_workspace(self, tmp_path):
        token = _current_workspace.set(str(tmp_path))
        try:
            args = self._backend()._base_args("n")
        finally:
            _current_workspace.reset(token)
        mount = args[args.index("-v") + 1]
        assert mount.endswith(":/work")
        assert os.path.normcase(mount[: -len(":/work")]) == os.path.normcase(str(tmp_path))

    def test_base_args_cpus_and_user(self):
        args = self._backend(docker_cpus=2.0, docker_user="1000:1000")._base_args("n")
        assert args[args.index("--cpus") + 1] == "2.0"
        assert args[args.index("--user") + 1] == "1000:1000"

    async def test_unavailable_returns_friendly_error(self, monkeypatch):
        """docker 不可用 → run_python 返回中文友好错（不抛、不崩对话）。"""
        b = self._backend()

        async def _fake_prepare():
            return "Docker 未安装或不在 PATH：……"
        monkeypatch.setattr(b, "_prepare", _fake_prepare)
        res = await b.run_python("print(1)")
        assert res.exit_code is None and "Docker" in res.stderr

    async def test_probe_binary_missing(self, monkeypatch):
        """探测：docker 可执行文件不存在 → 未安装提示。"""
        b = self._backend()

        async def _fake_probe(argv, *, timeout):
            return None, "", "docker 可执行文件不存在"
        monkeypatch.setattr(b, "_probe_exec", _fake_probe)
        msg = await b._do_prepare()
        assert msg and "未安装" in msg

    @pytest.mark.skipif(shutil.which("docker") is None, reason="本机未安装 docker CLI")
    async def test_real_docker_run_if_available(self):
        """端到端：docker 可用时容器内执行；daemon 未启动则跳过（优雅降级）。"""
        b = self._backend(timeout=180)
        err = await b._prepare()
        if err:
            pytest.skip(f"docker 不可用，跳过真跑：{err[:80]}")
        res = await b.run_python("print(6 * 7)")
        assert "42" in res.stdout and res.exit_code == 0


# ── 工具委派：python_repl / shell_command 走注入的后端 ──────
class TestToolDelegation:
    async def test_python_repl_uses_injected_subprocess(self, monkeypatch):
        from milu.tools.builtin.python_repl import python_repl

        monkeypatch.setenv("MILU_SBX_SECRET_KEY2", "LEAK")
        sp = SubprocessBackend(SandboxConfig(backend="subprocess"))
        token = _current_sandbox.set(sp)
        try:
            result = await python_repl(
                "import os; print(os.environ.get('MILU_SBX_SECRET_KEY2', 'CLEAN'))"
            )
        finally:
            _current_sandbox.reset(token)
        # 经子进程后端 → 密钥被清洗
        assert "LEAK" not in result
        assert "CLEAN" in result

    async def test_python_repl_default_local(self):
        """未注入后端时回退默认 LocalBackend，照常返回输出。"""
        from milu.tools.builtin.python_repl import python_repl

        result = await python_repl("print('hi_default')")
        assert "hi_default" in result

    async def test_shell_blacklist_applies_before_backend(self):
        """危险命令在工具层拦截，先于后端执行（无论注入哪个后端）。"""
        from milu.tools.builtin.shell_command import shell_command

        sp = SubprocessBackend(SandboxConfig(backend="subprocess"))
        token = _current_sandbox.set(sp)
        try:
            result = await shell_command(command="sudo rm something")
        finally:
            _current_sandbox.reset(token)
        assert "拦截" in result and "危险" in result


# ── 后端契约（ExecResult / aclose）─────────────────────────
class TestBackendContract:
    async def test_aclose_is_noop_safe(self):
        for b in (LocalBackend(SandboxConfig()), SubprocessBackend(SandboxConfig())):
            await b.aclose()
            await b.aclose()  # 重复调用安全

    def test_subclass_of_base(self):
        assert issubclass(LocalBackend, SandboxBackend)
        assert issubclass(SubprocessBackend, SandboxBackend)

    def test_exec_result_defaults(self):
        r = ExecResult()
        assert r.stdout == "" and r.stderr == "" and r.exit_code == 0 and r.timed_out is False
