"""子进程后端：在独立子进程中执行代码/命令（跨平台、零依赖的真隔离升级，默认后端）。

相对 LocalBackend 的提升（这些正是 local 后端在进程内 exec 下无法堵住的固有缺口）：
  - **超时能真杀**：到点 kill 子进程（local 的线程级超时杀不掉死循环）；
  - **资源限额**：经 RLIMIT_AS/RLIMIT_CPU 限内存与 CPU 时间——**内存硬限额实际只在
    Linux 可靠生效**（macOS 内核往往不强制 RLIMIT_AS，Windows 无 preexec_fn 跳过）；
  - **状态隔离**：崩溃/无限分配不波及 milu 主进程；
  - **环境清洗**：只透传最小必要环境变量 → 执行的代码看不到 {PROVIDER}_API_KEY 等密钥
    （local 的进程内 exec 能直接读 os.environ 拿到全部密钥）；
  - **受保护路径防护**：python 执行前注入 guarded-open（见 _wrap_python），拦截对 .env /
    milu 源码 / 项目配置的读写——与 local 的 guarded_open 对齐，堵住「env 已清洗但 .env
    文件仍能被 open() 读出密钥」的旁路，使 subprocess 在密钥保护上严格优于 local。

工作目录：默认**继承当前工作目录**（与 local 一致，`git`/`pytest`/相对路径写文件照常在
项目目录生效）；`ephemeral_workdir=True` 时改为每次一次性临时目录（更强隔离）。

边界说明：仍是同一主机、同一用户、同一文件系统（绝对路径仍可访问全盘），网络未隔离；
guarded-open 是「随手访问」防护（与 local 同档，determined 代码仍可经 io.open / os.open
等底层入口绕过）。要文件系统/网络级硬隔离请用 docker 后端（计划于后续版本）。
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile

from milu.sandbox.base import ExecResult, SandboxBackend
from milu.sandbox._proc import communicate_or_kill
from milu.tools._selfguard import get_protected_paths
from milu.resources import current_workspace

# 透传给子进程的环境变量白名单（其余一律不传，避免泄漏密钥）。
# PATH 与系统必要项保留，否则解释器/shell 无法定位可执行文件。
_COMMON_ENV_KEYS = ("PATH", "PYTHONUTF8", "LANG", "LC_ALL", "LC_CTYPE", "TZ")
_POSIX_ENV_KEYS = ("HOME", "TMPDIR", "USER", "SHELL")
# Windows 下 shell/解释器启动强依赖这些（缺 SystemRoot/COMSPEC 子进程直接起不来）
_WINDOWS_ENV_KEYS = (
    "SystemRoot", "SYSTEMROOT", "COMSPEC", "TEMP", "TMP", "PATHEXT",
    "WINDIR", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
    "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
)

# guarded-open 前导：覆盖子进程的 builtins.open，拦截对 milu 受保护路径的读写
# （与 LocalBackend 的 guarded_open 对齐）。受保护清单由父进程算好、以字面量注入
# （子进程 -I 隔离、不共享父状态）。__PD/__PF 两个字面量在 _wrap_python 中拼到前面。
_GUARD_BODY = (
    "import builtins as __b, os as __o\n"
    "__ro = __b.open\n"
    "def __blk(p):\n"
    "    try:\n"
    "        n = __o.path.normcase(__o.path.normpath(__o.path.abspath(p)))\n"
    "    except Exception:\n"
    "        return False\n"
    "    for d in __PD:\n"
    "        if n == d or n.startswith(d + __o.sep):\n"
    "            return True\n"
    "    return n in __PF\n"
    "def __go(file, mode='r', *a, **k):\n"
    "    if isinstance(file, (str, bytes, __o.PathLike)):\n"
    "        f = __o.fsdecode(file) if isinstance(file, bytes) else str(file)\n"
    "        if __blk(f):\n"
    "            raise PermissionError('安全限制：禁止在 python_repl 中访问 milu 受保护路径: ' + f)\n"
    "    return __ro(file, mode, *a, **k)\n"
    "__b.open = __go\n"
)


def _wrap_python(code: str) -> str:
    """把用户代码包成「注入 guarded-open → 以 <repl> 名编译执行」的子进程程序文本。

    - 受保护清单（__PD 目录 / __PF 文件）由父进程 get_protected_paths() 算好后以
      repr 字面量注入；selfguard 关闭时为空列表 → 不拦截。
    - 用 exec(compile(code, '<repl>', 'exec')) 而非直接把代码追加在前导后面，
      使用户代码的异常 traceback 行号保持从 1 开始（与原始片段一致）。
    """
    dirs, files = get_protected_paths()
    return (
        "__PD = " + repr(dirs) + "\n"
        + "__PF = " + repr(files) + "\n"
        + _GUARD_BODY
        + "__USER = " + repr(code) + "\n"
        + "exec(compile(__USER, '<repl>', 'exec'), {'__name__': '__main__'})\n"
    )


class SubprocessBackend(SandboxBackend):
    """独立子进程执行（跨平台零依赖）。"""

    name = "subprocess"

    # ── 环境 / 工作目录 / 资源限额 ────────────────────────────
    def _make_env(self) -> dict:
        """构造清洗后的最小环境（隐藏 API Key 等敏感变量）。"""
        src = os.environ
        keys = _COMMON_ENV_KEYS + (
            _WINDOWS_ENV_KEYS if sys.platform == "win32" else _POSIX_ENV_KEYS
        )
        env = {k: src[k] for k in keys if k in src}
        env["PYTHONUTF8"] = "1"
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        env.setdefault("PYTHONNOUSERSITE", "1")   # 不加载 ~/.local 用户站点包
        return env

    def _make_workdir(self):
        """返回 (cwd, cleanup)。优先级：固定 workdir > 一次性临时目录 > agent 工作区 > 进程目录。

        - 配置了 workdir：复用该目录（不清理）；
        - ephemeral_workdir=True：每次一次性临时目录、执行后清理（更强隔离）；
        - 注入了 agent 工作区（current_workspace）：在工作区执行（与 file_write 落点一致）；
        - 否则：继承进程当前工作目录（裸调用/单测的 hermetic 行为）。
        """
        if self.config.workdir:
            os.makedirs(self.config.workdir, exist_ok=True)
            return self.config.workdir, (lambda: None)
        if getattr(self.config, "ephemeral_workdir", False):
            d = tempfile.mkdtemp(prefix="milu_sbx_")

            def cleanup():
                shutil.rmtree(d, ignore_errors=True)

            return d, cleanup
        ws = current_workspace()
        if ws:
            os.makedirs(ws, exist_ok=True)
            return ws, (lambda: None)
        return os.getcwd(), (lambda: None)

    def _make_preexec(self):
        """POSIX 资源限额（preexec_fn 在 fork 后、exec 前的子进程中执行）。

        Windows 不支持 preexec_fn（需 Job Object，本版本不做），返回 None。
        """
        if sys.platform == "win32":
            return None
        mem = self.config.memory_mb
        cpu = self.config.cpu_seconds
        if not mem and not cpu:
            return None

        def _set_limits():
            import resource
            if mem:
                nbytes = int(mem) * 1024 * 1024
                # RLIMIT_AS 限地址空间；过低会让解释器/numpy 启动失败，故 try 容错。
                # 注：Linux 可靠强制；macOS 内核往往不强制（调用成功但不真正限制）。
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
                except (ValueError, OSError):
                    pass
            if cpu:
                try:
                    resource.setrlimit(resource.RLIMIT_CPU, (int(cpu), int(cpu)))
                except (ValueError, OSError):
                    pass

        return _set_limits

    # ── 通用收发：喂 stdin、超时连根杀进程树、解码 ──────────────
    async def _communicate(self, proc, stdin_data: str | None, timeout: int) -> ExecResult:
        # 超时不再只 proc.kill()（仅杀直接子进程，会留下 npx/node 等孤儿孙进程且不关管道），
        # 改用 communicate_or_kill：到点连根杀整棵进程树再回收残余输出（见 _proc.py）。
        data = stdin_data.encode("utf-8") if stdin_data is not None else None
        stdout, stderr, timed_out = await communicate_or_kill(proc, data, timeout)
        return ExecResult(
            stdout=stdout.decode("utf-8", errors="replace") if stdout else "",
            stderr=stderr.decode("utf-8", errors="replace") if stderr else "",
            exit_code=None if timed_out else proc.returncode,
            timed_out=timed_out,
        )

    def _spawn_kwargs(self, cwd: str) -> dict:
        kwargs: dict = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "cwd": cwd,
            "env": self._make_env(),
        }
        # POSIX：子进程自成新会话/进程组，超时时可整组 killpg 连根清理孙进程。
        # Windows 无此语义，靠 taskkill /T 按 PID 树清理（见 _proc.terminate_process_tree）。
        if sys.platform != "win32":
            kwargs["start_new_session"] = True
        preexec = self._make_preexec()
        if preexec is not None:
            kwargs["preexec_fn"] = preexec
        return kwargs

    # ── 后端接口实现 ─────────────────────────────────────────
    async def run_python(self, code: str, *, timeout: int | None = None) -> ExecResult:
        t = timeout or self.config.timeout
        cwd, cleanup = self._make_workdir()
        try:
            # -X utf8：显式启用 UTF-8 模式（-I 含 -E 会忽略 PYTHONUTF8 环境变量，
            #          故走命令行 -X 而非依赖 env）；
            # -I：隔离模式（忽略 PYTHONPATH/用户站点、sys.path[0] 不含脚本目录，
            #     防止 import 到 milu 项目源码）；'-' 从 stdin 读代码（避免命令行长度/转义问题）。
            argv = [sys.executable, "-X", "utf8", "-I", "-"]
            program = _wrap_python(code)   # 注入 guarded-open + 以 <repl> 名编译执行
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv, stdin=asyncio.subprocess.PIPE, **self._spawn_kwargs(cwd)
                )
            except Exception as e:
                return ExecResult(stderr=f"子进程启动失败: {e}", exit_code=None)
            return await self._communicate(proc, program, t)
        finally:
            cleanup()

    async def run_shell(self, command: str, *, timeout: int | None = None) -> ExecResult:
        t = timeout or self.config.timeout
        cwd, cleanup = self._make_workdir()
        try:
            try:
                proc = await asyncio.create_subprocess_shell(
                    command, **self._spawn_kwargs(cwd)
                )
            except Exception as e:
                return ExecResult(stderr=f"命令启动失败: {e}", exit_code=None)
            return await self._communicate(proc, None, t)
        finally:
            cleanup()
