"""本地后端：进程内执行 Python、宿主 Shell 执行命令（milu 的现状默认）。

隔离强度最低（与 milu 同进程 / 同主机、同权限），仅靠：
  - guarded open：拦截对 milu 受保护路径（源码/配置/.env）的读写
  - 危险命令黑名单（在 shell_command 工具层，后端无关）
  - 超时（python 为线程级软超时，杀不掉运行中的线程；shell 可真杀子进程）

工作区：shell 子进程在注入的 agent 工作区（current_workspace）中执行；但 python 是
**进程内** exec，与 milu 共享解释器，无法安全地临时切 CWD（并发会串扰），故其相对路径
写文件仍按进程目录——这是 local 后端的固有局限。需要 python 也落工作区请用默认的
subprocess 后端（子进程 CWD 可独立设为工作区）。file_write 工具不受影响（在工具层解析）。

适合可信环境 / 单人本地使用。需要真正隔离请改用 subprocess / docker 后端。
"""
from __future__ import annotations

import asyncio
import io
import os
import traceback
from contextlib import redirect_stdout

from milu.sandbox.base import ExecResult, SandboxBackend
from milu.tools._selfguard import is_protected_path
from milu.resources import current_workspace


def _make_guarded_open():
    """返回覆盖了受保护路径检查的 open（仅本地后端用，进程内执行的最后防线）。"""
    _real_open = open

    def _guarded_open(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            str_file = os.fsdecode(file) if isinstance(file, bytes) else str(file)
            if any(c in str(mode) for c in ("w", "a", "x")) and is_protected_path(str_file):
                raise PermissionError(
                    f"安全限制：禁止在 python_repl 中写入 milu 受保护路径: {str_file}"
                )
            if is_protected_path(str_file):
                raise PermissionError(
                    f"安全限制：禁止在 python_repl 中读取受保护文件: {str_file}"
                )
        return _real_open(file, mode, *args, **kwargs)

    return _guarded_open


def _exec_code(code: str) -> ExecResult:
    """在受限命名空间内 exec 代码，捕获 stdout 与异常 traceback。"""
    stdout_capture = io.StringIO()

    builtins_copy = dict(__builtins__) if isinstance(__builtins__, dict) else vars(__builtins__).copy()
    builtins_copy["open"] = _make_guarded_open()
    restricted_globals = {
        "__builtins__": builtins_copy,
        "__name__": "__main__",
    }

    try:
        with redirect_stdout(stdout_capture):
            exec(code, restricted_globals)  # noqa: S102
        return ExecResult(stdout=stdout_capture.getvalue(), exit_code=0)
    except Exception:
        return ExecResult(
            stdout=stdout_capture.getvalue(),
            stderr=traceback.format_exc(),
            exit_code=1,
        )


class LocalBackend(SandboxBackend):
    """进程内 / 宿主机执行（零依赖，默认后端）。"""

    name = "local"

    async def run_python(self, code: str, *, timeout: int | None = None) -> ExecResult:
        t = timeout or self.config.timeout
        loop = asyncio.get_event_loop()
        try:
            # 注意：run_in_executor 跑的是阻塞线程，wait_for 超时只让调用方返回、
            # 杀不掉线程（死循环会泄漏一个线程）。这是 local 后端的固有局限，
            # 需要可靠超时请用 subprocess / docker 后端。
            return await asyncio.wait_for(
                loop.run_in_executor(None, _exec_code, code),
                timeout=t,
            )
        except asyncio.TimeoutError:
            return ExecResult(exit_code=None, timed_out=True)

    async def run_shell(self, command: str, *, timeout: int | None = None) -> ExecResult:
        t = timeout or self.config.timeout
        # shell 子进程在 agent 工作区执行（与 file_write 落点一致）；进程内 python
        # 因共享解释器无法安全切 CWD，其相对路径仍按进程目录（已知局限，见模块说明）。
        cwd = self.config.workdir or current_workspace()
        if cwd:
            os.makedirs(cwd, exist_ok=True)
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        except Exception as e:
            return ExecResult(stderr=f"命令启动失败: {e}", exit_code=None)

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=t)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecResult(exit_code=None, timed_out=True)

        return ExecResult(
            stdout=stdout.decode("utf-8", errors="replace") if stdout else "",
            stderr=stderr.decode("utf-8", errors="replace") if stderr else "",
            exit_code=proc.returncode,
        )
