"""内置工具：Python 代码沙箱执行

在受限环境中执行 Python 代码片段，捕获 stdout 和异常。
"""
from __future__ import annotations

import asyncio
import io
import locale
import os
import sys
import traceback
from contextlib import redirect_stdout

from milu.tools.decorator import tool
from milu.tools._selfguard import is_protected_path

# 执行超时（秒）
_EXEC_TIMEOUT = 30


# ── Windows 中文编码修复 ────────────────────────────────────
# 在 Windows 中文系统上，locale.getencoding() 在解释器启动时
# 缓存为 cp936 (GBK)。subprocess._text_encoding() 使用该值决定
# 管道编解码方式，导致用户代码中的 subprocess 调用遇到 UTF-8 中文输出时
# 报 UnicodeDecodeError。
#
# 修复（两步缺一不可）：
#   1. os.environ["PYTHONUTF8"] = "1"
#      → 子进程 Python 启动时读取，启用 UTF-8 模式，stdout 使用 UTF-8
#   2. locale.getencoding monkey-patch → "utf-8"
#      → 父进程 subprocess 模块用 UTF-8 解码管道数据
if sys.platform == "win32" and not getattr(sys.flags, "utf8_mode", False):
    os.environ["PYTHONUTF8"] = "1"

    def _utf8_encoding(*_args, **_kwargs) -> str:
        return "utf-8"

    # 父进程 subprocess 解码管道用的编码函数：Python 3.11+ 是 locale.getencoding，
    # 3.10 及更早是 locale.getpreferredencoding。按存在性 patch——直接访问
    # locale.getencoding 在 3.10 上会 AttributeError，导致 import milu 崩溃
    # （Windows + Python 3.10 用户）。
    if hasattr(locale, "getencoding"):
        locale.getencoding = _utf8_encoding
    else:
        locale.getpreferredencoding = _utf8_encoding

    # 同步修复当前进程的 stdout/stderr
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _make_guarded_open():
    """返回一个覆盖了自我保护检查的 open 函数。"""
    _real_open = open

    def _guarded_open(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            str_file = os.fsdecode(file) if isinstance(file, bytes) else str(file)
            # 写保护：milu 自身源码和配置
            if any(c in str(mode) for c in ("w", "a", "x")) and is_protected_path(str_file):
                raise PermissionError(
                    f"安全限制：禁止在 python_repl 中写入 milu 受保护路径: {str_file}"
                )
            # 读保护：与写保护共用同一套受保护路径（含 .env 等凭据文件）
            if is_protected_path(str_file):
                raise PermissionError(
                    f"安全限制：禁止在 python_repl 中读取受保护文件: {str_file}"
                )
        return _real_open(file, mode, *args, **kwargs)

    return _guarded_open


def _exec_code(code: str) -> str:
    """在受限环境中执行代码并返回输出"""
    stdout_capture = io.StringIO()

    # 构建受限的全局命名空间，覆盖 open 以拦截对保护路径的写操作
    builtins_copy = dict(__builtins__) if isinstance(__builtins__, dict) else vars(__builtins__).copy()
    builtins_copy["open"] = _make_guarded_open()
    restricted_globals = {
        "__builtins__": builtins_copy,
        "__name__": "__main__",
    }

    try:
        with redirect_stdout(stdout_capture):
            exec(code, restricted_globals)  # noqa: S102
        output = stdout_capture.getvalue()
        return output if output else "(无输出)"
    except Exception:
        tb = traceback.format_exc()
        stdout_output = stdout_capture.getvalue()
        if stdout_output:
            return f"{stdout_output}\n错误:\n{tb}"
        return f"错误:\n{tb}"


@tool(name="python_repl", description="执行 Python 代码片段并返回输出，支持标准库")
async def python_repl(code: str) -> str:
    """
    在沙箱中执行 Python 代码。支持 import 标准库、print 输出。
    异常会被捕获并返回 traceback 信息。

    :param code: 要执行的 Python 代码
    """
    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _exec_code, code),
            timeout=_EXEC_TIMEOUT,
        )
        return result
    except asyncio.TimeoutError:
        return f"错误: 代码执行超时（{_EXEC_TIMEOUT}秒）"
    except Exception as e:
        return f"错误: {e}"
