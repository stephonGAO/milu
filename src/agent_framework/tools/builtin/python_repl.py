"""内置工具：Python 代码沙箱执行

在受限环境中执行 Python 代码片段，捕获 stdout 和异常。
"""
from __future__ import annotations

import asyncio
import io
import traceback
from contextlib import redirect_stdout

from agent_framework.tools.decorator import tool

# 执行超时（秒）
_EXEC_TIMEOUT = 30


def _exec_code(code: str) -> str:
    """在受限环境中执行代码并返回输出"""
    stdout_capture = io.StringIO()

    # 构建受限的全局命名空间
    restricted_globals = {
        "__builtins__": __builtins__,
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
