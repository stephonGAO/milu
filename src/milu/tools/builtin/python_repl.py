"""内置工具：Python 代码执行

把代码交给当前生效的沙箱后端执行（local 进程内 / subprocess 子进程 / docker 容器），
捕获 stdout 与异常。后端经 Agent.run() 入口的 ContextVar 注入；未注入时回退到模块级
默认 LocalBackend（保证脱离 Agent 直接调用/单测仍可运行）。隔离强度由部署的
sandbox.backend 配置决定，见 milu.sandbox 与 docs/沙箱执行方案设计.md。
"""
from __future__ import annotations

from milu.tools.decorator import tool
from milu.sandbox import ExecResult, current_backend


def _format_python_result(res: ExecResult, timeout: int) -> str:
    """把 ExecResult 格式化为回填 LLM 的文本（保持与改造前一致的输出形态）。"""
    if res.timed_out:
        return f"错误: 代码执行超时（{timeout}秒）"
    out = res.stdout or ""
    if (res.exit_code not in (0, None)) or res.stderr:
        err = res.stderr or ""
        return f"{out}\n错误:\n{err}" if out else f"错误:\n{err}"
    return out if out else "(无输出)"


@tool(name="python_repl", description="执行 Python 代码片段并返回输出，支持标准库")
async def python_repl(code: str) -> str:
    """
    执行 Python 代码。支持 import 标准库、print 输出。
    异常会被捕获并返回 traceback 信息。

    :param code: 要执行的 Python 代码
    """
    backend = current_backend()
    try:
        res = await backend.run_python(code)
    except Exception as e:
        return f"错误: {e}"
    return _format_python_result(res, backend.config.timeout)
