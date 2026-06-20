"""沙箱执行后端包：为 python_repl / shell_command 提供可插拔的执行隔离。

详见 base.py 顶部说明与 docs/沙箱执行方案设计.md。
"""
from milu.sandbox.base import (
    ExecResult,
    SandboxBackend,
    _current_sandbox,
    current_backend,
    default_backend,
)
from milu.sandbox.config import SandboxConfig, build_backend
from milu.sandbox.local import LocalBackend
from milu.sandbox.process import SubprocessBackend

__all__ = [
    "ExecResult",
    "SandboxBackend",
    "SandboxConfig",
    "LocalBackend",
    "SubprocessBackend",
    "build_backend",
    "current_backend",
    "default_backend",
    "_current_sandbox",
]
