"""沙箱执行后端：抽象基类、执行结果、ContextVar 注入。

milu 的代码执行工具（python_repl / shell_command）在「软门控」（模式 + AI 判定
决定要不要执行）之上，再叠加一层「执行时隔离」（决定放行后跑在哪、能碰什么）。
后端可插拔——同一套工具，按部署需要选不同隔离强度：

    LocalBackend       进程内 exec / 宿主 shell（现状，零依赖，默认）
    SubprocessBackend  独立子进程 + 资源限额 + 清洗环境（跨平台零依赖，推荐升级）
    DockerBackend      容器内执行（真隔离，需 Docker；计划于后续版本）

设计与既有 knowledge / memory / tracer 一致：后端实例经 run() 入口的 ContextVar
注入（asyncio 任务级隔离），工具通过 current_backend() 读取；未注入时回退到模块级
默认 LocalBackend，保证脱离 Agent 的单元测试/直接调用仍可运行（hermetic）。
"""
from __future__ import annotations

import contextvars
import locale
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milu.sandbox.config import SandboxConfig


# ── Windows 中文编码修复（自 python_repl 迁移至此，所有后端共享）─────────────
# Windows 中文系统上 locale.getencoding() 在解释器启动时缓存为 cp936(GBK)，
# subprocess 据此解码管道，遇到 UTF-8 中文输出报 UnicodeDecodeError。
# 两步修复：① PYTHONUTF8=1（子进程启动启用 UTF-8 模式，且经 env 传给子进程）；
#          ② locale 取编码函数 monkey-patch 为 utf-8（父进程解码管道用）。
# 在本模块导入时执行一次（python_repl 现导入本包，触发时机与改造前一致）。
def _apply_windows_utf8() -> None:
    if sys.platform != "win32" or getattr(sys.flags, "utf8_mode", False):
        return
    os.environ["PYTHONUTF8"] = "1"

    def _utf8_encoding(*_args, **_kwargs) -> str:
        return "utf-8"

    # 3.11+ 是 locale.getencoding，3.10 及更早是 locale.getpreferredencoding；
    # 直接访问不存在的属性会 AttributeError 崩 import milu，故按存在性 patch。
    if hasattr(locale, "getencoding"):
        locale.getencoding = _utf8_encoding
    else:
        locale.getpreferredencoding = _utf8_encoding

    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_apply_windows_utf8()


@dataclass
class ExecResult:
    """统一的执行结果（后端无关，由工具层格式化为回填 LLM 的文本）。"""

    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = 0   # 进程退出码；超时/无法获取时为 None
    timed_out: bool = False     # 是否因超时被强制终止


class SandboxBackend(ABC):
    """代码/命令执行后端抽象。

    子类持有一份 SandboxConfig（限额、镜像等），run_python / run_shell 返回
    统一的 ExecResult。timeout 形参为「本次调用」的覆盖值（shell_command 工具
    允许 LLM 传 timeout）；为 None 时回退到 config.timeout。
    """

    name: str = "base"

    def __init__(self, config: "SandboxConfig"):
        self.config = config

    @abstractmethod
    async def run_python(self, code: str, *, timeout: int | None = None) -> ExecResult:
        """执行一段 Python 代码。"""
        raise NotImplementedError

    @abstractmethod
    async def run_shell(self, command: str, *, timeout: int | None = None) -> ExecResult:
        """执行一条 Shell 命令。"""
        raise NotImplementedError

    async def aclose(self) -> None:
        """释放后端持有的资源（容器/连接池等）。Local/Subprocess 为 no-op。"""
        return None


# ── 注入：与 knowledge / tracer 同款 ContextVar（asyncio 任务级隔离）──────────
_current_sandbox: contextvars.ContextVar["SandboxBackend | None"] = contextvars.ContextVar(
    "_current_sandbox", default=None
)

# 模块级 hermetic 回退后端（无状态、可共享单例）：未经 Agent 注入时回退到它。
# 恒为 LocalBackend（进程内执行、零子进程开销），保证裸工具调用 / 单测的纯净与速度。
# 注意：这与「应用层默认后端」不同——后者由 SandboxConfig 默认（= subprocess）经
# config.json 驱动，仅在 CLI/Web 显式下传 sandbox 时生效；此处的回退恒为 local。
_default_backend: "SandboxBackend | None" = None


def default_backend() -> "SandboxBackend":
    """返回进程级 hermetic 回退后端（恒为 LocalBackend，懒构造单例）。"""
    global _default_backend
    if _default_backend is None:
        from milu.sandbox.config import SandboxConfig
        from milu.sandbox.local import LocalBackend
        _default_backend = LocalBackend(SandboxConfig(backend="local"))
    return _default_backend


def current_backend() -> "SandboxBackend":
    """当前生效后端：ContextVar 注入值优先，否则模块级默认 LocalBackend。"""
    return _current_sandbox.get() or default_backend()
