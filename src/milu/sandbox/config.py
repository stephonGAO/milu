"""沙箱配置与后端工厂。

SandboxConfig 是默认值的唯一真相源（分层配置 config.json 的 sandbox 分节从它派生，
见 milu.config._builtin_defaults）。build_backend() 把多种 spec 统一解析为后端实例。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from milu.sandbox.base import SandboxBackend

# config.json sandbox 分节暴露的可调字段（docker_mounts 为列表、本版本未启用 docker，
# 不入分节；workdir 默认 None）
_KNOWN_KEYS = (
    "backend", "timeout", "memory_mb", "cpu_seconds",
    "network", "max_output_chars", "workdir", "ephemeral_workdir", "docker_image",
)


@dataclass
class SandboxConfig:
    """沙箱执行配置（所有后端共享；部分字段仅特定后端生效）。"""

    backend: str = "subprocess"          # 执行后端：local | subprocess | docker（默认子进程隔离）
    timeout: int = 60                    # 默认执行超时（秒）；shell_command 的 timeout 形参可覆盖
    memory_mb: int | None = 1024         # 子进程地址空间上限 MB（POSIX RLIMIT_AS；None=不限）
    cpu_seconds: int | None = None       # 子进程 CPU 时间上限（POSIX RLIMIT_CPU；None=不限）
    network: bool = False                # 是否允许联网（仅 docker 生效；subprocess 本版本不隔离网络）
    max_output_chars: int = 8192         # shell 输出截断上限（字符）
    workdir: str | None = None           # 固定工作目录；None=继承当前工作目录（与 local 一致，编码工作流照常）
    ephemeral_workdir: bool = False      # True 时 subprocess 每次在一次性临时目录执行（更强隔离，相对路径写不落项目目录）
    docker_image: str = "python:3.12-slim"   # docker 后端镜像（计划于后续版本）
    docker_mounts: list = field(default_factory=list)  # docker 挂载（计划于后续版本）

    @classmethod
    def from_mapping(cls, m: dict | None) -> "SandboxConfig":
        """从（可能含无关键 / "//" 注释键的）映射构造，未知键安全忽略。"""
        if not m:
            return cls()
        kwargs = {k: m[k] for k in (*_KNOWN_KEYS, "docker_mounts") if k in m}
        return cls(**kwargs)


def _from_config(cfg: SandboxConfig) -> SandboxBackend:
    """按 cfg.backend 分发到具体后端实现。"""
    name = (cfg.backend or "local").strip().lower()
    if name == "local":
        from milu.sandbox.local import LocalBackend
        return LocalBackend(cfg)
    if name == "subprocess":
        from milu.sandbox.process import SubprocessBackend
        return SubprocessBackend(cfg)
    if name == "docker":
        raise ValueError(
            "docker 沙箱后端尚未在本版本提供（计划于后续版本），"
            "请将 sandbox.backend 设为 local 或 subprocess"
        )
    raise ValueError(f"未知沙箱后端: {cfg.backend!r}（可选：local / subprocess / docker）")


def build_backend(spec: "str | SandboxConfig | SandboxBackend | None") -> SandboxBackend:
    """把多种 spec 统一解析为后端实例。

    - None            → SandboxConfig 默认（= subprocess 隔离后端 + 默认限额）
    - "local"         → 进程内 / 宿主 shell 后端（零开销，可信场景）
    - SandboxConfig   → 按其 backend 字段构造
    - SandboxBackend  → 原样返回（程序化注入自定义后端）

    注意：这与「未注入时的 hermetic 回退」不同——后者恒为 LocalBackend
    （见 base.default_backend()），保证裸工具调用 / 单测零子进程开销。
    """
    if isinstance(spec, SandboxBackend):
        return spec
    if spec is None:
        cfg = SandboxConfig()
    elif isinstance(spec, SandboxConfig):
        cfg = spec
    elif isinstance(spec, str):
        cfg = replace(SandboxConfig(), backend=spec)
    else:
        raise TypeError(f"无法解析 sandbox 规格: {type(spec).__name__}")
    return _from_config(cfg)
