"""Docker 后端：在一次性容器中执行代码/命令（真隔离，多用户/不可信输入首选）。

相对 subprocess 的提升：**文件系统与网络级硬隔离**——容器只看得见挂进去的 agent
工作区（`-v {workspace}:/work`），宿主的 `.env` / 项目 / 系统目录一律不可见；默认
`--network none` 断网；`--read-only` 只读根 + tmpfs 临时区；限内存/CPU/PID；不传宿主
环境变量（`*_API_KEY` 等密钥天然不可见）；`--rm` 用后即焚。

**零硬依赖、可插拔**：只走 `docker` CLI 子进程，不引入 docker Python SDK，故
`pip install milu` 不需要任何额外包，`import` 本模块也不依赖 docker 存在。仅当
`sandbox.backend=docker` 且真正执行代码时才需要本机装好 Docker Engine；未安装 / 守护
进程未启动 / 镜像拉取失败时返回中文友好错（提示改用 subprocess），不崩溃对话。

注意：容器内是**镜像自带的 Python/包**（默认 python:3.12-slim），不是 milu 的 venv，
故 `import milu` / 第三方库需镜像内已有；代码经 stdin 喂给 `python -X utf8 -I -`。
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid

from milu.sandbox.base import ExecResult, SandboxBackend
from milu.resources import current_workspace

logger = logging.getLogger(__name__)

_UNSET = object()   # _prepare 结果缓存的未初始化哨兵


def docker_available_sync(timeout: float = 6.0) -> bool:
    """同步快探 docker daemon 是否就绪（供 CLI/serve 启动横幅判断，非执行路径）。

    docker 未安装 / daemon 未启 / 探测异常均返回 False。不抛异常。
    """
    import shutil
    import subprocess
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, timeout=timeout,
        )
        return r.returncode == 0
    except Exception:
        return False


class DockerBackend(SandboxBackend):
    """容器内执行（真隔离，需本机 Docker Engine；走 docker CLI、零 Python 依赖）。"""

    name = "docker"

    def __init__(self, config):
        super().__init__(config)
        # 懒探测结果缓存：None=就绪；str=不可用原因；_UNSET=未探测
        self._prepared = _UNSET

    # ── docker 命令行拼装 ─────────────────────────────────────
    def _base_args(self, name: str) -> list[str]:
        """构造 `docker run` 的隔离参数（不含镜像与入口命令）。

        工作区（current_workspace 或 config.workdir）以 `-v …:/work` 挂入并设为工作
        目录 → 产物持久化、且容器只能碰到这一处宿主目录；无工作区时用 tmpfs（全隔离）。
        """
        cfg = self.config
        args = [
            "docker", "run", "--rm", "-i", "--name", name,
            "--network", ("bridge" if cfg.network else "none"),
            "--pids-limit", "512",
            "--read-only",                              # 只读根文件系统
            "--tmpfs", "/tmp:rw,exec,size=64m",         # 可写临时区
            "-e", "PYTHONUTF8=1",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-e", "HOME=/tmp",
            "-w", "/work",
        ]
        if cfg.memory_mb:
            # memory-swap 设为与 memory 相同 → 禁用 swap，硬限内存
            args += ["--memory", f"{int(cfg.memory_mb)}m",
                     "--memory-swap", f"{int(cfg.memory_mb)}m"]
        cpus = getattr(cfg, "docker_cpus", None)
        if cpus:
            args += ["--cpus", str(cpus)]
        user = getattr(cfg, "docker_user", None)
        if user:
            args += ["--user", str(user)]
        ws = current_workspace() or (cfg.workdir or None)
        if ws:
            ws = os.path.abspath(ws)
            os.makedirs(ws, exist_ok=True)
            args += ["-v", f"{ws}:/work"]               # 挂载工作区 → /work（rw）
        else:
            args += ["--tmpfs", "/work:rw,exec,size=128m"]   # 无工作区 → 临时区，全隔离
        for m in (cfg.docker_mounts or []):
            args += ["-v", str(m)]
        return args

    # ── 懒探测：daemon 可用性 + 镜像就绪 ───────────────────────
    async def _prepare(self) -> str | None:
        """首次执行前确认 docker 可用并就绪。返回 None=就绪；str=不可用原因（缓存）。"""
        if self._prepared is not _UNSET:
            return self._prepared
        self._prepared = await self._do_prepare()
        return self._prepared

    async def _do_prepare(self) -> str | None:
        # 1) docker 是否安装 + 守护进程是否可用
        rc, _out, err = await self._probe_exec(
            ["docker", "version", "--format", "{{.Server.Version}}"], timeout=15)
        if rc is None:
            return ("Docker 未安装或不在 PATH：sandbox.backend=docker 需要本机安装 "
                    "Docker Engine；如不便安装可改用 `milu config set sandbox.backend "
                    "subprocess`（子进程隔离，零依赖）。")
        if rc != 0:
            return (f"Docker 守护进程不可用（{(err or '请确认 Docker 已启动').strip()[:200]}）；"
                    "可改用 sandbox.backend=subprocess。")
        # 2) 镜像是否就绪，缺则拉取（首次较慢）
        image = self.config.docker_image
        rc, _out, _err = await self._probe_exec(
            ["docker", "image", "inspect", image], timeout=20)
        if rc != 0:
            logger.info("[sandbox/docker] 本地无镜像 %s，开始拉取（首次较慢）…", image)
            rc, _out, perr = await self._probe_exec(["docker", "pull", image], timeout=600)
            if rc != 0:
                return f"拉取镜像 {image} 失败：{(perr or '').strip()[:300]}"
        return None

    async def _probe_exec(self, argv: list[str], *, timeout: int):
        """运行一条不带 stdin 的辅助 docker 命令，返回 (rc|None, stdout, stderr)。

        rc=None 表示 docker 可执行文件不存在（未安装）。
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return None, "", "docker 可执行文件不存在"
        except Exception as e:  # noqa: BLE001
            return None, "", str(e)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await proc.wait()
            except Exception:
                pass
            return 124, "", f"docker 命令超时（{timeout}s）"
        return (proc.returncode,
                out.decode("utf-8", "replace") if out else "",
                err.decode("utf-8", "replace") if err else "")

    # ── 容器执行（超时真杀容器）────────────────────────────────
    async def _container_run(self, image_cmd: list[str], stdin: str | None,
                             timeout: int) -> ExecResult:
        name = f"milu_sbx_{uuid.uuid4().hex[:12]}"
        argv = self._base_args(name) + [self.config.docker_image] + image_cmd
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:  # noqa: BLE001
            return ExecResult(stderr=f"docker 启动失败: {e}", exit_code=None)
        data = stdin.encode("utf-8") if stdin is not None else None
        try:
            out, err = await asyncio.wait_for(proc.communicate(input=data), timeout=timeout)
        except asyncio.TimeoutError:
            # 杀 docker run 客户端不一定停容器，按名字显式 kill（--rm 随后清理）
            await self._kill_container(name)
            proc.kill()
            try:
                await proc.wait()
            except Exception:
                pass
            return ExecResult(exit_code=None, timed_out=True)
        return ExecResult(
            stdout=out.decode("utf-8", "replace") if out else "",
            stderr=err.decode("utf-8", "replace") if err else "",
            exit_code=proc.returncode,
        )

    async def _kill_container(self, name: str) -> None:
        try:
            await self._probe_exec(["docker", "kill", name], timeout=10)
        except Exception:
            pass

    # ── 后端接口实现 ─────────────────────────────────────────
    async def run_python(self, code: str, *, timeout: int | None = None) -> ExecResult:
        err = await self._prepare()
        if err:
            return ExecResult(stderr=err, exit_code=None)
        # -X utf8 显式 UTF-8；-I 隔离模式；'-' 从 stdin 读代码
        return await self._container_run(
            ["python", "-X", "utf8", "-I", "-"], code, timeout or self.config.timeout)

    async def run_shell(self, command: str, *, timeout: int | None = None) -> ExecResult:
        err = await self._prepare()
        if err:
            return ExecResult(stderr=err, exit_code=None)
        return await self._container_run(
            ["sh", "-c", command], None, timeout or self.config.timeout)
