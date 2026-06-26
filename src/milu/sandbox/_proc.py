"""子进程超时收发的公共助手：连根杀进程树 + 健壮超时（local / subprocess 后端共用）。

为什么需要它（修复的真实 bug）：
  shell 命令的实际进程链常是 `shell → npx → node → …` 多层。原实现超时只做
  `proc.kill()`，**仅杀直接子进程**，留下一串孤儿孙进程继续在后台运行（如 playwright
  install 一直在下载）；更糟的是孤儿继承了 stdout/stderr 管道写端不关闭，父侧读管道
  **永远等不到 EOF**，而 `asyncio.wait_for(proc.communicate(), t)` 到点要取消
  `communicate()`，在 Windows Proactor 事件循环下这个取消会**卡死**，连超时分支都进不去
  —— 最终只能靠 Agent 层的总超时（默认 3600s）兜底，工具级 timeout 形同虚设。

两条修复缺一不可：
  ① 杀**进程树**而非单个进程（Windows 用 ``taskkill /F /T``；POSIX 用进程组 ``killpg``，
     要求子进程以 ``start_new_session=True`` 自成进程组）；
  ② 超时**不靠取消** ``communicate()``：改用 ``asyncio.wait``（到点只返回、不取消任务），
     先杀树关闭管道，再回收 ``communicate()`` 自然返回的残余输出。
"""
from __future__ import annotations

import asyncio
import os
import signal
import sys


async def terminate_process_tree(proc) -> None:
    """连根杀掉子进程及其所有后代。

    必须杀整棵树：只杀直接子进程会留下孤儿孙进程（继续占资源、且其继承的
    stdout/stderr 管道写端不关闭 → 父侧读管道永不 EOF）。任一步失败都回退到
    ``proc.kill()`` 至少杀掉直接子进程，绝不抛异常打断调用方。
    """
    pid = proc.pid
    if pid is None:
        return
    if sys.platform == "win32":
        # Windows 无进程组/会话语义，用 taskkill 按 PID 树强杀（/F 强制，/T 含所有子孙）。
        # taskkill 自身用 milu 父进程环境启动（PATH 含 System32），不受沙箱清洗影响。
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill", "/F", "/T", "/PID", str(pid),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    else:
        # POSIX：子进程已 start_new_session=True 自成进程组首领 → 杀整个进程组。
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass


async def communicate_or_kill(
    proc, input_data: bytes | None, timeout: int
) -> tuple[bytes, bytes, bool]:
    """喂 stdin、收 stdout/stderr，超时则连根杀进程树。

    关键：用 ``asyncio.wait``（到点只返回、不取消内层任务）替代 ``asyncio.wait_for``
    （到点取消 ``communicate()``，在 Windows Proactor 下会卡死）。超时后先杀树关闭管道，
    管道一关 ``communicate()`` 即可返回已缓冲的残余输出。

    返回 ``(stdout, stderr, timed_out)``，stdout/stderr 为 bytes（由调用方解码）。
    """
    task = asyncio.ensure_future(proc.communicate(input=input_data))
    done, _pending = await asyncio.wait({task}, timeout=timeout)
    if task in done:
        out, err = task.result()
        return out or b"", err or b"", False

    # 超时：先连根杀进程树 → 所有管道写端关闭 → communicate 自然返回残余输出
    await terminate_process_tree(proc)
    try:
        out, err = await asyncio.wait_for(task, timeout=10)
        return out or b"", err or b"", True
    except Exception:
        # 极端情况：杀树后 10s 内仍收不回（task 已被 wait_for 取消）→ 放弃残余输出
        if not task.done():
            task.cancel()
        return b"", b"", True
