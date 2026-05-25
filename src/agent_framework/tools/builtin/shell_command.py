"""内置工具：Shell 命令执行

执行系统 Shell 命令并返回输出。标记为 dangerous=True。
"""
from __future__ import annotations

import asyncio

from agent_framework.tools.decorator import tool

# 输出最大字符数
_MAX_OUTPUT_CHARS = 8192


@tool(name="shell_command", description="执行 Shell 命令并返回 stdout/stderr 输出", dangerous=True)
async def shell_command(command: str, timeout: int = 30) -> str:
    """
    在系统 Shell 中执行命令。这是一个危险操作，请谨慎使用。

    :param command: 要执行的 Shell 命令
    :param timeout: 超时时间（秒），默认 30
    """
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"错误: 命令执行超时（{timeout}秒），已强制终止"

        # 解码输出
        stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
        exit_code = proc.returncode

        # 拼接结果
        parts = []
        if stdout_text:
            parts.append(stdout_text)
        if stderr_text:
            parts.append(f"[stderr]\n{stderr_text}")
        parts.append(f"[退出码: {exit_code}]")

        result = "\n".join(parts)

        # 截断超长输出
        if len(result) > _MAX_OUTPUT_CHARS:
            result = result[:_MAX_OUTPUT_CHARS] + f"\n\n... [输出已截断，共 {len(result)} 字符]"

        return result

    except Exception as e:
        return f"错误: 命令执行失败 - {e}"
