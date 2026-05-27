"""内置工具：Shell 命令执行

执行系统 Shell 命令并返回输出。标记为 dangerous=True。
内置危险命令黑名单，拦截高风险操作。
"""
from __future__ import annotations

import asyncio
import re

from agent_framework.tools.decorator import tool

# 输出最大字符数
_MAX_OUTPUT_CHARS = 8192

# 危险命令黑名单（正则匹配，忽略大小写）
_DANGEROUS_PATTERNS = [
    r"rm\s+-\w*[rf]\w*[rf]\w*\s+/",            # rm -rf / rm -fr /xxx（删除根目录下任何内容）
    r"\bsudo\b",                                 # sudo 提权
    r"\bshutdown\b",                             # 关机
    r"\breboot\b",                               # 重启
    r"\bmkfs\b",                                 # 格式化磁盘
    r"\bdd\s+.*of=/dev/",                        # dd 写设备
    r">\s*/dev/sd",                              # 重定向到磁盘设备
    r">\s*/dev/nvme",                            # 重定向到 NVMe 设备
]


def _is_dangerous_command(command: str) -> str | None:
    """检查命令是否匹配危险命令黑名单。

    Returns:
        匹配到的危险模式描述，安全则返回 None
    """
    cmd = command.strip()
    for pattern in _DANGEROUS_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            return pattern
    return None


@tool(name="shell_command", description="执行 Shell 命令并返回 stdout/stderr 输出", dangerous=True)
async def shell_command(command: str, timeout: int = 30) -> str:
    """
    在系统 Shell 中执行命令。这是一个危险操作，请谨慎使用。

    :param command: 要执行的 Shell 命令
    :param timeout: 超时时间（秒），默认 30
    """
    # 危险命令拦截
    matched = _is_dangerous_command(command)
    if matched:
        return f"错误: 危险命令被拦截（匹配规则: {matched}），该命令禁止执行"

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
