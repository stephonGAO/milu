"""内置工具：Shell 命令执行

执行系统 Shell 命令并返回输出。
内置危险命令黑名单，拦截高风险操作。
内置安全命令白名单，talk 模式下允许白名单中的命令。
"""
from __future__ import annotations

import asyncio
import os
import re
import shlex

from milu.tools.decorator import tool
from milu.tools._selfguard import is_protected_path

# 输出最大字符数
_MAX_OUTPUT_CHARS = 8192

# 安全命令白名单（取命令的第一个词，即基础命令名）
_SAFE_COMMANDS = {
    # 文件查看
    "ls", "ll", "dir", "cat", "head", "tail", "less", "more",
    "wc", "file", "stat", "du", "df", "tree", "nl",
    # 搜索
    "find", "grep", "egrep", "fgrep", "which", "where", "locate",
    # 网络
    "curl", "wget", "ping", "traceroute", "nslookup", "dig",
    "netstat", "ss", "host",
    # 系统信息
    "ps", "top", "htop", "uname", "hostname", "whoami", "id",
    "uptime", "free", "lscpu", "lsblk", "env", "printenv", "echo",
    # 文本处理
    "awk", "sed", "sort", "uniq", "cut", "tr", "xargs", "diff",
    # 版本控制（只读）
    "git",
    # 其他
    "date", "cal", "pwd", "basename", "dirname", "realpath",
    "type", "command",
}

# 管道/重定向中可能包含写入操作的分隔符
_WRITE_INDICATORS = re.compile(r">\s*\S|>>\s*\S|\|\s*tee\b|\|\s*sudo\b")


def _is_safe_shell(args: dict) -> bool:
    """检查 shell 命令是否为安全操作（talk 模式白名单检查）。

    检查逻辑：
    1. 提取基础命令名（第一个词）
    2. 检查是否在白名单中
    3. 检查是否包含写入重定向（>、>>）或危险管道
    """
    command = str(args.get("command", "")).strip()
    if not command:
        return False

    # 包含写入重定向 → 非只读
    if _WRITE_INDICATORS.search(command):
        return False

    # 提取基础命令名（处理路径前缀如 /usr/bin/ls）
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()

    if not parts:
        return False

    base_cmd = parts[0]
    # 去掉路径前缀：/usr/bin/ls → ls
    if "/" in base_cmd:
        base_cmd = base_cmd.rsplit("/", 1)[-1]
    # Windows 路径
    if "\\" in base_cmd:
        base_cmd = base_cmd.rsplit("\\", 1)[-1]

    return base_cmd in _SAFE_COMMANDS

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
    # 自我保护：包管理器操作（可能覆盖 milu 自身）
    r"\bpip\s+(install|uninstall|download)\b",   # pip 安装/卸载
    r"\bpip3\s+(install|uninstall|download)\b",
    r"\buv\s+add\b|\buv\s+remove\b",            # uv 包管理
    r"\bpoetry\s+(add|remove|update)\b",        # poetry 包管理
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


@tool(
    name="shell_command",
    description="执行 Shell 命令并返回 stdout/stderr 输出",
    is_safe=False,
    safe_check=_is_safe_shell,
)
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

    # 自我保护：检测命令中是否包含对受保护路径的写操作
    if _WRITE_INDICATORS.search(command):
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        for token in tokens:
            if os.path.sep in token or "/" in token or token.startswith("."):
                if is_protected_path(token):
                    return f"错误: 安全限制——命令中包含对 milu 受保护路径的写操作，已拦截: {token}"

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
