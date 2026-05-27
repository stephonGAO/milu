"""测试内置工具 shell_command - Shell 命令执行"""
import sys

import pytest
from agent_framework.tools.builtin.shell_command import shell_command, _is_dangerous_command


class TestShellCommand:
    """shell_command 功能测试"""

    @pytest.mark.asyncio
    async def test_echo_command(self):
        """执行简单 echo 命令"""
        if sys.platform == "win32":
            result = await shell_command(command="echo hello")
        else:
            result = await shell_command(command="echo hello")
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_command_with_output(self):
        """命令输出被捕获"""
        if sys.platform == "win32":
            result = await shell_command(command="echo test_output")
        else:
            result = await shell_command(command="printf 'line1\\nline2'")
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_command_exit_code(self):
        """返回结果包含退出码信息"""
        if sys.platform == "win32":
            result = await shell_command(command="echo ok")
        else:
            result = await shell_command(command="true")
        # 成功命令不应包含错误
        assert "超时" not in result

    @pytest.mark.asyncio
    async def test_command_error(self):
        """执行失败命令"""
        if sys.platform == "win32":
            result = await shell_command(command="nonexistent_command_xyz")
        else:
            result = await shell_command(command="false")
        assert len(result) > 0  # 有输出（错误信息或退出码）

    @pytest.mark.asyncio
    async def test_command_timeout(self):
        """超时命令被终止"""
        if sys.platform == "win32":
            result = await shell_command(command="ping -n 100 localhost", timeout=2)
        else:
            result = await shell_command(command="sleep 100", timeout=2)
        assert "超时" in result

    @pytest.mark.asyncio
    async def test_output_truncation(self):
        """超长输出被截断"""
        if sys.platform == "win32":
            # Windows: 生成大量输出
            cmd = "for /L %i in (1,1,2000) do @echo line_%i_padding_text_to_make_it_longer"
        else:
            cmd = "for i in $(seq 1 2000); do echo \"line $i padding text to make output longer\"; done"
        result = await shell_command(command=cmd)
        assert "截断" in result

    @pytest.mark.asyncio
    async def test_tool_wrapper_metadata(self):
        """@tool 元数据正确"""
        wrapper = shell_command._tool_wrapper
        assert wrapper.name == "shell_command"
        assert wrapper.is_async is True
        assert wrapper.dangerous is True  # 关键：标记为危险


class TestDangerousCommandBlocklist:
    """危险命令黑名单检测"""

    @pytest.mark.parametrize("cmd", [
        "rm -rf /",
        "rm -rf / ",
        "rm -rf /tmp",
        "rm -fr /",
        "rm -fr /var/log",
        "sudo apt install foo",
        "sudo ls",
        "shutdown now",
        "shutdown -h now",
        "reboot",
        "reboot -f",
        "mkfs.ext4 /dev/sda1",
        "dd if=/dev/zero of=/dev/sda bs=1M",
        "> /dev/sda",
        "> /dev/nvme0n1",
    ])
    def test_dangerous_commands_detected(self, cmd):
        """危险命令应被检测"""
        assert _is_dangerous_command(cmd) is not None, f"应检测到危险命令: {cmd}"

    @pytest.mark.parametrize("cmd", [
        "echo hello",
        "ls -la",
        "rm -rf ./build",
        "rm -rf build/",
        "cat /etc/hosts",
        "find . -name '*.py'",
        "git status",
        "python script.py",
        "npm install",
        "rm file.txt",
    ])
    def test_safe_commands_pass(self, cmd):
        """安全命令不应被拦截"""
        assert _is_dangerous_command(cmd) is None, f"不应拦截安全命令: {cmd}"

    @pytest.mark.asyncio
    async def test_dangerous_command_blocked(self):
        """危险命令执行应被拦截"""
        result = await shell_command(command="sudo echo test")
        assert "拦截" in result
        assert "危险" in result

    @pytest.mark.asyncio
    async def test_safe_command_runs(self):
        """安全命令正常执行"""
        result = await shell_command(command="echo safe_test_123")
        assert "safe_test_123" in result
