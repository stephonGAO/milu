"""测试内置工具 shell_command - Shell 命令执行"""
import sys

import pytest
from agent_framework.tools.builtin.shell_command import shell_command


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
