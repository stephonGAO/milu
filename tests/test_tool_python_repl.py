"""测试内置工具 python_repl - Python 代码沙箱执行"""
import pytest
from agent_framework.tools.builtin.python_repl import python_repl


class TestPythonRepl:
    """python_repl 功能测试"""

    @pytest.mark.asyncio
    async def test_simple_expression(self):
        """简单表达式求值"""
        result = await python_repl("print(2 + 3)")
        assert "5" in result

    @pytest.mark.asyncio
    async def test_multiline_code(self):
        """多行代码执行"""
        code = "x = 10\ny = 20\nprint(x + y)"
        result = await python_repl(code)
        assert "30" in result

    @pytest.mark.asyncio
    async def test_import_stdlib(self):
        """可以导入标准库"""
        code = "import math\nprint(math.pi)"
        result = await python_repl(code)
        assert "3.14" in result

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """异常被捕获并显示 traceback"""
        result = await python_repl("1 / 0")
        assert "错误" in result or "Error" in result or "ZeroDivisionError" in result

    @pytest.mark.asyncio
    async def test_print_output(self):
        """print 输出被捕获"""
        result = await python_repl("for i in range(3): print(i)")
        assert "0" in result
        assert "1" in result
        assert "2" in result

    @pytest.mark.asyncio
    async def test_syntax_error(self):
        """语法错误被捕获"""
        result = await python_repl("def foo(:\n  pass")
        assert "错误" in result or "Error" in result or "SyntaxError" in result

    @pytest.mark.asyncio
    async def test_string_operations(self):
        """字符串操作"""
        result = await python_repl("s = 'Hello World'\nprint(s.upper())")
        assert "HELLO WORLD" in result

    @pytest.mark.asyncio
    async def test_tool_wrapper_metadata(self):
        """@tool 元数据正确"""
        wrapper = python_repl._tool_wrapper
        assert wrapper.name == "python_repl"
        assert wrapper.is_async is True
        assert wrapper.is_safe is True
