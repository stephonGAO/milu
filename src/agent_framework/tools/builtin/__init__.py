"""内置工具集 - 提供开箱即用的常用 Agent 工具。

使用方式:
    from agent_framework.tools.builtin import BUILTIN_TOOLS

    agent = Agent(
        llm=llm,
        tools=BUILTIN_TOOLS,  # 注册所有内置工具
    )

    # 或选择性导入
    from agent_framework.tools.builtin import file, web_search
    agent = Agent(llm=llm, tools=[file, web_search])

    # structured_output 需要工厂函数创建（注入 LLM 用于自修复）
    from agent_framework.tools.builtin import create_structured_output_tool
    so_tool = create_structured_output_tool(llm)
    agent = Agent(llm=llm, tools=[*BUILTIN_TOOLS, so_tool])
"""

from agent_framework.tools.builtin.datetime_tool import datetime_tool, get_current_time
from agent_framework.tools.builtin.http_request import http_request
from agent_framework.tools.builtin.file_tool import file
from agent_framework.tools.builtin.python_repl import python_repl
from agent_framework.tools.builtin.web_search import web_search
from agent_framework.tools.builtin.shell_command import shell_command
from agent_framework.tools.builtin.structured_output import (
    structured_output,
    create_structured_output_tool,
)

# 所有内置工具列表，可直接传给 Agent(tools=BUILTIN_TOOLS)
# 注意：structured_output 需要通过 create_structured_output_tool(llm) 创建后单独添加
BUILTIN_TOOLS = [
    datetime_tool,
    http_request,
    file,
    python_repl,
    web_search,
    shell_command,
]

__all__ = [
    "datetime_tool",
    "get_current_time",
    "http_request",
    "file",
    "python_repl",
    "web_search",
    "shell_command",
    "structured_output",
    "create_structured_output_tool",
    "BUILTIN_TOOLS",
]
