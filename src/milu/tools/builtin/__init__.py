"""内置工具集 - 提供开箱即用的常用 Agent 工具。

使用方式:
    from milu.tools.builtin import BUILTIN_TOOLS

    agent = Agent(
        llm=llm,
        tools=BUILTIN_TOOLS,  # 注册所有内置工具
    )

    # 或选择性导入
    from milu.tools.builtin import file_read, file_write, web_search
    agent = Agent(llm=llm, tools=[file_read, file_write, web_search])

    # structured_output 需要工厂函数创建（注入 LLM 用于自修复）
    from milu.tools.builtin import create_structured_output_tool
    so_tool = create_structured_output_tool(llm)
    agent = Agent(llm=llm, tools=[*BUILTIN_TOOLS, so_tool])
"""

from milu.tools.builtin.datetime_tool import datetime_tool, get_current_time
from milu.tools.builtin.http_request import http_request
from milu.tools.builtin.file_tool import file_read, file_write
from milu.tools.builtin.image_tool import image_read
from milu.tools.builtin.doc_tool import doc_read
from milu.tools.builtin.python_repl import python_repl
from milu.tools.builtin.web_search import web_search
from milu.tools.builtin.web_fetch import web_fetch
from milu.tools.builtin.shell_command import shell_command
from milu.tools.builtin.structured_output import (
    structured_output,
    create_structured_output_tool,
)
from milu.tools.builtin.todo_write import todo_read, todo_write
from milu.tools.builtin.memory_tool import memory_read, memory_write
from milu.tools.builtin.schedule_tool import schedule_create, schedule_manage

# 所有内置工具列表，可直接传给 Agent(tools=BUILTIN_TOOLS)
# 注意：structured_output 需要通过 create_structured_output_tool(llm) 创建后单独添加
# 注意：todo 工具（todo_write / todo_read）依赖 Agent.run() 注入的 ContextVar（session_dir），
#       在非 Agent 上下文中手动调用会抛 RuntimeError；正常通过 Agent 使用时无影响
# 注意：memory 工具（memory_write / memory_read）不在此列表——长期记忆默认关闭，
#       由 Agent(memory=True/"user_id") 开关启用时自动注册（见 agent.py）
BUILTIN_TOOLS = [
    datetime_tool,
    http_request,
    web_fetch,
    file_read,
    file_write,
    image_read,
    doc_read,
    python_repl,
    web_search,
    shell_command,
    todo_read,
    todo_write,
    schedule_create,
    schedule_manage,
]

__all__ = [
    "datetime_tool",
    "get_current_time",
    "http_request",
    "file_read",
    "file_write",
    "image_read",
    "doc_read",
    "python_repl",
    "web_search",
    "web_fetch",
    "shell_command",
    "structured_output",
    "create_structured_output_tool",
    "todo_read",
    "todo_write",
    "memory_read",
    "memory_write",
    "schedule_create",
    "schedule_manage",
    "BUILTIN_TOOLS",
]
