"""工具执行器 - 安全地执行工具调用并处理异常"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

from agent_framework.tools.registry import ToolRegistry
from agent_framework.agent.config import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionResult:
    """工具执行结果"""
    output: str          # 工具输出（始终为字符串，回填给 LLM）
    is_error: bool       # 是否为错误结果


class ToolExecutor:
    """工具执行器 - 负责安全地执行工具并处理异常"""

    def __init__(self, registry: ToolRegistry, config: AgentConfig):
        self._registry = registry
        self._config = config

    async def execute(self, tool_call: dict) -> ToolExecutionResult:
        """
        执行一个工具调用，异常不中断循环。

        Args:
            tool_call: OpenAI 格式的工具调用字典
                {"id": "call_xxx", "function": {"name": "...", "arguments": "..."}}

        Returns:
            ToolExecutionResult(output=结果文本, is_error=是否出错)
        """
        func_name = tool_call.get("function", {}).get("name", "")
        arguments_str = tool_call.get("function", {}).get("arguments", "{}")

        # 1. 查找工具
        wrapper = self._registry.get_tool(func_name)
        if not wrapper:
            return ToolExecutionResult(
                output=f"工具不存在: {func_name}",
                is_error=True
            )

        # 1.5 检查工具是否在休眠池（尚未激活）
        if self._registry.is_dormant(func_name):
            return ToolExecutionResult(
                output=f"工具 {func_name} 尚未激活。请先调用 activate_tools 激活它。",
                is_error=True,
            )

        # 2. 解析参数 JSON
        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError as e:
            return ToolExecutionResult(
                output=f"参数解析失败: {e}",
                is_error=True
            )

        # 3. 执行工具函数
        try:
            if wrapper.is_async:
                result = await wrapper.func(**arguments)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: wrapper.func(**arguments)
                )

            return ToolExecutionResult(
                output=str(result),
                is_error=False
            )

        except Exception as e:
            logger.warning(f"工具 {func_name} 执行异常: {e}")
            return ToolExecutionResult(
                output=f"执行异常: {e}",
                is_error=True
            )
