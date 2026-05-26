"""Agent 核心编排器 - LLM 调用、工具执行、对话历史管理"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator, Awaitable, Callable

from agent_framework.agent.config import AgentConfig
from agent_framework.tools.executor import ToolExecutor, ToolExecutionResult
from agent_framework.agent.history import ConversationHistory
from agent_framework.agent.events import (
    AgentDone,
    AgentError,
    AgentEvent,
    ReasoningDelta,
    TextDelta,
    ToolCallStart,
    ToolConfirmRequired,
    ToolResult,
)
from agent_framework.llm.base.message import Message, MessageRole
from agent_framework.llm.base.response import StreamChunk, TokenUsage
from agent_framework.llm.providers.base import BaseLLM
from agent_framework.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 合并工具调用片段
# ---------------------------------------------------------------------------

def _merge_tool_calls(
    buffer: list[dict],
    new_calls: list,
) -> list[dict]:
    """合并流式工具调用片段。

    流式 API 中 tool_call 的 id/name/arguments 可能被拆分成多个 chunk。
    此函数按 index 合并，name 和 arguments 通过字符串拼接累积。

    Args:
        buffer: 已累积的工具调用字典列表（会被原地修改）
        new_calls: 新到的工具调用片段列表

    Returns:
        buffer（原地修改后返回）
    """
    for tc in new_calls:
        idx = getattr(tc, "index", 0)
        call_id = getattr(tc, "id", None) or ""
        fn = getattr(tc, "function", None)
        fn_name = getattr(fn, "name", "") if fn else ""
        fn_args = getattr(fn, "arguments", "") if fn else ""

        # 查找是否已有该 index 的 buffer 条目
        existing = None
        for item in buffer:
            if item["_idx"] == idx:
                existing = item
                break

        if existing is None:
            # 新建条目
            buffer.append({
                "_idx": idx,
                "id": call_id,
                "function": {
                    "name": fn_name,
                    "arguments": fn_args,
                },
            })
        else:
            # 追加拼接
            if call_id:
                existing["id"] = call_id
            if fn_name:
                existing["function"]["name"] += fn_name
            if fn_args:
                existing["function"]["arguments"] += fn_args

    return buffer


def _merge_usage(total: TokenUsage, new: TokenUsage | None) -> TokenUsage:
    """累积 TokenUsage。"""
    if new is None:
        return total
    total.prompt_tokens += new.prompt_tokens
    total.completion_tokens += new.completion_tokens
    total.total_tokens += new.total_tokens
    total.reasoning_tokens += new.reasoning_tokens
    return total


# ---------------------------------------------------------------------------
# Agent 类
# ---------------------------------------------------------------------------

class Agent:
    """Agent 编排器 - 负责 LLM 调用、工具执行、对话历史管理。"""

    def __init__(
        self,
        llm: BaseLLM,
        system_prompt: str,
        tools: list | None = None,
        history: ConversationHistory | None = None,
        config: AgentConfig | None = None,
        on_confirm: Callable[[str, str], Awaitable[bool]] | None = None,
        mcp_config_path: str | None = None,
    ):
        self._llm = llm
        self._config = config or AgentConfig()
        self._history = history or ConversationHistory(
            max_tokens=self._config.max_total_tokens,
        )
        self._registry = ToolRegistry()
        if tools:
            self._registry.register_many(tools)
        self._executor = ToolExecutor(self._registry, self._config)
        self._on_confirm = on_confirm

        # MCP 配置文件路径
        self._mcp_config_path = mcp_config_path
        self._mcp_manager = None

        # 设置 system prompt
        self._history.set_system(Message(role=MessageRole.SYSTEM, content=system_prompt))

    # -- 属性 ----------------------------------------------------------------

    @property
    def history(self) -> ConversationHistory:
        """对话历史管理器。"""
        return self._history

    @property
    def tools(self) -> ToolRegistry:
        """工具注册表。"""
        return self._registry

    # -- 公开方法 ------------------------------------------------------------

    async def reset(self) -> None:
        """清空对话历史（保留 system 消息）。"""
        self._history.clear()

    # -- MCP 生命周期 --------------------------------------------------------

    async def connect_mcp(self) -> None:
        """连接所有 MCP 服务器，发现并注册工具。

        配置路径解析优先级：
        1. mcp_config_path 参数（构造函数传入）
        2. 环境变量 MCP_CONFIG_PATH（.env 或系统环境变量）
        3. 自动搜索 config/mcp_servers.json → ~/.agent_framework/mcp_servers.json
        """
        from agent_framework.tools.mcp.config import MCPServerConfig
        from agent_framework.tools.mcp.manager import MCPManager
        from dotenv import load_dotenv
        import os

        # 确定配置文件路径
        path = self._mcp_config_path
        if not path:
            load_dotenv()
            path = os.environ.get("MCP_CONFIG_PATH")

        configs = MCPServerConfig.load_file(path)  # path=None 时自动搜索

        if not configs:
            return

        self._mcp_manager = MCPManager(configs)
        wrappers = await self._mcp_manager.connect_all()
        for w in wrappers:
            self._registry.register_wrapper(w)
        logger.info("MCP 工具已注册: %d 个", len(wrappers))

    async def disconnect_mcp(self) -> None:
        """断开所有 MCP 服务器连接。"""
        if self._mcp_manager:
            await self._mcp_manager.disconnect_all()
            self._mcp_manager = None

    async def __aenter__(self):
        await self.connect_mcp()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect_mcp()
        return False

    async def run(
        self,
        user_input: str,
        **llm_kwargs,
    ) -> AsyncIterator[AgentEvent]:
        """运行 Agent 循环。

        Args:
            user_input: 用户输入
            **llm_kwargs: 传递给 LLM 的额外参数

        Yields:
            AgentEvent 子类实例
        """
        # 将用户输入加入历史
        self._history.add(Message(role=MessageRole.USER, content=user_input))

        total_usage = TokenUsage()
        turn_count = 0
        total_tool_calls = 0
        start_time = time.monotonic()
        final_text = ""

        while True:
            turn_count += 1

            # 1. 检查总超时
            elapsed = time.monotonic() - start_time
            if elapsed >= self._config.total_timeout:
                yield AgentError(
                    error_type="total_timeout",
                    message=f"总超时 {self._config.total_timeout}s",
                )
                return

            # 2. 获取截断后的历史消息
            messages = self._history.get_messages()

            # 3. 获取工具 schema
            tool_schemas = self._registry.get_schemas()
            if tool_schemas:
                llm_kwargs["tools"] = tool_schemas

            # 4. 流式调用 LLM
            tool_call_buffer: list[dict] = []
            turn_text_parts: list[str] = []

            try:
                async with asyncio.timeout(self._config.timeout):
                    async for chunk in self._llm.chat(messages, **llm_kwargs):
                        # 5. 产出文本/思考事件
                        if chunk.content:
                            turn_text_parts.append(chunk.content)
                            yield TextDelta(text=chunk.content)

                        if chunk.reasoning_content:
                            yield ReasoningDelta(text=chunk.reasoning_content)

                        # 6. 合并工具调用片段
                        if chunk.tool_calls:
                            _merge_tool_calls(tool_call_buffer, chunk.tool_calls)

                        # 累积 usage
                        if chunk.usage:
                            _merge_usage(total_usage, chunk.usage)

                        if chunk.finish_reason == "stop":
                            break
            except asyncio.TimeoutError:
                yield AgentError(
                    error_type="call_timeout",
                    message=f"单次调用超时 {self._config.timeout}s",
                )
                return

            # 7. 记录 assistant 消息到历史
            assistant_content = "".join(turn_text_parts) if turn_text_parts else None

            # 清理 buffer：移除内部字段，转为标准格式
            resolved_calls = []
            for item in tool_call_buffer:
                resolved_calls.append({
                    "id": item["id"],
                    "type": "function",
                    "function": {
                        "name": item["function"]["name"],
                        "arguments": item["function"]["arguments"],
                    },
                })

            assistant_msg = Message(
                role=MessageRole.ASSISTANT,
                content=assistant_content,
                tool_calls=resolved_calls if resolved_calls else None,
            )
            self._history.add(assistant_msg)
            final_text = assistant_content or final_text

            # 8. 无工具调用 → 结束
            if not resolved_calls:
                yield AgentDone(
                    final_text=final_text,
                    total_usage=total_usage,
                    turn_count=turn_count,
                )
                return

            # 9. 检查最大轮次
            if turn_count >= self._config.max_turns:
                yield AgentError(
                    error_type="max_turns",
                    message=f"超出最大轮次 {self._config.max_turns}",
                )
                return

            # 10. 执行每个工具调用
            for call in resolved_calls:
                # 检查工具调用次数上限
                total_tool_calls += 1
                if total_tool_calls > self._config.tool_call_limit:
                    yield AgentError(
                        error_type="tool_limit",
                        message=f"超出工具调用上限 {self._config.tool_call_limit}",
                    )
                    return

                tool_call_id = call.get("id", "")
                fn = call.get("function", {})
                tool_name = fn.get("name", "")
                tool_args = fn.get("arguments", "{}")

                # 产出 ToolCallStart 事件
                yield ToolCallStart(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    arguments=tool_args,
                )

                # 危险工具确认检查
                wrapper = self._registry.get_tool(tool_name)
                if (self._config.confirm_dangerous
                        and wrapper and wrapper.dangerous
                        and self._on_confirm):
                    approved = await self._on_confirm(tool_name, tool_args)
                    yield ToolConfirmRequired(
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        arguments=tool_args,
                        approved=approved,
                    )
                    if not approved:
                        reject_msg = f"用户拒绝了危险工具 {tool_name} 的执行"
                        yield ToolResult(
                            tool_name=tool_name,
                            tool_call_id=tool_call_id,
                            output=reject_msg,
                            is_error=True,
                        )
                        self._history.add(Message(
                            role=MessageRole.TOOL,
                            content=reject_msg,
                            tool_call_id=tool_call_id,
                            name=tool_name,
                        ))
                        continue

                # 通过 executor 执行
                exec_result: ToolExecutionResult = await self._executor.execute(call)

                # 产出 ToolResult 事件
                yield ToolResult(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    output=exec_result.output,
                    is_error=exec_result.is_error,
                )

                # 将工具结果加入历史
                self._history.add(Message(
                    role=MessageRole.TOOL,
                    content=exec_result.output,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                ))

            # 11. 继续循环，LLM 将看到工具结果
