"""Agent 核心编排器 - LLM 调用、工具执行、对话历史管理"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, Union

import os

from agent_framework.agent.config import AgentConfig, AgentMode
from agent_framework.tools.executor import ToolExecutor, ToolExecutionResult
from agent_framework.agent.history import ConversationHistory
from agent_framework.agent.events import (
    AgentDone,
    AgentError,
    AgentEvent,
    ConfirmResponse,
    HistoryCompacted,
    ReasoningDelta,
    SessionLoaded,
    SubAgentDone,
    SubAgentEvent,
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

# 计划工具名称集合 — 不可与其他工具同批并发执行
_PLAN_TOOLS = {"todo_write", "todo_read"}

# 元工具名称集合 — 用于发现和加载工具（MCP / Skill），不视为"已开始工作"
_META_TOOLS = {"list_catalog", "search_tools", "activate_tools", "load_skill", "compact"}

# 技能目录自动搜索路径
_SKILL_SEARCH_PATHS = [
    "skills",
    os.path.expanduser("~/.agent_framework/skills"),
]


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
        system_prompt: str = "",
        tools: list | None = None,
        history: ConversationHistory | None = None,
        config: AgentConfig | None = None,
        on_confirm: Callable[[str, str], Awaitable[Union[bool, ConfirmResponse]]] | None = None,
        mcp_config_path: str | None = None,
        register_catalog: bool = True,
        skills: list | None = None,
        skills_dir: str | None = None,
        register_skills: bool = True,
        prompt_dir: "str | os.PathLike | None" = None,
        prompt_variables: dict[str, str] | None = None,
    ):
        self._llm = llm
        self._config = config or AgentConfig()
        self._history = history or ConversationHistory(
            strategy="auto_compact",
            max_tokens=self._config.max_total_tokens,
            llm=llm,
        )

        # ── 会话持久化 ──
        self._session = None
        if self._config.session_enabled:
            from agent_framework.agent.session import Session
            base_dir = Path(self._config.session_dir or ".sessions")
            model = self._extract_model_name(llm)
            self._session = Session(
                Session.generate_id(), base_dir, model=model,
            )
            self._history.attach_session(self._session)

        self._registry = ToolRegistry()
        if tools:
            self._registry.register_many(tools)

        # 注册元工具（用于发现和激活休眠工具）
        # register_catalog=False 时跳过（如 SubAgent，避免 LLM 被无用的元工具误导）
        self._catalog_tools = []
        if register_catalog:
            from agent_framework.tools.catalog import create_catalog_tools
            self._catalog_tools = create_catalog_tools(self._registry)
            self._registry.register_many(self._catalog_tools)

        # ── 压缩元工具（压缩器由 history 内部管理）──
        if register_catalog and self._history.compact_enabled:
            from agent_framework.agent.compactor import create_compact_tool
            self._compact_tool = create_compact_tool(self)
            self._registry.register(self._compact_tool)

        self._executor = ToolExecutor(self._registry, self._config)
        self._on_confirm = on_confirm

        # MCP 配置文件路径
        self._mcp_config_path = mcp_config_path
        self._mcp_manager = None

        # 流程约束状态标记
        self._work_started = False   # 非计划工具是否已执行
        self._plan_created = False   # todo_write 是否已被成功调用

        # ── 自动发现 TodoManager（通过工具标记）──
        self._todo_manager = self._find_todo_manager()
        if self._todo_manager and self._session:
            self._rebind_todo_manager()

        # ── 技能（Skills）初始化 ──
        from agent_framework.skills.registry import SkillRegistry
        self._skill_registry = SkillRegistry()

        # 注册编程方式传入的技能
        if skills:
            for cfg in skills:
                self._skill_registry.add(cfg)

        # 扫描技能目录
        if skills_dir:
            self._skill_registry.load_from_directory(skills_dir)
        else:
            for search_path in _SKILL_SEARCH_PATHS:
                if os.path.isdir(search_path):
                    self._skill_registry.load_from_directory(search_path)
                    break

        # ── PromptBuilder 初始化 ──
        self._prompt_builder = None
        self._prompt_variables: dict[str, str] = prompt_variables or {}
        if prompt_dir:
            from agent_framework.prompts.builder import PromptBuilder
            self._prompt_builder = PromptBuilder(prompt_dir)

        # 保存原始 system prompt（每轮动态拼装，不在此处 set_system）
        self._system_prompt = system_prompt

        # 注册 load_skill 元工具（register_skills=False 时跳过，如 SubAgent）
        if register_skills:
            self._registry.register_wrapper(self._skill_registry.as_tool())

    # -- 属性 ----------------------------------------------------------------

    @property
    def history(self) -> ConversationHistory:
        """对话历史管理器。"""
        return self._history

    @property
    def tools(self) -> ToolRegistry:
        """工具注册表。"""
        return self._registry

    @property
    def skill_registry(self) -> "SkillRegistry":
        """技能注册表。"""
        return self._skill_registry

    @property
    def prompt_builder(self):
        """提示词构建器（None 表示未使用 prompt_dir）。"""
        return self._prompt_builder

    @property
    def session(self):
        """当前会话（None 表示未启用 session）。"""
        return self._session

    # -- 公开方法 ------------------------------------------------------------

    def set_mode(self, mode: "AgentMode | str") -> None:
        """运行时切换操作模式。

        :param mode: AgentMode 枚举或字符串（talk / auto / superwork）
        """
        if isinstance(mode, str):
            mode = AgentMode(mode)
        old = self._config.mode
        self._config.mode = mode
        logger.info("Agent 模式切换: %s → %s", old.value, mode.value)

    @property
    def mode(self) -> AgentMode:
        """当前操作模式。"""
        return self._config.mode

    async def reset(self) -> None:
        """清空对话历史（保留 system 消息）。"""
        self._history.clear()
        self._work_started = False
        self._plan_created = False
        if self._todo_manager:
            self._todo_manager.clear()

    def save_session(self) -> None:
        """保存当前会话元数据到 session.json。"""
        if self._session:
            self._session.save_metadata()

    def new_session(self) -> None:
        """保存当前会话，创建新会话，清空历史。"""
        from agent_framework.agent.session import Session as SessionClass

        if self._session:
            self._session.save_metadata()

        base_dir = self._session.base_dir if self._session else Path(self._config.session_dir or ".sessions")
        model = self._extract_model_name(self._llm)

        self._session = SessionClass(SessionClass.generate_id(), base_dir, model=model)
        self._history.clear()
        self._history.attach_session(self._session)
        self._work_started = False
        self._plan_created = False
        self._rebind_todo_manager()

    def load_session(self, session_id: str) -> int:
        """加载历史会话，恢复对话。返回消息数量。"""
        from agent_framework.agent.session import Session as SessionClass

        # 保存当前会话
        if self._session:
            self._session.save_metadata()

        base_dir = self._session.base_dir if self._session else Path(self._config.session_dir or ".sessions")
        self._session = SessionClass.load_session(session_id, base_dir)
        self._history.load_from_session(self._session)
        self._work_started = False
        self._plan_created = False
        self._rebind_todo_manager()
        return self._session.message_count

    # -- 内部方法 ------------------------------------------------------------

    @staticmethod
    def _is_read_only_call(wrapper, args_str: str) -> bool:
        """判断一次工具调用是否为只读操作。

        判断逻辑（按优先级）：
        1. wrapper.read_only == True → 只读
        2. wrapper.read_only_check 存在 → 调用检查函数
        3. wrapper.read_only_actions 不为 None → 检查 args 中的 action 字段
        4. 否则 → 非只读
        """
        if wrapper.read_only:
            return True
        if wrapper.read_only_check is not None:
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                return bool(wrapper.read_only_check(args))
            except (json.JSONDecodeError, AttributeError, TypeError):
                return False
        if wrapper.read_only_actions is not None:
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                action = args.get("action", "")
                return action in wrapper.read_only_actions
            except (json.JSONDecodeError, AttributeError):
                return False
        return False

    @staticmethod
    def _extract_model_name(llm) -> str:
        """从 LLM 实例安全提取模型名称。"""
        try:
            config = getattr(llm, '_model_config', None)
            if config is not None and not isinstance(config, type):
                model = getattr(config, 'model', None)
                if isinstance(model, str):
                    return model
        except Exception:
            pass
        return ""

    def _build_system_prompt(self) -> None:
        """动态拼装 system prompt 并写入历史（每轮调用一次）。

        拼装顺序：
        1. PromptBuilder 输出（prompt_dir 指定的文件目录）
        2. system_prompt 字符串（向后兼容的内联提示词）
        3. 技能目录元数据
        """
        parts: list[str] = []

        # 1. PromptBuilder 输出（每轮重读文件 → 热重载）
        if self._prompt_builder:
            try:
                prompt_text = self._prompt_builder.build(**self._prompt_variables)
                if prompt_text:
                    parts.append(prompt_text)
            except FileNotFoundError as e:
                logger.warning("提示词目录加载失败: %s", e)

        # 2. 原始 system_prompt（向后兼容）
        if self._system_prompt:
            parts.append(self._system_prompt)

        # 3. 技能目录元数据
        skill_catalog = self._skill_registry.describe_available()
        if skill_catalog:
            parts.append(f"\n\n## 可用技能\n{skill_catalog}")

        self._history.set_system(
            Message(role=MessageRole.SYSTEM, content="".join(parts))
        )

    def _find_todo_manager(self):
        """从已注册工具中查找 TodoManager（通过 _todo_manager 标记）。"""
        for name in self._registry.list_tools():
            wrapper = self._registry.get_tool(name)
            if wrapper and hasattr(wrapper.func, '_todo_manager'):
                return wrapper.func._todo_manager
        return None

    def _rebind_todo_manager(self):
        """将 TodoManager 的 plan 文件绑定到当前 session 目录。"""
        if self._todo_manager and self._session:
            self._todo_manager.set_plan_file(self._session.dir_path / "plan.json")

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
            if self._config.mcp_tools_active_by_default:
                self._registry.register_wrapper(w)  # 向后兼容模式
            else:
                # 注册到休眠池（register_dormant 自动从工具名提取 category）
                self._registry.register_dormant(w)
        logger.info(
            "MCP 工具已注册: %d 个（%s）",
            len(wrappers),
            "活跃" if self._config.mcp_tools_active_by_default else "休眠",
        )

    async def disconnect_mcp(self) -> None:
        """断开所有 MCP 服务器连接。"""
        if self._mcp_manager:
            await self._mcp_manager.disconnect_all()
            self._mcp_manager = None

    async def __aenter__(self):
        await self.connect_mcp()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.save_session()
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

        self._work_started = False
        # self._plan_created = False #如果之前创建过任务，但对话中断后又继续，agent就认为没有创建过就不能再使用工具更新了。

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

            # 1.5 每轮重建 system prompt（反映技能目录等动态状态）
            self._build_system_prompt()

            # 1.6 自动压缩（如果启用）
            if self._history.compact_enabled:
                original_count = len(self._history._messages)
                original_messages = self._history._messages
                compacted = await self._history.auto_compact()
                if compacted is not original_messages:
                    yield HistoryCompacted(
                        strategy="auto",
                        original_count=original_count,
                        compacted_count=len(compacted),
                    )

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
                            # 更新压缩器的 token 跟踪
                            if chunk.usage.prompt_tokens:
                                self._history.update_prompt_tokens(chunk.usage.prompt_tokens)

                        if chunk.finish_reason == "stop":
                            break
            except asyncio.TimeoutError:
                yield AgentError(
                    error_type="call_timeout",
                    message=f"单次调用超时 {self._config.timeout}s",
                )
                return
            except Exception as e:
                # 检测上下文过长错误，尝试应急压缩后重试
                error_str = str(e).lower()
                context_too_long_keywords = [
                    "context_length", "too_long", "token", "maximum",
                    "max_tokens", "reduce", "truncat",
                ]
                is_context_error = any(
                    kw in error_str for kw in context_too_long_keywords
                )
                if is_context_error and self._history.compact_enabled:
                    logger.info("检测到上下文过长错误，触发应急压缩: %s", e)
                    original_count = len(self._history._messages)
                    original_messages = self._history._messages
                    compacted = await self._history.reactive_compact()
                    if compacted is not original_messages:
                        yield HistoryCompacted(
                            strategy="reactive",
                            original_count=original_count,
                            compacted_count=len(compacted),
                        )
                        continue  # 压缩成功，重试本轮
                    # 压缩无效果，无法恢复，直接抛出
                    raise
                raise

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
                self.save_session()
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

            # 10. 执行工具调用
            # ── 10a. 检查工具调用次数上限（提前检查整批）──
            batch_size = len(resolved_calls)
            if total_tool_calls + batch_size > self._config.tool_call_limit:
                yield AgentError(
                    error_type="tool_limit",
                    message=f"超出工具调用上限 {self._config.tool_call_limit}",
                )
                return
            total_tool_calls += batch_size

            # ── 10a.5 计划工具冲突检查（在危险工具确认之前）──
            # 如果计划工具与其他工具混在同一批，立即拒绝，不进入确认流程
            batch_names = {
                c.get("function", {}).get("name", "")
                for c in resolved_calls
            }
            has_plan = bool(batch_names & _PLAN_TOOLS)
            # 元工具（发现/加载）与计划工具同批不视为冲突
            has_non_plan = bool(batch_names - _PLAN_TOOLS - _META_TOOLS)

            if has_plan and has_non_plan:
                plan_names = ", ".join(batch_names & _PLAN_TOOLS)
                other_names = ", ".join(batch_names - _PLAN_TOOLS)
                error_msg = (
                    f"错误：计划工具（{plan_names}）不能与其他工具（{other_names}）"
                    f"在同一批调用中同时执行。"
                    f"请先单独调用计划工具完成计划更新，"
                    f"然后在下一轮对话中再调用其他工具。"
                )
                for call in resolved_calls:
                    tc_id = call.get("id", "")
                    tn = call.get("function", {}).get("name", "")
                    ta = call.get("function", {}).get("arguments", "{}")
                    yield ToolCallStart(tool_name=tn, tool_call_id=tc_id, arguments=ta)
                    yield ToolResult(
                        tool_name=tn, tool_call_id=tc_id,
                        output=error_msg, is_error=True,
                    )
                    self._history.add(Message(
                        role=MessageRole.TOOL, content=error_msg,
                        tool_call_id=tc_id, name=tn,
                    ))
                continue  # 跳过确认循环和 asyncio.gather，直接进入下一轮

            # ── 10a.6 跨轮次执行顺序守卫 ──
            # 如果已经执行过非计划工具，禁止再创建计划
            if self._work_started and not self._plan_created:
                late_plan = batch_names & _PLAN_TOOLS
                if late_plan:
                    plan_names = ", ".join(late_plan)
                    error_msg = (
                        f"流程约束：已经开始执行任务后不能再创建计划（{plan_names}）。"
                        f"请直接继续完成工作，无需补做计划。"
                    )
                    for call in resolved_calls:
                        tc_id = call.get("id", "")
                        tn = call.get("function", {}).get("name", "")
                        ta = call.get("function", {}).get("arguments", "{}")
                        yield ToolCallStart(tool_name=tn, tool_call_id=tc_id, arguments=ta)
                        yield ToolResult(
                            tool_name=tn, tool_call_id=tc_id,
                            output=error_msg, is_error=True,
                        )
                        self._history.add(Message(
                            role=MessageRole.TOOL, content=error_msg,
                            tool_call_id=tc_id, name=tn,
                        ))
                    continue

            # ── 10a.7 talk 模式只读检查 ──
            # talk 模式下，阻止所有非只读工具调用
            if self._config.mode == AgentMode.TALK:
                blocked = []
                allowed = []
                for call in resolved_calls:
                    fn = call.get("function", {})
                    tool_name = fn.get("name", "")
                    tool_args = fn.get("arguments", "{}")
                    wrapper = self._registry.get_tool(tool_name)
                    if wrapper and self._is_read_only_call(wrapper, tool_args):
                        allowed.append(call)
                    else:
                        blocked.append(call)

                if blocked:
                    for call in blocked:
                        tc_id = call.get("id", "")
                        fn = call.get("function", {})
                        tn = fn.get("name", "")
                        ta = fn.get("arguments", "{}")
                        yield ToolCallStart(tool_name=tn, tool_call_id=tc_id, arguments=ta)
                        error_msg = (
                            f"[talk 模式] 工具 {tn} 不可用：当前为只读模式，"
                            f"仅允许只读操作。请用只读工具或指令完成需求。"
                        )
                        yield ToolResult(
                            tool_name=tn, tool_call_id=tc_id,
                            output=error_msg, is_error=True,
                        )
                        self._history.add(Message(
                            role=MessageRole.TOOL, content=error_msg,
                            tool_call_id=tc_id, name=tn,
                        ))
                    if not allowed:
                        continue  # 全部被阻止，跳过确认和执行
                    # 部分被阻止，只执行允许的
                    resolved_calls = allowed

            # ── 10b. 顺序发出 ToolCallStart + 危险工具确认 ──
            confirmed = []
            for call in resolved_calls:
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

                # 危险工具确认检查（superwork 和 talk 模式跳过确认，因为talk模式以在上面检查拦截）
                wrapper = self._registry.get_tool(tool_name)
                if (self._config.mode not in (AgentMode.SUPERWORK, AgentMode.TALK)
                        and self._config.confirm_dangerous
                        and wrapper and wrapper.dangerous
                        and self._on_confirm):
                    raw_result = await self._on_confirm(tool_name, tool_args)

                    # 兼容 bool 和 ConfirmResponse 两种返回
                    if isinstance(raw_result, ConfirmResponse):
                        approved = raw_result.approved
                        user_message = raw_result.message
                    else:
                        approved = bool(raw_result)
                        user_message = ""

                    yield ToolConfirmRequired(
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        arguments=tool_args,
                        approved=approved,
                        message=user_message,
                    )
                    if not approved:
                        if user_message:
                            reject_msg = (
                                f"用户拒绝了工具 {tool_name} 的执行，并给出指示：{user_message}"
                            )
                        else:
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

                confirmed.append({
                    "call": call,
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "wrapper": wrapper,
                })

            # ── 10c. 并发执行所有已确认的工具 ──
            if confirmed:
                async def _execute_one(item):
                    result = await self._executor.execute(item["call"])
                    return item, result

                results = await asyncio.gather(
                    *[_execute_one(item) for item in confirmed]
                )

                # ── 10d. 顺序处理结果：转发事件、产出 ToolResult ──
                for item, exec_result in results:
                    tool_name = item["tool_name"]
                    tool_call_id = item["tool_call_id"]
                    wrapper = item["wrapper"]

                    # 子代理事件转发
                    if wrapper and getattr(wrapper, '_is_subagent', False):
                        sub_events = getattr(wrapper, '_subagent_events', [])
                        sub_final_text = ""
                        sub_turn_count = 0
                        sub_total_usage = TokenUsage()
                        sub_is_error = False

                        for sevt in sub_events:
                            yield SubAgentEvent(subagent_name=tool_name, event=sevt)
                            if isinstance(sevt, AgentDone):
                                sub_final_text = sevt.final_text
                                sub_turn_count = sevt.turn_count
                                sub_total_usage = sevt.total_usage
                            elif isinstance(sevt, AgentError):
                                sub_is_error = True
                                sub_final_text = sevt.message

                        yield SubAgentDone(
                            subagent_name=tool_name,
                            final_text=sub_final_text,
                            turn_count=sub_turn_count,
                            total_usage=sub_total_usage,
                            is_error=sub_is_error,
                        )

                    # 产出 ToolResult 事件
                    yield ToolResult(
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        output=exec_result.output,
                        is_error=exec_result.is_error,
                    )

                    # 标记流程约束状态
                    if tool_name == "todo_write" and not exec_result.is_error:
                        self._plan_created = True
                    if tool_name not in _PLAN_TOOLS and tool_name not in _META_TOOLS:
                        self._work_started = True

                    # 将工具结果加入历史
                    self._history.add(Message(
                        role=MessageRole.TOOL,
                        content=exec_result.output,
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    ))

            # 11. 继续循环，LLM 将看到工具结果
