"""Agent 核心编排器 - LLM 调用、工具执行、对话历史管理"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Awaitable, Callable, Union

if TYPE_CHECKING:
    from milu.tools.mcp.manager import MCPManager

import os

from milu.agent.config import AgentConfig, AgentMode
from milu.tools.executor import ToolExecutor, ToolExecutionResult
from milu.agent.history import ConversationHistory
from milu.agent.events import (
    AgentDone,
    AgentError,
    AgentEvent,
    ConfirmResponse,
    HistoryCompacted,
    ReasoningDelta,
    SafetyCheckStart,
    SessionLoaded,
    SubAgentDone,
    SubAgentEvent,
    TextDelta,
    ToolCallPreparing,
    ToolCallStart,
    ToolConfirmRequired,
    ToolResult,
)
from milu.llm.base.message import Message, MessageRole
from milu.llm.base.response import StreamChunk, TokenUsage
from milu.llm.providers.base import BaseLLM
from milu.tools.registry import ToolRegistry
from milu.tools.builtin.todo_write import (
    _current_session_dir as _current_todo_session_dir,
    _current_plan_items as _current_todo_plan_items,
)
from milu.tools.builtin.memory_tool import (
    _current_memory_path,
    memory_file_path,
    memory_read,
    memory_write,
    render_memory_prompt,
)
from milu.tools.builtin.image_tool import (
    _current_pending_images,
    _current_vision_support,
)
from milu.tools.builtin.schedule_tool import _current_schedule_user
from milu.tools.builtin.knowledge_tool import (
    _current_knowledge,
    KnowledgeRuntime,
    kb_ingest,
    kb_manage,
    kb_search,
    render_knowledge_prompt,
)
from milu.knowledge import KnowledgeConfig
from milu.exceptions import content_safety_hint
from milu.llm.base.vision import build_user_content
from milu.agent.subagent import (
    _current_subagent_events,
    _current_parent_mode,
    _current_parent_confirm,
    _current_parent_judge,
    _current_parent_llm_kwargs,
    _INHERITED_LLM_KWARGS,
)

logger = logging.getLogger(__name__)

# 计划工具名称集合 — 不可与其他工具同批并发执行
_PLAN_TOOLS = {"todo_write", "todo_read"}

# 元工具名称集合 — 用于发现和加载工具（MCP / Skill），不视为"已开始工作"
_META_TOOLS = {"list_catalog", "search_tools", "activate_tools", "load_skill", "compact"}

# prompt_dir / skills_dir 的默认解析约定（见 __init__）：
#   None  → 用内置默认资源（仅顶层 Agent；子代理 register_catalog=False 保持精简，不注入）
#   路径  → 用该路径
# 顶层 Agent 与子代理用 register_catalog 区分：只有子代理会显式传 register_catalog=False。


# ---------------------------------------------------------------------------
# 合并工具调用片段
# ---------------------------------------------------------------------------

def _extract_tool_call_extra_content(tc) -> dict | None:
    """从原始 tool_call delta 提取 provider 透传的 extra_content。

    目前用于 Gemini 思考模型的 thought_signature——它位于
    tool_call.extra_content.google.thought_signature，多轮工具调用回传历史时【必须】
    原样带回（否则 400 "Function call is missing a thought_signature"）。
    OpenAI SDK 对未知字段 extra="allow"，可能以属性或 model_extra 暴露；值可能是
    dict 或 pydantic 模型，统一转成普通 dict 以便 JSON 序列化、入历史、回传。
    非 Gemini provider 不产生该字段，返回 None（通用、无副作用）。
    """
    raw = getattr(tc, "extra_content", None)
    if raw is None:
        model_extra = getattr(tc, "model_extra", None)
        if isinstance(model_extra, dict):
            raw = model_extra.get("extra_content")
    if raw is None:
        return None
    if hasattr(raw, "model_dump"):
        try:
            raw = raw.model_dump(exclude_none=True)
        except Exception:
            return None
    return raw if isinstance(raw, dict) and raw else None


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
        idx = getattr(tc, "index", None)
        call_id = getattr(tc, "id", None) or ""
        fn = getattr(tc, "function", None)
        fn_name = getattr(fn, "name", "") if fn else ""
        fn_args = getattr(fn, "arguments", "") if fn else ""
        # provider 透传字段（如 Gemini thought_signature）——通常随首片到达
        extra_content = _extract_tool_call_extra_content(tc)

        # 续接目标判定：
        # - 非空 id = 一个新工具调用的起始标志（OpenAI 流式里 id 只在首片出现一次）。
        #   即便多个并行调用 index 缺失或都为 0（Gemini 兼容层的已知行为），靠唯一 id
        #   也能正确拆分，不会把 researcher+coder 误并成 "researchercoder"。
        # - 无 id = 参数续片，按 index 归属；index 也缺失时归到最近一条（buffer[-1]）。
        existing = None
        if call_id:
            for item in buffer:
                if item.get("id") == call_id:
                    existing = item
                    break
        else:
            if idx is not None:
                for item in buffer:
                    if item["_idx"] == idx:
                        existing = item
                        break
            if existing is None and buffer:
                existing = buffer[-1]

        if existing is None:
            # 新建条目；index 缺失时用合成序号保证唯一
            item = {
                "_idx": idx if idx is not None else len(buffer),
                "id": call_id,
                "function": {
                    "name": fn_name,
                    "arguments": fn_args,
                },
            }
            if extra_content:
                item["extra_content"] = extra_content
            buffer.append(item)
        else:
            # 追加拼接（id 只在首片出现，不覆盖已有 id）
            if call_id and not existing.get("id"):
                existing["id"] = call_id
            if fn_name:
                existing["function"]["name"] += fn_name
            if fn_args:
                existing["function"]["arguments"] += fn_args
            # 后续片若携带 extra_content 也补上（一般首片即有）
            if extra_content and "extra_content" not in existing:
                existing["extra_content"] = extra_content

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


def _safe_reset(var, token) -> None:
    """重置 ContextVar token，容忍「生成器在其它 Context 中被 finalize」。

    消费方对 run() 提前 break / 客户端断连时，异步生成器被事件循环的
    finalizer 在新任务（新 Context）中执行 finally——此时 reset 会抛
    ValueError: Token was created in a different Context。原 Context 已随
    任务结束销毁，无需（也无法）重置，安全忽略即可；且不能因第一个 reset
    抛错而跳过后续 token 的重置。
    """
    try:
        var.reset(token)
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Agent 类
# ---------------------------------------------------------------------------

class Agent:
    """Agent 编排器 - 负责 LLM 调用、工具执行、对话历史管理。"""

    def __init__(
        self,
        llm: BaseLLM,
        system_prompt: str = "",
        tools: list | None = None,  # None→全套内置工具（顶层）；[]→无工具；列表→该列表
        history: ConversationHistory | None = None,
        config: AgentConfig | None = None,
        on_confirm: Callable[[str, str], Awaitable[Union[bool, ConfirmResponse]]] | None = None,
        mcp_config_path: str | None = None,
        register_catalog: bool = True,
        skills: list | None = None,
        skills_dir: "str | os.PathLike | None" = None,  # None→内置技能（顶层）；路径→该目录
        register_skills: bool = True,
        prompt_dir: "str | os.PathLike | None" = None,  # None→内置 main（顶层且无 system_prompt）；路径→该目录
        prompt_variables: dict[str, str] | None = None,
        session_id: str | None = None,
        mcp_manager: "MCPManager | None" = None,
        # ── Agent 能力参数（原属 AgentConfig，现为 Agent 直接参数）──
        mode: "AgentMode | str" = AgentMode.AUTO,           # 操作模式：talk/manual/auto/superwork
        session_enabled: bool = True,                       # 是否自动创建会话（对话日志持久化）
        session_dir: "str | os.PathLike | None" = None,     # None→用户级 ~/.milu/sessions
        mcp_tools_active_by_default: bool = False,          # MCP 工具是否默认进活跃池（否则休眠）
        subagents: "list | None" = None,                    # None→内置三件套（顶层）；[]→无；列表→自定义 SubAgentConfig
        memory: "bool | str" = False,                       # 长期记忆开关：False→关闭（默认）；True→启用（身份 "default"）；字符串→启用并按该用户标识隔离存储
        knowledge: "bool | str | KnowledgeConfig" = False,  # 向量知识库开关：False→关闭（默认）；True→启用（身份 "default"）；字符串→按该用户标识隔离；KnowledgeConfig→程序化定制
        schedule_user: "str | None" = None,                 # 定时任务用户标识：None→工具层退化为 "default"（CLI 单人）；多用户场景传 user_id（任务文件按用户隔离）
        judge_llm: "BaseLLM | bool | None" = None,          # auto 模式 AI 安全判定器：None→默认复用主 llm；False→关闭；实例→指定模型
        judge_rules: str = "",                              # 追加给判定器的自定义规则文本（如「禁止访问生产库」）
    ):
        self._llm = llm
        # 复制传入的 config，保证每个 Agent 持有独立 AgentConfig 实例（防御性隔离，
        # 避免调用方复用/事后修改同一 config 影响多个 Agent）。AgentConfig 现仅含不可变
        # 运行限额标量、运行期不再被原地修改，浅拷贝（dataclasses.replace）即可。
        self._config = dataclasses.replace(config) if config is not None else AgentConfig()

        # Agent 能力参数（实例级）：mode 运行期可变（set_mode），不再寄居于共享 config，
        # 因此天然无跨用户串扰风险（旧设计正是为此才要拷贝 config.mode）。
        self._mode = AgentMode(mode) if isinstance(mode, str) else mode
        self._session_enabled = session_enabled
        self._mcp_tools_active_by_default = mcp_tools_active_by_default
        # AI 安全判定器（auto 模式兜底安全网）解析（与 tools/subagents 同一约定）：
        #   None/True（默认）→ 复用主 llm（判定调用以低温度发起，见 judge.py）；
        #   False → 关闭判定；BaseLLM 实例 → 指定判定模型（建议便宜快速模型）。
        # judge_llm 为 AsyncOpenAI 封装，协程安全可跨 Agent 共享；
        # judge_rules 为不可变字符串。均无跨用户串扰风险。
        if judge_llm is None or judge_llm is True:
            self._judge_llm = llm
        elif judge_llm is False:
            self._judge_llm = None
        else:
            self._judge_llm = judge_llm
        self._judge_rules = judge_rules
        # session 根目录：None → 用户级默认（与 CWD 解耦）；否则用传入路径
        if session_dir is None:
            from milu.resources import default_session_dir
            self._session_dir = default_session_dir()
        else:
            self._session_dir = Path(session_dir)

        self._history = history or ConversationHistory(
            strategy="auto_compact",
            max_tokens=self._config.max_total_tokens,
            llm=llm,
        )

        # ── 会话持久化 ──
        self._session = None
        if self._session_enabled:
            from milu.agent.session import Session
            base_dir = self._session_dir
            model = self._extract_model_name(llm)
            # session_id 为 None → 生成随机 id（一次性会话）；
            # 显式传入 → 使用确定性 id，便于淘汰/重启/多 worker 共享存储后
            # 按 (user, session) 定位并恢复历史。
            sid = session_id or Session.generate_id()
            self._session = Session(sid, base_dir, model=model)
            # 若该 session 已有历史日志（重建/恢复场景），加载之；否则挂载空会话。
            if self._session.conversation_path.exists():
                self._history.load_from_session(self._session)
            else:
                self._history.attach_session(self._session)

        # 工具默认（与 prompt_dir/skills_dir 同一约定）：
        #   None + 顶层 Agent → 全套内置工具（开箱即用）；显式 [] → 无工具；显式列表 → 该列表
        #   子代理（register_catalog=False）传 None 时保持精简，不注入内置工具
        resolved_tools = tools
        if resolved_tools is None and register_catalog:
            from milu.tools.builtin import BUILTIN_TOOLS
            resolved_tools = BUILTIN_TOOLS
        self._registry = ToolRegistry()
        if resolved_tools:
            self._registry.register_many(resolved_tools)

        # ── 长期记忆（默认关闭，单开关启用）──
        # memory=False → 关闭；True → 启用（身份 "default"）；字符串 → 启用并作为
        # 用户标识（多用户场景传 user_id，按用户隔离）。存储与 session 解耦：
        # 用户级文件 ~/.milu/memory/{user_id}.json，同一标识跨会话共享。
        # 启用时注册 memory 工具 + 每轮把条目渲染进 system prompt 末尾（见
        # _build_system_prompt）；路径在 run() 入口经 ContextVar 注入工具层。
        self._memory_path: Path | None = None
        if memory:
            self._memory_path = memory_file_path(
                memory if isinstance(memory, str) else "default"
            )
            self._registry.register_many([memory_write, memory_read])

        # ── 向量知识库（默认关闭，与 memory 同款开关约定）──
        # knowledge=False → 关闭；True → 启用（身份 "default"）；字符串 → 按用户
        # 标识隔离（~/.milu/knowledge/{user_id}/）；KnowledgeConfig → 程序化定制
        # （embedding 厂商/模型/分块参数）。库纯净性：Agent 不读 config.json，
        # 应用/CLI 入口读分层配置后构造 KnowledgeConfig 传入。启用时注册 kb_*
        # 工具；KnowledgeRuntime 在 run() 入口经 ContextVar 注入工具层。
        self._knowledge: KnowledgeRuntime | None = None
        if knowledge:
            if isinstance(knowledge, KnowledgeConfig):
                kn_cfg = knowledge
            elif isinstance(knowledge, str):
                kn_cfg = KnowledgeConfig(user_id=knowledge)
            else:
                kn_cfg = KnowledgeConfig()
            self._knowledge = KnowledgeRuntime(kn_cfg)
            self._registry.register_many([kb_search, kb_ingest, kb_manage])

        # ── 定时任务用户标识（与 memory 平级的能力参数）──
        # None → 工具层 _resolve_user() 退化为 "default"（CLI 单人行为不变）；
        # 字符串 → schedule 工具按该用户隔离任务文件。run() 入口经 ContextVar
        # 注入工具层（显式注入 None 也起隔离作用：子代理不会继承父的标识）。
        self._schedule_user = schedule_user

        # 子代理默认（同一约定）：
        #   None + 顶层 Agent → 内置三件套（researcher/reader/coder）；显式 [] → 无；
        #   显式列表 → 自定义 SubAgentConfig。子代理自身 register_catalog=False，
        #   不会触发此默认 → 结构上保证子代理不嵌套子代理。
        #   mode / 确认回调由 run() 经 ContextVar 注入，子代理工具与本 Agent 同 llm。
        resolved_subagents = subagents
        if resolved_subagents is None and register_catalog:
            from milu.agent.subagent import builtin_subagent_configs
            resolved_subagents = builtin_subagent_configs()
        if resolved_subagents:
            from milu.agent.subagent import create_subagent_tools
            self._registry.register_many(create_subagent_tools(llm, resolved_subagents))

        # 注册元工具（用于发现和激活休眠工具）
        # register_catalog=False 时跳过（如 SubAgent，避免 LLM 被无用的元工具误导）
        self._catalog_tools = []
        if register_catalog:
            from milu.tools.catalog import create_catalog_tools
            self._catalog_tools = create_catalog_tools(self._registry)
            self._registry.register_many(self._catalog_tools)

        # ── 压缩元工具（压缩器由 history 内部管理）──
        if register_catalog and self._history.compact_enabled:
            from milu.agent.compactor import create_compact_tool
            self._compact_tool = create_compact_tool(self)
            self._registry.register(self._compact_tool)

        self._executor = ToolExecutor(self._registry, self._config)
        self._on_confirm = on_confirm

        # 可选钩子：等待人工确认期间进入此 async 上下文管理器，用于释放外部并发许可
        # （如 AgentPool 的全局 Semaphore），避免长时间人工确认占用并发名额。
        # None 表示不启用（独立使用 Agent 时为 no-op）。由 AgentPool 注入。
        self._confirm_wait_guard: Callable[[], "contextlib.AbstractAsyncContextManager"] | None = None

        # MCP 配置文件路径
        self._mcp_config_path = mcp_config_path
        # MCP 管理器与所有权：
        # - 未注入（默认）：本 Agent 在 connect_mcp() 时自建并「拥有」manager，
        #   disconnect 时断开。每个 Agent 一组 MCP 子进程（per-agent，进程独占）。
        # - 注入共享 manager：多个 Agent 复用同一组已连接的 MCP 子进程（省内存），
        #   本 Agent「不拥有」它——disconnect_mcp() 不会断开，生命周期由注入方统一管理。
        self._mcp_manager: "MCPManager | None" = mcp_manager
        self._owns_mcp = mcp_manager is None
        if mcp_manager is not None:
            # 共享 manager 应为「已连接」状态：直接注册其工具（不再拉起新进程）。
            self._register_mcp_tools(mcp_manager.get_tools())

        # 流程约束状态标记
        self._work_started = False   # 非计划工具是否已执行
        self._plan_created = False   # todo_write 是否已被成功调用

        # todo 计划的内存后端：无 session 时承载计划（同一 Agent 跨轮保留）。
        # 有 session 时不用它（计划落盘 session_dir/plan.json）。在 run() 入口择一注入。
        self._in_memory_plan: list[dict] = []

        # ── 技能（Skills）初始化 ──
        from milu.skills.registry import SkillRegistry
        self._skill_registry = SkillRegistry()

        # 注册编程方式传入的技能
        if skills:
            for cfg in skills:
                self._skill_registry.add(cfg)

        # 技能目录默认：None + 顶层 Agent → 内置技能目录；显式路径 → 该目录。
        # 子代理（register_catalog=False）保持精简，None 时不注入内置技能。
        resolved_skills_dir = skills_dir
        if resolved_skills_dir is None and register_catalog:
            from milu.resources import builtin_skills_dir
            resolved_skills_dir = str(builtin_skills_dir())
        if resolved_skills_dir:
            self._skill_registry.load_from_directory(str(resolved_skills_dir))

        # ── PromptBuilder 初始化（None 默认 + 与 system_prompt 互斥）──
        # prompt_dir 解析：
        #   显式路径                          → 该目录
        #   None + 有 system_prompt           → 只用 system_prompt（不接内置）
        #   None + 无 system_prompt + 顶层    → 内置 main 角色提示词（开箱即用）
        #   子代理（register_catalog=False）   → None 时不注入内置 main，保持精简
        resolved_prompt_dir = prompt_dir
        if resolved_prompt_dir is None and not system_prompt and register_catalog:
            from milu.resources import builtin_prompts_dir
            resolved_prompt_dir = builtin_prompts_dir("main")

        self._prompt_builder = None
        self._prompt_variables: dict[str, str] = prompt_variables or {}
        if resolved_prompt_dir:
            from milu.prompts.builder import PromptBuilder
            self._prompt_builder = PromptBuilder(resolved_prompt_dir)

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

        :param mode: AgentMode 枚举或字符串（talk / manual / auto / superwork）
        """
        if isinstance(mode, str):
            mode = AgentMode(mode)
        old = self._mode
        self._mode = mode
        logger.info("Agent 模式切换: %s → %s", old.value, mode.value)

    @property
    def mode(self) -> AgentMode:
        """当前操作模式。"""
        return self._mode

    def set_confirm_wait_guard(
        self, guard: "Callable[[], contextlib.AbstractAsyncContextManager] | None"
    ) -> None:
        """注入「等待人工确认」期间使用的 async 上下文管理器工厂。

        AgentPool 用它在确认等待期间临时释放全局并发许可（Semaphore），
        确认结束后重新获取，避免长时间人工确认阻塞占用 max_concurrent_runs 名额。

        :param guard: 无参可调用，返回一个 async context manager；None 表示停用。
        """
        self._confirm_wait_guard = guard

    async def reset(self) -> None:
        """清空对话历史（保留 system 消息），并切换到新的会话日志段。

        session 目录与 session_id 不变，旧日志段保留（归档）；新消息写入
        conversation.N.jsonl 新段，load_session 也只会恢复新段内容。
        """
        self._history.clear()
        if self._session is not None:
            self._session.reset()
        self._work_started = False
        self._plan_created = False
        # v2: TodoManager 已删除，plan 状态由 session/plan.json 承载
        # reset() 不需要清空 plan 文件（plan 跨 reset 保留）

    def save_session(self) -> None:
        """保存当前会话元数据到 session.json。"""
        if self._session:
            self._session.save_metadata()

    def new_session(self) -> None:
        """保存当前会话，创建新会话，清空历史。"""
        from milu.agent.session import Session as SessionClass

        if self._session:
            self._session.save_metadata()

        base_dir = self._session.base_dir if self._session else self._session_dir
        model = self._extract_model_name(self._llm)

        self._session = SessionClass(SessionClass.generate_id(), base_dir, model=model)
        self._history.clear()
        self._history.attach_session(self._session)
        self._work_started = False
        self._plan_created = False

    def load_session(self, session_id: str) -> int:
        """加载历史会话，恢复对话。返回消息数量。"""
        from milu.agent.session import Session as SessionClass

        # 保存当前会话
        if self._session:
            self._session.save_metadata()

        base_dir = self._session.base_dir if self._session else self._session_dir
        self._session = SessionClass.load_session(session_id, base_dir)
        self._history.load_from_session(self._session)
        self._work_started = False
        self._plan_created = False
        return self._session.message_count

    # -- 内部方法 ------------------------------------------------------------

    @staticmethod
    def _is_safe_call(wrapper, args_str: str) -> bool:
        """判断一次工具调用是否为安全操作。

        判断逻辑（按优先级）：
        1. wrapper.is_safe == True → 安全
        2. wrapper.safe_check 存在 → 调用检查函数
        3. 否则 → 不安全
        """
        if wrapper.is_safe:
            return True
        if wrapper.safe_check is not None:
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                return bool(wrapper.safe_check(args))
            except Exception:
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

    @staticmethod
    def _default_prompt_variables() -> dict[str, str]:
        """环境上下文变量（每轮重算），提示词模板可直接引用：

        - {{current_date}}：当前日期（ISO 格式），供时效判断
        - {{platform}}：操作系统（Windows / Linux / Darwin）
        - {{cwd}}：当前工作目录

        对齐业界实践（Claude Code 等在 system prompt 注入运行环境上下文）。
        用户经 prompt_variables 传入的同名变量优先于这些默认值。
        """
        import platform as _platform
        from datetime import date
        return {
            "current_date": date.today().isoformat(),
            "platform": _platform.system(),
            "cwd": str(Path.cwd()),
        }

    def _build_system_prompt(self) -> None:
        """动态拼装 system prompt 并写入历史（每轮调用一次）。

        拼装顺序：
        1. PromptBuilder 输出（prompt_dir 指定的文件目录）
        2. system_prompt 字符串（向后兼容的内联提示词）
        3. 技能目录元数据
        4. 知识库来源目录 + 检索路由规则（启用 knowledge 时）
        5. 长期记忆条目（启用 memory 时，固定在最后）
        """
        parts: list[str] = []

        # 1. PromptBuilder 输出（每轮重读文件 → 热重载；环境变量每轮重算保持新鲜）
        if self._prompt_builder:
            try:
                variables = {**self._default_prompt_variables(), **self._prompt_variables}
                prompt_text = self._prompt_builder.build(**variables)
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

        # 4. 知识库来源目录 + 检索路由规则（启用时每轮渲染；模型知道库里
        #    有什么才会主动 kb_search——经 mtime 缓存，文件未变不读盘）；
        #    auto_retrieve 启用时附带本轮前置自动检索结果块
        if self._knowledge is not None:
            parts.append(render_knowledge_prompt(
                self._knowledge.store, self._knowledge.auto_context,
            ))

        # 5. 长期记忆（启用时注入在最后；每轮重读文件，同一用户其他
        #    Agent/进程的新写入即时可见，与提示词热重载同理）
        if self._memory_path is not None:
            parts.append(render_memory_prompt(self._memory_path))

        self._history.set_system(
            Message(role=MessageRole.SYSTEM, content="".join(parts))
        )

    # -- MCP 生命周期 --------------------------------------------------------

    async def connect_mcp(self) -> None:
        """连接所有 MCP 服务器，发现并注册工具。

        配置路径解析优先级：
        1. mcp_config_path 参数（构造函数传入）
        2. 环境变量 MCP_CONFIG_PATH（.env 或系统环境变量）
        3. 自动搜索 config/mcp_servers.json → ~/.milu/mcp_servers.json
        """
        from milu.tools.mcp.config import MCPServerConfig
        from milu.tools.mcp.manager import MCPManager
        from milu._env import ensure_dotenv_loaded
        import os

        # 确定配置文件路径
        path = self._mcp_config_path
        if not path:
            ensure_dotenv_loaded()
            path = os.environ.get("MCP_CONFIG_PATH")

        # 幂等 / 共享保护：已有 manager（注入的共享 manager，或本 Agent 已连接过）
        # 则直接返回，绝不重复连接、绝不拉起第二组进程。
        if self._mcp_manager is not None:
            return

        configs = MCPServerConfig.load_file(path)  # path=None 时自动搜索

        if not configs:
            return

        self._mcp_manager = MCPManager(configs)
        wrappers = await self._mcp_manager.connect_all()
        self._register_mcp_tools(wrappers)
        logger.info(
            "MCP 工具已注册: %d 个（%s）",
            len(wrappers),
            "活跃" if self._mcp_tools_active_by_default else "休眠",
        )

    def _register_mcp_tools(self, wrappers: list) -> None:
        """把 MCP 工具按配置注册进工具表（活跃池或休眠池）。

        被 connect_mcp()（自建 manager）与 __init__（注入共享 manager）共用。
        """
        for w in wrappers:
            if self._mcp_tools_active_by_default:
                self._registry.register_wrapper(w)  # 向后兼容模式：直接进活跃池
            else:
                # 注册到休眠池（register_dormant 自动从工具名提取 category）
                self._registry.register_dormant(w)

    async def disconnect_mcp(self) -> None:
        """断开 MCP 服务器连接。

        仅断开本 Agent「拥有」的 manager；注入的共享 manager 不在此断开
        （由注入方——如 AgentPool——统一管理生命周期），避免一个用户结束就
        把其他用户共享的 MCP 进程关掉。
        """
        if self._mcp_manager and self._owns_mcp:
            await self._mcp_manager.disconnect_all()
            self._mcp_manager = None

    async def close_knowledge(self) -> None:
        """关闭知识库的 embedding 客户端，释放连接池。

        未启用 knowledge 时为 no-op；重复调用安全。与 disconnect_mcp 并列在
        __aexit__ / AgentPool 淘汰路径调用（Embedder 为 per-Agent 资源）。
        """
        if self._knowledge is not None:
            await self._knowledge.aclose()

    async def __aenter__(self):
        await self.connect_mcp()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.save_session()
        await self.disconnect_mcp()
        await self.close_knowledge()
        return False

    async def run(
        self,
        user_input: str,
        images: list[str] | None = None,
        **llm_kwargs,
    ) -> AsyncIterator[AgentEvent]:
        """运行 Agent 循环。

        Args:
            user_input: 用户输入
            images: 随消息附带的本地图片路径列表（可选）。需要 LLM 支持视觉
                （capabilities.supports_vision），不支持时降级为纯文本并记录警告。
                历史中只存轻量 image_path 引用块，发送 API 时才物化为 base64。
            **llm_kwargs: 传递给 LLM 的额外参数

        Yields:
            AgentEvent 子类实例
        """
        # 注入 per-run 状态到 ContextVar（asyncio 任务级隔离）：
        #   - todo 计划存储：有 session → 文件后端（session_dir）；无 session → 内存后端
        #   - 长期记忆文件路径：未启用 memory 时注入 None（显式隔离，子代理不会
        #     经 Context 继承父的记忆路径而意外写入）
        #   - 父 Agent 操作模式：供子代理工具继承（无需 get_parent_mode 回调）
        session_token = None
        plan_items_token = None
        if self._session is not None:
            session_token = _current_todo_session_dir.set(self._session.dir_path)
        else:
            plan_items_token = _current_todo_plan_items.set(self._in_memory_plan)
        memory_token = _current_memory_path.set(self._memory_path)
        # 知识库运行时（未启用时注入 None → 工具返回启用指引；子代理不继承）
        knowledge_token = _current_knowledge.set(self._knowledge)
        # 定时任务用户标识（未设置时注入 None → 工具层退化 "default"）
        schedule_user_token = _current_schedule_user.set(self._schedule_user)
        mode_token = _current_parent_mode.set(self._mode)
        # 确认回调透传：子代理继承父的 on_confirm（manual 模式不安全工具、
        # auto 模式 AI 判定 confirm 的调用同样走审批），委派不构成安全旁路
        confirm_token = _current_parent_confirm.set(self._on_confirm)
        # AI 安全判定器透传：子代理继承父的 judge_llm/judge_rules，委派不旁路判定
        judge_token = _current_parent_judge.set(
            (self._judge_llm, self._judge_rules) if self._judge_llm else None
        )
        # 思考设置透传：子代理继承父的 enable_thinking/thinking_level（关思考就全局关）。
        # 必须在下方 llm_kwargs 被注入 tools 之前捕获白名单子集（仅思考相关键）。
        parent_llm_kwargs_token = _current_parent_llm_kwargs.set({
            k: llm_kwargs[k] for k in _INHERITED_LLM_KWARGS if k in llm_kwargs
        })
        # 图片注入通路：image_read 工具把校验通过的路径 append 到此列表，
        # Agent 在工具批次结束后注入多模态 user 消息（见步骤 10e）
        pending_images: list[str] = []
        images_token = _current_pending_images.set(pending_images)
        vision_supported = bool(getattr(
            self._llm.capabilities, "supports_vision", False
        ))
        vision_token = _current_vision_support.set(vision_supported)
        try:
            if images and vision_supported:
                self._history.add(Message(
                    role=MessageRole.USER,
                    content=build_user_content(user_input, images),
                ))
            else:
                if images:
                    logger.warning(
                        "当前模型不支持视觉输入，已忽略 %d 张图片，降级为纯文本",
                        len(images),
                    )
                self._history.add(Message(role=MessageRole.USER, content=user_input))

            # 前置自动检索（knowledge.auto_retrieve，默认关）：每轮用户消息先检索
            # 知识库，达标片段经 _build_system_prompt 注入——不依赖模型决策调
            # kb_search，永不漏检；失败只记日志不阻断对话。每次 run 整体覆写。
            if self._knowledge is not None and self._knowledge.config.auto_retrieve:
                await self._knowledge.prepare_auto_context(user_input)

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
                turn_reasoning_parts: list[str] = []  # 累积思考增量，回写 assistant 消息
                announced_calls: set[int] = set()  # 已发 ToolCallPreparing 的 buffer 下标

                try:
                    async with asyncio.timeout(self._config.timeout):
                        async for chunk in self._llm.chat(messages, **llm_kwargs):
                            # 5. 产出文本/思考事件
                            if chunk.content:
                                turn_text_parts.append(chunk.content)
                                yield TextDelta(text=chunk.content)

                            if chunk.reasoning_content:
                                turn_reasoning_parts.append(chunk.reasoning_content)
                                yield ReasoningDelta(text=chunk.reasoning_content)

                            # 6. 合并工具调用片段
                            if chunk.tool_calls:
                                _merge_tool_calls(tool_call_buffer, chunk.tool_calls)
                                # 每个调用首次拿到名字时发一次 ToolCallPreparing——
                                # 正文结束后参数流式生成期（长代码可达数十秒）原本
                                # 零事件，前端无从显示活动状态（"聊天框静止"）
                                for idx, item in enumerate(tool_call_buffer):
                                    if (idx not in announced_calls
                                            and item["function"]["name"]):
                                        announced_calls.add(idx)
                                        yield ToolCallPreparing(
                                            tool_name=item["function"]["name"])

                            # 累积 usage
                            if chunk.usage:
                                _merge_usage(total_usage, chunk.usage)
                                # 更新压缩器的 token 跟踪
                                if chunk.usage.prompt_tokens:
                                    self._history.update_prompt_tokens(chunk.usage.prompt_tokens)

                            # 注意：不要在 finish_reason=="stop" 时立即 break。
                            # OpenAI 兼容流在 finish_reason 之后还会发一个「仅含 usage、
                            # choices 为空」的收尾 chunk（stream_options.include_usage=True）。
                            # 提前 break 会丢掉这次调用的 token 用量。这里继续把流读完，
                            # 流随后自然结束（[DONE]）；整体仍受外层 asyncio.timeout 兜底。
                            # （tool_calls 收尾本就不 break，故 usage 一直能采到。）
                except asyncio.TimeoutError:
                    yield AgentError(
                        error_type="call_timeout",
                        message=f"单次调用超时 {self._config.timeout}s",
                    )
                    return
                except Exception as e:
                    error_str = str(e).lower()

                    # ── 流式连接中断重试（网络抖动、服务端断连）──
                    _CONNECTION_ERROR_KEYWORDS = [
                        "incomplete chunked", "peer closed", "connection reset",
                        "broken pipe", "remote end closed", "connection aborted",
                    ]
                    is_conn_error = any(kw in error_str for kw in _CONNECTION_ERROR_KEYWORDS)
                    if is_conn_error:
                        _retry_count = getattr(self, '_conn_retry_count', 0) + 1
                        self._conn_retry_count = _retry_count
                        if _retry_count <= 3:
                            delay = min(2 ** _retry_count, 16)
                            logger.warning(
                                "流式连接中断（第 %d 次），%ds 后重试: %s",
                                _retry_count, delay, e,
                            )
                            await asyncio.sleep(delay)
                            continue  # 重试本轮 LLM 调用
                        # 超过重试上限，放弃
                        logger.error("流式连接中断超过重试上限: %s", e)
                        del self._conn_retry_count

                    # ── 限流 / 服务过载重试（429、engine overloaded 等瞬时错误）──
                    # 这类错误语义即「稍后重试」，指数退避后重试本轮即可恢复。
                    # 主代理与子代理共用本 run() 循环，故子代理（如 coder）撞到 429
                    # 也会自动退避重试，而非直接判失败。
                    _RATE_LIMIT_KEYWORDS = [
                        "429", "rate limit", "rate_limit", "too many requests",
                        "overload", "try again later", "engine_overloaded",
                        "server is busy", "service unavailable", "503",
                    ]
                    is_rate_error = any(kw in error_str for kw in _RATE_LIMIT_KEYWORDS)
                    if is_rate_error:
                        _retry_count = getattr(self, '_rate_retry_count', 0) + 1
                        self._rate_retry_count = _retry_count
                        if _retry_count <= 4:
                            delay = min(2 ** _retry_count, 30)
                            logger.warning(
                                "服务限流/过载（第 %d 次），%ds 后重试: %s",
                                _retry_count, delay, e,
                            )
                            await asyncio.sleep(delay)
                            continue  # 重试本轮 LLM 调用
                        # 超过重试上限，放弃
                        logger.error("服务限流/过载超过重试上限: %s", e)
                        del self._rate_retry_count

                    # ── 内容合规 / 安全风控拦截 → 不重试，直接给用户清晰提示 ──
                    # 国产模型常见（data_inspection_failed / content_filter / 敏感 等）。
                    # 与限流/网络不同，重试无益；主代理与子代理共用本 run() 循环，
                    # 故子代理（如 researcher）撞到风控也会产出结构化错误而非静默吞掉。
                    safety_hint = content_safety_hint(e)
                    if safety_hint is not None:
                        logger.warning("内容合规拦截: %s", e)
                        yield AgentError(
                            error_type="content_filter",
                            message=safety_hint,
                        )
                        return

                    # ── 上下文过长 → 应急压缩后重试 ──
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

                # 流式调用成功，重置瞬时错误重试计数器
                if hasattr(self, '_conn_retry_count'):
                    del self._conn_retry_count
                if hasattr(self, '_rate_retry_count'):
                    del self._rate_retry_count

                # 7. 记录 assistant 消息到历史
                assistant_content = "".join(turn_text_parts) if turn_text_parts else None
                # 思考内容随 assistant 消息保真存储——thinking 模型（如 Kimi）要求带
                # tool_calls 的 assistant 消息在回传时携带 reasoning_content，否则报错。
                # 由各 provider 的 _messages_to_dicts() 按需注入，默认不回传（见 Message）。
                assistant_reasoning = "".join(turn_reasoning_parts) if turn_reasoning_parts else None

                # 清理 buffer：移除内部字段，转为标准格式
                resolved_calls = []
                for item in tool_call_buffer:
                    call = {
                        "id": item["id"],
                        "type": "function",
                        "function": {
                            "name": item["function"]["name"],
                            "arguments": item["function"]["arguments"],
                        },
                    }
                    # provider 透传字段（如 Gemini thought_signature）原样保留——
                    # 入历史并在回传时由 to_dict() 透传，满足多轮工具调用校验。
                    if item.get("extra_content"):
                        call["extra_content"] = item["extra_content"]
                    resolved_calls.append(call)

                assistant_msg = Message(
                    role=MessageRole.ASSISTANT,
                    content=assistant_content,
                    tool_calls=resolved_calls if resolved_calls else None,
                    reasoning_content=assistant_reasoning,
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

                # ── 10a.7 talk 模式安全检查 ──
                # talk 模式下，阻止所有不安全工具调用
                if self._mode == AgentMode.TALK:
                    blocked = []
                    allowed = []
                    for call in resolved_calls:
                        fn = call.get("function", {})
                        tool_name = fn.get("name", "")
                        tool_args = fn.get("arguments", "{}")
                        wrapper = self._registry.get_tool(tool_name)
                        if wrapper and self._is_safe_call(wrapper, tool_args):
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
                                f"仅允许安全操作。请用安全工具或指令完成需求。"
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

                # ── 10a.8 auto 模式 AI 安全判定（批量一次调用）──
                # 分层与 Claude Code 对齐：is_safe/safe_check 确定性快路径不经过 AI，
                # 仅不安全调用的灰色地带交给判定器兜底。
                # 判定失败一律 fail-open（回退为直接执行，记录警告）。
                judge_verdicts: dict = {}
                if self._mode == AgentMode.AUTO and self._judge_llm is not None:
                    to_judge = []
                    for call in resolved_calls:
                        fn = call.get("function", {})
                        j_name = fn.get("name", "")
                        j_args = fn.get("arguments", "{}")
                        j_wrapper = self._registry.get_tool(j_name)
                        if j_wrapper is None:
                            continue  # 未知工具交由执行器报错
                        if self._is_safe_call(j_wrapper, j_args):
                            continue  # 安全快路径，零 AI 开销
                        to_judge.append(call)
                    if to_judge:
                        from milu.agent.judge import judge_tool_calls
                        # 判定是一次阻塞式 LLM 调用（数秒），期间无其他事件——
                        # 先发状态事件，让前端能显示「安全判定中」而非静止
                        yield SafetyCheckStart(tool_names=tuple(
                            c.get("function", {}).get("name", "") for c in to_judge
                        ))
                        try:
                            judge_verdicts = await judge_tool_calls(
                                self._judge_llm, user_input, to_judge,
                                self._registry, self._judge_rules,
                            )
                        except Exception as e:
                            logger.warning(
                                "安全判定器调用失败，fail-open 回退为直接执行: %s", e
                            )
                            judge_verdicts = {}

                # ── 10b. 顺序发出 ToolCallStart + 不安全工具确认 ──
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

                    # 不安全工具确认检查：
                    #   manual 模式：不安全工具需人工审批
                    #   auto   模式：不安全工具自主执行（配 judge_llm 时经 AI 判定，见下）
                    #   superwork  ：全部直接执行，不走此分支
                    wrapper = self._registry.get_tool(tool_name)
                    needs_confirm = (
                        self._mode == AgentMode.MANUAL
                        and not self._is_safe_call(wrapper, tool_args)
                    )

                    # AI 安全判定结果处理（仅 auto 模式的不安全调用有判定结果）：
                    #   allow → 落下去直接执行；confirm → 并入人工审批流程；deny → 拒绝
                    verdict = judge_verdicts.get(tool_call_id)
                    if verdict is not None:
                        if verdict.decision == "deny":
                            reject_msg = (
                                f"[安全判定] 工具 {tool_name} 调用被拒绝："
                                f"{verdict.reason or '判定为高风险操作'}。"
                                f"请调整方案或换用更安全的方式完成任务。"
                            )
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
                        if verdict.decision == "confirm":
                            if self._on_confirm:
                                needs_confirm = True  # 并入下方人工审批流程
                            else:
                                reject_msg = (
                                    f"[安全判定] 工具 {tool_name} 需要人工确认"
                                    f"（{verdict.reason or '影响范围不明'}），"
                                    f"但当前未配置确认回调，已拒绝执行。"
                                )
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
                        # allow → 不设 needs_confirm，直接落到执行

                    if needs_confirm and self._on_confirm:
                        # 等待人工确认期间，通过守卫释放外部并发许可（无注入时为 no-op），
                        # 避免长时间确认阻塞占用全局并发名额。
                        wait_guard = (
                            self._confirm_wait_guard()
                            if self._confirm_wait_guard
                            else contextlib.nullcontext()
                        )
                        async with wait_guard:
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
                                reject_msg = f"用户拒绝了工具 {tool_name} 的执行"
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
                        wrapper = item["wrapper"]
                        # v2: subagent 工具注入 per-call events 列表（替代 _last_events 闭包）
                        per_call_events: list[AgentEvent] = []
                        events_token = None
                        if wrapper and getattr(wrapper, '_is_subagent', False):
                            events_token = _current_subagent_events.set(per_call_events)
                        try:
                            result = await self._executor.execute(item["call"])
                        finally:
                            if events_token is not None:
                                _current_subagent_events.reset(events_token)
                        return item, result, per_call_events

                    results = await asyncio.gather(
                        *[_execute_one(item) for item in confirmed]
                    )

                    # ── 10d. 顺序处理结果：转发事件、产出 ToolResult ──
                    for item, exec_result, per_call_events in results:
                        tool_name = item["tool_name"]
                        tool_call_id = item["tool_call_id"]
                        wrapper = item["wrapper"]

                        # v2: 子代理事件从 per-call 列表读取（无闭包共享）
                        if wrapper and getattr(wrapper, '_is_subagent', False):
                            sub_final_text = ""
                            sub_turn_count = 0
                            sub_total_usage = TokenUsage()
                            sub_is_error = False

                            for sevt in per_call_events:
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
                            
                            per_call_events.clear()

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

                # ── 10e. 注入 image_read 加载的图片（多模态 user 消息）──
                # 协议顺序合法：assistant(tool_calls) → tool 消息×N → user 消息。
                # 历史中只存轻量 image_path 引用块（不含 base64），
                # 下一轮发送 API 时由 provider 层物化为 base64 data URL。
                if pending_images:
                    self._history.add(Message(
                        role=MessageRole.USER,
                        content=build_user_content(
                            "[image_read] 以上为工具加载的图片，请直接查看并继续任务",
                            pending_images,
                        ),
                    ))
                    pending_images.clear()

                # 11. 继续循环，LLM 将看到工具结果

        finally:
            if session_token is not None:
                _safe_reset(_current_todo_session_dir, session_token)
            if plan_items_token is not None:
                _safe_reset(_current_todo_plan_items, plan_items_token)
            _safe_reset(_current_memory_path, memory_token)
            _safe_reset(_current_knowledge, knowledge_token)
            _safe_reset(_current_schedule_user, schedule_user_token)
            _safe_reset(_current_parent_mode, mode_token)
            _safe_reset(_current_parent_confirm, confirm_token)
            _safe_reset(_current_parent_judge, judge_token)
            _safe_reset(_current_parent_llm_kwargs, parent_llm_kwargs_token)
            _safe_reset(_current_pending_images, images_token)
            _safe_reset(_current_vision_support, vision_token)