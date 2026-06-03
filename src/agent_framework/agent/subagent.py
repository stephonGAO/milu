"""子代理（SubAgent）— 无状态版本

v2 变更：
- 删除 _last_events 闭包变量
- 每次工具调用通过 ContextVar 注入 per-call events 列表
- 工具函数和工厂签名不变
"""
from __future__ import annotations

import contextvars
import dataclasses
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

from agent_framework.agent.config import AgentConfig, AgentMode
from agent_framework.agent.events import AgentDone, AgentError, AgentEvent
from agent_framework.agent.history import ConversationHistory
from agent_framework.tools.decorator import tool

if TYPE_CHECKING:
    from agent_framework.llm.providers.base import BaseLLM

logger = logging.getLogger(__name__)

# ── per-call events 注入（asyncio 任务级隔离）──────────

_current_subagent_events: contextvars.ContextVar[list[AgentEvent] | None] = contextvars.ContextVar(
    "current_subagent_events", default=None
)

# ── 父 Agent 操作模式注入（asyncio 任务级隔离）──────────
#
# Agent.run() 在入口把自身 mode 写入此 ContextVar，子代理工具在执行时读取它来继承
# 父模式。相比旧的 get_parent_mode 回调，这种方式无需「先建 Agent 再回填闭包」，
# 子代理工具可在 Agent 之前创建并正常通过 tools=[...] 传入。get_parent_mode 仍保留
# 为可选显式覆盖（传入时优先于 ContextVar）。
_current_parent_mode: contextvars.ContextVar[AgentMode | None] = contextvars.ContextVar(
    "current_parent_mode", default=None
)


# ── 数据模型 ──────────────────────────────────────────────


@dataclass
class SubAgentConfig:
    """单个子代理的配置。

    :param name: 工具名（如 "researcher"），父 LLM 通过此名称调用
    :param description: 工具描述，出现在父 LLM 的 tool schema 中
    :param system_prompt: 子代理的系统提示（可选，可与 prompt_dir 组合使用）
    :param tools: 子代理可用的工具列表（@tool 装饰的函数）
    :param config: 子代理的 AgentConfig（None 时使用默认值：
        max_turns=50, timeout=120, total_timeout=600, confirm_dangerous=False）
    :param history_max_turns: 子代理对话历史最大轮数
    :param history_max_tokens: 子代理对话历史 token 上限（None 为不限）
    :param llm_kwargs: 传递给子代理 LLM 的额外参数（如 web_search=True, enable_thinking=True）
    :param skills: 子代理可用的技能列表（SkillConfig 实例）
    :param skills_dir: 子代理技能目录路径（None 时不自动扫描）
    :param prompt_dir: 提示词文件目录路径（None 时不使用文件化提示词）
    :param prompt_variables: 提示词变量，替换文件中的 {{key}}
    """
    name: str
    description: str
    system_prompt: str = ""
    tools: list = field(default_factory=list)
    config: AgentConfig | None = None
    history_max_turns: int = 50
    history_max_tokens: int | None = None
    llm_kwargs: dict = field(default_factory=dict)
    skills: list | None = None
    skills_dir: str | None = None
    prompt_dir: "str | None" = None
    prompt_variables: dict[str, str] | None = None


# ── 内部：子代理默认配置 ──────────────────────────────────────

_DEFAULT_SUBAGENT_CONFIG = AgentConfig(
    max_turns=50,
    timeout=120.0,
    total_timeout=600,
    tool_call_limit=50,
)


# ── 工厂函数 ──────────────────────────────────────────────


def create_subagent_tools(
    llm: "BaseLLM",
    subagents: list[SubAgentConfig],
    get_parent_mode: Optional[Callable[[], AgentMode]] = None,
) -> list:
    """创建子代理工具列表。

    每个 SubAgentConfig 生成一个 @tool 装饰的异步函数。
    父 LLM 调用该工具时，内部创建全新的 Agent 实例执行任务，
    执行完毕后将精炼结果返回给父 Agent。

    v2: 工具函数本身无 _last_events 闭包。Agent 在每次工具调用前
    通过 ContextVar `_current_subagent_events` 注入 per-call 事件列表。

    :param llm: 共享的 LLM 实例（BaseLLM 无状态，安全共享）
    :param subagents: 子代理配置列表
    :param get_parent_mode: 获取父 Agent 当前模式的回调（用于模式继承）
    :return: 工具函数列表，可传给 Agent(tools=...)

    用法：
        tools = create_subagent_tools(llm, [
            SubAgentConfig(name="researcher", ...),
            SubAgentConfig(name="coder", ...),
        ])
        agent = Agent(llm=llm, tools=[*BUILTIN_TOOLS, *tools])
    """
    return [_create_single_subagent_tool(llm, cfg, get_parent_mode) for cfg in subagents]


def _create_single_subagent_tool(
    llm: "BaseLLM",
    cfg: SubAgentConfig,
    get_parent_mode: Optional[Callable[[], AgentMode]] = None,
):
    """为单个 SubAgentConfig 创建工具函数。

    利用闭包捕获 llm 和 cfg，每次调用创建全新的 Agent 实例。
    """
    # 延迟导入避免循环依赖
    from agent_framework.agent.agent import Agent

    @tool(
        name=cfg.name,
        description=cfg.description,
    )
    async def _subagent_tool(task: str) -> str:
        """
        :param task: 委派给子代理的任务描述或问题
        """
        # v2: 从 ContextVar 取出 per-call events 列表（Agent 在调用前已 set）
        events = _current_subagent_events.get()
        if events is None:
            raise RuntimeError(
                f"子代理 {cfg.name} 必须在 Agent.run() 上下文中调用"
                "（Agent 应在调用前通过 ContextVar 注入 per-call events 列表）"
            )
        events.clear()

        # 每次调用创建全新 Agent → 完全的历史隔离
        sub_config = cfg.config or _DEFAULT_SUBAGENT_CONFIG

        # 继承父 Agent 的操作模式：显式回调优先，否则读 ContextVar（Agent.run 已注入）
        parent_mode = (
            get_parent_mode() if get_parent_mode is not None
            else _current_parent_mode.get()
        )
        if parent_mode is not None:
            sub_config = dataclasses.replace(sub_config, mode=parent_mode)
        else:
            sub_config = dataclasses.replace(sub_config)

        sub_config.session_enabled = False  # 子代理不创建独立 session，结果已在主 agent 日志中记录

        sub_history = ConversationHistory(
            max_turns=cfg.history_max_turns,
            max_tokens=cfg.history_max_tokens,
        )

        sub_agent = Agent(
            llm=llm,
            system_prompt=cfg.system_prompt,
            tools=cfg.tools if cfg.tools else None,
            history=sub_history,
            config=sub_config,
            register_catalog=False,  # 子代理不注册mcp元工具，避免误导
            skills=cfg.skills,
            skills_dir=cfg.skills_dir,
            register_skills=bool(cfg.skills or cfg.skills_dir),
            prompt_dir=cfg.prompt_dir,
            prompt_variables=cfg.prompt_variables,
        )
        # 不调用 create_subagent_tools → 子代理不能嵌套子代理（结构性保证）

        final_text: Optional[str] = None
        try:
            async for event in sub_agent.run(task, **cfg.llm_kwargs):
                events.append(event)
                if isinstance(event, AgentDone):
                    final_text = event.final_text
                elif isinstance(event, AgentError):
                    return f"[{cfg.name}] 错误: {event.message}"
        except Exception as e:
            logger.warning("子代理 %s 执行异常: %s", cfg.name, e)
            events.append(
                AgentError(error_type="subagent_crash", message=str(e))
            )
            return f"[{cfg.name}] 子代理执行失败: {e}"

        return f"[{cfg.name}]: {final_text or '(无结果返回)'}"

    # 仅标记 _is_subagent（不再设置 _subagent_events；Agent 通过 ContextVar 注入 per-call events）
    _subagent_tool._tool_wrapper._is_subagent = True

    return _subagent_tool
