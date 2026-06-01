"""子代理（SubAgent）— 让父 Agent 将专项任务委派给独立的子 Agent

每个子代理以工具形式暴露给父 LLM：
  - 独立的系统提示、工具集、对话历史（完全隔离）
  - 每次调用创建全新 Agent 实例，历史不累积
  - 执行完毕只将精炼结果返回父 Agent（不污染父上下文）
  - 不可嵌套子代理（v1 结构性保证）

使用方式：
    from agent_framework.agent.subagent import SubAgentConfig, create_subagent_tools

    subagent_tools = create_subagent_tools(
        llm=llm,
        subagents=[
            SubAgentConfig(
                name="researcher",
                description="调研助手：擅长搜索和整理信息",
                system_prompt="你是一个专业的调研助手...",
                tools=[web_search, http_request],
            ),
        ],
    )
    agent = Agent(llm=llm, tools=[*BUILTIN_TOOLS, *subagent_tools])
"""
from __future__ import annotations

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

    # 每次调用时存储子代理事件（供父 Agent 读取转发）
    _last_events: list[AgentEvent] = []

    @tool(
        name=cfg.name,
        description=cfg.description,
    )
    async def _subagent_tool(task: str) -> str:
        """
        :param task: 委派给子代理的任务描述或问题
        """
        _last_events.clear()

        # 每次调用创建全新 Agent → 完全的历史隔离
        sub_config = cfg.config or _DEFAULT_SUBAGENT_CONFIG

        # 继承父 Agent 的操作模式
        if get_parent_mode is not None:
            parent_mode = get_parent_mode()
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

        try:
            async for event in sub_agent.run(task, **cfg.llm_kwargs):
                _last_events.append(event)
        except Exception as e:
            logger.warning("子代理 %s 执行异常: %s", cfg.name, e)
            _last_events.append(
                AgentError(error_type="subagent_crash", message=str(e))
            )
            return f"[{cfg.name}] 子代理执行失败: {e}"

        # 从终止事件提取结果
        for event in reversed(_last_events):
            if isinstance(event, AgentDone):
                return f"[{cfg.name}]: {event.final_text}"
            if isinstance(event, AgentError):
                return f"[{cfg.name}] 错误: {event.message}"

        return f"[{cfg.name}]: (无结果返回)"

    # 在 ToolWrapper 上附加事件存储和标记（供 Agent 循环读取）
    _subagent_tool._tool_wrapper._subagent_events = _last_events
    _subagent_tool._tool_wrapper._is_subagent = True

    return _subagent_tool
