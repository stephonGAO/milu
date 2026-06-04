"""子代理（SubAgent）— 无状态版本

v2 变更：
- 删除 _last_events 闭包变量
- 每次工具调用通过 ContextVar 注入 per-call events 列表
- 工具函数和工厂签名不变
"""
from __future__ import annotations

import contextvars
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

# ── 父 Agent 确认回调注入（asyncio 任务级隔离）──────────
#
# Agent.run() 在入口把自身 on_confirm 写入此 ContextVar，子代理创建时继承它。
# 由此，manual 模式下子代理内部的不安全工具（file_write / shell_command 等）、
# auto 模式下被 AI 判定为 confirm 的调用会走与父 Agent 相同的人工确认流程，
# 而不是因为子代理没接 on_confirm 而被静默放行——
# 「委派给子代理」不再成为绕过安全审批的旁路。
# 注意：确认仍通过回调实时发生；但 ToolConfirmRequired 事件会随子代理事件列表在
# 工具调用结束后才以 SubAgentEvent 形式回放（事件是事后记录，回调才是阻塞点）。
_current_parent_confirm: contextvars.ContextVar["Callable | None"] = contextvars.ContextVar(
    "current_parent_confirm", default=None
)

# ── 父 Agent AI 安全判定器注入（asyncio 任务级隔离）──────────
#
# Agent.run() 在入口把 (judge_llm, judge_rules) 写入此 ContextVar（未启用时为 None），
# 子代理创建时继承——auto 模式下子代理内的不安全工具同样经过 AI 安全判定，
# 委派不构成判定旁路。judge_llm 为 AsyncOpenAI 封装，协程安全可共享。
_current_parent_judge: contextvars.ContextVar["tuple | None"] = contextvars.ContextVar(
    "current_parent_judge", default=None
)


# ── 数据模型 ──────────────────────────────────────────────


@dataclass
class SubAgentConfig:
    """单个子代理的配置。

    :param name: 工具名（如 "researcher"），父 LLM 通过此名称调用
    :param description: 工具描述，出现在父 LLM 的 tool schema 中
    :param system_prompt: 子代理的系统提示（可选，可与 prompt_dir 组合使用）
    :param tools: 子代理可用的工具列表（@tool 装饰的函数）
    :param config: 子代理的 AgentConfig 运行限额（None 时使用默认值：
        max_turns=50, timeout=120, total_timeout=600, tool_call_limit=50）。
        注：操作模式由父 Agent 继承（非此处配置），session 固定关闭。
    :param history_max_turns: 子代理对话历史最大轮数
    :param history_max_tokens: 子代理对话历史 token 上限（None 为不限）
    :param llm_kwargs: 传递给子代理 LLM 的额外参数（如 web_search=True, enable_thinking=True）
    :param skills: 子代理可用的技能列表（SkillConfig 实例）
    :param skills_dir: 子代理技能目录路径（None 时不自动扫描）
    :param role: 内置角色名（main/coder/researcher/reviewer）。设置后自动套用该角色的
        内置提示词目录——便利写法，免去手写 prompt_dir；显式 prompt_dir 优先于 role。
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
    role: str | None = None
    prompt_dir: "str | None" = None
    prompt_variables: dict[str, str] | None = None


# ── 内部：子代理默认配置 ──────────────────────────────────────

_DEFAULT_SUBAGENT_CONFIG = AgentConfig(
    max_turns=50,
    timeout=120.0,
    total_timeout=600,
    tool_call_limit=50,
)


# ── 内置子代理预设 ────────────────────────────────────────


def builtin_subagent_configs(
    include: "tuple[str, ...] | list[str]" = ("researcher", "reader", "coder"),
) -> list[SubAgentConfig]:
    """返回内置子代理配置列表（生产级预设，配套内置角色提示词与技能）。

    默认三件套（按「上下文隔离 / 权限收窄 / 可并行」标准选型）：
      - researcher 调研员：开放网络信息获取（web_search/web_fetch/datetime），只读
      - reader     长内容阅读员：长文档/日志/网页定向提取（file_read/web_fetch），只读
      - coder      编码执行员：计算/脚本/数据处理（python_repl/file_read/file_write）

    可选（不在默认集）：
      - reviewer   审查员：用干净上下文复查代码/成果（file_read/python_repl + code-review 技能）

    :param include: 要启用的子代理名称集合，如 ("researcher", "reader", "coder", "reviewer")
    :return: list[SubAgentConfig]，传给 create_subagent_tools(llm, configs) 即可

    用法：
        from agent_framework import builtin_subagent_configs, create_subagent_tools
        tools = create_subagent_tools(llm, builtin_subagent_configs())
        agent = Agent(llm=llm, tools=[*BUILTIN_TOOLS, *tools])
    """
    # 延迟导入：避免 import 包时就拉起全部内置工具依赖
    from agent_framework.resources import builtin_skills_dir
    from agent_framework.skills.config import SkillConfig
    from agent_framework.tools.builtin import (
        datetime_tool,
        file_read,
        file_write,
        python_repl,
        web_fetch,
        web_search,
    )

    deep_research_md = builtin_skills_dir() / "deep-research.md"
    code_review_md = builtin_skills_dir() / "code-review.md"

    presets: dict[str, SubAgentConfig] = {
        "researcher": SubAgentConfig(
            name="researcher",
            description=(
                "调研员：开放网络信息获取与多源交叉验证。"
                "当需要搜索资料、对比方案、汇总时效性信息时委派；"
                "返回带来源 URL 和可信度的精炼结论。"
                "简单常识问题不要委派，直接回答。"
            ),
            role="researcher",
            tools=[web_search, web_fetch, datetime_tool],
            skills=[SkillConfig.from_file(str(deep_research_md))]
            if deep_research_md.exists() else None,
        ),
        "reader": SubAgentConfig(
            name="reader",
            description=(
                "长内容阅读员：定向消化长文档/日志/代码文件/网页，只回传与问题相关的结论。"
                "当需要「读完某个大文件或网页后回答特定问题」时委派（可同批委派多个并行阅读）；"
                "返回结构化答案 + 出处定位（文件:行号 / URL）。"
                "短内容（几十行以内）不要委派，自己读。"
            ),
            role="reader",
            tools=[file_read, web_fetch],
        ),
        "coder": SubAgentConfig(
            name="coder",
            description=(
                "编码执行员：编写并运行 Python 代码完成计算、脚本、数据处理、文件生成。"
                "当任务需要实际执行代码或产出文件时委派；"
                "返回结果摘要 + 产物文件路径 + 复现方式。"
                "注意：继承父 Agent 的操作模式与确认流程，委派不构成安全旁路。"
            ),
            role="coder",
            tools=[python_repl, file_read, file_write],
        ),
        "reviewer": SubAgentConfig(
            name="reviewer",
            description=(
                "审查员：用独立干净的上下文复查代码或工作成果（正确性/安全性/可维护性）。"
                "当需要对已完成的代码或方案做无偏审查时委派；"
                "返回问题清单（按严重程度排序）+ 改进建议。"
            ),
            role="reviewer",
            tools=[file_read, python_repl],
            skills=[SkillConfig.from_file(str(code_review_md))]
            if code_review_md.exists() else None,
        ),
    }

    unknown = [n for n in include if n not in presets]
    if unknown:
        raise ValueError(
            f"未知的内置子代理: {', '.join(unknown)}。可选：{', '.join(presets)}"
        )
    return [presets[n] for n in include]


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

        # 每次调用创建全新 Agent → 完全的历史隔离。
        # config 仅含运行限额（Agent.__init__ 会做防御性拷贝，此处直接传引用即可）。
        sub_config = cfg.config or _DEFAULT_SUBAGENT_CONFIG

        # 继承父 Agent 的操作模式：显式回调优先，否则读 ContextVar（Agent.run 已注入）。
        # mode 现为 Agent 直接参数，不再寄居于 config。
        parent_mode = (
            get_parent_mode() if get_parent_mode is not None
            else _current_parent_mode.get()
        )
        sub_mode = parent_mode if parent_mode is not None else AgentMode.AUTO

        sub_history = ConversationHistory(
            max_turns=cfg.history_max_turns,
            max_tokens=cfg.history_max_tokens,
        )

        # 继承父 Agent 的 AI 安全判定器（Agent.run 经 ContextVar 注入）。
        # 父未启用时传 False 显式关闭——若传 None 会触发子 Agent 的
        # 「默认复用自身 llm」规则，导致父关了判定子又自行开启。
        parent_judge = _current_parent_judge.get()
        judge_llm, judge_rules = parent_judge if parent_judge else (False, "")

        # role 便利字段：未显式给 prompt_dir 时，套用该内置角色的提示词目录
        prompt_dir = cfg.prompt_dir
        if prompt_dir is None and cfg.role:
            from agent_framework.resources import builtin_prompts_dir
            prompt_dir = builtin_prompts_dir(cfg.role)

        sub_agent = Agent(
            llm=llm,
            system_prompt=cfg.system_prompt,
            tools=cfg.tools if cfg.tools else None,
            history=sub_history,
            config=sub_config,
            mode=sub_mode,            # 继承父 Agent 操作模式
            # 继承父 Agent 的人工确认回调（Agent.run 经 ContextVar 注入）：
            # manual 模式不安全工具、auto 模式 AI 判定 confirm 的调用同样需要审批，
            # 委派不构成安全旁路
            on_confirm=_current_parent_confirm.get(),
            # 继承父 Agent 的 AI 安全判定器：auto 模式下子代理同样经过判定
            judge_llm=judge_llm,
            judge_rules=judge_rules,
            session_enabled=False,    # 子代理不创建独立 session，结果已在主 agent 日志中记录
            register_catalog=False,   # 子代理不注册 mcp 元工具，避免误导
            skills=cfg.skills,
            skills_dir=cfg.skills_dir,
            register_skills=bool(cfg.skills or cfg.skills_dir),
            prompt_dir=prompt_dir,
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
