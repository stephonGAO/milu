"""从 Settings 构建 LLM 与 Agent（CLI 专用，依托 Agent「全配默认」最简创建）。"""
from __future__ import annotations

from milu.agent import Agent
from milu.llm.providers import ModelRegistry
from milu.llm.providers.base import BaseLLM

from milu.cli.config import Settings
from milu.cli.render import confirm_unsafe


def build_llm(s: Settings) -> BaseLLM:
    """按设置创建 LLM 实例（鉴权懒加载，构造时不触网）。

    web_search / enable_thinking 作为构造默认值传入，不支持的厂商由
    BaseLLM._validate_params 静默丢弃。
    """
    kwargs = {
        "model": s.model,
        "web_search": s.web_search,
        "enable_thinking": s.enable_thinking,
    }
    if s.api_key:
        kwargs["api_key"] = s.api_key
    # 显式覆盖上下文窗口（内置表未收录的模型用）；None 时由 provider 按模型名解析
    if s.context_window:
        kwargs["context_window"] = s.context_window
    return ModelRegistry.create(s.provider, **kwargs)


def apply_security(s: Settings) -> None:
    """按配置应用安全设置（在 build_agent 之前调用）。"""
    from milu.tools._selfguard import set_enabled
    set_enabled(s.selfguard_enabled)


def build_agent(s: Settings) -> Agent:
    """按设置构建顶层 Agent（最简创建，其余全部交给 Agent 的「全配默认」）。

    - 运行限额 / 压缩参数来自分层配置（settings.agent / settings.compact）
    - tools / skills / prompt 不传 → 自动注入全套内置工具、内置技能、内置 main 角色提示词
    - subagents：None → 内置三件套（researcher/reader/coder）；--no-subagents → [] 关闭
    - on_confirm：manual 模式人工审批 + auto 模式 AI 判定器转人工时的交互式确认
    """
    from milu.agent.config import AgentConfig, CompactConfig
    from milu.agent.history import ConversationHistory

    apply_security(s)
    llm = build_llm(s)
    # s.agent 含 mode/session_enabled（非 AgentConfig 字段），故只取运行限额字段构造。
    # 兜底默认从 AgentConfig() 派生（单一真相源），避免与 dataclass 默认值脱节。
    _def = AgentConfig()
    agent_config = AgentConfig(
        max_turns=s.agent.get("max_turns", _def.max_turns),
        timeout=s.agent.get("timeout", _def.timeout),
        total_timeout=s.agent.get("total_timeout", _def.total_timeout),
        max_total_tokens=s.agent.get("max_total_tokens", _def.max_total_tokens),
        tool_call_limit=s.agent.get("tool_call_limit", _def.tool_call_limit),
    )
    history = ConversationHistory(
        strategy="auto_compact",
        max_tokens=agent_config.max_total_tokens,
        llm=llm,
        compact_config=CompactConfig(**s.compact) if s.compact else None,
    )
    # 向量知识库：config.json 的 knowledge.enabled 为真时启用（CLI 单人，身份 "default"）。
    # 库纯净性：分层配置在此（应用入口）转为 KnowledgeConfig 下传，Agent 不读 config.json。
    knowledge = False
    if s.knowledge.get("enabled"):
        from milu.knowledge import KnowledgeConfig
        knowledge = KnowledgeConfig.from_mapping(s.knowledge)
    return Agent(
        llm=llm,
        mode=s.mode,
        session_enabled=s.session_enabled,
        config=agent_config,
        history=history,
        subagents=None if s.use_subagents else [],
        knowledge=knowledge,
        on_confirm=confirm_unsafe,
    )


def build_scheduler_engine(echo: bool):
    """构造 CLI 形态的调度引擎（daemon 与 chat 嵌入共用，无 agent_pool）。

    :param echo: daemon 前台传 True（控制台回显）；chat 嵌入传 False（静默，
        不污染 REPL，结果走 outbox/系统弹窗/日志文件）
    :return: (engine, store, data_dir) 三元组——store 供启动时统计任务数，
        data_dir 供构造 SchedulerLock
    """
    from milu.config import load_config
    from milu.resources import user_data_dir
    from milu.scheduler import ScheduleEngine, ScheduleStore

    data_dir = user_data_dir()
    store = ScheduleStore(data_dir)
    engine = ScheduleEngine(
        store,
        log_dir=data_dir / "scheduler_logs",
        config=load_config().to_scheduler_config(),
        echo=echo,
    )
    return engine, store, data_dir
