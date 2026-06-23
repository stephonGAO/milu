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
    # 运行追踪：config.json 的 observability.enabled 为真时启用（本地落盘 trace.jsonl
    # + runs.jsonl 索引）。同款应用层开关约定；顺带清理过期的散件 trace。
    trace = False
    if s.observability.get("enabled"):
        from milu.observability import TraceConfig, cleanup_old_traces
        trace = TraceConfig.from_mapping(s.observability)
        try:
            cleanup_old_traces(trace.retention_days)
        except Exception:
            pass
    # 代码执行沙箱后端：分层配置 sandbox 分节转 SandboxConfig 下传（默认 subprocess
    # 子进程隔离；用户在 config.json 改 sandbox.backend=local 可退回零开销本地执行）。
    from milu.sandbox import SandboxConfig
    sandbox = SandboxConfig.from_mapping(s.sandbox)
    # agent 工作区：相对路径文件读写 + 沙箱 CWD 的落点。CLI 单人无需按用户隔离，
    # 默认 ~/.milu/workspace（config 的 agent.workspace 或环境变量 MILU_WORKSPACE 可覆盖）。
    from milu.resources import workspace_dir
    workspace = str(workspace_dir(base=s.agent.get("workspace") or None))
    # 工作区围栏（multiuser=strict 会经分层配置把 agent.workspace_jail 设为 true）
    workspace_jail = bool(s.agent.get("workspace_jail", False))
    return Agent(
        llm=llm,
        mode=s.mode,
        session_enabled=s.session_enabled,
        config=agent_config,
        history=history,
        subagents=None if s.use_subagents else [],
        knowledge=knowledge,
        trace=trace,
        sandbox=sandbox,
        workspace=workspace,
        workspace_jail=workspace_jail,
        on_confirm=confirm_unsafe,
    )


def build_gateway_pool(s: Settings):
    """为 `milu gateway` 构建多用户 AgentPool（含 per-用户隔离 + strict 透传）。

    与 Web 服务的 agent_factory 同款：每个 (user_id, session_id) 构造一个 per-用户
    Agent，**工作区按 user_id 隔离**（避免多用户经 file_read/file_write 互看产物），
    sandbox/workspace_jail/knowledge/trace 从分层配置派生。`load_config()` 已对
    multiuser=strict 应用最高层覆盖（sandbox.backend=docker + workspace_jail=true），
    故这里读到的就是生效后的隔离基线——把生产踩过的「from_llm 默认工厂不自动套
    strict」坑直接堵在产品默认里。

    :return: (AgentPool, MiluConfig) ——后者供调用方打印部署/隔离状态横幅。
    """
    from milu.agent import Agent
    from milu.agent.config import AgentConfig
    from milu.config import load_config
    from milu.resources import workspace_dir
    from milu.sandbox import SandboxConfig
    from milu.serving.pool import AgentPool

    apply_security(s)
    mc = load_config()  # 已应用 multiuser=strict 覆盖
    llm = build_llm(s)

    _def = AgentConfig()
    agent_config = AgentConfig(
        max_turns=s.agent.get("max_turns", _def.max_turns),
        timeout=s.agent.get("timeout", _def.timeout),
        total_timeout=s.agent.get("total_timeout", _def.total_timeout),
        max_total_tokens=s.agent.get("max_total_tokens", _def.max_total_tokens),
        tool_call_limit=s.agent.get("tool_call_limit", _def.tool_call_limit),
    )

    # 闭包内用到的分层配置快照（一次读取，避免每次构造都打开文件）
    sandbox_cfg = dict(mc.sandbox)
    workspace_base = mc.agent.get("workspace", "") or None
    workspace_jail = bool(mc.agent.get("workspace_jail", False))
    knowledge_cfg = dict(mc.knowledge)
    obs_cfg = dict(mc.observability)
    mode = s.mode
    session_enabled = s.session_enabled
    use_subagents = s.use_subagents

    def factory(user_id: str, session_id: str, _llm):
        sandbox = SandboxConfig.from_mapping(sandbox_cfg)
        workspace = str(workspace_dir(user_id=user_id, base=workspace_base))
        knowledge = False
        if knowledge_cfg.get("enabled"):
            from milu.knowledge import KnowledgeConfig
            knowledge = KnowledgeConfig.from_mapping(knowledge_cfg, user_id=user_id)
        trace = False
        if obs_cfg.get("enabled"):
            from milu.observability import TraceConfig
            trace = TraceConfig.from_mapping(obs_cfg, user_id=user_id)
        return Agent(
            llm=llm,
            config=agent_config,
            mode=mode,
            session_enabled=session_enabled,
            session_id=AgentPool._derive_session_id(user_id, session_id),
            schedule_user=user_id,          # 定时任务按 user_id 隔离
            knowledge=knowledge,            # 向量知识库（默认关），按 user_id 隔离
            trace=trace,                    # 运行追踪（本地落盘）
            subagents=None if use_subagents else [],
            sandbox=sandbox,                # 代码执行后端（strict 下为 docker）
            workspace=workspace,            # agent 工作区（按用户隔离）
            workspace_jail=workspace_jail,  # 工作区围栏（strict 多用户）
        )

    pool = AgentPool(
        llm_factory=lambda uid, sid: llm,
        agent_factory=factory,
        config=mc.to_pool_config(),
        agent_config=agent_config,
    )
    return pool, mc


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
