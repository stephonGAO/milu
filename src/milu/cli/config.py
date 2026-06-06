"""CLI 参数层：把文件配置（MiluConfig）与命令行参数合并为一次运行的 Settings。

文件配置的分层合并（项目 config/milu.json ← 用户 ~/.milu/config.json）在 milu.config
完成；本模块只负责在其之上叠加 CLI 参数，并解析 API Key（CLI 参数 > 环境变量）。

优先级（高 → 低）：
  provider / model / mode / 各参数：CLI 参数 > 文件配置（用户级 > 项目级）> 内置默认
  api_key：CLI 参数 > 环境变量 {PROVIDER}_API_KEY（密钥不再落 config.json）
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from milu.config import DEFAULT_MODELS, DEFAULT_PROVIDER, MiluConfig, load_config

__all__ = [
    "DEFAULT_MODELS",
    "DEFAULT_PROVIDER",
    "Settings",
    "env_key_name",
    "load_config",
    "resolve_settings",
]


def env_key_name(provider: str) -> str:
    """该厂商的 API Key 环境变量名：{PROVIDER}_API_KEY。"""
    return f"{provider.upper()}_API_KEY"


@dataclass
class Settings:
    """一次运行最终生效的设置（文件配置 + CLI 参数 + 环境变量密钥 合并后的结果）。"""

    provider: str
    model: str
    api_key: str | None
    mode: str
    session_enabled: bool
    web_search: bool
    enable_thinking: bool
    use_mcp: bool = True
    use_subagents: bool = True
    session_id: str | None = None
    selfguard_enabled: bool = True
    show_subagent_events: bool = False   # 是否在 CLI 渲染子代理内部事件（默认隐藏）
    # 运行限额 / 压缩分节（供 builder 构造 AgentConfig / CompactConfig）
    agent: dict = field(default_factory=dict)
    compact: dict = field(default_factory=dict)


def resolve_settings(config: MiluConfig, args) -> Settings:
    """在合并后的文件配置上叠加 CLI 参数，得到最终 Settings。

    :raises ValueError: 厂商无内置默认模型且未显式指定 model 时。
    """
    llm = config.llm
    agent = config.agent
    default_models = config.default_models

    provider = getattr(args, "provider", None) or llm.get("provider") or DEFAULT_PROVIDER

    # 模型解析：CLI 显式 model 始终优先；否则配置里的 model 只在「未切换厂商」
    # （解析出的 provider == 配置 llm.provider）时沿用，避免给 deepseek 套上
    # qwen 的模型名；都没有则回退到厂商的内置默认模型。
    cli_model = getattr(args, "model", None)
    config_provider = llm.get("provider") or DEFAULT_PROVIDER
    config_model = llm.get("model")
    if cli_model:
        model = cli_model
    elif config_model and provider == config_provider:
        model = config_model
    else:
        model = default_models.get(provider)
    if not model:
        raise ValueError(
            f"厂商 '{provider}' 没有内置默认模型，请用 --model 指定，"
            f"或在 config.json 的 default_models 中补充。"
        )

    api_key = getattr(args, "api_key", None) or os.environ.get(env_key_name(provider))

    mode = getattr(args, "mode", None) or agent.get("mode") or "auto"

    session_enabled = bool(agent.get("session_enabled", True))
    if getattr(args, "no_session", False):
        session_enabled = False

    security = config.security
    display = config.display
    return Settings(
        provider=provider,
        model=model,
        api_key=api_key,
        mode=mode,
        session_enabled=session_enabled,
        web_search=bool(llm.get("web_search", True)),
        enable_thinking=bool(llm.get("enable_thinking", False)),
        use_mcp=not getattr(args, "no_mcp", False),
        use_subagents=not getattr(args, "no_subagents", False),
        session_id=getattr(args, "session", None),
        selfguard_enabled=bool(security.get("selfguard_enabled", True)),
        show_subagent_events=bool(display.get("show_subagent_events", False)),
        agent=dict(agent),
        compact=dict(config.compact),
    )
