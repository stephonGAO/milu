"""CLI 配置：~/.milu/config.json 的读写，及厂商/模型/Key 的解析。

配置文件落在 user_data_dir()（默认 ~/.milu，可被环境变量
MILU_HOME 覆盖），与会话目录同源、与 CWD 解耦。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from milu.resources import user_data_dir

# 默认厂商（用户未显式指定 --provider 且配置文件未设置时）
DEFAULT_PROVIDER = "qwen"

# 各厂商的默认模型。已核对可用：qwen / deepseek / openai / gemini / anthropic / doubao；
# kimi / glm / minimax 给的是合理默认值，按需用 `config set model <名>` 或 --model 覆盖。
DEFAULT_MODELS: dict[str, str] = {
    "qwen": "qwen3.6-plus",
    "deepseek": "deepseek-v4-flash",
    "kimi": "kimi-k2.6",
    "glm": "glm-5",
    "minimax": "MiniMax-M3",
    "doubao": "doubao-pro",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3.5-flash",
    "anthropic": "claude-sonnet-4.6",
}

# config.json 中允许通过 `config set <key> <value>` 修改的标量字段及其类型，
# 用于命令行字符串值的类型转换。
SCALAR_FIELDS: dict[str, type] = {
    "provider": str,
    "model": str,
    "mode": str,
    "session_enabled": bool,
    "web_search": bool,
    "enable_thinking": bool,
}

_TRUTHY = {"1", "true", "yes", "on", "y"}
_FALSY = {"0", "false", "no", "off", "n"}


def coerce_scalar(key: str, value: str):
    """把命令行传入的字符串值转换为字段的目标类型。"""
    typ = SCALAR_FIELDS.get(key)
    if typ is bool:
        low = value.strip().lower()
        if low in _TRUTHY:
            return True
        if low in _FALSY:
            return False
        raise ValueError(f"'{key}' 需要布尔值（true/false），收到：{value!r}")
    return value


def env_key_name(provider: str) -> str:
    """该厂商的 API Key 环境变量名：{PROVIDER}_API_KEY。"""
    return f"{provider.upper()}_API_KEY"


@dataclass
class CLIConfig:
    """持久化的 CLI 配置（~/.milu/config.json）。"""

    provider: str | None = None
    model: str | None = None
    mode: str = "auto"
    session_enabled: bool = True
    web_search: bool = True
    enable_thinking: bool = False
    api_keys: dict[str, str] = field(default_factory=dict)

    @classmethod
    def path(cls) -> Path:
        """配置文件绝对路径。"""
        return user_data_dir() / "config.json"

    @classmethod
    def load(cls) -> "CLIConfig":
        """读取配置文件。不存在→默认实例；损坏→默认实例 + 告警。"""
        p = cls.path()
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"[警告] 配置文件读取失败，将使用默认值：{p} — {e}")
            return cls()
        # 只采用已知字段，忽略陌生键，避免 __init__ 报错
        known = set(cls().__dataclass_fields__)
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def save(self) -> Path:
        """写回配置文件（自动创建父目录），返回路径。"""
        p = self.path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return p


@dataclass
class Settings:
    """一次运行最终生效的设置（CLI 参数 / 环境变量 / 配置文件 / 默认值合并后的结果）。"""

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


def resolve_settings(config: CLIConfig, args) -> Settings:
    """合并 CLI 参数 / 环境变量 / 配置文件 / 内置默认，得到最终 Settings。

    优先级（高 → 低）：
      provider / model / mode：CLI 参数 > 配置文件 > 内置默认
      api_key：CLI 参数 > 环境变量 {PROVIDER}_API_KEY > 配置文件 api_keys
               （均无则为 None，交由 BaseLLM 在首次调用时从环境变量兜底/报错）

    :raises ValueError: 厂商无内置默认模型且未显式指定 model 时。
    """
    provider = getattr(args, "provider", None) or config.provider or DEFAULT_PROVIDER

    # 模型解析：CLI 显式 model 始终优先；否则配置文件里的 model 只在「未切换厂商」
    # （解析出的 provider == 配置 model 所属的厂商）时才沿用，避免给 deepseek 套上
    # qwen 的模型名；都没有则回退到厂商的内置默认模型。
    cli_model = getattr(args, "model", None)
    config_model_provider = config.provider or DEFAULT_PROVIDER
    if cli_model:
        model = cli_model
    elif config.model and provider == config_model_provider:
        model = config.model
    else:
        model = DEFAULT_MODELS.get(provider)
    if not model:
        raise ValueError(
            f"厂商 '{provider}' 没有内置默认模型，请用 --model 指定，"
            f"或运行 `config set model <模型名>`。"
        )

    api_key = (
        getattr(args, "api_key", None)
        or os.environ.get(env_key_name(provider))
        or config.api_keys.get(provider)
    )

    mode = getattr(args, "mode", None) or config.mode or "auto"

    session_enabled = config.session_enabled
    if getattr(args, "no_session", False):
        session_enabled = False

    return Settings(
        provider=provider,
        model=model,
        api_key=api_key,
        mode=mode,
        session_enabled=session_enabled,
        web_search=config.web_search,
        enable_thinking=config.enable_thinking,
        use_mcp=not getattr(args, "no_mcp", False),
        use_subagents=not getattr(args, "no_subagents", False),
        session_id=getattr(args, "session", None),
    )
