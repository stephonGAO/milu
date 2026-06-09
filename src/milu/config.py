"""统一分层配置体系 —— 一处总览/调整全部可调参数。

分层优先级（高 → 低）：
    CLI 参数 > 用户级 ~/.milu/config.json > 项目级 config/milu.json > 代码内 dataclass 默认值

设计原则：
- **不重复默认值**：基线从现有 dataclass 派生（`AgentConfig` / `CompactConfig` /
  `AgentPoolConfig`），默认值在代码里仍只有一份；config.json 只承载「覆盖」。
- **库纯净**：本模块只在应用/CLI 入口被显式调用（`load_config()`），不侵入
  `Agent` / `AgentPool` 的构造路径。直接 `Agent(llm)` 仍走 dataclass 默认，单测 hermetic。
- **职责分离**：`.env` 只放密钥与必要的进程级开关；可调参数全部在 config.json。

配置分节：
    agent   —— mode / session_enabled / llm（该 Agent 用的模型对象：
               provider / model / web_search / enable_thinking）/ AgentConfig 运行限额
    compact —— CompactConfig 上下文压缩
    pool    —— AgentPoolConfig 多用户资源池（可序列化子集）
    scheduler —— SchedulerConfig 定时任务调度引擎（并发上限/任务超时/通知开关）
    knowledge —— 向量知识库（enabled 开关 + KnowledgeConfig 派生的 embedding/分块参数；
               user_id 为运行时身份、api_key 走 .env，均不入 config.json）
    default_models —— 各厂商默认模型表（仅供查看/参考；agent.llm.model 留 null 时按它取默认）
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from milu.i18n import t
from milu.resources import project_config_path, user_config_path

logger = logging.getLogger(__name__)

# 默认厂商（llm.provider 未设置时）
DEFAULT_PROVIDER = "qwen"

# 各厂商默认模型。已核对可用：qwen / deepseek / openai / gemini / anthropic / doubao；
# kimi / glm / minimax 为合理默认值，按需用 `config set` 或 --model 覆盖。
DEFAULT_MODELS: dict[str, str] = {
    "qwen": "qwen3.6-plus",
    "deepseek": "deepseek-v4-flash",
    "kimi": "kimi-k2.6",
    "glm": "glm-5",
    "minimax": "MiniMax-M3",
    "doubao": "doubao-seed-2-0-lite-260428",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3.5-flash",
    "anthropic": "claude-sonnet-4.6",
}

# agent 分节里属于 AgentConfig 的字段（mode / session_enabled 为 Agent 直接参数，不在此列）
_AGENT_CONFIG_KEYS = (
    "max_turns", "timeout", "total_timeout", "max_total_tokens", "tool_call_limit",
)
# pool 分节暴露的可调字段（mcp_config_path 走环境变量/显式传参，不入 config.json）
_POOL_KEYS = (
    "max_agents", "max_concurrent_runs", "idle_ttl_seconds",
    "sweep_interval_seconds", "acquire_timeout", "shared_mcp",
)
# scheduler 分节暴露的可调字段（SchedulerConfig 全量）
_SCHEDULER_KEYS = ("max_concurrent_tasks", "task_timeout", "notify")
# knowledge 分节暴露的可调字段（KnowledgeConfig 子集：user_id 是运行时身份、
# api_key 走 .env / {PROVIDER}_API_KEY，均不入 config.json）
_KNOWLEDGE_KEYS = (
    "embedding_provider", "embedding_model",
    "chunk_size", "chunk_overlap", "top_k", "min_score",
    "auto_retrieve", "auto_top_k", "auto_min_score", "batch_size",
)

_TRUTHY = {"1", "true", "yes", "on", "y"}
_FALSY = {"0", "false", "no", "off", "n"}

# 进程内只对废弃 api_keys 字段告警一次
_warned_legacy_keys = False


def _builtin_defaults() -> dict:
    """基线默认配置：全部从现有 dataclass 派生，默认值不在此重复定义。"""
    from milu.agent.config import AgentConfig, CompactConfig
    from milu.knowledge.config import KnowledgeConfig
    from milu.scheduler.engine import SchedulerConfig
    from milu.serving.pool import AgentPoolConfig

    pool_defaults = AgentPoolConfig()
    scheduler_defaults = SchedulerConfig()
    knowledge_defaults = KnowledgeConfig()
    return {
        "agent": {
            "mode": "auto",
            "session_enabled": True,
            # 该 Agent 使用的模型对象（model 留 null → 按 default_models[provider] 取默认）
            "llm": {
                "provider": DEFAULT_PROVIDER,
                "model": None,
                "web_search": True,
                "enable_thinking": False,
            },
            **asdict(AgentConfig()),
        },
        "compact": asdict(CompactConfig()),
        "pool": {k: getattr(pool_defaults, k) for k in _POOL_KEYS},
        "scheduler": {k: getattr(scheduler_defaults, k) for k in _SCHEDULER_KEYS},
        "knowledge": {
            # enabled 是应用层开关（CLI/服务入口据此决定是否给 Agent 传 knowledge），
            # 类比 agent.session_enabled，不属于 KnowledgeConfig dataclass。
            # 默认开：空库时 auto_retrieve 在 is_empty() 处早退、零开销零 numpy 依赖，
            # 未入库用户无成本；真正入库者已配好 embedding key 方能 kb_ingest。
            "enabled": True,
            **{k: getattr(knowledge_defaults, k) for k in _KNOWLEDGE_KEYS},
        },
        "security": {
            "selfguard_enabled": True,   # 禁止 Agent 读写 milu 自身代码和配置文件
        },
        "display": {
            # 是否把子代理（researcher/reader/coder 等）的内部事件输出到 CLI / Web 前端。
            # 默认关闭——子代理照常运行，仅不展示其内部 thinking/工具/文本细节，保持界面精简。
            "show_subagent_events": False,
        },
        "default_models": dict(DEFAULT_MODELS),
        # CLI 界面语言（zh/en，默认中文）；运行时可经 --lang / MILU_LANG 覆盖
        "lang": "zh",
    }


def _deep_merge(base: dict, overlay: dict) -> dict:
    """分节深合并：overlay 仅覆盖出现的键，未出现的保留 base 值。"""
    result = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _read_json(path: Path) -> dict | None:
    """读 JSON 文件；不存在→None，损坏→None + 告警（绝不抛出，配置坏不致命）。"""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("[配置] 读取失败，已忽略：%s — %s", path, e)
        return None
    if not isinstance(data, dict):
        logger.warning("[配置] 顶层非对象，已忽略：%s", path)
        return None
    return _strip_legacy(data)


def _strip_legacy(d: dict) -> dict:
    """剥离废弃的 api_keys 字段（密钥改走 .env / 环境变量），进程内告警一次。"""
    global _warned_legacy_keys
    if "api_keys" in d:
        if not _warned_legacy_keys:
            logger.warning(
                "[配置] 已忽略废弃字段 api_keys —— API Key 请改用 .env "
                "或环境变量 {PROVIDER}_API_KEY。"
            )
            _warned_legacy_keys = True
        d = {k: v for k, v in d.items() if k != "api_keys"}
    return d


@dataclass
class MiluConfig:
    """合并后的生效配置（基线 ← 项目 ← 用户）。`data` 为嵌套分节字典。"""

    data: dict

    @classmethod
    def load(cls) -> "MiluConfig":
        """加载并分层合并：基线 ← 项目 config/milu.json ← 用户 ~/.milu/config.json。"""
        merged = _builtin_defaults()
        for path in (project_config_path(), user_config_path()):
            loaded = _read_json(path)
            if loaded:
                merged = _deep_merge(merged, loaded)
        return cls(merged)

    # ── 分节便捷访问 ──────────────────────────────────────
    @property
    def llm(self) -> dict:
        """该 Agent 使用的模型对象（嵌套在 agent 下，便捷访问 agent.llm）。"""
        return self.data["agent"]["llm"]

    @property
    def agent(self) -> dict:
        return self.data["agent"]

    @property
    def compact(self) -> dict:
        return self.data["compact"]

    @property
    def pool(self) -> dict:
        return self.data["pool"]

    @property
    def scheduler(self) -> dict:
        return self.data.get("scheduler", {})

    @property
    def knowledge(self) -> dict:
        return self.data.get("knowledge", {})

    @property
    def security(self) -> dict:
        return self.data.get("security", {})

    @property
    def display(self) -> dict:
        return self.data.get("display", {})

    @property
    def default_models(self) -> dict:
        return self.data["default_models"]

    # ── 转为运行期 dataclass（供 builder / pool 消费）────────
    def to_agent_config(self):
        """构造 AgentConfig（仅取运行限额字段，mode/session_enabled 另行透传）。"""
        from milu.agent.config import AgentConfig
        a = self.agent
        return AgentConfig(**{k: a[k] for k in _AGENT_CONFIG_KEYS if k in a})

    def to_compact_config(self):
        """构造 CompactConfig。"""
        from milu.agent.config import CompactConfig
        return CompactConfig(**self.compact)

    def to_pool_config(self):
        """构造 AgentPoolConfig（可序列化子集，mcp_config_path 留默认）。"""
        from milu.serving.pool import AgentPoolConfig
        p = self.pool
        return AgentPoolConfig(**{k: p[k] for k in _POOL_KEYS if k in p})

    def to_scheduler_config(self):
        """构造 SchedulerConfig（调度引擎参数）。"""
        from milu.scheduler.engine import SchedulerConfig
        s = self.scheduler
        return SchedulerConfig(**{k: s[k] for k in _SCHEDULER_KEYS if k in s})

    def to_knowledge_config(self, user_id: str = "default"):
        """构造 KnowledgeConfig（knowledge.enabled 开关由调用方自行判断）。"""
        from milu.knowledge.config import KnowledgeConfig
        return KnowledgeConfig.from_mapping(self.knowledge, user_id=user_id)

    # ── dotted 读取 ──────────────────────────────────────
    def get(self, dotted: str):
        """按点号路径读取生效值，如 'agent.max_turns'。不存在抛 KeyError。"""
        node = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(dotted)
            node = node[part]
        return node


def load_config() -> MiluConfig:
    """加载分层合并后的生效配置（应用/CLI 入口调用）。"""
    return MiluConfig.load()


# ── 用户级写入：config set / config init ─────────────────────

def _coerce_value(current, raw: str):
    """按当前生效值的类型把命令行字符串转成目标类型。

    - bool 字段：true/false/on/off/...（区分大小写无关）
    - int / float 字段：直接转换
    - None 字段（如 model / max_total_tokens）：能转 int/float 则转，否则保留字符串
    - str 字段：原样
    """
    # bool 须在 int 之前判断（bool 是 int 的子类）
    if isinstance(current, bool):
        low = raw.strip().lower()
        if low in _TRUTHY:
            return True
        if low in _FALSY:
            return False
        raise ValueError(t("需要布尔值（true/false），收到：{raw}", raw=repr(raw)))
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if current is None:
        for conv in (int, float):
            try:
                return conv(raw)
            except ValueError:
                continue
        return raw
    return raw


def set_user_value(dotted: str, raw: str) -> tuple[Path, object]:
    """把 dotted 配置项写入用户级 ~/.milu/config.json（稀疏，只存覆盖项）。

    :return: (写入路径, 转换后的值)。
    :raises ValueError: 配置项在默认树中不存在，或类型转换失败。
    """
    eff = load_config()
    try:
        current = eff.get(dotted)
    except KeyError:
        raise ValueError(t("未知配置项：{dotted}", dotted=dotted))
    if isinstance(current, dict):
        raise ValueError(t("'{dotted}' 是配置分节，请设置其下具体项（如 {dotted}.xxx）", dotted=dotted))
    try:
        value = _coerce_value(current, raw)
    except ValueError as e:
        raise ValueError(f"'{dotted}' {e}")

    path = user_config_path()
    raw_user = _read_json(path) or {}
    node = raw_user
    parts = dotted.split(".")
    for p in parts[:-1]:
        nxt = node.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            node[p] = nxt
        node = nxt
    node[parts[-1]] = value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw_user, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, value


def write_project_template(path: Path | None = None) -> Path:
    """把全量默认树写到项目级 config/milu.json（config init）。"""
    path = path or project_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_builtin_defaults(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
