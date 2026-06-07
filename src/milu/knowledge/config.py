"""知识库配置 dataclass —— 默认值的唯一来源。

config.json 的 `knowledge` 分节从本 dataclass 派生基线（见 milu/config.py
`_builtin_defaults()`），默认值在代码里只有一份。

字段取舍：
- `user_id` 是运行时身份（多用户隔离维度），不入 config.json；
- `api_key` 走 `.env` / 环境变量 `{PROVIDER}_API_KEY`（密钥不落 config.json），
  仅供程序化显式传入。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass
class KnowledgeConfig:
    """向量知识库配置。

    `Agent(knowledge=True)` 等价于 `Agent(knowledge=KnowledgeConfig())`；
    应用/CLI 入口可读分层配置后构造本对象传入（库纯净性：Agent 不读 config.json）。
    """

    user_id: str = "default"          # 知识库归属标识（多用户隔离维度）
    embedding_provider: str = "qwen"  # embedding 厂商（独立于对话模型；claude/deepseek/kimi 无 embedding API）
    embedding_model: str = ""         # 空 → 用厂商默认 embedding 模型（见 embedder._EMBEDDING_PROVIDERS）
    api_key: str | None = None        # None → 环境变量 {PROVIDER}_API_KEY
    chunk_size: int = 800             # 分块目标长度（字符）
    chunk_overlap: int = 200          # 超长段落滑动窗口的重叠长度（字符）
    top_k: int = 5                    # kb_search 默认返回片段数
    min_score: float = 0.35           # 检索相似度阈值：低于该值的片段不返回（防把无关内容喂给 LLM
                                      # 诱导幻觉；基准评测正负分离度 +0.388，0.35 居正负区间之间。0 = 不过滤）
    auto_retrieve: bool = False       # 前置自动检索：每轮用户消息自动 embedding+检索，达标片段注入
                                      # system prompt（不依赖模型决策调 kb_search，永不漏检；代价每轮
                                      # +1 次 embedding ~200ms。适合知识库为核心场景的部署）
    auto_top_k: int = 3               # 自动注入的片段数上限（独立于 kb_search 的 top_k，更紧——
                                      # 自动注入每轮都发生，按"高质量、省上下文"原则收窄）
    auto_min_score: float = 0.5       # 自动注入的相似度门槛（高于 kb_search 的 min_score——自动路径
                                      # 走高精度，边缘命中由模型经 kb_search 以 0.35 阈值捞回，不损召回。
                                      # 评测基准：正确命中分布 0.41~0.76，0.5 保留其中高置信段）
    batch_size: int = 10              # embedding 单批条数（DashScope 等厂商单批上限 10，取最保守值）

    @classmethod
    def from_mapping(cls, data: dict, user_id: str = "default") -> "KnowledgeConfig":
        """从配置分节字典构造（忽略未知键与 enabled 开关；user_id 由调用方指定）。

        供 CLI/服务入口把 config.json 的 `knowledge` 分节转为本对象，
        分节里的 `enabled` 等非字段键自动忽略。
        """
        allowed = {f.name for f in dataclasses.fields(cls)} - {"user_id"}
        kwargs = {k: v for k, v in data.items() if k in allowed}
        return cls(user_id=user_id, **kwargs)
