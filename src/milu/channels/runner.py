"""AgentRunner —— 把入站消息跑过 milu Agent 得到回复（接 milu 的唯一处）。

这是「平台无关核心」：每个渠道把消息规整成 InboundMessage 后交给它，它持有一个
`AgentPool`，按 (channel, user) 取到该用户独立的 Agent 实例跑一轮，取 `AgentDone`
的最终文本回出去。替代了此前手写 run.py 里的 `on_text` 接线——任何渠道复用同一个
AgentRunner，自动获得 per-用户隔离、milu 全配默认、以及（经 agent_kwargs 透传的）
strict 沙箱隔离与视觉输入能力。
"""
from __future__ import annotations

import logging
import os
from typing import Callable

from milu.agent.events import AgentDone
from milu.serving.pool import AgentPool, AgentPoolConfig

from .base import InboundMessage, OutboundMessage
from .media import DOC_EXTENSIONS

logger = logging.getLogger("milu.channels.runner")

# Agent 没产出有效回复时的兜底语（空回复会被渠道丢弃，体验不佳）
DEFAULT_FALLBACK = "（抱歉，我暂时没想到怎么回答，换个说法再试试？）"

# 只发媒体、无配文时的默认指令（否则 agent.run("") 没有可回应的文本）
DEFAULT_MEDIA_PROMPT = "请查看用户随本条消息发来的附件并回应。"

# 自定义 (channel,user)->（user_id, session_id) 的映射钩子
SessionOf = Callable[[InboundMessage], "tuple[str, str]"]


def _format_attachments(images: list[str], files: list[str]) -> str:
    """组装「本条附件清单」说明块（附在用户消息末尾）。

    实践教训：只给「请查看图片」这类干瘪指令时，模型在 file-agent 人设下会主动
    跟用户解释「这图片/文件不在我工作区」之类的内部细节——既误导（图片其实已存进
    工作区、文件也能读）又吓人。故这里**显式告知**：附件已下载到本地、可直接处理、
    **回应里别提保存位置**，并锚定到「本条消息」防止跨条串到上一条附件。
    """
    if not images and not files:
        return ""
    lines = [
        "[附件] 用户随本条消息发来以下内容，均已下载到本地工作区、可直接处理。"
        "请基于这些附件作答；回复里不要提及文件是否保存、存放在哪等内部细节。",
    ]
    for p in images:
        lines.append(
            f"- 图片：{p}（已随本条消息提供给你视觉，可直接看图作答；如需进一步处理可用 image_read）"
        )
    for p in files:
        ext = os.path.splitext(p)[1].lower()
        hint = "请先用 doc_read 读取" if ext in DOC_EXTENSIONS else "请先用 file_read 读取"
        lines.append(f"- 文件：{p}（{hint}其内容，再据此回应）")
    return "\n".join(lines)


def _normalize_admins(spec) -> set[str]:
    """把 GATEWAY_ADMINS 规格规整为身份串集合。

    接受逗号分隔字符串（兼容中文逗号）或可迭代对象。每项是 "channel:user_id"
    （精确到渠道）或裸 "user_id"（跨渠道匹配同名用户）。
    """
    if not spec:
        return set()
    if isinstance(spec, str):
        items = spec.replace("，", ",").split(",")
    else:
        items = list(spec)
    return {str(x).strip() for x in items if str(x).strip()}


class AgentRunner:
    """持有 AgentPool，把 InboundMessage 跑过 Agent 得到 OutboundMessage。

    既是 Gateway 的 dispatch 实现（`__call__`），也负责 pool 生命周期
    （`start`/`close`，由 Gateway 在 lifespan 中调用）。
    """

    def __init__(
        self,
        pool: AgentPool,
        *,
        fallback_text: str = DEFAULT_FALLBACK,
        session_of: SessionOf | None = None,
        commands: bool = False,
        admins: "str | set[str] | list[str] | None" = None,
    ) -> None:
        """
        :param commands: 是否拦截以 / 开头的消息当作命令处理（见 channels/commands.py）。
            **默认 False（不识别命令）**——/ 消息照常作为普通输入喂给 LLM；显式 True
            才开启识别。信息类命令开放给所有人、敏感类仅 admins（见 commands.py）。
        :param admins: 管理员白名单（敏感命令如切换模式/切会话仅其可用）。None 时读
            环境变量 GATEWAY_ADMINS（逗号分隔的 "channel:user_id" 或裸 "user_id"）；
            传 ""/空集合则明确「无管理员」（忽略环境变量）。
        """
        self._pool = pool
        self._fallback = fallback_text
        self._session_of = session_of
        self._commands = commands
        self._admins = _normalize_admins(
            os.environ.get("GATEWAY_ADMINS", "") if admins is None else admins
        )

    # ── 便利构造：单个共享 LLM 一行起池 ──────────────────────────────
    @classmethod
    def from_llm(
        cls,
        llm,
        *,
        agent_kwargs: dict | None = None,
        pool_config: AgentPoolConfig | None = None,
        fallback_text: str = DEFAULT_FALLBACK,
        session_of: SessionOf | None = None,
        commands: bool = False,
        admins: "str | set[str] | list[str] | None" = None,
    ) -> "AgentRunner":
        """用单个共享 LLM 实例创建（AsyncOpenAI 协程安全，多用户共享安全）。

        :param agent_kwargs: 透传给每个 per-用户 Agent 的构造参数。strict 隔离
            （sandbox="docker" / workspace_jail=True）即经此显式传入——AgentPool
            默认工厂不会自动套用 config 的 multiuser=strict，必须在这里显式给。
        :param pool_config: 池配置（并发/容量/TTL）；None 时由 from_llm 按分层配置加载。
        :param commands: 是否启用 / 命令；admins: 管理员白名单（详见 __init__）。
        """
        pool = AgentPool.from_llm(
            llm, config=pool_config, agent_kwargs=agent_kwargs or None
        )
        return cls(
            pool, fallback_text=fallback_text, session_of=session_of,
            commands=commands, admins=admins,
        )

    @property
    def pool(self) -> AgentPool:
        return self._pool

    @property
    def admins(self) -> set[str]:
        """管理员白名单身份串集合（供启动横幅展示数量）。"""
        return self._admins

    async def start(self) -> None:
        await self._pool.start()

    async def close(self) -> None:
        await self._pool.stop()

    # ── dispatch 实现 ─────────────────────────────────────────────────
    def _keys(self, inbound: InboundMessage) -> tuple[str, str]:
        """派生 (user_id, session_id)。默认按 渠道:用户 隔离、一人一会话。"""
        if self._session_of is not None:
            return self._session_of(inbound)
        uid = f"{inbound.channel}:{inbound.user_id}"
        sid = uid if inbound.session_id is None else f"{inbound.channel}:{inbound.session_id}"
        return uid, sid

    def _is_admin(self, inbound: InboundMessage) -> bool:
        """调用者是否管理员：身份串 "channel:user_id" 或裸 "user_id" 命中白名单。"""
        if not self._admins:
            return False
        return (
            f"{inbound.channel}:{inbound.user_id}" in self._admins
            or inbound.user_id in self._admins
        )

    def _compose_input(self, inbound: InboundMessage) -> tuple[str, list[str] | None]:
        """把入站消息规整为喂给 Agent 的 (文本, 图片路径列表)。

        - 图片 → 经 `agent.run(images=...)` 走视觉物化；
        - 图片/文件 → 在消息尾追加统一「本条附件清单」，告知已下载到本地、可直接
          处理、别提保存细节（防模型输出「文件不在我工作区」之类的误导废话）；
        - 只发媒体无配文时补一句默认指令，否则 `agent.run("")` 没有可回应的文本。
        """
        text = inbound.text or ""
        images = inbound.images or None
        files = inbound.files or []
        if not (images or files):
            return text, images
        if not text.strip():
            text = DEFAULT_MEDIA_PROMPT
        manifest = _format_attachments(inbound.images or [], files)
        if manifest:
            text = f"{text}\n\n{manifest}"
        return text, images

    async def __call__(self, inbound: InboundMessage) -> OutboundMessage | None:
        """跑一轮 Agent，返回回复（异常/空回复回退兜底语，绝不向渠道抛异常）。"""
        user_id, session_id = self._keys(inbound)
        text = (inbound.text or "").strip()
        # / 开头 → 作为命令处理（不喂 LLM），权限分层见 channels/commands.py
        if self._commands and text.startswith("/"):
            return await self._run_command(inbound, user_id, session_id, text)

        text_in, images = self._compose_input(inbound)
        reply = ""
        try:
            async with self._pool.acquire(user_id=user_id, session_id=session_id) as h:
                async for evt in h.agent.run(text_in, images=images):
                    if isinstance(evt, AgentDone):
                        reply = evt.final_text
        except Exception:
            logger.exception("Agent 运行失败 channel=%s user=%s", inbound.channel, user_id)
            return OutboundMessage(text=self._fallback)
        return OutboundMessage(text=reply or self._fallback)

    async def _run_command(
        self, inbound: InboundMessage, user_id: str, session_id: str, cmd: str
    ) -> OutboundMessage | None:
        """在该用户的 Agent 上执行一个 / 命令并返回文本回复。"""
        from .commands import run_command

        is_admin = self._is_admin(inbound)
        identity = f"{inbound.channel}:{inbound.user_id}"
        try:
            async with self._pool.acquire(user_id=user_id, session_id=session_id) as h:
                reply = await run_command(
                    h.agent, cmd, is_admin=is_admin, identity=identity
                )
        except Exception:
            logger.exception(
                "命令执行失败 channel=%s user=%s cmd=%s",
                inbound.channel, user_id, cmd.split(maxsplit=1)[0],
            )
            return OutboundMessage(text="命令执行失败，请稍后再试。")
        return OutboundMessage(text=reply) if reply else None
