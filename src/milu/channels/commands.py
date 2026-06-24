"""Gateway 渠道的 / 命令处理（IM 友好：纯文本进、纯文本出）。

与 CLI REPL（`cli/repl.py`，ANSI 终端渲染）和 Web（`serving/web/app.py`，SSE 事件流）
的命令集对齐，但收敛为「单条纯文本回复」，适配 IM 平台（微信/飞书/Telegram）一问一答
的形态——一条 / 消息进来，回一段文本出去。

**权限分层（面向公网陌生用户）**：
- **信息类**（只读 / 仅作用于自己）对所有人开放：
  /help /whoami /history /tools /skills /plan /memory /mode(查看) /reset /compact
- **敏感类**（改模式 / 切会话 / 泄露内部系统提示词）仅管理员（`GATEWAY_ADMINS` 白名单）：
  /mode <模式>(切换) /prompt /new /load /save /sessions
  非管理员调用 → 返回「需要管理员权限」提示（陌生人不能经 /mode superwork 提权）。

是否管理员由调用方（`AgentRunner`）按 `GATEWAY_ADMINS` 判定后以 `is_admin` 传入；
本模块只负责「分层 + 执行 + 文本化」，不读环境变量、不碰渠道协议。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("milu.channels.commands")


def _own_session_prefix(identity: str) -> str:
    """由调用者身份串推出「本用户会话命名空间」前缀，用于把 /sessions 过滤到自己名下。

    网关派生会话 ID 形如 ``sanitize("channel:user")__sanitize("channel:user")``
    （见 AgentPool._derive_session_id，对非 [A-Za-z0-9_.-] 字符替换为 "_"）。故本用户
    所有确定性会话都以 ``sanitize(identity) + "__"`` 开头；按此前缀过滤即只见自己的会话，
    且 "__" 边界避免 "a" 误配 "a2" 这类前缀粘连。identity 为空时返回 ""（不匹配任何会话）。
    """
    if not identity:
        return ""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", identity) + "__"

# IM 单条消息长度上限保护（各平台不一：微信 ~2KB、Telegram 4096 字符），取保守值；
# 超长截断并加省略提示，避免平台因超限整条丢弃或报错。
_MAX_LEN = 3500

# 帮助文本：信息类（所有人）与敏感类（仅管理员）分列展示
_HELP_INFO = [
    "/help        显示帮助",
    "/whoami      查看你的身份 ID 与权限",
    "/history     查看对话历史",
    "/tools       查看可用工具",
    "/skills      查看可用技能",
    "/plan        查看当前会话计划",
    "/memory      查看长期记忆",
    "/mode        查看当前操作模式",
    "/reset       重置对话（清空上下文与计划）",
    "/compact     手动压缩对话历史",
    "/new         新建会话",
    "/sessions    查看你自己的历史会话",
]
_HELP_ADMIN = [
    "/mode <模式>  切换模式（talk/manual/auto/superwork）",
    "/prompt      查看当前系统提示词",
    "/load <id>   加载历史会话",
    "/save        保存当前会话",
]

_DENY_ADMIN = (
    "该命令需要管理员权限。\n"
    "（让服务端把你的身份 ID 加入 GATEWAY_ADMINS 后即可使用；发送 /whoami 查看你的 ID）"
)


def is_command(text: str) -> bool:
    """文本是否为 / 命令（去除首尾空白后以 / 开头）。"""
    return text.lstrip().startswith("/")


def _clip(text: str) -> str:
    """超长文本截断，保证不超过 IM 单条消息上限。"""
    if len(text) <= _MAX_LEN:
        return text
    return text[: _MAX_LEN - 12] + "\n…（已截断）"


def _read_plan_items(agent) -> list[dict]:
    """读取当前会话计划（todo 工具持久化在 session 目录的 plan.json）。"""
    if not agent.session:
        return []
    plan_file = agent.session.dir_path / "plan.json"
    if not plan_file.exists():
        return []
    try:
        return json.loads(plan_file.read_text(encoding="utf-8")).get("items", [])
    except (OSError, ValueError):
        return []


def _text_of(content) -> str:
    """把消息 content 规整为纯文本（多模态消息提取文本块 + 图片计数摘要）。"""
    if isinstance(content, list):
        imgs = sum(1 for b in content if isinstance(b, dict)
                   and b.get("type") in ("image_path", "image_url"))
        txt = " ".join(b.get("text", "") for b in content
                       if isinstance(b, dict) and b.get("type") == "text")
        return (f"[图片×{imgs}] " if imgs else "") + txt
    return content or ""


async def run_command(
    agent, cmd: str, *, is_admin: bool = False, identity: str = ""
) -> str:
    """执行一个 / 命令，返回纯文本回复（绝不抛异常；敏感命令需 is_admin=True）。

    :param agent: 该 (渠道,用户) 对应的 per-用户 Agent 实例
    :param cmd: 完整命令文本（如 "/mode auto"）
    :param is_admin: 调用者是否在 GATEWAY_ADMINS 白名单内
    :param identity: 调用者身份串 "channel:user_id"（供 /whoami 回显）
    """
    try:
        return _clip(await _dispatch(agent, cmd, is_admin=is_admin, identity=identity))
    except Exception as e:  # noqa: BLE001 — 命令兜底，绝不向渠道抛
        logger.exception("命令执行异常 cmd=%s", cmd)
        return f"命令执行异常：{e}"


async def _dispatch(agent, cmd: str, *, is_admin: bool, identity: str) -> str:
    parts = cmd.split(maxsplit=1)
    head = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    # ── 信息类（所有人）──────────────────────────────────────────────
    if head == "/help":
        return "\n".join([
            "可用命令：",
            *(f"  {x}" for x in _HELP_INFO),
            "",
            "管理员命令（需在 GATEWAY_ADMINS 白名单）：",
            *(f"  {x}" for x in _HELP_ADMIN),
        ])

    if head == "/whoami":
        return (
            f"你的身份 ID：{identity or '未知'}\n"
            f"权限：{'管理员' if is_admin else '普通用户'}"
        )

    if head == "/history":
        # 跳过 SYSTEM（系统提示词体量大、对用户无意义，否则一条就撑爆长度上限）
        msgs = [m for m in agent.history.all_messages if m.role.value != "system"]
        lines = [f"=== 对话历史（{len(msgs)} 条）==="]
        for m in msgs:
            raw = _text_of(m.content).replace("\n", " ")
            raw = raw[:200] + "…" if len(raw) > 200 else raw
            lines.append(f"[{m.role.value.upper():<9}] {raw}")
        return "\n".join(lines)

    if head == "/tools":
        active = agent.tools.list_tools()
        lines = [f"=== 活跃工具（{len(active)} 个）==="]
        for name in active:
            w = agent.tools.get_tool(name)
            tag = "" if (w and w.is_safe) else " [需确认]"
            lines.append(f"  {name}{tag}")
        dormant = agent.tools.list_dormant_tools()
        if dormant:
            lines.append(f"\n休眠工具：{len(dormant)} 个（LLM 可按需激活）")
        return "\n".join(lines)

    if head == "/skills":
        names = agent.skill_registry.list_names()
        if not names:
            return "暂无可用技能。"
        lines = [f"=== 可用技能（{len(names)} 个）==="]
        for name in names:
            cfg = agent.skill_registry.get(name)
            lines.append(f"  {name}：{cfg.description}")
        return "\n".join(lines)

    if head == "/plan":
        items = _read_plan_items(agent)
        if not items:
            return "暂无会话计划。"
        markers = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
        lines = [f"=== 会话计划（{len(items)} 条）==="]
        for it in items:
            lines.append(f"  {markers.get(it.get('status'), '[ ]')} {it.get('content', '')}")
        return "\n".join(lines)

    if head == "/memory":
        path = getattr(agent, "_memory_path", None)
        if path is None:
            return "长期记忆未启用。"
        p = Path(path)
        if not p.exists():
            return "暂无长期记忆。"
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else data.get("items", [])
        except (OSError, ValueError):
            items = []
        if not items:
            return "暂无长期记忆。"
        lines = [f"=== 长期记忆（{len(items)} 条）==="]
        for it in items:
            lines.append(f"  · {it.get('content', '')}")
        return "\n".join(lines)

    if head == "/reset":
        await agent.reset()
        return "对话已重置，上下文与计划已清空。"

    if head == "/compact":
        if not agent._history.compact_enabled:
            return "上下文压缩未启用。"
        n0 = len(agent.history._messages)
        compacted, summary = await agent._history.manual_compact()
        agent.history.replace_all(compacted)
        agent._history._log_compaction(compacted)
        return f"=== 压缩完成（{n0} → {len(compacted)} 条）===\n{summary[:600]}"

    if head == "/new":
        # 自建会话仅影响自己，开放给所有人（与 /reset 同属自助操作）
        agent.new_session()
        return f"新会话已创建：{agent.session.session_id}" if agent.session else "会话功能未启用。"

    if head == "/sessions":
        # 开放给所有人，但**只列调用者自己名下**的会话——网关多用户共用一个会话根目录
        # （session ID 内含各自 user_id），不过滤会泄露他人会话，故按本用户命名空间前缀筛。
        from milu.agent.session import Session
        from milu.resources import default_session_dir
        base = Path(agent.session.base_dir) if agent.session else default_session_dir()
        prefix = _own_session_prefix(identity)
        cur = agent.session.session_id if agent.session else None
        ss = [
            s for s in Session.list_sessions(base)
            if prefix and (s.get("session_id") or "").startswith(prefix)
        ]
        if not ss:
            return "暂无你自己的历史会话。"
        lines = [f"=== 你的历史会话（{len(ss)} 个）==="]
        for s in ss:
            mark = " ← 当前" if s.get("session_id") == cur else ""
            lines.append(f"  {s.get('session_id')}  {s.get('message_count', 0)} 条{mark}")
        return "\n".join(lines)

    if head == "/mode":
        cur = agent.mode.value if hasattr(agent.mode, "value") else str(agent.mode)
        if not arg:
            return (
                f"当前模式：{cur}\n"
                "可选：talk / manual / auto / superwork\n"
                "切换模式需管理员权限：/mode <模式>"
            )
        # 带参数 = 切换模式 → 敏感操作，需管理员
        if not is_admin:
            return _DENY_ADMIN
        try:
            agent.set_mode(arg)
            return f"模式已切换为：{arg}"
        except ValueError:
            return f"无效模式：{arg}（可选 talk/manual/auto/superwork）"

    # ── 敏感类（仅管理员）────────────────────────────────────────────
    if head == "/prompt":
        if not is_admin:
            return _DENY_ADMIN
        agent._build_system_prompt()
        msgs = agent.history.all_messages
        content = _text_of(msgs[0].content) if msgs else ""
        return f"=== 系统提示词（{len(content)} 字符）===\n{content}"

    if head == "/save":
        if not is_admin:
            return _DENY_ADMIN
        agent.save_session()
        if agent.session:
            return f"会话已保存：{agent.session.session_id}"
        return "会话功能未启用。"

    if head == "/load":
        if not is_admin:
            return _DENY_ADMIN
        if not arg:
            return "用法：/load <会话ID>"
        try:
            n = agent.load_session(arg)
            return f"会话已加载：{arg}（{n} 条消息）"
        except FileNotFoundError:
            return f"会话不存在：{arg}"

    return f"未知命令：{head}（发送 /help 查看帮助）"
