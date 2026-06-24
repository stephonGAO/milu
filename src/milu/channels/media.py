"""网关媒体下载落盘 —— 把各渠道的图片/文件存进 per-用户工作区，供视觉/文件工具读取。

各渠道收到图片/文件类消息时，用自己的平台 API 把媒体下载为二进制，再经本模块
落盘到「该用户工作区下的 `_incoming` 子目录」，把绝对路径填进 InboundMessage：

- **图片** → `InboundMessage.images`，AgentRunner 经 `agent.run(images=...)` 走视觉物化
  （base64 注入，不经文件工具，故不受 workspace_jail 围栏影响）。
- **文件** → `InboundMessage.files`，AgentRunner 追加附件说明，模型用 doc_read/file_read 读取。

**为何落工作区内**：strict 部署（workspace_jail=true）下 doc_read/file_read 被关进
该用户工作区，下载文件必须落在工作区内才读得到。这里用与 gateway agent_factory
**同一 user_id（"channel:user_id"）**派生工作区路径，使下载文件正好落在该用户
Agent 的工作区里。图片虽不经文件工具、不受围栏限制，也统一落这里便于管理与清理。
"""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

from milu.llm.base.vision import IMAGE_EXTENSIONS, MAX_IMAGE_BYTES
from milu.resources import workspace_dir

logger = logging.getLogger("milu.channels.media")

# 下载附件保留天数（落盘时顺带清理过期文件，防工作区无限增长）
MEDIA_TTL_SECONDS = 7 * 24 * 3600

# content-type → 图片扩展名（平台资源下载常只给 content-type，不带文件名）
_CT_IMAGE_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
}


def _workspace_base() -> str | None:
    """读分层配置的 agent.workspace（与 gateway agent_factory 同源），失败回退 None。"""
    try:
        from milu.config import load_config

        return load_config().agent.get("workspace") or None
    except Exception:
        return None


def media_dir(channel: str, user_id: str) -> Path:
    """该用户的网关媒体落点：per-用户工作区下的 `_incoming` 子目录（顺带清理过期文件）。"""
    root = workspace_dir(user_id=f"{channel}:{user_id}", base=_workspace_base())
    d = root / "_incoming"
    d.mkdir(parents=True, exist_ok=True)
    _cleanup(d)
    return d


def _cleanup(d: Path, ttl: float = MEDIA_TTL_SECONDS) -> None:
    """清理过期下载（按 mtime）。尽力而为，失败静默，不阻断落盘。"""
    cutoff = time.time() - ttl
    try:
        for f in d.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
    except OSError:
        pass


def _safe_name(name: str) -> str:
    """文件名安全化：只留字母数字与 _.-，取尾部 120 字符（保留扩展名）。"""
    name = re.sub(r"[^A-Za-z0-9_.\-]", "_", os.path.basename(name or ""))
    return name[-120:] or "file"


def _write(channel: str, user_id: str, data: bytes, name: str) -> Path:
    """把二进制落盘到该用户媒体目录，加毫秒时间戳前缀防同名覆盖，返回绝对路径。"""
    d = media_dir(channel, user_id)
    path = d / f"{int(time.time() * 1000)}_{_safe_name(name)}"
    path.write_bytes(data)
    return path.resolve()


def _image_ext(content_type: str = "", filename: str = "") -> str:
    """推断图片扩展名：优先 filename 的合法图片后缀，其次 content-type，默认 .jpg。"""
    ext = os.path.splitext(filename)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return ext
    ct = (content_type or "").split(";")[0].strip().lower()
    return _CT_IMAGE_EXT.get(ct, ".jpg")


def save_image(
    channel: str,
    user_id: str,
    data: bytes,
    *,
    content_type: str = "",
    filename: str = "",
    media_id: str = "",
) -> str | None:
    """把下载到的图片落盘，返回绝对路径字符串；超限/空数据返回 None。

    文件名带合法图片后缀则沿用，否则用 media_id + 推断后缀（视觉层按扩展名识别格式）。
    """
    if not data:
        return None
    if len(data) > MAX_IMAGE_BYTES:
        logger.warning(
            "图片超过 %.1fMB 上限，跳过 channel=%s", MAX_IMAGE_BYTES / 1024 / 1024, channel
        )
        return None
    name = _safe_name(filename)
    if os.path.splitext(name)[1].lower() not in IMAGE_EXTENSIONS:
        name = f"{_safe_name(media_id) or 'image'}{_image_ext(content_type, filename)}"
    return str(_write(channel, user_id, data, name))
