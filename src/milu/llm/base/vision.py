"""多模态图片内容支持 - 轻量引用块与发送时物化

设计:
  - 对话历史/会话日志中只存轻量图片引用块（不存 base64）:
        {"type": "image_path", "path": "D:/.../test.png"}
    收益：conversation.jsonl 不被 base64 撑爆、token 估算不失真、会话可序列化恢复。
  - 发送 API 时在 provider 层单点物化为标准 OpenAI 多模态块:
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    物化点：BaseLLM._messages_to_dicts()（8 个 OpenAI 兼容厂商）与
    ChatGPTLLM._messages_to_input()（Responses API 的 input_image）。
  - 物化失败（文件被删/不可读）不抛异常，降级为 text 占位块，保证对话不中断。
"""
from __future__ import annotations

import base64
import logging
import os

logger = logging.getLogger(__name__)

# 支持的图片扩展名 → MIME 类型
IMAGE_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

IMAGE_EXTENSIONS = frozenset(IMAGE_MIME_TYPES)

# 单张图片大小上限（base64 后约 +33%，过大请求会被多数厂商拒绝）
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def image_mime_type(path: str) -> str | None:
    """按扩展名返回图片 MIME 类型，非支持的扩展名返回 None。"""
    ext = os.path.splitext(path)[1].lower()
    return IMAGE_MIME_TYPES.get(ext)


def encode_image_data_url(path: str) -> str:
    """读取本地图片并编码为 data URL（data:<mime>;base64,...）。

    :raises OSError: 文件不可读
    :raises ValueError: 扩展名不是支持的图片格式
    """
    mime = image_mime_type(path)
    if mime is None:
        raise ValueError(f"不支持的图片格式: {path}")
    with open(path, "rb") as f:
        data = f.read()
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def build_user_content(text: str, image_paths: list[str]) -> list:
    """构造多模态 user 消息 content：图片引用块在前，文本块在后。"""
    blocks: list = [
        {"type": "image_path", "path": os.path.abspath(p)} for p in image_paths
    ]
    blocks.append({"type": "text", "text": text})
    return blocks


def materialize_content(content):
    """物化消息 content：把轻量 image_path 块转为 base64 data URL 的 image_url 块。

    - content 为 str/None：原样返回（纯文本消息零开销）
    - content 为 list：逐块处理——
        image_path → image_url(data URL)；文件不可读时降级为 text 占位块（绝不抛异常）
        其余块（text / image_url 等）原样透传
    """
    if not isinstance(content, list):
        return content

    materialized: list = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image_path":
            path = block.get("path", "")
            try:
                materialized.append({
                    "type": "image_url",
                    "image_url": {"url": encode_image_data_url(path)},
                })
            except (OSError, ValueError) as e:
                logger.warning("图片物化失败，降级为占位文本: %s (%s)", path, e)
                materialized.append({
                    "type": "text",
                    "text": f"[图片不可用: {path}]",
                })
        else:
            materialized.append(block)
    return materialized
