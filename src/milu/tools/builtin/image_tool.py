"""内置工具：图片读取（视觉输入）— 无状态版本

让视觉模型"看到"本地图片的桥梁：
  1. LLM 调用 image_read(path)，工具校验后把路径登记到 per-run 待注入列表
  2. Agent 在该批工具结果回填历史后，追加注入一条多模态 user 消息
     （轻量 image_path 引用块，不含 base64，见 llm/base/vision.py）
  3. 下一轮 LLM 调用时，provider 层把引用块物化为 base64 data URL，
     视觉模型即可直接查看图片内容

设计原则（与 todo/subagent 一致）：
- 工具函数无闭包、无模块级可变状态
- per-run 状态由 Agent.run() 入口经 ContextVar 注入（asyncio 任务级隔离）
- 工具批次经 asyncio.gather 并发执行时，子任务继承 Context 副本但共享
  同一 list 对象，append 跨任务可见
"""
from __future__ import annotations

import contextvars
import json
import os

from milu.llm.base.vision import IMAGE_EXTENSIONS, MAX_IMAGE_BYTES, image_mime_type
from milu.tools.decorator import tool

# ── 状态注入（asyncio 任务级隔离）──────────────────────
#
# _current_pending_images：Agent.run() 入口注入空列表；image_read 校验通过后
#   append 绝对路径，Agent 在工具批次结束后读取并注入多模态 user 消息。
#   默认 None = 非 Agent.run() 上下文（无注入通路，工具返回明确错误）。
# _current_vision_support：Agent.run() 入口注入 llm.capabilities.supports_vision，
#   不支持视觉的模型（如 deepseek）直接拒绝并给出换模型建议。
#   默认 True = 独立使用工具时不做能力拦截。

_current_pending_images: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "current_pending_images", default=None
)

_current_vision_support: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "current_vision_support", default=True
)


def _err(msg: str, **extra) -> str:
    return json.dumps({"success": False, "error": msg, **extra}, ensure_ascii=False)


@tool(name="image_read", description=(
    "读取本地图片并注入对话，供视觉模型直接查看分析。\n"
    "支持 png/jpg/jpeg/webp/gif/bmp，单张不超过 10MB。\n"
    "用户提到本地图片路径（截图、照片、图表等）时优先使用本工具，"
    "不要用 file_read（无法处理二进制图片）。"
    "调用成功后图片会出现在下一轮对话中，届时直接描述/分析所见内容即可。"
), is_safe=True)
async def image_read(path: str) -> str:
    """
    读取本地图片并注入对话上下文。

    :param path: 图片文件路径（绝对路径或相对当前目录）
    """
    if not _current_vision_support.get():
        return _err(
            "当前模型不支持视觉输入，无法查看图片。"
            "请改用支持视觉的模型（如 qwen-vl-plus、glm-4v、kimi-latest、"
            "doubao-1.5-vision-pro 等），或让用户描述图片内容。"
        )

    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        return _err(f"文件不存在: {abs_path}")

    if image_mime_type(abs_path) is None:
        ext = os.path.splitext(abs_path)[1] or "(无扩展名)"
        return _err(
            f"不支持的图片格式: {ext}",
            supported=sorted(IMAGE_EXTENSIONS),
        )

    size = os.path.getsize(abs_path)
    if size > MAX_IMAGE_BYTES:
        return _err(
            f"图片过大: {size / 1024 / 1024:.1f}MB，上限 {MAX_IMAGE_BYTES // 1024 // 1024}MB。"
            "请压缩或缩小图片后重试。"
        )

    pending = _current_pending_images.get()
    if pending is None:
        return _err(
            "image_read 需要在 Agent 对话上下文中使用（无图片注入通路）。"
        )

    if abs_path not in pending:  # 同批去重
        pending.append(abs_path)

    return json.dumps({
        "success": True,
        "path": abs_path,
        "size_bytes": size,
        "format": image_mime_type(abs_path),
        "note": "图片已注入对话，下一轮可直接查看并分析其内容",
    }, ensure_ascii=False)
