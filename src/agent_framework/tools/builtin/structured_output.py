"""内置工具：结构化输出解析与验证

三层处理机制：
  1. 解析层 - 处理 LLM 常见输出噪声（markdown 包裹、注释、尾部逗号）
  2. 验证层 - jsonschema Draft7 对照 schema 逐字段校验，返回精确错误路径
  3. 修复层 - 验证失败时反馈给 LLM 自修复（默认 deepseek-v4-flash，最多重试 2 次）
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

import jsonschema

from agent_framework.tools.decorator import tool
from agent_framework.llm.base.message import Message, MessageRole
from agent_framework.llm.providers.base import BaseLLM

# 修复层默认使用的模型（轻量快速）
_DEFAULT_FIX_MODEL = "deepseek-v4-flash"
_DEFAULT_FIX_PROVIDER = "deepseek"


def _get_default_fix_llm() -> BaseLLM:
    """获取修复层默认 LLM 实例（延迟导入避免循环依赖）"""
    from agent_framework.llm.providers import ModelRegistry
    return ModelRegistry.create(_DEFAULT_FIX_PROVIDER, model=_DEFAULT_FIX_MODEL)


# ── 公共 API ──────────────────────────────────────────────


async def structured_output(
    raw: str,
    schema: dict,
    auto_fix: bool = True,
    _fix_llm: Optional[BaseLLM] = None,
) -> dict:
    """
    解析和验证 LLM 输出的结构化数据。

    :param raw: agent 原始输出字符串
    :param schema: JSON Schema dict，描述期望格式
    :param auto_fix: 验证失败时是否调用 LLM 自修复
    :param _fix_llm: 用于自修复的 LLM 实例，默认使用 deepseek-v4-flash

    返回: {"success": bool, "data": ..., "errors": [...], "attempts": int}
    """
    result = _try_parse_and_validate(raw, schema)
    if result["success"]:
        return result

    if not auto_fix:
        return result

    # 获取修复用 LLM
    fix_llm = _fix_llm or _get_default_fix_llm()

    # 最多修复 2 次
    for attempt in range(2):
        fixed_raw = await _ask_llm_to_fix(fix_llm, raw, schema, result["errors"])
        result = _try_parse_and_validate(fixed_raw, schema)
        result["attempts"] = attempt + 2
        if result["success"]:
            return result

    return result


def create_structured_output_tool(fix_llm: Optional[BaseLLM] = None):
    """
    创建带 LLM 自修复能力的 structured_output Agent 工具。

    :param fix_llm: 用于自修复的 LLM 实例，默认使用 deepseek-v4-flash

    用法：
        # 使用默认修复模型（deepseek-v4-flash）
        tool = create_structured_output_tool()
        agent = Agent(llm=main_llm, tools=[tool])

        # 指定自定义修复模型
        from agent_framework.llm.providers import ModelRegistry
        custom_llm = ModelRegistry.create("deepseek", model="deepseek-chat")
        tool = create_structured_output_tool(fix_llm=custom_llm)
        agent = Agent(llm=main_llm, tools=[tool])
    """

    @tool(name="structured_output", description=(
        "解析和验证结构化输出。接收原始文本和 JSON Schema，"
        "自动清理 LLM 输出噪声（markdown 包裹、注释、尾部逗号），"
        "对照 schema 验证字段类型和必填项，验证失败时自动修复"
    ))
    async def _tool_func(
        raw: str,
        schema: dict,
        auto_fix: bool = True,
    ) -> str:
        """
        :param raw: 原始输出文本
        :param schema: 期望的 JSON Schema（对象格式）
        :param auto_fix: 验证失败时是否自动修复
        """
        result = await structured_output(raw, schema, auto_fix, _fix_llm=fix_llm)
        return json.dumps(result, ensure_ascii=False)

    return _tool_func


# ── 内部实现 ──────────────────────────────────────────────


def _try_parse_and_validate(raw: str, schema: dict) -> dict:
    """解析 + 验证，返回结果字典"""
    # 第一层：解析
    try:
        data = _extract_json(raw)
    except (json.JSONDecodeError, ValueError) as e:
        return {
            "success": False,
            "data": None,
            "errors": [f"JSON 解析失败: {e}"],
            "attempts": 1,
        }

    # 第二层：验证
    validator = jsonschema.Draft7Validator(schema)
    errors = [
        f"{' -> '.join(str(p) for p in e.absolute_path) or 'root'}: {e.message}"
        for e in validator.iter_errors(data)
    ]
    if errors:
        return {"success": False, "data": data, "errors": errors, "attempts": 1}

    return {"success": True, "data": data, "errors": [], "attempts": 1}


def _extract_json(raw: str) -> Any:
    """处理 LLM 常见输出噪声，提取 JSON"""
    text = raw.strip()

    # 剥除 markdown 代码块包裹
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()

    # 去除行注释（// ...）
    text = re.sub(r"//[^\n]*", "", text)

    # 修复尾部逗号: ,} 或 ,]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # 如果字符串前面有解释性文字，找到第一个 { 或 [
    if not text.startswith(("{", "[")):
        idx = min(
            (text.find(c) for c in ("{", "[") if c in text),
            default=-1,
        )
        if idx != -1:
            text = text[idx:]

    return json.loads(text)


async def _ask_llm_to_fix(
    llm: BaseLLM,
    raw: str,
    schema: dict,
    errors: list[str],
) -> str:
    """把错误信息反馈给 LLM，要求修复后只输出 JSON"""
    prompt = (
        "以下 JSON 输出未能通过 schema 验证，请修复并只输出合法 JSON，不要任何解释：\n\n"
        f"原始输出：\n{raw}\n\n"
        f"期望 Schema：\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        f"验证错误：\n" + "\n".join(f"- {e}" for e in errors)
    )

    messages = [
        Message(role=MessageRole.SYSTEM, content="你是一个 JSON 修复工具，只输出合法 JSON"),
        Message(role=MessageRole.USER, content=prompt),
    ]

    chunks = []
    async for chunk in llm.chat(messages, temperature=0):
        if chunk.content:
            chunks.append(chunk.content)

    return "".join(chunks)
