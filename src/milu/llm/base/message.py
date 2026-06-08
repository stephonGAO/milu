"""统一消息类型定义 - 兼容所有厂商的消息格式"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MessageRole(str, Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """
    统一消息结构，兼容所有厂商的消息格式。

    属性:
        role: 消息角色（system/user/assistant/tool）
        content: 消息内容，str为纯文本，list为多模态内容
        tool_calls: 工具调用信息（仅assistant消息使用）
        tool_call_id: 工具调用ID（仅tool消息使用）
        name: 工具/函数名称（仅tool消息使用）
        reasoning_content: 思考过程文本（仅assistant消息使用）。
            用于保真存储 thinking 模型产出的推理内容。⚠️【默认不进 to_dict()】——
            多数 provider（如 DeepSeek-R1）明确要求回传时【不带】reasoning_content，
            带了反而报错。需要回传的 provider（Kimi thinking 模型要求带 tool_calls 的
            assistant 消息必须回传 reasoning_content）自行在 _messages_to_dicts() 注入。
    """
    role: MessageRole
    content: str | list | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    reasoning_content: str | None = None

    def to_dict(self) -> dict:
        """转换为OpenAI API兼容的字典格式，自动排除值为None的字段。

        注意：reasoning_content 故意【不】序列化进 dict——它是 provider 间不可移植的
        字段（带了会让 DeepSeek 等报错），由需要它的 provider 在自身的
        _messages_to_dicts() 中按需注入。
        """
        result: dict = {"role": self.role.value}
        if self.content is not None:
            result["content"] = self.content
        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            result["name"] = self.name
        return result
