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
    """
    role: MessageRole
    content: str | list | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict:
        """转换为OpenAI API兼容的字典格式，自动排除值为None的字段。"""
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
