"""BaseLLM 抽象基类和 ModelCapabilities 能力描述符"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from openai import AsyncOpenAI

from agent_framework.exceptions import AuthenticationError
from agent_framework.models.message import Message
from agent_framework.models.response import StreamChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelCapabilities:
    """
    模型能力描述符 - 声明某个厂商/模型支持的功能集合。

    每个厂商实例化一个 ModelCapabilities，明确声明自身支持和不支持的功能。
    调用方可通过 model.capabilities 查询能力，实现运行时动态UI展示。
    使用 frozen=True 确保创建后不可修改。
    """
    # 基础能力
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_json_mode: bool = False
    supports_web_search: bool = False
    supports_thinking: bool = False
    supports_embedding: bool = False

    # 多模态理解能力
    supports_vision: bool = False
    supports_audio_understand: bool = False
    supports_video: bool = False
    supports_document: bool = False

    # 多模态生成能力
    supports_image_generation: bool = False
    supports_audio_generation: bool = False

    # 模型规格
    max_context_window: int = 8192
    supported_output_formats: tuple = ("text",)


class BaseLLM(ABC):
    """
    所有厂商模型的抽象基类。

    子类需要实现:
        - provider_name: 厂商名称（用于环境变量查找等）
        - base_url: API基础URL
        - capabilities: 该厂商的能力声明
        - _get_available_param_names(): 该厂商支持的参数名集合
        - chat(): 流式聊天接口

    基类提供:
        - OpenAI客户端管理（懒加载）
        - API Key获取（构造函数 > 环境变量）
        - 参数校验和过滤
        - 可用参数查询
    """

    def __init__(self, api_key: str | None = None, model: str = "", **kwargs):
        """
        初始化LLM实例。

        参数:
            api_key: API密钥，为None时从环境变量读取
            model: 模型名称
            **kwargs: 其他配置参数
        """
        self._api_key = api_key
        self.model = model
        self._extra_kwargs = kwargs
        self._client: AsyncOpenAI | None = None

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """厂商标识名，如 'qwen', 'kimi' 等"""
        ...

    @property
    @abstractmethod
    def base_url(self) -> str:
        """API基础URL"""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """该厂商/模型的能力声明"""
        ...

    @abstractmethod
    def _get_available_param_names(self) -> set[str]:
        """返回该厂商支持的所有参数名称集合。"""
        ...

    @abstractmethod
    async def chat(self, messages: list[Message], **kwargs) -> AsyncIterator[StreamChunk]:
        """统一的流式聊天接口。"""
        ...

    def _get_api_key(self) -> str:
        """获取API密钥。优先级：构造函数传入 > 环境变量 {PROVIDER_NAME}_API_KEY"""
        if self._api_key:
            return self._api_key
        env_key = f"{self.provider_name.upper()}_API_KEY"
        key = os.environ.get(env_key)
        if not key:
            raise AuthenticationError(
                f"未找到API Key。请通过构造函数传入或设置环境变量 {env_key}"
            )
        return key

    def _get_client(self) -> AsyncOpenAI:
        """获取或创建OpenAI异步客户端实例（懒加载）"""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._get_api_key(),
                base_url=self.base_url,
            )
        return self._client

    def get_available_params(self) -> dict[str, dict]:
        """
        返回当前模型可用的参数及其元信息。
        返回格式: {参数名: {"type": 类型, "default": 默认值}}
        仅包含该厂商实际支持的参数。
        """
        all_params = {
            "temperature": {"type": float, "default": None},
            "top_p": {"type": float, "default": None},
            "max_tokens": {"type": int, "default": None},
            "stop": {"type": list, "default": None},
            "frequency_penalty": {"type": float, "default": None},
            "presence_penalty": {"type": float, "default": None},
            "web_search": {"type": bool, "default": False},
            "web_search_strategy": {"type": str, "default": "auto"},
            "enable_thinking": {"type": bool, "default": False},
            "thinking_level": {"type": str, "default": "medium"},
            "tools": {"type": list, "default": None},
            "tool_choice": {"type": str, "default": "auto"},
            "image_size": {"type": str, "default": "1024x1024"},
            "image_quality": {"type": str, "default": "standard"},
            "num_images": {"type": int, "default": 1},
            "voice": {"type": str, "default": None},
            "audio_format": {"type": str, "default": "mp3"},
            "speed": {"type": float, "default": 1.0},
        }
        available_names = self._get_available_param_names()
        return {name: info for name, info in all_params.items() if name in available_names}

    def _validate_params(self, params: dict) -> dict:
        """校验并过滤参数：移除不支持的参数并记录警告。"""
        available = self._get_available_param_names()
        validated = {}
        for key, value in params.items():
            if key in available:
                validated[key] = value
            else:
                logger.warning(f"[{self.provider_name}] 参数 '{key}' 不被支持，已忽略")
        return validated

    def _messages_to_dicts(self, messages: list[Message]) -> list[dict]:
        """将Message对象列表转换为API兼容的字典列表"""
        return [msg.to_dict() for msg in messages]
