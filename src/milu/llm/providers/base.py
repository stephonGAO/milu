"""BaseLLM 抽象基类和 ModelCapabilities 能力描述符"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from openai import AsyncOpenAI

from milu.llm.base.exceptions import AuthenticationError
from milu.llm.base.message import Message
from milu.llm.base.response import StreamChunk

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
        # 幂等、可通过 MILU_NO_DOTENV 关闭的 .env 加载（不再每次静默扫描 CWD）
        from milu._env import ensure_dotenv_loaded
        ensure_dotenv_loaded()
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

    async def aclose(self) -> None:
        """关闭底层 HTTP 客户端，释放连接池。

        懒加载客户端尚未创建时为空操作。多用户/多租户场景下，
        当某个 LLM 实例被淘汰（如 KeyedLLMProvider 的 LRU 回收）或服务关闭时，
        应调用本方法及时释放连接池，避免连接泄漏。调用后再次 chat() 会重新懒建客户端。
        """
        client = self._client
        if client is not None:
            self._client = None
            await client.close()

    # 注释：
    # 这里是构造时候可能会用到的默认参数，实际用到多少是在实现的时候的chat运行的时候
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

    
    # 注释：
    # 用户在构建实例（因为要兼容多平台所以多这一步，正常原生api只在运行时传入参数）的时候，
    # 或者chat运行的时候，对传入的参数做校验，
    # 因为是多供应商平台，所以如果是真实存在的参数，就验证通过放在validated中生效，
    # 如果不是这个供应商实际可用的参数就丢弃不生效。
    # chat运行时序晚于构建llm实例，所有运行时参数可以覆盖构造时参数。
    # 所有大模型在chat实现中必须有这一步。
    def _validate_params(self, params: dict) -> dict:
        """校验并过滤参数：合并构造时默认值，移除不支持的参数并记录警告。

        优先级：chat() 调用时传入的参数 > 构造时传入的参数（_extra_kwargs）
        """
        # 构造时的 kwargs 作为默认值，调用chat时附带的 params 可覆盖
        merged = {**self._extra_kwargs, **params}#chat运行 - 覆盖 ->构造
        available = self._get_available_param_names()#获取真实可用的参数
        validated = {}
        for key, value in merged.items():
            if key in available:
                validated[key] = value
            else:
                logger.warning(f"[{self.provider_name}] 参数 '{key}' 不被支持，已忽略")
        return validated

    def _messages_to_dicts(self, messages: list[Message]) -> list[dict]:
        """将Message对象列表转换为API兼容的字典列表"""
        return [msg.to_dict() for msg in messages]
