"""
模型厂商包 - 包含所有厂商实现和注册表。

使用方式:
    from agent_framework.llm.providers import ModelRegistry

    # 工厂方式创建
    model = ModelRegistry.create("qwen", model="qwen-max")

    # 或直接导入
    from agent_framework.llm.providers.qwen import QwenLLM
    model = QwenLLM(model="qwen-max")
"""

from __future__ import annotations

from agent_framework.llm.providers.base import BaseLLM, ModelCapabilities


class ModelRegistry:
    """
    模型注册表 - 管理所有厂商的LLM实现。

    提供统一的工厂方法创建模型实例，支持动态注册新厂商。
    各厂商模块加载时自动注册到此类。
    """

    _providers: dict[str, type[BaseLLM]] = {}

    @classmethod
    def register(cls, name: str, provider_class: type[BaseLLM]) -> None:
        """注册一个厂商实现。"""
        cls._providers[name] = provider_class

    @classmethod
    def create(cls, provider: str, **kwargs) -> BaseLLM:
        """根据厂商名创建模型实例。"""
        if provider not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(f"不支持的厂商: {provider}，可用: {available}")
        return cls._providers[provider](**kwargs)

    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有已注册的厂商名称"""
        return list(cls._providers.keys())


# 导入所有厂商模块以触发自动注册
from agent_framework.llm.providers import qwen
from agent_framework.llm.providers import kimi
from agent_framework.llm.providers import glm
from agent_framework.llm.providers import deepseek
from agent_framework.llm.providers import minimax
from agent_framework.llm.providers import doubao

__all__ = [
    "ModelRegistry",
    "BaseLLM",
    "ModelCapabilities",
]
