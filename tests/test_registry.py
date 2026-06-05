"""测试 ModelRegistry 注册表"""

import pytest
from milu.llm.providers import ModelRegistry
from milu.llm.providers.base import BaseLLM, ModelCapabilities
from milu.llm.base.message import Message
from milu.llm.base.response import StreamChunk


class FakeLLM(BaseLLM):
    """测试用的假LLM实现"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def base_url(self) -> str:
        return "https://fake.api/v1"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    def _get_available_param_names(self) -> set[str]:
        return {"temperature"}

    async def chat(self, messages, **kwargs):
        yield StreamChunk(content="fake")


class TestModelRegistry:
    def setup_method(self):
        """每个测试前清空注册表"""
        ModelRegistry._providers.clear()

    def test_register_and_create(self):
        ModelRegistry.register("fake", FakeLLM)
        model = ModelRegistry.create("fake", api_key="key", model="test")
        assert isinstance(model, FakeLLM)
        assert model.model == "test"

    def test_list_providers(self):
        ModelRegistry.register("fake", FakeLLM)
        assert "fake" in ModelRegistry.list_providers()

    def test_create_unknown_provider(self):
        with pytest.raises(ValueError, match="不支持的厂商"):
            ModelRegistry.create("unknown")

    def test_register_duplicate_overwrites(self):
        ModelRegistry.register("fake", FakeLLM)
        ModelRegistry.register("fake", FakeLLM)
        assert "fake" in ModelRegistry.list_providers()

    def test_create_passes_kwargs(self):
        ModelRegistry.register("fake", FakeLLM)
        model = ModelRegistry.create("fake", api_key="my-key", model="my-model")
        assert model.model == "my-model"
