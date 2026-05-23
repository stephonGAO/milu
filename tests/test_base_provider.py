"""测试 BaseLLM 抽象基类和 ModelCapabilities"""

import pytest
from agent_framework.llm.providers.base import BaseLLM, ModelCapabilities
from agent_framework.llm.base.message import Message, MessageRole
from agent_framework.llm.base.response import StreamChunk


# ==================== ModelCapabilities 测试 ====================


class TestModelCapabilities:
    """测试 ModelCapabilities 数据类的默认值、不可变性和自定义值"""

    def test_default_capabilities(self):
        """测试所有能力字段的默认值"""
        caps = ModelCapabilities()
        # 基础能力
        assert caps.supports_streaming is True
        assert caps.supports_function_calling is False
        assert caps.supports_json_mode is False
        assert caps.supports_web_search is False
        assert caps.supports_thinking is False
        assert caps.supports_embedding is False
        # 多模态理解能力
        assert caps.supports_vision is False
        assert caps.supports_audio_understand is False
        assert caps.supports_video is False
        assert caps.supports_document is False
        # 多模态生成能力
        assert caps.supports_image_generation is False
        assert caps.supports_audio_generation is False
        # 模型规格
        assert caps.max_context_window == 8192
        assert caps.supported_output_formats == ("text",)

    def test_frozen(self):
        """测试 frozen=True：创建后不可修改"""
        caps = ModelCapabilities()
        with pytest.raises(AttributeError):
            caps.supports_vision = True

    def test_custom_capabilities(self):
        """测试自定义能力配置"""
        caps = ModelCapabilities(
            supports_function_calling=True,
            supports_thinking=True,
            max_context_window=131072,
        )
        assert caps.supports_function_calling is True
        assert caps.supports_thinking is True
        assert caps.max_context_window == 131072

    def test_equality(self):
        """测试两个相同配置的 ModelCapabilities 相等"""
        caps1 = ModelCapabilities(supports_vision=True)
        caps2 = ModelCapabilities(supports_vision=True)
        assert caps1 == caps2

    def test_supported_output_formats_custom(self):
        """测试自定义输出格式"""
        caps = ModelCapabilities(supported_output_formats=("text", "json", "markdown"))
        assert caps.supported_output_formats == ("text", "json", "markdown")


# ==================== ConcreteLLM 测试辅助类 ====================


class ConcreteLLM(BaseLLM):
    """用于测试的具体LLM实现"""

    def __init__(self, api_key: str = "test-key", model: str = "test-model", **kwargs):
        super().__init__(api_key=api_key, model=model, **kwargs)

    @property
    def provider_name(self) -> str:
        return "test_provider"

    @property
    def base_url(self) -> str:
        return "https://api.test.com/v1"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_function_calling=True,
            supports_web_search=True,
            supports_thinking=True,
            max_context_window=32768,
        )

    def _get_available_param_names(self) -> set[str]:
        return {
            "temperature", "top_p", "max_tokens", "stop",
            "frequency_penalty", "presence_penalty",
            "web_search", "enable_thinking", "thinking_level",
        }

    async def chat(self, messages, **kwargs):
        """模拟流式聊天，产出一个测试数据块"""
        yield StreamChunk(content="test")


# ==================== BaseLLM 测试 ====================


class TestBaseLLM:
    """测试 BaseLLM 抽象基类的各项功能"""

    def test_provider_name(self):
        """测试厂商名称属性"""
        llm = ConcreteLLM()
        assert llm.provider_name == "test_provider"

    def test_base_url(self):
        """测试基础URL属性"""
        llm = ConcreteLLM()
        assert llm.base_url == "https://api.test.com/v1"

    def test_capabilities(self):
        """测试能力声明"""
        llm = ConcreteLLM()
        caps = llm.capabilities
        assert caps.supports_function_calling is True
        assert caps.supports_web_search is True
        assert caps.supports_thinking is True
        assert caps.max_context_window == 32768

    def test_model_name(self):
        """测试模型名称设置"""
        llm = ConcreteLLM(model="my-model")
        assert llm.model == "my-model"

    def test_model_name_default(self):
        """测试默认模型名称"""
        llm = ConcreteLLM()
        assert llm.model == "test-model"

    # ---------- 可用参数测试 ----------

    def test_get_available_params_contains_expected(self):
        """测试可用参数包含预期参数"""
        llm = ConcreteLLM()
        params = llm.get_available_params()
        assert "temperature" in params
        assert "top_p" in params
        assert "max_tokens" in params
        assert "web_search" in params
        assert "enable_thinking" in params
        assert "thinking_level" in params

    def test_get_available_params_excludes_unsupported(self):
        """测试可用参数不包含不支持的参数"""
        llm = ConcreteLLM()
        params = llm.get_available_params()
        # image_size、voice 等不在 ConcreteLLM 的支持列表中
        assert "image_size" not in params
        assert "voice" not in params
        assert "audio_format" not in params
        assert "num_images" not in params

    def test_get_available_params_format(self):
        """测试可用参数的返回格式"""
        llm = ConcreteLLM()
        params = llm.get_available_params()
        for name, info in params.items():
            assert "type" in info
            assert "default" in info

    # ---------- 参数校验测试 ----------

    def test_validate_params_keeps_valid(self):
        """测试参数校验保留有效参数"""
        llm = ConcreteLLM()
        result = llm._validate_params({"temperature": 0.7, "web_search": True})
        assert result["temperature"] == 0.7
        assert result["web_search"] is True

    def test_validate_params_filters_unsupported(self):
        """测试参数校验过滤不支持的参数"""
        llm = ConcreteLLM()
        result = llm._validate_params({"temperature": 0.7, "image_size": "1024x1024"})
        assert result["temperature"] == 0.7
        assert "image_size" not in result

    def test_validate_params_empty_input(self):
        """测试空参数输入"""
        llm = ConcreteLLM()
        result = llm._validate_params({})
        assert result == {}

    def test_validate_params_all_unsupported(self):
        """测试全部参数都不支持的情况"""
        llm = ConcreteLLM()
        result = llm._validate_params({"image_size": "1024", "voice": "alloy"})
        assert result == {}

    def test_validate_params_warns_on_filter(self, caplog):
        """测试过滤不支持参数时记录警告日志"""
        import logging
        with caplog.at_level(logging.WARNING):
            llm = ConcreteLLM()
            llm._validate_params({"unknown_param": "value"})
        assert "unknown_param" in caplog.text
        assert "test_provider" in caplog.text

    # ---------- API Key 获取测试 ----------

    def test_get_api_key_from_constructor(self):
        """测试从构造函数获取API Key"""
        llm = ConcreteLLM(api_key="my-key")
        assert llm._get_api_key() == "my-key"

    def test_get_api_key_from_env(self, monkeypatch):
        """测试从环境变量获取API Key"""
        monkeypatch.setenv("TEST_PROVIDER_API_KEY", "env-key")
        llm = ConcreteLLM(api_key=None)
        assert llm._get_api_key() == "env-key"

    def test_get_api_key_constructor_takes_priority(self, monkeypatch):
        """测试构造函数优先级高于环境变量"""
        monkeypatch.setenv("TEST_PROVIDER_API_KEY", "env-key")
        llm = ConcreteLLM(api_key="constructor-key")
        assert llm._get_api_key() == "constructor-key"

    def test_get_api_key_missing_raises(self, monkeypatch):
        """测试API Key缺失时抛出 AuthenticationError"""
        monkeypatch.delenv("TEST_PROVIDER_API_KEY", raising=False)
        llm = ConcreteLLM(api_key=None)
        from agent_framework.llm.base.exceptions import AuthenticationError
        with pytest.raises(AuthenticationError):
            llm._get_api_key()

    # ---------- 客户端管理测试 ----------

    def test_client_lazy_init(self):
        """测试客户端懒加载，初始化时不创建客户端"""
        llm = ConcreteLLM(api_key="test-key")
        assert llm._client is None

    def test_get_client_creates_instance(self):
        """测试获取客户端时创建 AsyncOpenAI 实例"""
        from openai import AsyncOpenAI
        llm = ConcreteLLM(api_key="test-key")
        client = llm._get_client()
        assert isinstance(client, AsyncOpenAI)

    def test_get_client_reuses_instance(self):
        """测试客户端复用，多次调用返回同一实例"""
        llm = ConcreteLLM(api_key="test-key")
        client1 = llm._get_client()
        client2 = llm._get_client()
        assert client1 is client2

    # ---------- 消息转换测试 ----------

    def test_messages_to_dicts(self):
        """测试 Message 对象列表转换为字典列表"""
        llm = ConcreteLLM()
        messages = [
            Message(role=MessageRole.SYSTEM, content="你是助手"),
            Message(role=MessageRole.USER, content="你好"),
        ]
        result = llm._messages_to_dicts(messages)
        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "你是助手"}
        assert result[1] == {"role": "user", "content": "你好"}

    def test_messages_to_dicts_empty(self):
        """测试空消息列表转换"""
        llm = ConcreteLLM()
        result = llm._messages_to_dicts([])
        assert result == []

    # ---------- 抽象类不可直接实例化测试 ----------

    def test_cannot_instantiate_base_directly(self):
        """测试不能直接实例化 BaseLLM"""
        with pytest.raises(TypeError):
            BaseLLM(api_key="key", model="model")


# ==================== 异步测试 ====================


class TestBaseLLMAsync:
    """测试 BaseLLM 的异步功能"""

    @pytest.mark.asyncio
    async def test_chat_yields_stream_chunks(self):
        """测试 chat 方法产出 StreamChunk"""
        llm = ConcreteLLM()
        messages = [Message(role=MessageRole.USER, content="你好")]
        chunks = []
        async for chunk in llm.chat(messages):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert isinstance(chunks[0], StreamChunk)
        assert chunks[0].content == "test"
