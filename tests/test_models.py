"""测试数据模型：Message, StreamChunk, TokenUsage, Config"""

from milu.llm.base.message import Message, MessageRole
from milu.llm.base.response import StreamChunk, TokenUsage
from milu.llm.base.config import (
    ModelConfig,
    WebSearchConfig,
    ThinkingConfig,
    FunctionCallingConfig,
    ImageGenerationConfig,
    AudioGenerationConfig,
)


class TestMessageRole:
    def test_role_values(self):
        assert MessageRole.SYSTEM == "system"
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.TOOL == "tool"

    def test_role_is_string(self):
        role = MessageRole.USER
        assert isinstance(role, str)
        assert role == "user"


class TestMessage:
    def test_basic_text_message(self):
        msg = Message(role=MessageRole.USER, content="你好")
        assert msg.role == MessageRole.USER
        assert msg.content == "你好"
        assert msg.tool_calls is None
        assert msg.tool_call_id is None
        assert msg.name is None

    def test_system_message(self):
        msg = Message(role=MessageRole.SYSTEM, content="你是一个助手")
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "你是一个助手"

    def test_multimodal_message(self):
        content = [
            {"type": "text", "text": "这张图是什么？"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        msg = Message(role=MessageRole.USER, content=content)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_assistant_message_with_tool_calls(self):
        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "北京"}'}}]
        msg = Message(role=MessageRole.ASSISTANT, tool_calls=tool_calls)
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

    def test_tool_message(self):
        msg = Message(role=MessageRole.TOOL, content="晴天，25°C", tool_call_id="call_1", name="get_weather")
        assert msg.tool_call_id == "call_1"
        assert msg.name == "get_weather"

    def test_message_to_dict_basic(self):
        msg = Message(role=MessageRole.USER, content="你好")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "你好"}

    def test_message_to_dict_excludes_none(self):
        msg = Message(role=MessageRole.USER, content="你好")
        d = msg.to_dict()
        assert "tool_calls" not in d
        assert "tool_call_id" not in d
        assert "name" not in d

    def test_message_to_dict_with_tool_calls(self):
        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "test"}}]
        msg = Message(role=MessageRole.ASSISTANT, tool_calls=tool_calls)
        d = msg.to_dict()
        assert "tool_calls" in d

    def test_message_to_dict_with_tool_result(self):
        msg = Message(role=MessageRole.TOOL, content="结果", tool_call_id="call_1", name="func")
        d = msg.to_dict()
        assert d["tool_call_id"] == "call_1"
        assert d["name"] == "func"

    def test_message_to_dict_multimodal(self):
        content = [
            {"type": "text", "text": "描述图片"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        msg = Message(role=MessageRole.USER, content=content)
        d = msg.to_dict()
        assert d["content"] == content


class TestTokenUsage:
    def test_default_values(self):
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.reasoning_tokens == 0

    def test_custom_values(self):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, reasoning_tokens=30)
        assert usage.total_tokens == 150
        assert usage.reasoning_tokens == 30


class TestStreamChunk:
    def test_content_chunk(self):
        chunk = StreamChunk(content="你好")
        assert chunk.content == "你好"
        assert chunk.reasoning_content is None
        assert chunk.finish_reason is None

    def test_reasoning_chunk(self):
        chunk = StreamChunk(reasoning_content="让我想想...")
        assert chunk.reasoning_content == "让我想想..."
        assert chunk.content is None

    def test_finish_chunk_with_usage(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunk = StreamChunk(finish_reason="stop", usage=usage)
        assert chunk.finish_reason == "stop"
        assert chunk.usage.total_tokens == 15

    def test_tool_calls_chunk(self):
        chunk = StreamChunk(tool_calls=[{"id": "call_1", "function": {"name": "test"}}])
        assert len(chunk.tool_calls) == 1


# ==================== Config Tests ====================


class TestModelConfig:
    def test_default_values(self):
        cfg = ModelConfig(model="qwen-max")
        assert cfg.model == "qwen-max"
        assert cfg.temperature is None
        assert cfg.top_p is None
        assert cfg.max_tokens is None
        assert cfg.stop is None
        assert cfg.stream is True
        assert cfg.frequency_penalty is None
        assert cfg.presence_penalty is None

    def test_custom_values(self):
        cfg = ModelConfig(
            model="kimi-k2.5",
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            stop=["\n\n", "END"],
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )
        assert cfg.temperature == 0.7
        assert cfg.top_p == 0.9
        assert cfg.max_tokens == 2048
        assert cfg.stop == ["\n\n", "END"]
        assert cfg.frequency_penalty == 0.5
        assert cfg.presence_penalty == 0.3

    def test_to_dict_excludes_none(self):
        cfg = ModelConfig(model="qwen-max")
        d = cfg.to_dict()
        assert d == {"model": "qwen-max", "stream": True}

    def test_to_dict_includes_non_none(self):
        cfg = ModelConfig(model="qwen-max", temperature=0.8, max_tokens=1024)
        d = cfg.to_dict()
        assert d == {"model": "qwen-max", "temperature": 0.8, "max_tokens": 1024, "stream": True}

    def test_to_dict_always_includes_stream(self):
        cfg = ModelConfig(model="test")
        d = cfg.to_dict()
        assert d["stream"] is True


class TestWebSearchConfig:
    def test_default_values(self):
        cfg = WebSearchConfig(model="qwen-max")
        assert cfg.web_search is False
        assert cfg.web_search_strategy == "auto"

    def test_custom_values(self):
        cfg = WebSearchConfig(model="qwen-max", web_search=True, web_search_strategy="always")
        assert cfg.web_search is True
        assert cfg.web_search_strategy == "always"

    def test_to_dict_excludes_none(self):
        cfg = WebSearchConfig(model="qwen-max")
        d = cfg.to_dict()
        assert d["model"] == "qwen-max"
        assert d["web_search"] is False
        assert d["web_search_strategy"] == "auto"
        assert d["stream"] is True

    def test_inherits_from_model_config(self):
        cfg = WebSearchConfig(model="kimi-k2.5", temperature=0.5, web_search=True)
        d = cfg.to_dict()
        assert d["temperature"] == 0.5
        assert d["web_search"] is True


class TestThinkingConfig:
    def test_default_values(self):
        cfg = ThinkingConfig(model="qwen-max")
        assert cfg.enable_thinking is False
        assert cfg.thinking_level == "medium"

    def test_custom_values(self):
        cfg_low = ThinkingConfig(model="qwen-max", enable_thinking=True, thinking_level="low")
        assert cfg_low.enable_thinking is True
        assert cfg_low.thinking_level == "low"

        cfg_high = ThinkingConfig(model="qwen-max", enable_thinking=True, thinking_level="high")
        assert cfg_high.enable_thinking is True
        assert cfg_high.thinking_level == "high"

    def test_thinking_level_three_levels(self):
        for level in ["low", "medium", "high"]:
            cfg = ThinkingConfig(model="test", enable_thinking=True, thinking_level=level)
            assert cfg.thinking_level == level

    def test_to_dict_excludes_none(self):
        cfg = ThinkingConfig(model="qwen-max")
        d = cfg.to_dict()
        assert d["model"] == "qwen-max"
        assert d["enable_thinking"] is False
        assert d["thinking_level"] == "medium"
        assert d["stream"] is True

    def test_inherits_from_model_config(self):
        cfg = ThinkingConfig(model="qwen-max", temperature=0.9, max_tokens=4096, enable_thinking=True)
        d = cfg.to_dict()
        assert d["temperature"] == 0.9
        assert d["max_tokens"] == 4096
        assert d["enable_thinking"] is True


class TestFunctionCallingConfig:
    def test_default_values(self):
        cfg = FunctionCallingConfig(model="qwen-max")
        assert cfg.tools is None
        assert cfg.tool_choice == "auto"

    def test_custom_values(self):
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        cfg = FunctionCallingConfig(model="qwen-max", tools=tools, tool_choice="required")
        assert cfg.tools == tools
        assert cfg.tool_choice == "required"

    def test_tool_choice_dict(self):
        cfg = FunctionCallingConfig(
            model="qwen-max",
            tool_choice={"type": "function", "function": {"name": "specific_tool"}},
        )
        assert cfg.tool_choice["type"] == "function"

    def test_to_dict_excludes_none(self):
        cfg = FunctionCallingConfig(model="qwen-max")
        d = cfg.to_dict()
        assert d["model"] == "qwen-max"
        assert "tools" not in d
        assert d["tool_choice"] == "auto"
        assert d["stream"] is True

    def test_inherits_from_model_config(self):
        tools = [{"type": "function", "function": {"name": "search"}}]
        cfg = FunctionCallingConfig(model="qwen-max", temperature=0.6, tools=tools, tool_choice="required")
        d = cfg.to_dict()
        assert d["temperature"] == 0.6
        assert d["tools"] == tools
        assert d["tool_choice"] == "required"


class TestImageGenerationConfig:
    def test_default_values(self):
        cfg = ImageGenerationConfig(model="dall-e-3")
        assert cfg.image_size == "1024x1024"
        assert cfg.image_quality == "standard"
        assert cfg.num_images == 1

    def test_custom_values(self):
        cfg = ImageGenerationConfig(
            model="dall-e-3",
            image_size="1024x1792",
            image_quality="hd",
            num_images=4,
        )
        assert cfg.image_size == "1024x1792"
        assert cfg.image_quality == "hd"
        assert cfg.num_images == 4

    def test_to_dict_excludes_none(self):
        cfg = ImageGenerationConfig(model="dall-e-3")
        d = cfg.to_dict()
        assert d["model"] == "dall-e-3"
        assert d["image_size"] == "1024x1024"
        assert d["image_quality"] == "standard"
        assert d["num_images"] == 1
        assert d["stream"] is True

    def test_inherits_from_model_config(self):
        cfg = ImageGenerationConfig(model="dall-e-3", temperature=0.5)
        d = cfg.to_dict()
        assert d["temperature"] == 0.5


class TestAudioGenerationConfig:
    def test_default_values(self):
        cfg = AudioGenerationConfig(model="tts-1")
        assert cfg.voice is None
        assert cfg.audio_format == "mp3"
        assert cfg.speed == 1.0

    def test_custom_values(self):
        cfg = AudioGenerationConfig(
            model="tts-1",
            voice="alloy",
            audio_format="wav",
            speed=1.5,
        )
        assert cfg.voice == "alloy"
        assert cfg.audio_format == "wav"
        assert cfg.speed == 1.5

    def test_to_dict_excludes_none(self):
        cfg = AudioGenerationConfig(model="tts-1")
        d = cfg.to_dict()
        assert d["model"] == "tts-1"
        assert "voice" not in d
        assert d["audio_format"] == "mp3"
        assert d["speed"] == 1.0
        assert d["stream"] is True

    def test_inherits_from_model_config(self):
        cfg = AudioGenerationConfig(model="tts-1", temperature=0.5, voice="nova", speed=0.8)
        d = cfg.to_dict()
        assert d["temperature"] == 0.5
        assert d["voice"] == "nova"
        assert d["speed"] == 0.8
