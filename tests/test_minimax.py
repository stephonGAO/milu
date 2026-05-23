"""测试 MiniMaxLLM - MiniMax模型实现"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import MockChunk, MockChoice, MockDelta, MockUsage


class TestMiniMaxCapabilities:
    """验证 MiniMaxLLM 的能力标志是否与规格一致"""

    def test_capabilities(self):
        from agent_framework.providers.minimax import MiniMaxLLM

        llm = MiniMaxLLM(api_key="test", model="minimax-speech-02-turbo")
        caps = llm.capabilities

        # 基础能力
        assert caps.supports_streaming is True
        assert caps.supports_function_calling is True
        assert caps.supports_json_mode is True
        # MiniMax 不支持联网搜索和思考模式
        assert caps.supports_web_search is False
        assert caps.supports_thinking is False
        assert caps.supports_embedding is True

        # 多模态理解能力
        assert caps.supports_vision is True
        assert caps.supports_audio_understand is True
        assert caps.supports_video is False
        assert caps.supports_document is False

        # 多模态生成能力 - 支持图片和音频生成
        assert caps.supports_image_generation is True
        assert caps.supports_audio_generation is True

        # 模型规格 - 1M超长上下文
        assert caps.max_context_window == 1000000


class TestMiniMaxAvailableParams:
    """验证 MiniMaxLLM 支持的参数名集合"""

    def test_available_params_include_basic(self):
        from agent_framework.providers.minimax import MiniMaxLLM

        llm = MiniMaxLLM(api_key="test", model="minimax-speech-02-turbo")
        params = llm._get_available_param_names()

        # 基础参数
        assert "temperature" in params
        assert "top_p" in params
        assert "max_tokens" in params
        assert "stop" in params
        assert "frequency_penalty" in params
        assert "presence_penalty" in params

    def test_available_params_include_tools(self):
        from agent_framework.providers.minimax import MiniMaxLLM

        llm = MiniMaxLLM(api_key="test", model="minimax-speech-02-turbo")
        params = llm._get_available_param_names()

        assert "tools" in params
        assert "tool_choice" in params

    def test_available_params_include_image_params(self):
        from agent_framework.providers.minimax import MiniMaxLLM

        llm = MiniMaxLLM(api_key="test", model="minimax-speech-02-turbo")
        params = llm._get_available_param_names()

        # 图片生成参数
        assert "image_size" in params
        assert "image_quality" in params
        assert "num_images" in params

    def test_available_params_include_audio_params(self):
        from agent_framework.providers.minimax import MiniMaxLLM

        llm = MiniMaxLLM(api_key="test", model="minimax-speech-02-turbo")
        params = llm._get_available_param_names()

        # 音频生成参数
        assert "voice" in params
        assert "audio_format" in params
        assert "speed" in params

    def test_no_web_search_or_thinking_params(self):
        from agent_framework.providers.minimax import MiniMaxLLM

        llm = MiniMaxLLM(api_key="test", model="minimax-speech-02-turbo")
        params = llm._get_available_param_names()

        # MiniMax 不支持联网搜索和思考模式参数
        assert "web_search" not in params
        assert "web_search_strategy" not in params
        assert "enable_thinking" not in params
        assert "thinking_level" not in params


class TestMiniMaxChat:
    """测试 MiniMaxLLM 流式聊天功能"""

    @pytest.mark.asyncio
    async def test_basic_chat(self):
        """基础聊天：3个chunk（内容"你"+"好"+结束带usage），验证StreamChunk字段"""
        from agent_framework.models.message import Message, MessageRole
        from agent_framework.providers.minimax import MiniMaxLLM

        # 构造模拟的流式响应：3个chunk
        chunks = [
            # chunk 1: 内容 "你"
            MockChunk(
                choices=[MockChoice(delta=MockDelta(content="你", role="assistant"))]
            ),
            # chunk 2: 内容 "好"
            MockChunk(
                choices=[MockChoice(delta=MockDelta(content="好"))]
            ),
            # chunk 3: 结束，带usage
            MockChunk(
                choices=[MockChoice(delta=MockDelta(), finish_reason="stop")],
                usage=MockUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        ]

        # 将chunks包装成异步迭代器
        async def mock_stream():
            for chunk in chunks:
                yield chunk

        llm = MiniMaxLLM(api_key="test-key", model="minimax-speech-02-turbo")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        with patch.object(llm, "_get_client", return_value=mock_client):
            messages = [Message(role=MessageRole.USER, content="你好")]
            results = []
            async for chunk in llm.chat(messages):
                results.append(chunk)

        # 验证返回了3个 StreamChunk
        assert len(results) == 3

        # chunk 1: 内容 "你"
        assert results[0].content == "你"
        assert results[0].finish_reason is None

        # chunk 2: 内容 "好"
        assert results[1].content == "好"
        assert results[1].finish_reason is None

        # chunk 3: 结束，带usage
        assert results[2].content is None
        assert results[2].finish_reason == "stop"
        assert results[2].usage is not None
        assert results[2].usage.prompt_tokens == 10
        assert results[2].usage.completion_tokens == 5
        assert results[2].usage.total_tokens == 15
